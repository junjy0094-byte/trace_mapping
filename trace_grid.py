"""TraceGridMapper: sub-pixel rasterisation and per-cell copper fraction.

The mapper builds one boolean sub-pixel bitmap and derives two outputs:
  _raster_bitmap  – reused by plot.py for the left display panel
  fractions       – per-cell copper density (block-averaged from bitmap)

Cache integration: process.py saves/loads the bitmap so that the full
parse+rasterise phase is skipped on repeated runs with the same parameters.

even_odd here is the legacy fill-rule raster heuristic. By default the
copper geometry handed in via `copper=` has already been resolved from the
Gerber file's own polarity/region-nesting information (see
gerber_layer.GerberLayer.copper), so 'merged' mode already renders holes
correctly and even_odd is only needed for the old approximation.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TraceGridMapper:
    """
    Divides a bounding region into an NxM grid and computes
    copper area fraction per cell.

    Supports two modes:
    - Merged mode: copper is a single (Multi)Polygon from unary_union
    - Individual mode: copper_polys is a list of separate polygons
      (no_merge mode). Uses STRtree spatial index for performance.

    Grid geometry can be uniform (nx/ny + bounds) or custom non-uniform
    (x_edges / y_edges). When custom edges are given they define both the
    cell boundaries and the overall bounds of the mapped region.

    Attributes:
        copper: Shapely geometry of merged copper regions (or None)
        copper_polys: List of individual copper polygons (or None)
        nx, ny: Grid divisions in X and Y (derived from edges when custom)
        bounds: (xmin, ymin, xmax, ymax) override, or auto from copper
        x_edges: Optional 1D array of nx+1 strictly-increasing X cell edges
        y_edges: Optional 1D array of ny+1 strictly-increasing Y cell edges
        even_odd: If True, apply even-odd fill rule per cell (odd=filled,
                  even=empty). Requires copper_polys (individual mode).
        fractions: 2D numpy array [ny, nx] of copper fractions (0~1)
    """
    copper: object = None
    copper_polys: list = None
    nx: int = 20
    ny: int = 20
    bounds: Optional[Tuple[float, float, float, float]] = None
    x_edges: Optional[np.ndarray] = None
    y_edges: Optional[np.ndarray] = None
    even_odd: bool = False
    min_display_pixels: int = 600
    fractions: np.ndarray = field(default=None, repr=False)
    _raster_bitmap: np.ndarray = field(default=None, repr=False)
    _raster_sub: int = field(default=0, repr=False)

    def __post_init__(self):
        if self.x_edges is not None:
            self.x_edges = np.asarray(self.x_edges, dtype=np.float64).ravel()
            if self.x_edges.size < 2:
                raise ValueError("x_edges must contain at least 2 values")
            if not np.all(np.diff(self.x_edges) > 0):
                raise ValueError("x_edges must be strictly increasing")
            self.nx = int(self.x_edges.size - 1)
        if self.y_edges is not None:
            self.y_edges = np.asarray(self.y_edges, dtype=np.float64).ravel()
            if self.y_edges.size < 2:
                raise ValueError("y_edges must contain at least 2 values")
            if not np.all(np.diff(self.y_edges) > 0):
                raise ValueError("y_edges must be strictly increasing")
            self.ny = int(self.y_edges.size - 1)

        if self.x_edges is not None and self.y_edges is not None:
            self.bounds = (float(self.x_edges[0]), float(self.y_edges[0]),
                           float(self.x_edges[-1]), float(self.y_edges[-1]))

        if self.bounds is None:
            if self.copper is not None:
                self.bounds = self.copper.bounds
            elif self.copper_polys:
                all_b = [g.bounds for g in self.copper_polys]
                self.bounds = (
                    min(b[0] for b in all_b),
                    min(b[1] for b in all_b),
                    max(b[2] for b in all_b),
                    max(b[3] for b in all_b),
                )
            else:
                raise ValueError("Either copper or copper_polys must be provided")

    @property
    def x_edges_arr(self) -> np.ndarray:
        """Return X-cell edge coordinates (custom if provided, else uniform)."""
        if self.x_edges is not None:
            return self.x_edges
        xmin, _, xmax, _ = self.bounds
        return np.linspace(xmin, xmax, self.nx + 1)

    @property
    def y_edges_arr(self) -> np.ndarray:
        """Return Y-cell edge coordinates (custom if provided, else uniform)."""
        if self.y_edges is not None:
            return self.y_edges
        _, ymin, _, ymax = self.bounds
        return np.linspace(ymin, ymax, self.ny + 1)

    @property
    def custom_grid(self) -> bool:
        return self.x_edges is not None or self.y_edges is not None

    def _fractions_from_bitmap(self):
        """Block-average self._raster_bitmap into the ny×nx density grid.

        Uses a summed-area-table so arbitrary (bitmap-H, bitmap-W) shapes
        and non-uniform cell edges work even when not evenly divisible.
        """
        bitmap = self._raster_bitmap
        if bitmap is None:
            raise RuntimeError("no raster bitmap available")
        H, W = bitmap.shape
        xmin, ymin, xmax, ymax = self.bounds

        if not self.custom_grid:
            # Auto-detect SUB when bitmap came from cache without a recorded SUB.
            if self._raster_sub <= 0 and self.nx > 0 and self.ny > 0:
                if H % self.ny == 0 and W % self.nx == 0 \
                        and (H // self.ny) == (W // self.nx):
                    self._raster_sub = H // self.ny

            # Fast path: bitmap is an exact (ny*SUB, nx*SUB) tiling.
            if self._raster_sub > 0 and H == self.ny * self._raster_sub \
                    and W == self.nx * self._raster_sub:
                S = self._raster_sub
                self.fractions = bitmap.astype(np.float64).reshape(
                    self.ny, S, self.nx, S
                ).mean(axis=(1, 3))
                return

        # General path: summed-area-table on integer image, fully vectorised.
        # Works uniformly for both equal-division and custom-edge grids.
        ii = np.zeros((H + 1, W + 1), dtype=np.int64)
        ii[1:, 1:] = bitmap.astype(np.int64).cumsum(axis=0).cumsum(axis=1)

        if self.x_edges is not None:
            xs_i = np.clip(
                np.round((self.x_edges - xmin) / (xmax - xmin) * W),
                0, W).astype(np.int64)
        else:
            xs_i = np.linspace(0, W, self.nx + 1).round().astype(np.int64)
        if self.y_edges is not None:
            ys_i = np.clip(
                np.round((self.y_edges - ymin) / (ymax - ymin) * H),
                0, H).astype(np.int64)
        else:
            ys_i = np.linspace(0, H, self.ny + 1).round().astype(np.int64)

        x0s, x1s = xs_i[:-1], xs_i[1:]
        y0s, y1s = ys_i[:-1], ys_i[1:]

        sums = (ii[np.ix_(y1s, x1s)] - ii[np.ix_(y0s, x1s)]
                - ii[np.ix_(y1s, x0s)] + ii[np.ix_(y0s, x0s)])
        areas = (y1s - y0s)[:, None] * (x1s - x0s)[None, :]
        fractions = np.zeros_like(sums, dtype=np.float64)
        np.divide(sums, areas, out=fractions, where=areas > 0)
        self.fractions = fractions

    def _compute_raster(self, mode):
        """Fast grid computation using point-sampling rasterisation.

        For uniform grids, sub-pixel count (SUB) is chosen so the bitmap
        has at least self.min_display_pixels per axis; floor of 5 keeps
        density estimates accurate on very coarse grids.

        For custom (non-uniform) grids the bitmap is oversampled so the
        thinnest cell still has at least 5 sub-pixels, while respecting
        min_display_pixels as a floor on total bitmap size.

        mode: 'even-odd' | 'merged' | 'individual'
        """
        import shapely
        from shapely.geometry import Polygon as SP, MultiPolygon as MP

        xmin, ymin, xmax, ymax = self.bounds
        x_range = xmax - xmin
        y_range = ymax - ymin

        if self.custom_grid:
            # Scale raster so every cell - even the smallest - gets >=5 px.
            dx_cells = np.diff(self.x_edges_arr)
            dy_cells = np.diff(self.y_edges_arr)
            nx_s = max(self.min_display_pixels, self.nx * 5,
                       int(np.ceil(5.0 * x_range / float(dx_cells.min()))))
            ny_s = max(self.min_display_pixels, self.ny * 5,
                       int(np.ceil(5.0 * y_range / float(dy_cells.min()))))
            SUB = 0  # non-uniform, no single per-cell sub-pixel count
        else:
            SUB = max(5, int(np.ceil(self.min_display_pixels / max(self.nx, self.ny))))
            nx_s, ny_s = self.nx * SUB, self.ny * SUB
        sx = (xmax - xmin) / nx_s
        sy = (ymax - ymin) / ny_s

        xs = xmin + (np.arange(nx_s) + 0.5) * sx
        ys = ymin + (np.arange(ny_s) + 0.5) * sy

        if mode == "even-odd":
            grid = np.zeros((ny_s, nx_s), dtype=np.uint16)
        else:
            grid = np.zeros((ny_s, nx_s), dtype=np.bool_)

        src = [self.copper] if mode == "merged" else self.copper_polys
        n_src = len(src)

        for k, geom in enumerate(src):
            if n_src > 200 and (k + 1) % 500 == 0:
                print(f"    rasterising {k+1}/{n_src} polygons ...", flush=True)

            if isinstance(geom, MP):
                simple = list(geom.geoms)
            elif isinstance(geom, SP):
                simple = [geom]
            else:
                continue

            for sp in simple:
                if sp.is_empty:
                    continue
                pxmin, pymin, pxmax, pymax = sp.bounds
                c0 = max(0, int((pxmin - xmin) / sx))
                c1 = min(nx_s, int(np.ceil((pxmax - xmin) / sx)))
                r0 = max(0, int((pymin - ymin) / sy))
                r1 = min(ny_s, int(np.ceil((pymax - ymin) / sy)))
                if c0 >= c1 or r0 >= r1:
                    continue

                sub_x = xs[c0:c1]
                sub_y = ys[r0:r1]
                XX, YY = np.meshgrid(sub_x, sub_y)

                # shapely.contains_xy uses GEOS's prepared-geometry predicate
                # (spatially indexed) rather than a plain per-point ray cast
                # against every edge, and it already excludes interior rings
                # (holes) as part of standard polygon containment -- an
                # order-of-magnitude faster than matplotlib's Path.contains_points
                # for complex, many-vertex polygons, with no separate hole pass.
                mask = shapely.contains_xy(sp, XX, YY)

                if mode == "even-odd":
                    grid[r0:r1, c0:c1] += mask
                else:
                    grid[r0:r1, c0:c1] |= mask

        # Apply fill rule: even-odd → 1=fill, 2=empty, 3+=always fill
        if mode == "even-odd":
            bitmap = (grid > 0) & (grid != 2)
        else:
            bitmap = grid

        self._raster_bitmap = bitmap
        self._raster_sub = SUB

        if self.custom_grid:
            # Non-uniform: fall back to the shared SAT averager.
            self._fractions_from_bitmap()
        else:
            self.fractions = bitmap.astype(np.float64).reshape(
                self.ny, SUB, self.nx, SUB
            ).mean(axis=(1, 3))

    def compute(self):
        """Compute copper fraction for each grid cell."""
        dx_cells = np.diff(self.x_edges_arr)
        dy_cells = np.diff(self.y_edges_arr)
        if dx_cells.min() <= 0 or dy_cells.min() <= 0:
            raise ValueError(
                f"Invalid grid: bounds={self.bounds}, nx={self.nx}, ny={self.ny}, "
                f"custom={self.custom_grid}")

        total = self.nx * self.ny

        # Fast path: cached raster already injected; derive fractions only.
        if self._raster_bitmap is not None and not (self.copper_polys or self.copper):
            t0 = time.time()
            self._fractions_from_bitmap()
            elapsed = time.time() - t0
            nonzero = np.count_nonzero(self.fractions)
            print(f"Reused cached raster {self._raster_bitmap.shape} → "
                  f"{self.nx}x{self.ny} cells in {elapsed:.2f}s  |  "
                  f"non-zero: {nonzero}/{total}  |  "
                  f"avg fraction: {self.fractions.mean():.4f}")
            return self.fractions

        if self.even_odd:
            if not self.copper_polys:
                raise ValueError("even_odd mode requires copper_polys (individual polygons)")
            mode = "even-odd"
        elif self.copper is not None:
            mode = "merged"
        elif self.copper_polys:
            mode = "individual"
        else:
            raise ValueError("No copper geometry provided")

        print(f"Computing {self.nx}x{self.ny} = {total} cells (mode={mode}) ... ",
              flush=True)
        t0 = time.time()

        self._compute_raster(mode)

        elapsed = time.time() - t0
        nonzero = np.count_nonzero(self.fractions)
        print(f"  Done in {elapsed:.1f}s  |  "
              f"non-zero cells: {nonzero}/{total}  |  "
              f"avg fraction: {self.fractions.mean():.4f}")
        return self.fractions

    @property
    def grid_info(self):
        """Return grid metadata dict.

        For non-uniform grids, 'dx' / 'dy' report the mean cell size; use
        'x_edges' / 'y_edges' for per-cell boundaries.
        """
        xmin, ymin, xmax, ymax = self.bounds
        x_edges = self.x_edges_arr
        y_edges = self.y_edges_arr
        return {
            'nx': self.nx, 'ny': self.ny,
            'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
            'dx': float(np.mean(np.diff(x_edges))),
            'dy': float(np.mean(np.diff(y_edges))),
            'x_edges': x_edges,
            'y_edges': y_edges,
            'custom': self.custom_grid,
        }

    def to_csv(self, filepath: str):
        """Export fractions to CSV (row=Y index, col=X index)."""
        if self.fractions is None:
            self.compute()
        header_lines = [
            f"Trace Mapping Grid: {self.nx}x{self.ny}",
            f"Bounds: {self.bounds}",
            f"Grid: {'custom (non-uniform)' if self.custom_grid else 'uniform'}",
        ]
        if self.custom_grid:
            x_edges_str = ','.join(f'{v:.6f}' for v in self.x_edges_arr)
            y_edges_str = ','.join(f'{v:.6f}' for v in self.y_edges_arr)
            header_lines.append(f"X edges: {x_edges_str}")
            header_lines.append(f"Y edges: {y_edges_str}")
        header_lines.append("Row=Y(bot->top), Col=X(left->right), Value=copper fraction")
        header = "\n".join(header_lines)
        np.savetxt(filepath, self.fractions, delimiter=',', fmt='%.6f',
                   header=header)
        print(f"Saved: {filepath}")

    def to_dict_array(self):
        """Return list of dicts [{ix, iy, cx, cy, fraction}, ...] for non-zero cells."""
        if self.fractions is None:
            self.compute()
        x_edges = self.x_edges_arr
        y_edges = self.y_edges_arr
        cx = 0.5 * (x_edges[:-1] + x_edges[1:])
        cy = 0.5 * (y_edges[:-1] + y_edges[1:])
        records = []
        for j in range(self.ny):
            for i in range(self.nx):
                f = self.fractions[j, i]
                if f > 0:
                    records.append({
                        'ix': i, 'iy': j,
                        'cx': float(cx[i]), 'cy': float(cy[j]),
                        'fraction': f,
                    })
        return records
