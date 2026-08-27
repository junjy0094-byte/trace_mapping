"""TraceGridMapper: sub-pixel rasterisation and per-cell copper fraction.

The mapper builds one boolean sub-pixel bitmap and derives two outputs:
  _raster_bitmap  – reused by plot.py for the left display panel
  fractions       – per-cell copper density (block-averaged from bitmap)

Sub-pixels are square on any board or grid aspect ratio, which needs a
sub-division count per axis (_raster_sub is the (sub_x, sub_y) pair) rather
than one shared by both: a sub-pixel spans cell_w/sub_x by cell_h/sub_y, so
a single shared count stretches it by exactly the cell aspect ratio. See
_square_subdivisions().

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


_MIN_SUB = 5           # sub-pixels per cell per axis, floor for density accuracy
_ASPECT_TOL = 0.005    # accept a sub-pixel this far from square
_MAX_RASTER_PIXELS = 32_000_000   # ... but never grow the bitmap past this
_SEARCH_LIMIT = 4096   # bound on the sub-division search per axis


def _ceil_div(a, b):
    """Integer ceiling division, both arguments positive."""
    return -(-int(a) // int(b))


def _square_subdivisions(cell_w, cell_h, nx, ny, min_px_x, min_px_y,
                         min_sub=_MIN_SUB, tol=_ASPECT_TOL,
                         max_pixels=_MAX_RASTER_PIXELS):
    """Per-axis sub-pixel counts (sub_x, sub_y) that make sub-pixels square.

    A sub-pixel measures cell_w/sub_x by cell_h/sub_y, so it is square only
    when sub_x : sub_y matches the cell aspect ratio cell_w : cell_h. One
    shared sub-division count -- what this used to use -- therefore yields
    square sub-pixels only for square cells; on a rectangular board (or any
    nx/ny that does not match the board's aspect) it stretched every
    sub-pixel by exactly the cell aspect ratio.

    Both counts stay whole, so the bitmap remains an exact
    (ny*sub_y, nx*sub_x) tiling of the cell grid and per-cell fractions stay
    exact block means -- no cell edge is snapped to a pixel boundary.

    That wholeness is also why the counts are searched rather than derived:
    at the 5-sub-pixel floor there are too few sub-pixels for the nearest
    integer pair to land on an awkward aspect ratio, and the fix is to
    spend a few more of them. The search walks each axis up from its floor
    and takes the first pair within `tol` of square. An elongated cell
    genuinely needs proportionally more sub-pixels along its long side, so
    the only ceiling is `max_pixels` on the whole bitmap; if even that
    cannot buy a square sub-pixel, the squarest affordable pair wins and
    _compute_raster() reports the shortfall.

    min_px_x / min_px_y are per-axis floors on the bitmap size; the result
    always meets them.
    """
    if cell_w <= 0 or cell_h <= 0:
        return min_sub, min_sub

    base_x = max(min_sub, _ceil_div(min_px_x, nx))
    base_y = max(min_sub, _ceil_div(min_px_y, ny))
    ratio = cell_w / cell_h                     # the wanted sub_x : sub_y
    # Budget is on the finished bitmap, not on growth over the floor: a
    # 22:1 cell needs 22x more sub-pixels across than down, and at a 5x18
    # floor that is still only a half-megapixel raster.
    budget = max(base_x * base_y, max_pixels // max(1, nx * ny))

    best_ok = None    # (total, sub_x, sub_y) -- within tol, smallest raster
    best_any = None   # (err, total, sub_x, sub_y) -- squarest within budget

    def offer(sub_x, sub_y):
        """Score one candidate; True once it is square enough to stop."""
        nonlocal best_ok, best_any
        total = sub_x * sub_y
        if sub_x < base_x or sub_y < base_y or total > budget:
            return False
        err = abs(ratio * sub_y / sub_x - 1.0)   # pixel width / height - 1
        if best_any is None or (err, total) < (best_any[0], best_any[1]):
            best_any = (err, total, sub_x, sub_y)
        if err <= tol:
            if best_ok is None or total < best_ok[0]:
                best_ok = (total, sub_x, sub_y)
            return True
        return False

    offer(base_x, base_y)   # always in budget, so best_any is never None
    for sub_y in range(base_y, base_y + _SEARCH_LIMIT):
        sub_x = max(base_x, int(round(sub_y * ratio)))
        if sub_x * sub_y > budget:
            break       # product only grows from here
        if offer(sub_x, sub_y):
            break
    for sub_x in range(base_x, base_x + _SEARCH_LIMIT):
        sub_y = max(base_y, int(round(sub_x / ratio)))
        if sub_x * sub_y > budget:
            break
        if offer(sub_x, sub_y):
            break

    if best_ok is not None:
        return best_ok[1], best_ok[2]
    return best_any[2], best_any[3]


def _square_dims(x_range, y_range, min_px_x, min_px_y):
    """Bitmap dimensions (nx_s, ny_s) with square pixels, meeting both floors.

    Used where no exact cell tiling is possible anyway (non-uniform grids):
    pick one pixel edge length small enough for both floors and derive each
    axis from it, so pixels come out square to within one pixel.
    """
    s = min(x_range / max(1.0, min_px_x), y_range / max(1.0, min_px_y))
    if s <= 0:
        return int(min_px_x), int(min_px_y)
    return (max(int(min_px_x), int(round(x_range / s))),
            max(int(min_px_y), int(round(y_range / s))))


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
        min_display_pixels: Minimum sub-pixel raster size along the board's
                  longer axis. The shorter axis is sized from it so that
                  sub-pixels stay square on any board aspect ratio.
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
    _raster_sub: tuple = field(default=(0, 0), repr=False)

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

    @property
    def sub_pixels(self) -> Tuple[int, int]:
        """(sub_x, sub_y) sub-pixels per cell, (0, 0) when not an exact tiling.

        Normalises _raster_sub, which older caches stored as one scalar
        shared by both axes.
        """
        sub = self._raster_sub
        if isinstance(sub, (tuple, list, np.ndarray)):
            if len(sub) != 2:
                return (0, 0)
            return (int(sub[0]), int(sub[1]))
        return (int(sub), int(sub))

    @property
    def pixel_size(self) -> Tuple[float, float]:
        """(width, height) of one raster sub-pixel, in board units."""
        if self._raster_bitmap is None:
            raise RuntimeError("no raster bitmap available")
        h, w = self._raster_bitmap.shape
        xmin, ymin, xmax, ymax = self.bounds
        return ((xmax - xmin) / w, (ymax - ymin) / h)

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
            sub_x, sub_y = self.sub_pixels
            # Auto-detect when the bitmap came from a cache with no recorded
            # sub-division. The two axes need not agree: square pixels on a
            # rectangular board mean sub_x != sub_y.
            if (sub_x <= 0 or sub_y <= 0) and self.nx > 0 and self.ny > 0:
                if H % self.ny == 0 and W % self.nx == 0:
                    sub_x, sub_y = W // self.nx, H // self.ny
                    self._raster_sub = (sub_x, sub_y)

            # Fast path: bitmap is an exact (ny*sub_y, nx*sub_x) tiling.
            if sub_x > 0 and sub_y > 0 and H == self.ny * sub_y \
                    and W == self.nx * sub_x:
                self.fractions = bitmap.astype(np.float64).reshape(
                    self.ny, sub_y, self.nx, sub_x
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

        Sub-pixels are kept square whatever the board or grid aspect
        ratio, so the display panel is undistorted and every sample covers
        the same area (see _square_subdivisions / _square_dims). That needs
        a sub-division count per axis, not one shared by both.

        For uniform grids the bitmap is (ny*sub_y, nx*sub_x) -- still an
        exact tiling of the cell grid -- sized so the physically longer
        axis has at least self.min_display_pixels; a floor of 5 sub-pixels
        per cell per axis keeps density estimates accurate on coarse grids.

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
            min_px_x = max(self.min_display_pixels, self.nx * _MIN_SUB,
                           int(np.ceil(_MIN_SUB * x_range / float(dx_cells.min()))))
            min_px_y = max(self.min_display_pixels, self.ny * _MIN_SUB,
                           int(np.ceil(_MIN_SUB * y_range / float(dy_cells.min()))))
            nx_s, ny_s = _square_dims(x_range, y_range, min_px_x, min_px_y)
            SUB = (0, 0)  # non-uniform, no exact per-cell sub-pixel tiling
        else:
            # min_display_pixels sizes the physically longer axis; with square
            # pixels that is the one carrying the most of them.
            min_px_x, min_px_y = self.nx * _MIN_SUB, self.ny * _MIN_SUB
            if x_range >= y_range:
                min_px_x = max(min_px_x, self.min_display_pixels)
            else:
                min_px_y = max(min_px_y, self.min_display_pixels)
            SUB = _square_subdivisions(x_range / self.nx, y_range / self.ny,
                                       self.nx, self.ny, min_px_x, min_px_y)
            nx_s, ny_s = self.nx * SUB[0], self.ny * SUB[1]
        sx = (xmax - xmin) / nx_s
        sy = (ymax - ymin) / ny_s

        # Sub-pixels should be square whatever the board aspect ratio; say so
        # when an extreme one cannot be squared inside the pixel budget,
        # rather than silently handing back stretched pixels.
        if abs(sx / sy - 1.0) > _ASPECT_TOL:
            print(f"    WARNING: sub-pixels are {sx/sy:.2f}:1, not square -- "
                  f"squaring a {(xmax-xmin)/(ymax-ymin):.4g}:1 region on a "
                  f"{self.nx}x{self.ny} grid would exceed the "
                  f"{_MAX_RASTER_PIXELS/1e6:.0f}M-pixel raster budget.")

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

                XX, YY = np.meshgrid(xs[c0:c1], ys[r0:r1])

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
                self.ny, SUB[1], self.nx, SUB[0]
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
