"""
Gerber Trace Mapping Tool
=========================
Parses Gerber/Artwork (.art, .gbr) files and computes copper area fraction
on an N x M grid for ANSYS APDL trace mapping.

Dependencies:
    pip install gerber shapely numpy matplotlib

Usage:
    python trace_mapper.py layer1.art --nx 20 --ny 20
    python trace_mapper.py layer1.art layer2.art --nx 50 --ny 50 --dpi 150

Merge Resolution Control:
    --merge-tolerance 0.001   (default: 0, no coordinate snapping)
    --no-merge                (skip unary_union, keep individual polygons)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MplRect
from matplotlib.collections import PatchCollection
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import argparse
import time

# ---------------------------------------------------------------------------
#  Gerber -> Shapely polygon conversion helpers
# ---------------------------------------------------------------------------

def _line_to_shapely(prim):
    """Trace line segment -> buffered LineString."""
    from shapely.geometry import LineString
    line = LineString([prim.start, prim.end])
    ap = prim.aperture
    if hasattr(ap, 'diameter'):
        # Round aperture -> round end caps
        return line.buffer(ap.diameter / 2.0, resolution=16, cap_style=1)
    elif hasattr(ap, 'width') and hasattr(ap, 'height'):
        # Rectangular aperture -> approximate with square caps
        half_w = max(ap.width, ap.height) / 2.0
        return line.buffer(half_w, resolution=4, cap_style=3)
    return None


def _arc_to_shapely(prim, num_segments=64):
    """Arc primitive -> buffered arc polyline."""
    from shapely.geometry import LineString
    import math

    cx, cy = prim.center
    sx, sy = prim.start
    ex, ey = prim.end
    r = math.hypot(sx - cx, sy - cy)

    start_angle = math.atan2(sy - cy, sx - cx)
    end_angle = math.atan2(ey - cy, ex - cx)

    # Determine sweep direction
    if hasattr(prim, 'direction') and prim.direction == 'clockwise':
        if end_angle >= start_angle:
            end_angle -= 2 * math.pi
    else:
        if end_angle <= start_angle:
            end_angle += 2 * math.pi

    angles = np.linspace(start_angle, end_angle, num_segments)
    pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]

    if len(pts) < 2:
        return None

    line = LineString(pts)
    ap = prim.aperture
    if hasattr(ap, 'diameter'):
        return line.buffer(ap.diameter / 2.0, resolution=16, cap_style=1)
    elif hasattr(ap, 'width'):
        half_w = max(ap.width, ap.height) / 2.0
        return line.buffer(half_w, resolution=4, cap_style=3)
    return line.buffer(0.001)  # fallback thin arc


def _region_to_shapely(prim):
    """Region (filled polygon) -> Shapely Polygon."""
    from shapely.geometry import Polygon
    import gerber.primitives as gp

    coords = []
    for seg in prim.primitives:
        if isinstance(seg, gp.Line):
            coords.append(seg.start)
        elif isinstance(seg, gp.Arc):
            # Linearize arc within region boundary
            import math
            cx, cy = seg.center
            sx, sy = seg.start
            r = math.hypot(sx - cx, sy - cy)
            sa = math.atan2(sy - cy, sx - cx)
            ea = math.atan2(seg.end[1] - cy, seg.end[0] - cx)
            if hasattr(seg, 'direction') and seg.direction == 'clockwise':
                if ea >= sa:
                    ea -= 2 * math.pi
            else:
                if ea <= sa:
                    ea += 2 * math.pi
            for a in np.linspace(sa, ea, 32):
                coords.append((cx + r * math.cos(a), cy + r * math.sin(a)))

    if len(coords) < 3:
        return None
    return Polygon(coords)


def _flash_to_shapely(prim):
    """Flash (pad) primitive -> Shapely geometry at prim.position."""
    from shapely.geometry import Point, box
    import gerber.primitives as gp

    x, y = prim.position

    if isinstance(prim, gp.Circle):
        return Point(x, y).buffer(prim.radius, resolution=32)

    elif isinstance(prim, gp.Rectangle):
        hw, hh = prim.width / 2.0, prim.height / 2.0
        return box(x - hw, y - hh, x + hw, y + hh)

    elif isinstance(prim, gp.Obround):
        from shapely.geometry import LineString
        hw, hh = prim.width / 2.0, prim.height / 2.0
        if prim.width >= prim.height:
            # Horizontal obround
            line = LineString([(x - hw + hh, y), (x + hw - hh, y)])
            return line.buffer(hh, resolution=32, cap_style=1)
        else:
            line = LineString([(x, y - hh + hw), (x, y + hh - hw)])
            return line.buffer(hw, resolution=32, cap_style=1)

    elif isinstance(prim, gp.Polygon):
        from shapely.geometry import Polygon
        if hasattr(prim, 'vertices') and prim.vertices:
            return Polygon(prim.vertices)

    return None


# ---------------------------------------------------------------------------
#  Interactive polygon exclusion picker
# ---------------------------------------------------------------------------

def pick_exclude_polygons(polys, title="Click polygons to exclude, then close window"):
    """Show all polygons interactively. Click to toggle exclude (red).
    Close the window to confirm. Returns list of non-excluded polygons.

    Args:
        polys: list of Shapely Polygon/MultiPolygon geometries
        title: window title

    Returns:
        filtered: list of polygons that were NOT excluded
    """
    from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.set_aspect('equal')
    ax.set_title(title)

    # Track exclusion state per polygon index
    excluded = set()
    patch_map = {}  # matplotlib patch -> polygon index

    # Draw each polygon as a clickable patch
    for i, geom in enumerate(polys):
        sub_polys = []
        if isinstance(geom, MultiPolygon):
            sub_polys = list(geom.geoms)
        elif isinstance(geom, ShapelyPolygon):
            sub_polys = [geom]
        else:
            continue

        for sp in sub_polys:
            x, y = sp.exterior.xy
            patch = ax.fill(x, y, fc='darkorange', ec='gray', lw=0.5,
                            alpha=0.7, picker=True)[0]
            patch_map[patch] = i

    ax.autoscale_view()

    def on_pick(event):
        patch = event.artist
        if patch not in patch_map:
            return
        idx = patch_map[patch]

        if idx in excluded:
            # Re-include: back to orange
            excluded.discard(idx)
            for p, pidx in patch_map.items():
                if pidx == idx:
                    p.set_facecolor('darkorange')
                    p.set_alpha(0.7)
        else:
            # Exclude: turn red + semi-transparent
            excluded.add(idx)
            for p, pidx in patch_map.items():
                if pidx == idx:
                    p.set_facecolor('red')
                    p.set_alpha(0.3)

        # Update count in title
        ax.set_title(f"{title}  [excluded: {len(excluded)}/{len(polys)}]")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

    # Filter out excluded polygons
    filtered = [p for i, p in enumerate(polys) if i not in excluded]
    print(f"  Interactive selection: excluded {len(excluded)}, "
          f"kept {len(filtered)}/{len(polys)} polygons")
    return filtered


# ---------------------------------------------------------------------------
#  GerberLayer: parse a single .art/.gbr file
# ---------------------------------------------------------------------------

@dataclass
class GerberLayer:
    """Represents one copper layer parsed from a Gerber file."""
    filepath: str
    name: str = ""
    merge_tolerance: float = 0.0
    no_merge: bool = False
    interactive: bool = False
    _parsed: object = field(default=None, repr=False)
    _copper: object = field(default=None, repr=False)  # shapely geometry (merged)
    _copper_polys: list = field(default_factory=list, repr=False)  # individual polygons

    def load(self):
        """Parse the Gerber file using pcb-tools."""
        from gerber.rs274x import read
        print(f"Loading: {self.filepath}")
        self._parsed = read(self.filepath)
        if not self.name:
            self.name = Path(self.filepath).stem
        print(f"  Bounds: {self._parsed.bounds}")
        print(f"  Primitives: {len(self._parsed.primitives)}")
        return self

    def to_polygons(self):
        """Convert all primitives to Shapely geometries.

        Behavior depends on merge settings:
        - no_merge=True: keep individual polygons (no unary_union),
          avoids fine trace merging entirely.
        - merge_tolerance > 0: use set_precision before union to control
          the coordinate grid resolution. Smaller value = finer resolution.
        - merge_tolerance == 0 (default): standard unary_union with full precision.
        """
        import gerber.primitives as gp
        from shapely.ops import unary_union

        if self._parsed is None:
            self.load()

        polys = []
        skip_count = 0

        for prim in self._parsed.primitives:
            try:
                geom = None
                if isinstance(prim, gp.Line):
                    geom = _line_to_shapely(prim)
                elif isinstance(prim, gp.Arc):
                    geom = _arc_to_shapely(prim)
                elif isinstance(prim, gp.Region):
                    geom = _region_to_shapely(prim)
                elif isinstance(prim, (gp.Circle, gp.Rectangle, gp.Obround, gp.Polygon)):
                    geom = _flash_to_shapely(prim)
                else:
                    skip_count += 1
                    continue

                if geom is not None and geom.is_valid and not geom.is_empty:
                    polys.append(geom)
                elif geom is not None and not geom.is_valid:
                    geom = geom.buffer(0)  # attempt to fix invalid geometry
                    if geom.is_valid and not geom.is_empty:
                        polys.append(geom)
            except Exception as e:
                skip_count += 1

        if skip_count > 0:
            print(f"  Skipped {skip_count} unsupported/failed primitives")

        if not polys:
            raise ValueError(f"No valid polygons from {self.filepath}")

        # Store individual polygons (always needed for no_merge mode and bounds)
        self._copper_polys = polys

        # --- INTERACTIVE EXCLUSION ---
        if self.interactive:
            print(f"  Opening interactive picker for {self.name} ...")
            print(f"  Click polygons to exclude (they turn red), then close the window.")
            self._copper_polys = pick_exclude_polygons(
                self._copper_polys,
                title=f"{self.name}: click to exclude, then close window",
            )
            polys = self._copper_polys
            if not polys:
                raise ValueError(f"All polygons excluded for {self.filepath}")

        if self.no_merge:
            # --- NO MERGE MODE ---
            # Skip unary_union entirely; keep polygons separate.
            # This preserves fine traces that would otherwise be merged
            # by coordinate snapping in the union algorithm.
            print(f"  No-merge mode: keeping {len(polys)} individual polygons")
            self._copper = None  # signal that we use _copper_polys
        else:
            # --- MERGE MODE ---
            print(f"  Merging {len(polys)} polygons ", end="", flush=True)

            if self.merge_tolerance > 0:
                # Apply precision grid before union.
                # set_precision snaps coordinates to a grid of the given size.
                # Larger tolerance = coarser grid = more merging of nearby features.
                # Smaller tolerance = finer grid = preserves fine traces.
                try:
                    from shapely import set_precision
                    print(f"(tolerance={self.merge_tolerance}) ... ", end="", flush=True)
                    snapped = [set_precision(g, grid_size=self.merge_tolerance) for g in polys]
                    snapped = [g for g in snapped if g.is_valid and not g.is_empty]
                    if not snapped:
                        print("WARNING: all geometries became invalid after snapping, using originals")
                        snapped = polys
                except ImportError:
                    # Shapely < 2.0 doesn't have set_precision at module level
                    print("(set_precision unavailable, using raw union) ... ", end="", flush=True)
                    snapped = polys
            else:
                print("... ", end="", flush=True)
                snapped = polys

            t0 = time.time()
            self._copper = unary_union(snapped)
            print(f"done ({time.time()-t0:.1f}s)")

        return self._copper if self._copper is not None else self._copper_polys

    @property
    def copper(self):
        """Merged copper geometry. None if no_merge mode."""
        if self._copper is None and not self._copper_polys:
            self.to_polygons()
        return self._copper

    @property
    def copper_polys(self):
        """List of individual copper polygons (always available)."""
        if not self._copper_polys:
            self.to_polygons()
        return self._copper_polys

    @property
    def bounds(self):
        """(xmin, ymin, xmax, ymax) of copper geometry."""
        if self._copper is not None:
            return self._copper.bounds
        # Compute bounds from individual polygons
        if not self._copper_polys:
            self.to_polygons()
        all_bounds = [g.bounds for g in self._copper_polys]
        xmin = min(b[0] for b in all_bounds)
        ymin = min(b[1] for b in all_bounds)
        xmax = max(b[2] for b in all_bounds)
        ymax = max(b[3] for b in all_bounds)
        return (xmin, ymin, xmax, ymax)


# ---------------------------------------------------------------------------
#  TraceGridMapper: N x M grid copper fraction calculation
# ---------------------------------------------------------------------------

@dataclass
class TraceGridMapper:
    """
    Divides a bounding region into an NxM grid and computes
    copper area fraction per cell.

    Supports two modes:
    - Merged mode: copper is a single (Multi)Polygon from unary_union
    - Individual mode: copper_polys is a list of separate polygons
      (no_merge mode). Uses STRtree spatial index for performance.

    Attributes:
        copper: Shapely geometry of merged copper regions (or None)
        copper_polys: List of individual copper polygons (or None)
        nx, ny: Grid divisions in X and Y
        bounds: (xmin, ymin, xmax, ymax) override, or auto from copper
        even_odd: If True, apply even-odd fill rule per cell (odd overlaps=filled,
                  even overlaps=empty). Requires copper_polys (individual mode).
        fractions: 2D numpy array [ny, nx] of copper fractions (0~1)
    """
    copper: object = None
    copper_polys: list = None
    nx: int = 20
    ny: int = 20
    bounds: Optional[Tuple[float, float, float, float]] = None
    even_odd: bool = False
    min_display_pixels: int = 600
    fractions: np.ndarray = field(default=None, repr=False)
    _raster_bitmap: np.ndarray = field(default=None, repr=False)
    _raster_sub: int = field(default=0, repr=False)

    def __post_init__(self):
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

    def _fractions_from_bitmap(self):
        """Block-average ``self._raster_bitmap`` into the ``ny × nx`` density
        grid, using a summed-area-table so arbitrary (bitmap-H, bitmap-W)
        shapes work even when they are not evenly divisible by (ny, nx).

        Called directly on cache hits (where ``_raster_bitmap`` already exists)
        and also for the fast-path where nx/ny changed but the raster didn't.
        """
        bitmap = self._raster_bitmap
        if bitmap is None:
            raise RuntimeError("no raster bitmap available")
        H, W = bitmap.shape

        # Auto-detect SUB (when bitmap came from cache without a recorded SUB).
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
        ii = np.zeros((H + 1, W + 1), dtype=np.int64)
        ii[1:, 1:] = bitmap.astype(np.int64).cumsum(axis=0).cumsum(axis=1)

        xs_i = np.linspace(0, W, self.nx + 1).round().astype(np.int64)
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

        Builds one sub-pixel bitmap and derives two outputs:

        * ``self._raster_bitmap`` — boolean image after the custom XOR rule,
          re-used by ``plot_comparison`` for the left panel (no second pass
          over the polygons, no polygon-level symmetric_difference).
        * ``self.fractions`` — per-cell density (block-averaged from the
          bitmap) used by the right-panel heat map.

        Sub-pixel count (``SUB``) is chosen so the resulting bitmap has at
        least ``self.min_display_pixels`` per axis, giving a sharp left-panel
        display without a separate render pass.  A floor of 5 keeps the
        density estimate accurate even for very coarse grids.  Increase
        ``min_display_pixels`` for sharper detail (cost grows ~quadratically).

        mode: 'even-odd' | 'merged' | 'individual'
        """
        from matplotlib.path import Path as MplPath
        from shapely.geometry import Polygon as SP, MultiPolygon as MP

        SUB = max(5, int(np.ceil(self.min_display_pixels / max(self.nx, self.ny))))
        xmin, ymin, xmax, ymax = self.bounds
        nx_s, ny_s = self.nx * SUB, self.ny * SUB
        sx = (xmax - xmin) / nx_s
        sy = (ymax - ymin) / ny_s

        # Centre of each sub-pixel
        xs = xmin + (np.arange(nx_s) + 0.5) * sx
        ys = ymin + (np.arange(ny_s) + 0.5) * sy

        if mode == "even-odd":
            # uint16 is enough for any realistic overlap count and halves memory
            grid = np.zeros((ny_s, nx_s), dtype=np.uint16)
        else:
            grid = np.zeros((ny_s, nx_s), dtype=np.bool_)

        # Collect source polygon list
        if mode == "merged":
            src = [self.copper]
        else:
            src = self.copper_polys

        n_src = len(src)
        for k, geom in enumerate(src):
            if n_src > 200 and (k + 1) % 500 == 0:
                print(f"    rasterising {k+1}/{n_src} polygons ...", flush=True)

            # Expand to simple Polygon objects
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

                # Build the sub-pixel grid lazily; use broadcasting-style
                # concatenation to avoid the full np.meshgrid copy.
                sub_x = xs[c0:c1]
                sub_y = ys[r0:r1]
                pts = np.empty((sub_y.size * sub_x.size, 2), dtype=np.float64)
                pts[:, 0] = np.repeat(sub_x[np.newaxis, :], sub_y.size, axis=0).ravel()
                pts[:, 1] = np.repeat(sub_y[:, np.newaxis], sub_x.size, axis=1).ravel()

                inside = MplPath(np.asarray(sp.exterior.coords)).contains_points(pts)
                for ring in sp.interiors:
                    inside &= ~MplPath(np.asarray(ring.coords)).contains_points(pts)

                mask = inside.reshape(r1 - r0, c1 - c0)
                if mode == "even-odd":
                    grid[r0:r1, c0:c1] += mask
                else:
                    grid[r0:r1, c0:c1] |= mask

        # Apply the custom fill rule once, at raster level:
        #   even-odd  → 1=fill, 2=empty, 3+ always fill
        #   other     → any coverage = fill
        if mode == "even-odd":
            bitmap = (grid > 0) & (grid != 2)
        else:
            bitmap = grid  # already bool

        # Cache for reuse by plot_comparison's left panel.
        self._raster_bitmap = bitmap
        self._raster_sub = SUB

        # Per-cell density = block-average of the boolean bitmap.
        self.fractions = bitmap.astype(np.float64).reshape(
            self.ny, SUB, self.nx, SUB
        ).mean(axis=(1, 3))

    def compute(self):
        """Compute copper fraction for each grid cell."""
        xmin, ymin, xmax, ymax = self.bounds
        dx = (xmax - xmin) / self.nx
        dy = (ymax - ymin) / self.ny
        cell_area = dx * dy

        if cell_area <= 0:
            raise ValueError(f"Invalid grid: bounds={self.bounds}, nx={self.nx}, ny={self.ny}")

        total = self.nx * self.ny

        # Fast path: a cached raster was already injected (load_cache / direct
        # assignment); just derive per-cell fractions for current nx/ny.
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
        """Return grid metadata dict."""
        xmin, ymin, xmax, ymax = self.bounds
        return {
            'nx': self.nx, 'ny': self.ny,
            'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
            'dx': (xmax - xmin) / self.nx,
            'dy': (ymax - ymin) / self.ny,
        }

    def to_csv(self, filepath: str):
        """Export fractions to CSV (row=Y index, col=X index)."""
        if self.fractions is None:
            self.compute()
        header = (f"# Trace Mapping Grid: {self.nx}x{self.ny}\n"
                  f"# Bounds: {self.bounds}\n"
                  f"# Row=Y(bot->top), Col=X(left->right), Value=copper fraction")
        np.savetxt(filepath, self.fractions, delimiter=',', fmt='%.6f',
                   header=header)
        print(f"Saved: {filepath}")

    def to_dict_array(self):
        """Return list of dicts: [{ix, iy, cx, cy, fraction}, ...] for non-zero cells.
        Useful for APDL integration (future step)."""
        if self.fractions is None:
            self.compute()
        info = self.grid_info
        records = []
        for j in range(self.ny):
            for i in range(self.nx):
                f = self.fractions[j, i]
                if f > 0:
                    records.append({
                        'ix': i, 'iy': j,
                        'cx': info['xmin'] + (i + 0.5) * info['dx'],
                        'cy': info['ymin'] + (j + 0.5) * info['dy'],
                        'fraction': f,
                    })
        return records


# ---------------------------------------------------------------------------
#  Visualization
# ---------------------------------------------------------------------------

def plot_copper(layer: GerberLayer, ax=None, color='darkorange', alpha=0.7):
    """Plot copper geometry outline."""
    from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Get polygon list depending on mode
    if layer.copper is not None:
        geom = layer.copper
        if isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        else:
            polys = [geom]
    else:
        polys = layer.copper_polys

    for poly in polys:
        if not isinstance(poly, ShapelyPolygon):
            continue
        x, y = poly.exterior.xy
        ax.fill(x, y, fc=color, ec='none', alpha=alpha)
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, fc='white', ec='none', alpha=1.0)

    ax.set_aspect('equal')
    ax.set_title(f"Copper: {layer.name}")
    return ax


def plot_fraction_map(mapper: TraceGridMapper, ax=None, cmap='YlOrRd',
                      title=None, show_grid=True):
    """Plot copper fraction heatmap on the grid."""
    if mapper.fractions is None:
        mapper.compute()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    info = mapper.grid_info
    extent = [info['xmin'], info['xmax'], info['ymin'], info['ymax']]

    im = ax.imshow(mapper.fractions, origin='lower', extent=extent,
                   cmap=cmap, vmin=0, vmax=1, aspect='equal',
                   interpolation='nearest')

    if show_grid:
        for i in range(mapper.nx + 1):
            x = info['xmin'] + i * info['dx']
            ax.axvline(x, color='gray', lw=0.3, alpha=0.5)
        for j in range(mapper.ny + 1):
            y = info['ymin'] + j * info['dy']
            ax.axhline(y, color='gray', lw=0.3, alpha=0.5)

    plt.colorbar(im, ax=ax, label='Copper Fraction', shrink=0.8)
    ax.set_title(title or f"Trace Mapping ({mapper.nx}x{mapper.ny})")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return ax


def plot_evenodd_copper(mapper: TraceGridMapper, layer_name="", ax=None,
                        color='darkorange'):
    """Show even-odd copper result derived from already-computed fractions.

    Uses the mapper.fractions grid (product of even-odd per-cell computation)
    to display which cells are filled vs empty.  Cells with fraction > 0 are
    copper; cells with fraction == 0 are empty (background or even-overlap hole).
    """
    import matplotlib.colors as mcolors

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    if mapper.fractions is None:
        mapper.compute()

    info = mapper.grid_info
    extent = [info['xmin'], info['xmax'], info['ymin'], info['ymax']]

    # Build two-color image: copper color where fraction > 0, white elsewhere.
    r, g, b, _ = mcolors.to_rgba(color)
    rgba = np.ones((*mapper.fractions.shape, 4))  # white background
    mask = mapper.fractions > 0
    rgba[mask] = [r, g, b, 0.85]       # filled cells → copper color
    rgba[~mask] = [1.0, 1.0, 1.0, 1.0]  # empty cells  → white

    ax.imshow(rgba, origin='lower', extent=extent, aspect='equal',
              interpolation='nearest')
    ax.set_aspect('equal')
    ax.set_title(f"Cu even-odd: {layer_name}  ({mapper.nx}×{mapper.ny} grid)")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return ax


def plot_comparison(layer: GerberLayer, mapper: TraceGridMapper):
    """Side-by-side: custom-XOR copper bitmap with grid overlay vs fraction heatmap.

    Left panel : copper region after the custom XOR rule (1=fill, 2=empty,
                 3+=always fill) rendered straight from the cached sub-pixel
                 bitmap that ``mapper.compute`` already produced — no
                 polygon-level symmetric_difference pass.
    Right panel: per-cell copper fraction (density) heat map.
    """
    import matplotlib.colors as mcolors

    if mapper.fractions is None:
        mapper.compute()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    info = mapper.grid_info
    extent = [info['xmin'], info['xmax'], info['ymin'], info['ymax']]

    # --- Left panel: render the cached XOR bitmap as an image ---
    bitmap = mapper._raster_bitmap
    if bitmap is not None:
        r, g, b, _ = mcolors.to_rgba('gold')
        rgba = np.empty((*bitmap.shape, 4), dtype=np.float32)
        rgba[..., 0] = np.where(bitmap, r, 1.0)
        rgba[..., 1] = np.where(bitmap, g, 1.0)
        rgba[..., 2] = np.where(bitmap, b, 1.0)
        rgba[..., 3] = 1.0
        ax1.imshow(rgba, origin='lower', extent=extent, aspect='equal',
                   interpolation='nearest')
    else:
        # Fallback (shouldn't happen once compute() ran)
        ax1.set_facecolor('white')

    # Grid overlay
    for i in range(mapper.nx + 1):
        x = info['xmin'] + i * info['dx']
        ax1.axvline(x, color='gray', lw=0.3, alpha=0.5)
    for j in range(mapper.ny + 1):
        y = info['ymin'] + j * info['dy']
        ax1.axhline(y, color='gray', lw=0.3, alpha=0.5)

    ax1.set_xlim(info['xmin'], info['xmax'])
    ax1.set_ylim(info['ymin'], info['ymax'])
    ax1.set_aspect('equal')
    mode_str = "custom XOR" if mapper.even_odd else "coverage"
    ax1.set_title(f"Copper ({mode_str}): {layer.name}  ({mapper.nx}×{mapper.ny} grid)")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # --- Right panel: fraction heatmap ---
    plot_fraction_map(mapper, ax=ax2, title=f"{layer.name} -- Grid Mapping")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
#  Multi-layer batch processing
# ---------------------------------------------------------------------------

def _exclude_largest_polygons(polys, n):
    """Remove the *n* largest polygons (by area) from the list.

    Useful for discarding outer-border polygons that come from the
    Gerber board outline.  Returns the filtered list and prints which
    polygons were removed.
    """
    if n <= 0 or not polys or n >= len(polys):
        if n >= len(polys) and polys:
            print(f"  Warning: exclude count ({n}) >= total polygons ({len(polys)}), "
                  "skipping exclusion.")
        return polys

    indexed = sorted(enumerate(polys), key=lambda x: x[1].area, reverse=True)
    exclude_indices = set(idx for idx, _ in indexed[:n])
    filtered = [p for i, p in enumerate(polys) if i not in exclude_indices]
    print(f"  Excluded {n} largest polygon(s) by area:")
    for idx, poly in indexed[:n]:
        print(f"    polygon #{idx}: area = {poly.area:.6f}")
    return filtered


def collect_art_files(paths: List[str], extensions=('.art', '.gbr')) -> List[str]:
    """
    Resolve input paths to a flat list of Gerber files.
    Accepts any mix of file paths and directory paths.
    Directories are scanned (non-recursive) for matching extensions.
    """
    files = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            found = sorted(f for f in pp.iterdir()
                           if f.is_file() and f.suffix.lower() in extensions)
            if not found:
                print(f"  Warning: no {extensions} files in {pp}")
            files.extend(str(f) for f in found)
        elif pp.is_file():
            files.append(str(pp))
        else:
            print(f"  Warning: path not found -- {pp}")
    return files


def compute_shared_bounds(layers: List[GerberLayer]):
    """Compute unified bounding box across all layers."""
    all_bounds = [l.bounds for l in layers]
    xmin = min(b[0] for b in all_bounds)
    ymin = min(b[1] for b in all_bounds)
    xmax = max(b[2] for b in all_bounds)
    ymax = max(b[3] for b in all_bounds)
    return (xmin, ymin, xmax, ymax)


# ---------------------------------------------------------------------------
#  Raster cache: reuse sub-pixel bitmaps across nx/ny re-runs.
#
#  The cache stores the post-custom-XOR bitmap (NOT per-cell fractions) so
#  the user can resample onto any (nx, ny) grid instantly via a summed-area
#  table.  Cache hits skip the entire Gerber parse + rasterisation phase.
# ---------------------------------------------------------------------------

CACHE_VERSION = 1
CACHE_DIRNAME = ".trace_cache"


def _raster_cache_key(filepath: str, params: dict) -> Tuple[str, dict]:
    """Return (hash, canonical-params). Hash covers file identity + every
    parameter that affects the rasterised bitmap (but NOT nx/ny)."""
    import hashlib, json, os
    st = os.stat(filepath)
    canon = {
        'file': os.path.abspath(filepath),
        'mtime_ns': st.st_mtime_ns,
        'size': st.st_size,
        'v': CACHE_VERSION,
        **params,
    }
    blob = json.dumps(canon, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12], canon


def _raster_cache_path(filepath: str, key_hash: str, outdir=None) -> Path:
    stem = Path(filepath).stem
    root = Path(outdir) if outdir else Path(filepath).parent
    return root / CACHE_DIRNAME / f"{stem}_{key_hash}.npz"


def _save_raster_cache(path: Path, bitmap: np.ndarray,
                       bounds: Tuple[float, float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # uint8 halves disk / mem vs bool (which numpy stores as 1 byte anyway,
    # but uint8 is unambiguous across numpy versions).
    np.savez_compressed(
        path,
        version=np.int32(CACHE_VERSION),
        bitmap=bitmap.astype(np.uint8),
        bounds=np.array(bounds, dtype=np.float64),
    )


def _load_raster_cache(path: Path):
    """Return (bitmap_bool, bounds_tuple) or None if missing/corrupt."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as d:
            if int(d['version']) != CACHE_VERSION:
                return None
            bitmap = d['bitmap'].astype(bool)
            bounds = tuple(d['bounds'].tolist())
            return bitmap, bounds
    except Exception as e:
        print(f"  Cache read failed ({path.name}): {e}")
        return None


def process_layers(filepaths: List[str], nx=20, ny=20,
                   bounds=None, shared_bounds=True,
                   export_csv=True, plot=True, show=False, outdir=None,
                   merge_tolerance=0.0, no_merge=False, interactive=False,
                   even_odd=True, exclude_largest=0,
                   min_display_pixels=600, cache=True):
    """
    Process multiple Gerber layer files.

    Args:
        shared_bounds: If True (default), all layers share one bounding box
                       so that grid cells align across layers.
        outdir: Output directory for CSV/PNG. Default = same dir as input file.
        merge_tolerance: Coordinate grid size for set_precision before union.
                         0 = no snapping (default). Larger values merge more aggressively.
        no_merge: If True, skip unary_union entirely. Individual polygons are
                  used with STRtree spatial index for grid computation. This
                  preserves fine traces that would otherwise merge.
        even_odd: If True (default), apply even-odd fill rule per cell.
                  Odd overlaps = filled, even overlaps = empty (hollow interior).
                  Enables hollow shapes: big rect + small rect inside = hole.
        exclude_largest: Number of largest polygons (by area) to exclude per layer.
                         Useful for removing outer board-outline polygons. 0 = none.
        cache: If True (default), reuse cached sub-pixel rasters keyed on
               file content + all rasterisation params (EXCLUDING nx/ny).
               Lets you change nx/ny and get instant re-mapping from a
               previously rendered bitmap.  Disabled automatically in
               ``interactive`` mode.
    Returns:
        dict: {layer_name: TraceGridMapper}
    """
    if not filepaths:
        print("No files to process.")
        return {}

    skip_merge = no_merge or even_odd
    cache_enabled = cache and not interactive

    # Polygon-affecting subset of params: these feed BOTH the meta cache
    # (own bounds after filtering) and the raster cache (bitmap).
    poly_params = {
        'merge_tolerance': merge_tolerance,
        'no_merge': skip_merge,
        'even_odd': even_odd,
        'exclude_largest': exclude_largest,
    }
    raster_params = {
        **poly_params,
        'min_display_pixels': min_display_pixels,
    }

    def _parse_layer(fp: str) -> Optional[GerberLayer]:
        try:
            layer = GerberLayer(filepath=fp,
                                merge_tolerance=merge_tolerance,
                                no_merge=skip_merge,
                                interactive=interactive)
            layer.load()
            layer.to_polygons()
            if exclude_largest > 0 and layer.copper_polys:
                layer._copper_polys = _exclude_largest_polygons(
                    layer.copper_polys, exclude_largest)
            return layer
        except Exception as e:
            print(f"  ERROR loading {fp}: {e}")
            return None

    # --- Step 1: Determine each file's own bounds ---------------------------
    # Try the meta cache first (params + file mtime/size).  On hit, skip the
    # Gerber parse entirely for now and defer to Step 3 if we end up needing
    # polygons (raster cache miss).
    parsed: dict = {}            # fp -> GerberLayer (if already parsed)
    own_bounds: dict = {}        # fp -> (xmin, ymin, xmax, ymax)

    for fp in filepaths:
        ob = None
        if cache_enabled:
            meta_hash, _ = _raster_cache_key(fp, {**poly_params, 'kind': 'meta'})
            meta_path = _raster_cache_path(fp, 'meta_' + meta_hash, outdir)
            if meta_path.exists():
                try:
                    with np.load(meta_path, allow_pickle=False) as d:
                        ob = tuple(d['bounds'].tolist())
                        print(f"  Meta cache hit: {Path(fp).name}  bounds={ob}")
                except Exception:
                    ob = None
        if ob is None:
            layer = _parse_layer(fp)
            if layer is None:
                continue
            ob = layer.bounds
            parsed[fp] = layer
            if cache_enabled:
                meta_hash, _ = _raster_cache_key(fp, {**poly_params, 'kind': 'meta'})
                meta_path = _raster_cache_path(fp, 'meta_' + meta_hash, outdir)
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(meta_path, bounds=np.array(ob, dtype=np.float64))
        own_bounds[fp] = ob

    if not own_bounds:
        print("No layers loaded successfully.")
        return {}

    # --- Step 2: Determine effective bounds per layer -----------------------
    if bounds is not None:
        effective = {fp: bounds for fp in own_bounds}
        print(f"\nUsing user-specified bounds: {bounds}")
    elif shared_bounds and len(own_bounds) > 1:
        all_b = list(own_bounds.values())
        sb = (min(b[0] for b in all_b), min(b[1] for b in all_b),
              max(b[2] for b in all_b), max(b[3] for b in all_b))
        effective = {fp: sb for fp in own_bounds}
        print(f"\nShared bounds across {len(own_bounds)} layers: {sb}")
    else:
        effective = dict(own_bounds)

    # --- Step 3: Per-layer raster (from cache or fresh) + fractions --------
    results = {}
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)

    for fp, eff_b in effective.items():
        name = Path(fp).stem

        raster_params_fp = {**raster_params,
                            'bounds': tuple(round(v, 9) for v in eff_b)}
        bitmap = cached_bounds = None
        if cache_enabled:
            r_hash, _ = _raster_cache_key(fp, raster_params_fp)
            r_path = _raster_cache_path(fp, r_hash, outdir)
            loaded = _load_raster_cache(r_path)
            if loaded is not None:
                bitmap, cached_bounds = loaded
                print(f"  Raster cache hit: {name}  bitmap={bitmap.shape}")

        if bitmap is None:
            # Need polygons + rasterization
            layer = parsed.get(fp) or _parse_layer(fp)
            if layer is None:
                continue
            parsed[fp] = layer

            if even_odd:
                mapper = TraceGridMapper(
                    copper_polys=layer.copper_polys,
                    nx=nx, ny=ny, bounds=eff_b,
                    even_odd=True,
                    min_display_pixels=min_display_pixels,
                )
            elif layer.no_merge or layer.copper is None:
                mapper = TraceGridMapper(
                    copper_polys=layer.copper_polys,
                    nx=nx, ny=ny, bounds=eff_b,
                    min_display_pixels=min_display_pixels,
                )
            else:
                mapper = TraceGridMapper(
                    copper=layer.copper,
                    nx=nx, ny=ny, bounds=eff_b,
                    min_display_pixels=min_display_pixels,
                )
            mapper.compute()

            if cache_enabled:
                _save_raster_cache(r_path, mapper._raster_bitmap, eff_b)
                print(f"  Raster cache saved: {r_path.name}")
        else:
            # Cache hit path: create empty mapper, inject bitmap, derive fractions.
            mapper = TraceGridMapper(
                nx=nx, ny=ny, bounds=cached_bounds,
                even_odd=even_odd,
                min_display_pixels=min_display_pixels,
            )
            mapper._raster_bitmap = bitmap
            mapper.compute()

        stem = Path(fp).stem
        out_base = Path(outdir) if outdir else Path(fp).parent

        if export_csv:
            csv_path = out_base / f"{stem}.csv"
            mapper.to_csv(str(csv_path))

        if plot:
            stub = parsed.get(fp) or type('LayerStub', (), {
                'name': name, 'filepath': fp,
            })()
            fig = plot_comparison(stub, mapper)
            png_path = out_base / f"{stem}.png"
            fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
            if not show:
                plt.close(fig)
            print(f"Plot saved: {png_path}")

        results[name] = mapper

    # --- Step 4: All-layer summary plot ---
    if plot and len(results) > 1:
        n = len(results)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.atleast_1d(axes).flatten()

        for idx, (name, mp) in enumerate(results.items()):
            plot_fraction_map(mp, ax=axes[idx],
                              title=name, show_grid=(nx <= 30))
        for idx in range(len(results), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f"All Layers -- {nx}x{ny} Grid", fontsize=14)
        fig.tight_layout()
        first_fp = next(iter(effective.keys()))
        summary_path = (Path(outdir) if outdir
                        else Path(first_fp).parent) / "all_layers_summary.png"
        fig.savefig(str(summary_path), dpi=150, bbox_inches='tight')
        if not show:
            plt.close(fig)
        print(f"Summary plot saved: {summary_path}")

    return results


# ---------------------------------------------------------------------------
#  GUI entry point (tkinter)
# ---------------------------------------------------------------------------

def gui_main():
    """Launch a simple tkinter GUI for the Gerber Trace Mapper."""
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
    import threading
    import sys
    import io

    root = tk.Tk()
    root.title("Gerber Trace Mapper")
    root.geometry("720x620")
    root.resizable(True, True)

    # ---- File selection ----
    file_frame = ttk.LabelFrame(root, text="Gerber Files (.art / .gbr)")
    file_frame.pack(fill='x', padx=8, pady=(8, 4))

    file_listbox = tk.Listbox(file_frame, height=4, selectmode=tk.EXTENDED)
    file_listbox.pack(side='left', fill='both', expand=True, padx=(4, 0), pady=4)
    file_scroll = ttk.Scrollbar(file_frame, orient='vertical',
                                command=file_listbox.yview)
    file_scroll.pack(side='left', fill='y', pady=4)
    file_listbox.config(yscrollcommand=file_scroll.set)

    btn_frame = tk.Frame(file_frame)
    btn_frame.pack(side='left', padx=4, pady=4)

    def browse_files():
        paths = filedialog.askopenfilenames(
            title="Select Gerber files",
            filetypes=[("Gerber / Artwork", "*.art *.gbr"),
                       ("All files", "*.*")])
        for p in paths:
            if p not in file_listbox.get(0, tk.END):
                file_listbox.insert(tk.END, p)

    def browse_dir():
        d = filedialog.askdirectory(title="Select directory containing Gerber files")
        if d:
            if d not in file_listbox.get(0, tk.END):
                file_listbox.insert(tk.END, d)

    def remove_selected():
        for idx in reversed(file_listbox.curselection()):
            file_listbox.delete(idx)

    ttk.Button(btn_frame, text="Add Files", command=browse_files).pack(fill='x', pady=1)
    ttk.Button(btn_frame, text="Add Dir", command=browse_dir).pack(fill='x', pady=1)
    ttk.Button(btn_frame, text="Remove", command=remove_selected).pack(fill='x', pady=1)

    # ---- Parameters ----
    param_frame = ttk.LabelFrame(root, text="Grid Parameters")
    param_frame.pack(fill='x', padx=8, pady=4)

    ttk.Label(param_frame, text="NX:").grid(row=0, column=0, padx=4, pady=2, sticky='e')
    nx_var = tk.StringVar(value="20")
    ttk.Entry(param_frame, textvariable=nx_var, width=8).grid(row=0, column=1, padx=4)

    ttk.Label(param_frame, text="NY:").grid(row=0, column=2, padx=4, pady=2, sticky='e')
    ny_var = tk.StringVar(value="20")
    ttk.Entry(param_frame, textvariable=ny_var, width=8).grid(row=0, column=3, padx=4)

    ttk.Label(param_frame, text="Merge Tolerance:").grid(
        row=0, column=4, padx=4, pady=2, sticky='e')
    tol_var = tk.StringVar(value="0.0")
    ttk.Entry(param_frame, textvariable=tol_var, width=10).grid(row=0, column=5, padx=4)

    ttk.Label(param_frame, text="Display Pixels:").grid(
        row=1, column=0, padx=4, pady=2, sticky='e')
    disp_var = tk.StringVar(value="600")
    ttk.Entry(param_frame, textvariable=disp_var, width=8).grid(row=1, column=1, padx=4)
    ttk.Label(param_frame, text="(left-panel raster, larger=sharper/slower)").grid(
        row=1, column=2, columnspan=4, padx=4, pady=2, sticky='w')

    # ---- Output directory ----
    out_frame = ttk.LabelFrame(root, text="Output Directory (blank = same as input)")
    out_frame.pack(fill='x', padx=8, pady=4)
    outdir_var = tk.StringVar(value="")
    ttk.Entry(out_frame, textvariable=outdir_var).pack(
        side='left', fill='x', expand=True, padx=4, pady=4)

    def browse_outdir():
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            outdir_var.set(d)

    ttk.Button(out_frame, text="Browse", command=browse_outdir).pack(
        side='left', padx=4, pady=4)

    # ---- Options ----
    opt_frame = ttk.LabelFrame(root, text="Options")
    opt_frame.pack(fill='x', padx=8, pady=4)

    even_odd_var = tk.BooleanVar(value=True)
    no_merge_var = tk.BooleanVar(value=False)
    interactive_var = tk.BooleanVar(value=False)
    shared_bounds_var = tk.BooleanVar(value=True)
    export_csv_var = tk.BooleanVar(value=True)
    plot_var = tk.BooleanVar(value=True)
    show_var = tk.BooleanVar(value=True)

    ttk.Checkbutton(opt_frame, text="Even-Odd fill", variable=even_odd_var).grid(
        row=0, column=0, padx=6, pady=2, sticky='w')
    ttk.Checkbutton(opt_frame, text="No Merge", variable=no_merge_var).grid(
        row=0, column=1, padx=6, pady=2, sticky='w')
    ttk.Checkbutton(opt_frame, text="Interactive exclude", variable=interactive_var).grid(
        row=0, column=2, padx=6, pady=2, sticky='w')
    ttk.Checkbutton(opt_frame, text="Shared bounds", variable=shared_bounds_var).grid(
        row=1, column=0, padx=6, pady=2, sticky='w')
    ttk.Checkbutton(opt_frame, text="Export CSV", variable=export_csv_var).grid(
        row=1, column=1, padx=6, pady=2, sticky='w')
    ttk.Checkbutton(opt_frame, text="Generate plots", variable=plot_var).grid(
        row=1, column=2, padx=6, pady=2, sticky='w')
    ttk.Checkbutton(opt_frame, text="Show plots", variable=show_var).grid(
        row=1, column=3, padx=6, pady=2, sticky='w')

    exclude_var = tk.BooleanVar(value=False)
    exclude_n_var = tk.StringVar(value="1")
    ttk.Checkbutton(opt_frame, text="Exclude largest poly", variable=exclude_var).grid(
        row=2, column=0, padx=6, pady=2, sticky='w')
    ef = tk.Frame(opt_frame)
    ef.grid(row=2, column=1, padx=6, pady=2, sticky='w')
    ttk.Label(ef, text="Count:").pack(side='left')
    ttk.Spinbox(ef, from_=1, to=20, textvariable=exclude_n_var,
                width=4).pack(side='left', padx=2)

    cache_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(opt_frame, text="Use raster cache", variable=cache_var).grid(
        row=2, column=2, padx=6, pady=2, sticky='w')

    # ---- Log output ----
    log_frame = ttk.LabelFrame(root, text="Log")
    log_frame.pack(fill='both', expand=True, padx=8, pady=4)
    log_text = tk.Text(log_frame, height=12, state='disabled', wrap='word')
    log_text.pack(fill='both', expand=True, padx=4, pady=4)
    log_scroll = ttk.Scrollbar(log_text, orient='vertical', command=log_text.yview)
    log_scroll.pack(side='right', fill='y')
    log_text.config(yscrollcommand=log_scroll.set)

    # ``log`` is safe to call from any thread: it marshals to the Tk main
    # thread via ``root.after(0, ...)``.  Previously calling it from the
    # worker thread triggered "main thread is not in main loop" errors.
    def _append_log(msg: str):
        log_text.config(state='normal')
        log_text.insert(tk.END, msg)
        log_text.see(tk.END)
        log_text.config(state='disabled')

    def log(msg):
        try:
            root.after(0, lambda m=msg: _append_log(m))
        except RuntimeError:
            # Tk already torn down; drop the message silently.
            pass

    # ---- Run button ----
    run_frame = tk.Frame(root)
    run_frame.pack(fill='x', padx=8, pady=(0, 8))

    def run_processing():
        paths = list(file_listbox.get(0, tk.END))
        if not paths:
            messagebox.showwarning("No files", "Please add at least one Gerber file.")
            return

        try:
            nx = int(nx_var.get())
            ny = int(ny_var.get())
        except ValueError:
            messagebox.showerror("Invalid", "NX and NY must be integers.")
            return
        try:
            merge_tol = float(tol_var.get())
        except ValueError:
            messagebox.showerror("Invalid", "Merge tolerance must be a number.")
            return
        try:
            disp_pix = max(50, int(disp_var.get()))
        except ValueError:
            messagebox.showerror("Invalid", "Display pixels must be an integer.")
            return

        outdir = outdir_var.get().strip() or None
        excl_n = 0
        if exclude_var.get():
            try:
                excl_n = int(exclude_n_var.get())
            except ValueError:
                excl_n = 1
        # Snapshot EVERY Tk variable on the main thread; Tk is not
        # thread-safe and reading vars from the worker triggers
        # "main thread is not in main loop".
        opts = {
            'shared_bounds': shared_bounds_var.get(),
            'export_csv': export_csv_var.get(),
            'no_merge': no_merge_var.get(),
            'interactive': interactive_var.get(),
            'even_odd': even_odd_var.get(),
            'cache': cache_var.get(),
        }
        do_plot = plot_var.get()
        do_show = show_var.get()
        run_btn.config(state='disabled')

        def worker():
            # Redirect stdout to log widget
            old_stdout = sys.stdout
            sys.stdout = _LogWriter(log)
            try:
                files = collect_art_files(paths)
                if not files:
                    log("No Gerber files found in the given paths.\n")
                    return
                log(f"Found {len(files)} Gerber file(s)\n")

                # Compute only – no matplotlib calls in this thread
                results = process_layers(
                    filepaths=files,
                    nx=nx, ny=ny,
                    shared_bounds=opts['shared_bounds'],
                    export_csv=opts['export_csv'],
                    plot=False,
                    show=False,
                    outdir=outdir,
                    merge_tolerance=merge_tol,
                    no_merge=opts['no_merge'],
                    interactive=opts['interactive'],
                    even_odd=opts['even_odd'],
                    exclude_largest=excl_n,
                    min_display_pixels=disp_pix,
                    cache=opts['cache'],
                )

                # Summary
                log("\n=== Summary ===\n")
                for name, mapper in results.items():
                    info = mapper.grid_info
                    log(f"  {name}: {info['nx']}x{info['ny']} grid, "
                        f"avg Cu = {mapper.fractions.mean():.4f}, "
                        f"max = {mapper.fractions.max():.4f}\n")

                log("Done.\n")

                # Schedule matplotlib work on the main (tkinter) thread
                root.after(0, lambda: _finish_plots(results, files, outdir))

            except Exception as e:
                log(f"\nERROR: {e}\n")
                import traceback
                log(traceback.format_exc())
                root.after(0, lambda: run_btn.config(state='normal'))
            finally:
                sys.stdout = old_stdout

        def _finish_plots(results, files, out):
            """Generate plots & savefig on the main thread (matplotlib requirement).

            ``plot_comparison`` now only needs ``layer.name`` and
            ``layer.filepath`` for the title/output path, so skip the
            expensive reparse of every Gerber file that was previously
            required for polygon-level re-rendering.
            """
            try:
                if not (do_plot or do_show):
                    return

                fp_by_name = {Path(fp).stem: fp for fp in files}

                for name, mapper in results.items():
                    fp = fp_by_name.get(name)
                    if fp is None:
                        continue
                    stub = type('LayerStub', (), {'name': name, 'filepath': fp})()
                    out_base = Path(out) if out else Path(fp).parent

                    if do_plot:
                        fig = plot_comparison(stub, mapper)
                        png_path = out_base / f"{name}.png"
                        fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
                        if not do_show:
                            plt.close(fig)
                        log(f"Plot saved: {png_path}\n")

                if do_show:
                    plt.show()
            except Exception as e:
                log(f"\nPlot ERROR: {e}\n")
            finally:
                run_btn.config(state='normal')

        threading.Thread(target=worker, daemon=True).start()

    def clear_cache():
        """Delete every .trace_cache entry next to the listed files or in outdir."""
        import shutil
        roots = set()
        od = outdir_var.get().strip()
        if od:
            roots.add(Path(od))
        for p in file_listbox.get(0, tk.END):
            pp = Path(p)
            roots.add(pp if pp.is_dir() else pp.parent)
        removed = 0
        for r in roots:
            c = r / CACHE_DIRNAME
            if c.exists():
                try:
                    shutil.rmtree(c)
                    removed += 1
                except Exception as e:
                    log(f"  Failed to remove {c}: {e}\n")
        log(f"Cleared cache in {removed} location(s)\n")

    run_btn = ttk.Button(run_frame, text="Run", command=run_processing)
    run_btn.pack(side='left', padx=4)
    ttk.Button(run_frame, text="Clear Cache", command=clear_cache).pack(
        side='left', padx=4)
    ttk.Button(run_frame, text="Quit", command=root.destroy).pack(side='right', padx=4)

    root.mainloop()


class _LogWriter:
    """Redirect print() output to a callback function."""
    def __init__(self, callback):
        self._cb = callback

    def write(self, text):
        if text:
            self._cb(text)

    def flush(self):
        pass


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Gerber Trace Mapping: compute copper fraction on NxM grid')
    parser.add_argument('paths', nargs='*',
                        help='Gerber file(s) and/or directories containing .art/.gbr '
                             '(omit to launch GUI)')
    parser.add_argument('--gui', action='store_true',
                        help='Launch graphical user interface')
    parser.add_argument('--nx', type=int, default=20, help='Grid X divisions (default: 20)')
    parser.add_argument('--ny', type=int, default=20, help='Grid Y divisions (default: 20)')
    parser.add_argument('--no-shared-bounds', action='store_true',
                        help='Use per-layer bounds instead of unified bounds')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory for CSV/PNG (default: same as input)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    parser.add_argument('--no-csv', action='store_true', help='Skip CSV export')
    parser.add_argument('--show', action='store_true', help='Display plots interactively')
    parser.add_argument('--interactive', action='store_true',
                        help='Open interactive picker to visually exclude polygons '
                             'before merge. Click polygons to exclude (turn red), '
                             'then close the window to proceed.')

    # --- New merge resolution controls ---
    merge_group = parser.add_argument_group(
        'Merge Resolution',
        'Control how copper polygons are merged. Fine traces may be lost '
        'during unary_union due to coordinate snapping. Use these options '
        'to preserve fine trace geometry.')
    merge_group.add_argument(
        '--merge-tolerance', type=float, default=0.0,
        help='Coordinate grid size for Shapely set_precision before union. '
             '0 = full precision (default). '
             'Typical values: 1e-6 (very fine), 1e-4 (fine), 1e-2 (coarse). '
             'Smaller = preserves finer traces, larger = more aggressive merging.')
    merge_group.add_argument(
        '--no-merge', action='store_true',
        help='Skip polygon union entirely. Each polygon is kept separate and '
             'grid fractions are computed using STRtree spatial index. '
             'Best for preserving very fine traces that otherwise merge.')
    merge_group.add_argument(
        '--no-even-odd', action='store_true',
        help='Disable even-odd fill rule (default: even-odd is ON). '
             'With even-odd OFF, overlapping areas are summed (clamped to 1). '
             'Use this only if hollow shapes are not needed.')
    parser.add_argument(
        '--exclude-largest', type=int, default=0, metavar='N',
        help='Exclude the N largest polygons (by area) per layer. '
             'Useful for removing outer board-outline polygons. (default: 0)')
    parser.add_argument(
        '--display-pixels', type=int, default=600, metavar='N',
        help='Minimum sub-pixel raster size per axis for the left-panel '
             'display (default: 600). Larger = sharper detail, slower '
             '(cost ~quadratic). e.g. 1200, 2000, 4000.')
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Disable reuse of cached sub-pixel rasters. By default, the '
             'raster (NOT nx/ny) is cached per file so changing nx/ny '
             'recomputes density instantly from the same bitmap.')
    parser.add_argument(
        '--clear-cache', action='store_true',
        help='Delete the .trace_cache directory next to each input path '
             '(or in --outdir) before running.')

    args = parser.parse_args()

    # If no files given (or --gui flag), launch GUI
    if not args.paths or args.gui:
        gui_main()
        return

    use_even_odd = not args.no_even_odd

    files = collect_art_files(args.paths)
    print(f"\nFound {len(files)} Gerber file(s):")
    for f in files:
        print(f"  {f}")

    if args.clear_cache:
        import shutil
        roots = {Path(args.outdir)} if args.outdir else set()
        roots.update(Path(p).parent if Path(p).is_file() else Path(p)
                     for p in args.paths)
        for r in roots:
            c = r / CACHE_DIRNAME
            if c.exists():
                shutil.rmtree(c)
                print(f"Cleared cache: {c}")

    if use_even_odd:
        print("\nMode: EVEN-ODD (per-cell union with spatial index)")
    elif args.no_merge:
        print("\nMode: NO-MERGE (individual polygons, area sum, STRtree spatial index)")
    elif args.merge_tolerance > 0:
        print(f"\nMode: MERGE with tolerance={args.merge_tolerance}")
    else:
        print("\nMode: MERGE (full precision)")
    print()

    results = process_layers(
        filepaths=files,
        nx=args.nx, ny=args.ny,
        shared_bounds=not args.no_shared_bounds,
        export_csv=not args.no_csv,
        plot=not args.no_plot,
        show=args.show,
        outdir=args.outdir,
        merge_tolerance=args.merge_tolerance,
        no_merge=args.no_merge,
        interactive=args.interactive,
        even_odd=use_even_odd,
        exclude_largest=args.exclude_largest,
        min_display_pixels=args.display_pixels,
        cache=not args.no_cache,
    )

    if args.show:
        plt.show()

    # Print summary
    print("\n=== Summary ===")
    for name, mapper in results.items():
        info = mapper.grid_info
        print(f"  {name}: {info['nx']}x{info['ny']} grid, "
              f"avg Cu fraction = {mapper.fractions.mean():.4f}, "
              f"max = {mapper.fractions.max():.4f}")


if __name__ == '__main__':
    main()
