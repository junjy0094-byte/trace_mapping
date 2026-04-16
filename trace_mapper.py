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
    fractions: np.ndarray = field(default=None, repr=False)

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

    def _compute_raster(self, mode):
        """Fast grid computation using point-sampling rasterisation.

        Instead of expensive Shapely polygon-polygon intersection per cell,
        creates a fine sub-pixel grid and tests point-in-polygon with
        matplotlib Path (C-optimised).  The sub-pixel bitmap is then
        block-averaged to produce the per-cell copper fraction.

        mode: 'even-odd' | 'merged' | 'individual'
        """
        from matplotlib.path import Path as MplPath
        from shapely.geometry import Polygon as SP, MultiPolygon as MP

        SUB = 5                               # sub-samples per cell per axis
        xmin, ymin, xmax, ymax = self.bounds
        nx_s, ny_s = self.nx * SUB, self.ny * SUB
        sx = (xmax - xmin) / nx_s
        sy = (ymax - ymin) / ny_s

        # Centre of each sub-pixel
        xs = xmin + (np.arange(nx_s) + 0.5) * sx
        ys = ymin + (np.arange(ny_s) + 0.5) * sy

        if mode == "even-odd":
            grid = np.zeros((ny_s, nx_s), dtype=np.int32)
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

                gx, gy = np.meshgrid(xs[c0:c1], ys[r0:r1])
                pts = np.column_stack([gx.ravel(), gy.ravel()])

                inside = MplPath(np.array(sp.exterior.coords)).contains_points(pts)
                for ring in sp.interiors:
                    inside &= ~MplPath(np.array(ring.coords)).contains_points(pts)

                mask = inside.reshape(r1 - r0, c1 - c0)
                if mode == "even-odd":
                    grid[r0:r1, c0:c1] += mask
                else:
                    grid[r0:r1, c0:c1] |= mask

        # Collapse sub-pixels → grid cells
        if mode == "even-odd":
            sampled = (grid % 2 == 1).astype(np.float64)
        else:
            sampled = grid.astype(np.float64)

        self.fractions = sampled.reshape(
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

def _get_design_copper_geometry(layer: GerberLayer):
    """Return copper geometry for design-view plotting (actual Cu area only)."""
    from shapely.ops import unary_union

    # Prefer merged copper if already available.
    if layer.copper is not None:
        return layer.copper

    # In no-merge/even-odd runs, layer.copper can be None; for plotting the
    # actual design Cu shape, stitch all Cu polygons once for display.
    if layer.copper_polys:
        return unary_union(layer.copper_polys)

    return None


def _grid_blocks_to_geometry(mapper: TraceGridMapper, min_fraction=0.0):
    """Build stitched geometry from grid blocks used for density evaluation.

    Each grid cell with fraction >= min_fraction is converted to its cell
    rectangle and then stitched via unary_union.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union

    if mapper.fractions is None:
        mapper.compute()

    info = mapper.grid_info
    dx, dy = info['dx'], info['dy']
    xmin, ymin = info['xmin'], info['ymin']

    cells = []
    for j in range(mapper.ny):
        y0 = ymin + j * dy
        y1 = y0 + dy
        for i in range(mapper.nx):
            if mapper.fractions[j, i] >= min_fraction:
                x0 = xmin + i * dx
                x1 = x0 + dx
                cells.append(box(x0, y0, x1, y1))

    if not cells:
        return None
    return unary_union(cells)


def plot_copper(geom, layer_name="", ax=None, color='darkorange', alpha=0.7):
    """Plot filled copper geometry from a Shapely geometry."""
    from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if geom is None or geom.is_empty:
        polys = []
    else:
        if isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        else:
            polys = [geom]

    for poly in polys:
        if not isinstance(poly, ShapelyPolygon):
            continue
        x, y = poly.exterior.xy
        ax.fill(x, y, fc=color, ec='none', alpha=alpha)
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, fc='white', ec='none', alpha=1.0)

    ax.set_aspect('equal')
    ax.set_title(f"Design Cu: {layer_name}")
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
                        color='darkorange', filled_threshold=0.5):
    """Show even-odd copper result derived from already-computed fractions.

    Uses the mapper.fractions grid (product of even-odd per-cell computation)
    to display which cells are filled vs empty. Cells with fraction >=
    filled_threshold are treated as copper.
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
    # A strict >0 mask can make almost every cell appear filled due to
    # sub-pixel edge hits. Use a configurable threshold to recover pattern.
    mask = mapper.fractions >= filled_threshold
    rgba[mask] = [r, g, b, 0.85]       # filled cells → copper color
    rgba[~mask] = [1.0, 1.0, 1.0, 1.0]  # empty cells  → white

    ax.imshow(rgba, origin='lower', extent=extent, aspect='equal',
              interpolation='nearest')
    ax.set_aspect('equal')
    ax.set_title(
        f"Cu even-odd: {layer_name}  ({mapper.nx}×{mapper.ny} grid, th={filled_threshold:g})"
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return ax


def plot_comparison(layer: GerberLayer, mapper: TraceGridMapper):
    """Side-by-side: raw copper artwork vs fraction heatmap.

    Left panel : actual Gerber Cu drawing (yellow) with grid overlay
    Right panel: copper fraction heatmap (grid mapping result)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    design_geom = _get_design_copper_geometry(layer)
    plot_copper(design_geom, layer_name=f"{layer.name} (Gerber Cu)", ax=ax1)

    # Overlay mapping grid so each cell shows the original Gerber drawing inside.
    info = mapper.grid_info
    for i in range(mapper.nx + 1):
        x = info['xmin'] + i * info['dx']
        ax1.axvline(x, color='gray', lw=0.3, alpha=0.5)
    for j in range(mapper.ny + 1):
        y = info['ymin'] + j * info['dy']
        ax1.axhline(y, color='gray', lw=0.3, alpha=0.5)

    plot_fraction_map(mapper, ax=ax2, title=f"{layer.name} -- Grid Mapping")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
#  Multi-layer batch processing
# ---------------------------------------------------------------------------

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


def process_layers(filepaths: List[str], nx=20, ny=20,
                   bounds=None, shared_bounds=True,
                   export_csv=True, plot=True, show=False, outdir=None,
                   merge_tolerance=0.0, no_merge=False, interactive=False,
                   even_odd=True):
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
    Returns:
        dict: {layer_name: TraceGridMapper}
    """
    if not filepaths:
        print("No files to process.")
        return {}

    # --- Step 1: Load & parse all layers ---
    # When even_odd mode is active we only need individual polygons,
    # so skip the expensive unary_union merge entirely.
    skip_merge = no_merge or even_odd
    layers: List[GerberLayer] = []
    for fp in filepaths:
        try:
            layer = GerberLayer(filepath=fp,
                                merge_tolerance=merge_tolerance,
                                no_merge=skip_merge,
                                interactive=interactive)
            layer.load()
            layer.to_polygons()
            layers.append(layer)
        except Exception as e:
            print(f"  ERROR loading {fp}: {e}")

    if not layers:
        print("No layers loaded successfully.")
        return {}

    # --- Step 2: Determine bounds ---
    if bounds is not None:
        unified_bounds = bounds
        print(f"\nUsing user-specified bounds: {unified_bounds}")
    elif shared_bounds and len(layers) > 1:
        unified_bounds = compute_shared_bounds(layers)
        print(f"\nShared bounds across {len(layers)} layers: {unified_bounds}")
    else:
        unified_bounds = None

    # --- Step 3: Compute grid fractions for each layer ---
    results = {}
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)

    for layer in layers:
        b = unified_bounds if unified_bounds else None

        # Create mapper with appropriate mode
        if even_odd:
            # Even-odd fill rule: hollow shapes supported, fast per-cell XOR.
            # Requires individual polygons (no global merge).
            mapper = TraceGridMapper(
                copper_polys=layer.copper_polys,
                nx=nx, ny=ny, bounds=b,
                even_odd=True,
            )
        elif layer.no_merge or layer.copper is None:
            mapper = TraceGridMapper(
                copper_polys=layer.copper_polys,
                nx=nx, ny=ny, bounds=b,
            )
        else:
            mapper = TraceGridMapper(
                copper=layer.copper,
                nx=nx, ny=ny, bounds=b,
            )

        mapper.compute()

        # Determine output path
        stem = Path(layer.filepath).stem
        out_base = Path(outdir) if outdir else Path(layer.filepath).parent

        if export_csv:
            csv_path = out_base / f"{stem}.csv"
            mapper.to_csv(str(csv_path))

        if plot:
            fig = plot_comparison(layer, mapper)
            png_path = out_base / f"{stem}.png"
            fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
            if not show:
                plt.close(fig)
            print(f"Plot saved: {png_path}")

        results[layer.name] = mapper

    # --- Step 4: All-layer summary plot ---
    if plot and len(layers) > 1:
        n = len(layers)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.atleast_1d(axes).flatten()

        for idx, layer in enumerate(layers):
            plot_fraction_map(results[layer.name], ax=axes[idx],
                              title=layer.name, show_grid=(nx <= 30))
        for idx in range(len(layers), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f"All Layers -- {nx}x{ny} Grid", fontsize=14)
        fig.tight_layout()
        summary_path = (Path(outdir) if outdir
                        else Path(layers[0].filepath).parent) / "all_layers_summary.png"
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

    # ---- Log output ----
    log_frame = ttk.LabelFrame(root, text="Log")
    log_frame.pack(fill='both', expand=True, padx=8, pady=4)
    log_text = tk.Text(log_frame, height=12, state='disabled', wrap='word')
    log_text.pack(fill='both', expand=True, padx=4, pady=4)
    log_scroll = ttk.Scrollbar(log_text, orient='vertical', command=log_text.yview)
    log_scroll.pack(side='right', fill='y')
    log_text.config(yscrollcommand=log_scroll.set)

    def log(msg):
        log_text.config(state='normal')
        log_text.insert(tk.END, msg)
        log_text.see(tk.END)
        log_text.config(state='disabled')
        root.update_idletasks()

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

        outdir = outdir_var.get().strip() or None
        run_btn.config(state='disabled')
        do_plot = plot_var.get()
        do_show = show_var.get()

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
                    shared_bounds=shared_bounds_var.get(),
                    export_csv=export_csv_var.get(),
                    plot=False,
                    show=False,
                    outdir=outdir,
                    merge_tolerance=merge_tol,
                    no_merge=no_merge_var.get(),
                    interactive=interactive_var.get(),
                    even_odd=even_odd_var.get(),
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
            """Generate plots & savefig on the main thread (matplotlib requirement)."""
            try:
                if do_plot or do_show:
                    # Reload layers minimally for plotting (need polygon data)
                    layers_by_name = {}
                    for fp in files:
                        try:
                            layer = GerberLayer(filepath=fp, no_merge=True)
                            layer.load()
                            layer.to_polygons()
                            layers_by_name[layer.name] = layer
                        except Exception:
                            pass

                    for name, mapper in results.items():
                        layer = layers_by_name.get(name)
                        if layer is None:
                            continue
                        stem = Path(layer.filepath).stem
                        out_base = Path(out) if out else Path(layer.filepath).parent

                        if do_plot:
                            fig = plot_comparison(layer, mapper)
                            png_path = out_base / f"{stem}.png"
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

    run_btn = ttk.Button(run_frame, text="Run", command=run_processing)
    run_btn.pack(side='left', padx=4)
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
