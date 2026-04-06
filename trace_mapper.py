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
#  GerberLayer: parse a single .art/.gbr file
# ---------------------------------------------------------------------------

@dataclass
class GerberLayer:
    """Represents one copper layer parsed from a Gerber file."""
    filepath: str
    name: str = ""
    merge_tolerance: float = 0.0
    no_merge: bool = False
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
        fractions: 2D numpy array [ny, nx] of copper fractions (0~1)
    """
    copper: object = None
    copper_polys: list = None
    nx: int = 20
    ny: int = 20
    bounds: Optional[Tuple[float, float, float, float]] = None
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

    def _compute_merged(self):
        """Compute fractions using merged copper geometry (original approach)."""
        from shapely.geometry import box
        from shapely import prepared

        xmin, ymin, xmax, ymax = self.bounds
        dx = (xmax - xmin) / self.nx
        dy = (ymax - ymin) / self.ny
        cell_area = dx * dy

        self.fractions = np.zeros((self.ny, self.nx), dtype=np.float64)
        prep_copper = prepared.prep(self.copper)

        for j in range(self.ny):
            for i in range(self.nx):
                x0 = xmin + i * dx
                y0 = ymin + j * dy
                cell = box(x0, y0, x0 + dx, y0 + dy)

                if not prep_copper.intersects(cell):
                    continue
                if prep_copper.contains(cell):
                    self.fractions[j, i] = 1.0
                    continue
                intersection = self.copper.intersection(cell)
                self.fractions[j, i] = intersection.area / cell_area

    def _compute_individual(self):
        """Compute fractions using individual polygons with STRtree spatial index.

        This avoids unary_union entirely, so fine traces that are geometrically
        separate remain separate. Each cell's fraction is computed by summing
        intersection areas of all polygons that touch that cell, then clamping
        to [0, 1] to handle any overlaps.
        """
        from shapely.geometry import box
        from shapely import strtree

        xmin, ymin, xmax, ymax = self.bounds
        dx = (xmax - xmin) / self.nx
        dy = (ymax - ymin) / self.ny
        cell_area = dx * dy

        self.fractions = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Build spatial index over copper polygons
        tree = strtree.STRtree(self.copper_polys)

        for j in range(self.ny):
            for i in range(self.nx):
                x0 = xmin + i * dx
                y0 = ymin + j * dy
                cell = box(x0, y0, x0 + dx, y0 + dy)

                # Query spatial index for candidate polygons
                try:
                    # Shapely >= 2.0 API
                    candidates = tree.query(cell)
                except TypeError:
                    # Shapely < 2.0 API
                    candidates = tree.query(cell)

                if len(candidates) == 0:
                    continue

                total_area = 0.0
                for idx in candidates:
                    if isinstance(idx, int):
                        poly = self.copper_polys[idx]
                    else:
                        poly = idx  # Shapely < 2.0 returns geometries directly

                    if poly.contains(cell):
                        total_area = cell_area
                        break
                    intersection = poly.intersection(cell)
                    if not intersection.is_empty:
                        total_area += intersection.area

                # Clamp to 1.0 (overlapping polygons could exceed cell_area)
                self.fractions[j, i] = min(total_area / cell_area, 1.0)

    def compute(self):
        """Compute copper fraction for each grid cell."""
        xmin, ymin, xmax, ymax = self.bounds
        dx = (xmax - xmin) / self.nx
        dy = (ymax - ymin) / self.ny
        cell_area = dx * dy

        if cell_area <= 0:
            raise ValueError(f"Invalid grid: bounds={self.bounds}, nx={self.nx}, ny={self.ny}")

        total = self.nx * self.ny

        if self.copper is not None:
            mode = "merged"
        elif self.copper_polys:
            mode = "individual"
        else:
            raise ValueError("No copper geometry provided")

        print(f"Computing {self.nx}x{self.ny} = {total} cells (mode={mode}) ... ",
              flush=True)
        t0 = time.time()

        if mode == "merged":
            self._compute_merged()
        else:
            self._compute_individual()

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


def plot_comparison(layer: GerberLayer, mapper: TraceGridMapper):
    """Side-by-side: raw copper vs fraction heatmap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plot_copper(layer, ax=ax1)
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
                   export_csv=True, plot=True, outdir=None,
                   merge_tolerance=0.0, no_merge=False):
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
    Returns:
        dict: {layer_name: TraceGridMapper}
    """
    if not filepaths:
        print("No files to process.")
        return {}

    # --- Step 1: Load & parse all layers ---
    layers: List[GerberLayer] = []
    for fp in filepaths:
        try:
            layer = GerberLayer(filepath=fp,
                                merge_tolerance=merge_tolerance,
                                no_merge=no_merge)
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
        if layer.no_merge or layer.copper is None:
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
        plt.close(fig)
        print(f"Summary plot saved: {summary_path}")

    return results


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Gerber Trace Mapping: compute copper fraction on NxM grid')
    parser.add_argument('paths', nargs='+',
                        help='Gerber file(s) and/or directories containing .art/.gbr')
    parser.add_argument('--nx', type=int, default=20, help='Grid X divisions (default: 20)')
    parser.add_argument('--ny', type=int, default=20, help='Grid Y divisions (default: 20)')
    parser.add_argument('--no-shared-bounds', action='store_true',
                        help='Use per-layer bounds instead of unified bounds')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory for CSV/PNG (default: same as input)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    parser.add_argument('--no-csv', action='store_true', help='Skip CSV export')
    parser.add_argument('--show', action='store_true', help='Display plots interactively')

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

    args = parser.parse_args()

    files = collect_art_files(args.paths)
    print(f"\nFound {len(files)} Gerber file(s):")
    for f in files:
        print(f"  {f}")

    if args.no_merge:
        print("\nMode: NO-MERGE (individual polygons, STRtree spatial index)")
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
        outdir=args.outdir,
        merge_tolerance=args.merge_tolerance,
        no_merge=args.no_merge,
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
