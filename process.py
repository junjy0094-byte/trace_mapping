"""Batch processing orchestrator: parse -> rasterise -> export per layer.

process_layers() is the main entry point. Workflow:
  Step 1  Determine each layer's own bounds (meta cache or fresh parse).
  Step 2  Compute effective bounds (shared across layers, or per-layer).
  Step 3  Rasterise each layer (raster cache or fresh compute) and export.
  Step 4  Multi-layer summary plot (when >1 layer).

Helpers:
  collect_art_files(paths)      -- resolve file/dir paths to .art/.gbr list
  compute_shared_bounds(layers) -- unified bounding box across layers

Bounds: each layer's extents come from its Gerber file by default, shared
across layers when shared_bounds=True. Passing `bounds` (CLI --bounds, or
the GUI's Mapping Bounds fields) overrides both and maps every layer over
one explicit box, so grids line up across runs and across boards.

Default fill resolution: use_polarity=True resolves each layer's copper
from its own Gerber level polarity (%LPD*%/%LPC*%) and region contour
nesting -- see gerber_layer.GerberLayer -- instead of the legacy even-odd
raster heuristic (even_odd=True).
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from gerber_layer import GerberLayer
from trace_grid import TraceGridMapper
from cache import (CACHE_DIRNAME, _file_identity_hash, _raster_params_hash,
                   _raster_cache_path, _save_raster_cache, _load_raster_cache)
from plot import plot_comparison, plot_fraction_map
from apdl_export import write_reference_full_model_apdl


def load_grid_csv(path: str) -> np.ndarray:
    """Load a column-vector CSV of grid coordinates.

    Accepts any cell containing a parseable float; non-numeric rows (e.g.
    headers) are skipped. Returns a 1D float64 array sorted ascending.
    Raises ValueError if fewer than 2 distinct values are found.
    """
    vals = []
    with open(path, 'r', newline='') as f:
        for row in csv.reader(f):
            for cell in row:
                s = cell.strip()
                if not s:
                    continue
                try:
                    vals.append(float(s))
                except ValueError:
                    continue
    if len(vals) < 2:
        raise ValueError(f"Grid CSV {path!r} must contain at least 2 numeric values")
    arr = np.array(vals, dtype=np.float64)
    arr = np.unique(arr)  # sort + dedupe
    if arr.size < 2:
        raise ValueError(f"Grid CSV {path!r} must contain at least 2 distinct values")
    return arr


def collect_art_files(paths: List[str], extensions=('.art', '.gbr')) -> List[str]:
    """Resolve input paths to a flat list of Gerber files.
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


def normalize_bounds(bounds):
    """Validate a user-supplied bounding box -> (xmin, ymin, xmax, ymax) floats.

    None passes through, meaning "derive the bounds from the Gerber files".
    """
    if bounds is None:
        return None
    vals = [float(v) for v in bounds]
    if len(vals) != 4:
        raise ValueError(
            f"bounds needs exactly 4 numbers (xmin ymin xmax ymax), got {len(vals)}")
    xmin, ymin, xmax, ymax = vals
    if not (xmax > xmin and ymax > ymin):
        raise ValueError(
            f"invalid bounds {tuple(vals)}: require xmin < xmax and ymin < ymax")
    return (xmin, ymin, xmax, ymax)


@dataclass
class _GridContext:
    """Resolved grid geometry plus the parameter sets that key the caches.

    process_layers() (which writes caches) and load_cached_mappers() (which
    reads them for "Show Saved Plot") have to agree on every one of these,
    or a Run produces caches the viewer cannot find. They used to derive
    them from two parallel copies of the same code; deriving them once here
    is what keeps the two in step.
    """
    nx: int
    ny: int
    x_edges: Optional[np.ndarray]
    y_edges: Optional[np.ndarray]
    custom_grid: bool
    skip_merge: bool
    use_polarity: bool
    min_display_pixels: int
    poly_params: dict
    raster_params: dict

    def meta_path(self, filepath, file_hash):
        """Cache file holding this layer's own bounds (no raster)."""
        return _raster_cache_path(
            filepath, file_hash,
            _raster_params_hash({**self.poly_params, 'kind': 'meta'}),
            is_meta=True)

    def raster_path(self, filepath, file_hash, eff_bounds):
        """Cache file holding this layer's sub-pixel bitmap."""
        params = {**self.raster_params,
                  'bounds': tuple(round(float(v), 9) for v in eff_bounds)}
        if self.custom_grid:
            # Non-uniform rasters are sized by the smallest cell, so the
            # bitmap shape differs from any uniform-grid cache for the same
            # bounds/params. Tag the key with the edge signature.
            params['x_edges'] = [round(float(v), 9) for v in self.x_edges]
            params['y_edges'] = [round(float(v), 9) for v in self.y_edges]
        return _raster_cache_path(filepath, file_hash,
                                  _raster_params_hash(params),
                                  min_display_pixels=self.min_display_pixels)

    def grid_kwargs(self, bounds, even_odd):
        """TraceGridMapper constructor arguments for this grid."""
        kw = dict(nx=self.nx, ny=self.ny, bounds=bounds, even_odd=even_odd,
                  min_display_pixels=self.min_display_pixels)
        if self.custom_grid:
            kw['x_edges'] = self.x_edges
            kw['y_edges'] = self.y_edges
        return kw

    def effective_bounds(self, own_bounds, bounds, shared_bounds, verbose=False):
        """Map each file to the bounding box its grid should span.

        User-supplied `bounds` wins over both the shared box and each
        layer's own, so one explicit box can be applied across every layer.
        """
        if self.custom_grid:
            box = (float(self.x_edges[0]), float(self.y_edges[0]),
                   float(self.x_edges[-1]), float(self.y_edges[-1]))
            if verbose:
                print(f"\nUsing custom-grid bounds: {box}")
            return {fp: box for fp in own_bounds}
        if bounds is not None:
            box = normalize_bounds(bounds)
            if verbose:
                print(f"\nUsing user-specified bounds: {box}")
            return {fp: box for fp in own_bounds}
        if shared_bounds and len(own_bounds) > 1:
            all_b = list(own_bounds.values())
            box = (min(b[0] for b in all_b), min(b[1] for b in all_b),
                   max(b[2] for b in all_b), max(b[3] for b in all_b))
            if verbose:
                print(f"\nShared bounds across {len(own_bounds)} layers: {box}")
            return {fp: box for fp in own_bounds}
        return dict(own_bounds)


def _build_grid_context(nx, ny, merge_tolerance, no_merge, even_odd,
                        use_polarity, exclude_largest, min_display_pixels,
                        x_coords_csv=None, y_coords_csv=None,
                        x_edges=None, y_edges=None, verbose=False):
    """Resolve grid edges and cache parameters once, for both entry points."""
    if x_coords_csv:
        x_edges = load_grid_csv(x_coords_csv)
    if y_coords_csv:
        y_edges = load_grid_csv(y_coords_csv)
    if (x_edges is None) != (y_edges is None):
        raise ValueError("Custom grid requires BOTH x and y coordinate sources.")

    custom_grid = x_edges is not None and y_edges is not None
    if custom_grid:
        x_edges = np.asarray(x_edges, dtype=np.float64)
        y_edges = np.asarray(y_edges, dtype=np.float64)
        nx = int(x_edges.size - 1)
        ny = int(y_edges.size - 1)
        if verbose:
            print(f"Custom grid enabled: {nx}x{ny} cells from provided edges "
                  f"(X: {x_edges[0]:.4f}..{x_edges[-1]:.4f}, "
                  f"Y: {y_edges[0]:.4f}..{y_edges[-1]:.4f})")

    skip_merge = no_merge or even_odd
    # Polarity resolution only applies when a merge actually happens.
    effective_use_polarity = use_polarity and not skip_merge

    poly_params = {
        'merge_tolerance': merge_tolerance,
        'no_merge': skip_merge,
        'even_odd': even_odd,
        'use_polarity': effective_use_polarity,
        'exclude_largest': exclude_largest,
    }
    return _GridContext(
        nx=nx, ny=ny, x_edges=x_edges, y_edges=y_edges,
        custom_grid=custom_grid, skip_merge=skip_merge,
        use_polarity=effective_use_polarity,
        min_display_pixels=min_display_pixels,
        poly_params=poly_params,
        raster_params={**poly_params, 'min_display_pixels': min_display_pixels},
    )


def process_layers(filepaths: List[str], nx=20, ny=20,
                   bounds=None, shared_bounds=True,
                   export_csv=True, plot=True, show=False, outdir=None,
                   merge_tolerance=0.0, no_merge=False, interactive=False,
                   even_odd=False, use_polarity=True, exclude_largest=0,
                   min_display_pixels=600, cache=True,
                   x_coords_csv: Optional[str] = None,
                   y_coords_csv: Optional[str] = None,
                   x_edges: Optional[np.ndarray] = None,
                   y_edges: Optional[np.ndarray] = None,
                   export_apdl: bool = False,
                   apdl_stride: int = 1):
    """
    Process multiple Gerber layer files.

    Args:
        bounds: Explicit (xmin, ymin, xmax, ymax) to map every layer over,
                instead of the extents read from the Gerber files. Takes
                precedence over shared_bounds, so one box applies to all
                layers and stays identical across runs; ignored when a
                custom coordinate grid is given, since its edges already
                fix the bounds.
        shared_bounds: If True (default), all layers share one bounding box
                       so that grid cells align across layers. Only consulted
                       when `bounds` is None.
        outdir: Output directory for CSV/PNG. Default = same dir as input file.
        merge_tolerance: Coordinate grid size for set_precision before union.
                         0 = no snapping (default). Larger values merge more.
        no_merge: If True, skip unary_union entirely. Individual polygons are
                  used with STRtree spatial index for grid computation.
        even_odd: If True, apply the legacy even-odd fill rule per cell
                  (odd overlaps = filled, even overlaps = empty) instead of
                  resolving fill/hole from the Gerber file itself. Default
                  False -- use_polarity below is the modern replacement.
        use_polarity: If True (default), resolve copper fill/hole from the
                      Gerber file's own %LPD*%/%LPC*% level polarity and
                      region contour nesting, instead of guessing via
                      even-odd. Ignored when even_odd or no_merge is set.
        exclude_largest: Number of largest polygons (by area) to exclude per layer.
                         Useful for removing outer board-outline polygons. 0 = none.
        cache: If True (default), reuse cached sub-pixel rasters keyed on
               file content + all rasterisation params (EXCLUDING nx/ny).
               Lets you change nx/ny and get instant re-mapping from a
               previously rendered bitmap.  Disabled automatically in
               interactive mode.
        x_coords_csv / y_coords_csv: Paths to column-vector CSV files
               containing the X / Y grid edge coordinates for a custom
               non-uniform grid. When both are provided, nx/ny and
               shared_bounds are ignored and the grid edges define the
               bounds and cell boundaries for every layer.
        x_edges / y_edges: Alternative direct-array form of x_coords_csv /
               y_coords_csv, primarily for programmatic callers.
        export_apdl: If True, also write an APDL macro (.mac) per layer --
               the "reference full model": one 2D element per raster
               sub-pixel, MAT=1 (Cu) / MAT=2 (PPG), matching the display
               panel exactly. See apdl_export.write_reference_full_model_apdl.
               Element count scales with min_display_pixels and can be
               very large; use apdl_stride to bound it.
        apdl_stride: Use every Nth raster sub-pixel per axis for the
               reference full model instead of all of them. 1 (default)
               matches the display panel exactly.
    Returns:
        dict: {layer_name: TraceGridMapper}
    """
    if not filepaths:
        print("No files to process.")
        return {}

    bounds = normalize_bounds(bounds)
    ctx = _build_grid_context(
        nx, ny, merge_tolerance, no_merge, even_odd, use_polarity,
        exclude_largest, min_display_pixels,
        x_coords_csv, y_coords_csv, x_edges, y_edges, verbose=True)
    nx, ny = ctx.nx, ctx.ny
    x_edges, y_edges = ctx.x_edges, ctx.y_edges
    skip_merge = ctx.skip_merge
    effective_use_polarity = ctx.use_polarity
    cache_enabled = cache and not interactive

    def _parse_layer(fp: str) -> Optional[GerberLayer]:
        try:
            layer = GerberLayer(filepath=fp,
                                merge_tolerance=merge_tolerance,
                                no_merge=skip_merge,
                                interactive=interactive,
                                use_polarity=effective_use_polarity)
            layer.load()
            layer.to_polygons()
            if exclude_largest > 0 and layer.copper_polys:
                layer.exclude_largest_polygons(exclude_largest)
            return layer
        except Exception as e:
            print(f"  ERROR loading {fp}: {e}")
            return None

    # --- Step 1: Determine each file's own bounds ---------------------------
    # Try the meta cache first (keyed on file content hash + poly params).
    # On hit, skip the Gerber parse entirely; defer to Step 3 on raster miss.
    parsed: dict = {}
    own_bounds: dict = {}
    file_hashes: dict = {}  # fp -> _file_identity_hash(fp), computed once per file

    for fp in filepaths:
        ob = None
        meta_path = None
        if cache_enabled:
            f_hash = _file_identity_hash(fp)
            file_hashes[fp] = f_hash
            meta_path = ctx.meta_path(fp, f_hash)
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
            if cache_enabled and meta_path is not None:
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(meta_path, bounds=np.array(ob, dtype=np.float64))
                print(f"  Meta cache saved: {meta_path.name}")
        own_bounds[fp] = ob

    if not own_bounds:
        print("No layers loaded successfully.")
        return {}

    # --- Step 2: Determine effective bounds per layer -----------------------
    effective = ctx.effective_bounds(own_bounds, bounds, shared_bounds,
                                     verbose=True)

    # --- Step 3: Per-layer raster (from cache or fresh) + fractions --------
    results = {}
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)

    for fp, eff_b in effective.items():
        name = Path(fp).stem

        bitmap = cached_bounds = None
        cached_sub = (0, 0)
        r_path = None
        if cache_enabled:
            f_hash = file_hashes.get(fp) or _file_identity_hash(fp)
            r_path = ctx.raster_path(fp, f_hash, eff_b)
            loaded = _load_raster_cache(r_path)
            if loaded is not None:
                bitmap, cached_bounds, cached_sub = loaded
                print(f"  Raster cache hit: {name}  bitmap={bitmap.shape}  "
                      f"sub={cached_sub}  [{r_path.name}]")

        if bitmap is None:
            layer = parsed.get(fp) or _parse_layer(fp)
            if layer is None:
                continue
            parsed[fp] = layer

            grid_kw = ctx.grid_kwargs(eff_b, even_odd)
            grid_kw.pop('even_odd')

            if even_odd:
                mapper = TraceGridMapper(
                    copper_polys=layer.copper_polys,
                    even_odd=True,
                    **grid_kw,
                )
            elif layer.no_merge or layer.copper is None:
                mapper = TraceGridMapper(
                    copper_polys=layer.copper_polys,
                    **grid_kw,
                )
            else:
                mapper = TraceGridMapper(
                    copper=layer.copper,
                    **grid_kw,
                )
            mapper.compute()

            if cache_enabled and r_path is not None:
                _save_raster_cache(r_path, mapper._raster_bitmap, eff_b,
                                   sub=mapper._raster_sub)
                print(f"  Raster cache saved: {r_path.name}")
        else:
            # Cache hit: inject bitmap and derive fractions for current grid.
            mapper = TraceGridMapper(**ctx.grid_kwargs(cached_bounds, even_odd))
            mapper._raster_bitmap = bitmap
            mapper._raster_sub = cached_sub
            mapper.compute()

        stem = Path(fp).stem
        out_base = Path(outdir) if outdir else Path(fp).parent

        if export_csv:
            csv_path = out_base / f"{stem}.csv"
            mapper.to_csv(str(csv_path))

        if export_apdl:
            apdl_path = out_base / f"{stem}_reference_full.mac"
            write_reference_full_model_apdl(mapper, str(apdl_path), stride=apdl_stride)

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


class CacheMissError(FileNotFoundError):
    """Raised by load_cached_mappers when any required cache file is absent."""
    def __init__(self, missing: List[str], kind: str = "cache", expected=None):
        self.missing = list(missing)
        self.kind = kind
        self.expected = list(expected) if expected else []
        names = ", ".join(Path(m).name for m in missing) or "(none)"
        msg = (f"Missing {kind} for: {names}. "
               "Run the full processing once first to generate the cache files.")
        if self.expected:
            # The filename carries the parameter hash, so showing it tells the
            # user their settings no longer match the Run that wrote the cache.
            msg += "\nLooked for: " + ", ".join(self.expected)
        super().__init__(msg)


def load_cached_mappers(filepaths: List[str], nx=20, ny=20,
                        bounds=None, shared_bounds=True,
                        merge_tolerance=0.0, no_merge=False,
                        even_odd=False, use_polarity=True, exclude_largest=0,
                        min_display_pixels=600,
                        x_coords_csv: Optional[str] = None,
                        y_coords_csv: Optional[str] = None,
                        x_edges: Optional[np.ndarray] = None,
                        y_edges: Optional[np.ndarray] = None):
    """Reconstruct TraceGridMapper objects purely from saved cache files.

    Shares _build_grid_context() with process_layers, so the cache keys the
    two derive are the same by construction rather than by two copies of the
    same code staying in step. Every argument that affects a key -- `bounds`
    included -- must match the Run that wrote the cache.

    Raises CacheMissError when any requested file lacks its meta or raster
    cache so the caller can prompt the user to run once.

    Returns: {layer_name: TraceGridMapper}
    """
    if not filepaths:
        raise CacheMissError([], kind="input files")

    bounds = normalize_bounds(bounds)
    ctx = _build_grid_context(
        nx, ny, merge_tolerance, no_merge, even_odd, use_polarity,
        exclude_largest, min_display_pixels,
        x_coords_csv, y_coords_csv, x_edges, y_edges)
    nx, ny = ctx.nx, ctx.ny

    # Step 1: meta cache -> per-file bounds (no parsing allowed).
    own_bounds: dict = {}
    file_hashes: dict = {}
    missing_meta: List[str] = []
    for fp in filepaths:
        if not Path(fp).exists():
            missing_meta.append(fp)
            continue
        try:
            f_hash = _file_identity_hash(fp)
        except OSError:
            missing_meta.append(fp)
            continue
        file_hashes[fp] = f_hash
        meta_path = ctx.meta_path(fp, f_hash)
        if not meta_path.exists():
            missing_meta.append(fp)
            continue
        try:
            with np.load(meta_path, allow_pickle=False) as d:
                own_bounds[fp] = tuple(d['bounds'].tolist())
        except Exception:
            missing_meta.append(fp)
    if missing_meta:
        raise CacheMissError(missing_meta, kind="meta cache")

    # Step 2: effective bounds (identical rules to process_layers).
    effective = ctx.effective_bounds(own_bounds, bounds, shared_bounds)

    # Step 3: raster cache -> mapper (fast path: compute() just derives fractions).
    results = {}
    missing_raster: List[str] = []
    for fp, eff_b in effective.items():
        r_path = ctx.raster_path(fp, file_hashes[fp], eff_b)
        loaded = _load_raster_cache(r_path)
        if loaded is None:
            missing_raster.append(fp)
            continue
        bitmap, cached_bounds, cached_sub = loaded

        mapper = TraceGridMapper(**ctx.grid_kwargs(cached_bounds, even_odd))
        mapper._raster_bitmap = bitmap
        mapper._raster_sub = cached_sub
        mapper.compute()
        results[Path(fp).stem] = mapper

    if missing_raster:
        raise CacheMissError(missing_raster, kind="raster cache",
                             expected=[str(ctx.raster_path(fp, file_hashes[fp],
                                                           effective[fp]).name)
                                       for fp in missing_raster])

    return results
