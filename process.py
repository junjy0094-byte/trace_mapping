"""Batch processing orchestrator: parse -> rasterise -> export per layer.

process_layers() is the main entry point. Workflow:
  Step 1  Determine each layer's own bounds (meta cache or fresh parse).
  Step 2  Compute effective bounds (shared across layers, or per-layer).
  Step 3  Rasterise each layer (raster cache or fresh compute) and export.
  Step 4  Multi-layer summary plot (when >1 layer).

Helpers:
  collect_art_files(paths)           -- resolve file/dir paths to .art/.gbr list
  compute_shared_bounds(layers)      -- unified bounding box across layers
  _exclude_largest_polygons(polys,n) -- remove N largest polygons by area
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

from gerber_layer import GerberLayer
from trace_grid import TraceGridMapper
from cache import (CACHE_DIRNAME, _file_identity_hash, _raster_params_hash,
                   _raster_cache_path, _save_raster_cache, _load_raster_cache)
from plot import plot_comparison, plot_fraction_map


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


def _exclude_largest_polygons(polys, n):
    """Remove the n largest polygons (by area) from the list.

    Useful for discarding outer-border polygons from the Gerber board outline.
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


def process_layers(filepaths: List[str], nx=20, ny=20,
                   bounds=None, shared_bounds=True,
                   export_csv=True, plot=True, show=False, outdir=None,
                   merge_tolerance=0.0, no_merge=False, interactive=False,
                   even_odd=True, exclude_largest=0,
                   min_display_pixels=600, cache=True,
                   x_coords_csv: Optional[str] = None,
                   y_coords_csv: Optional[str] = None,
                   x_edges: Optional[np.ndarray] = None,
                   y_edges: Optional[np.ndarray] = None):
    """
    Process multiple Gerber layer files.

    Args:
        shared_bounds: If True (default), all layers share one bounding box
                       so that grid cells align across layers.
        outdir: Output directory for CSV/PNG. Default = same dir as input file.
        merge_tolerance: Coordinate grid size for set_precision before union.
                         0 = no snapping (default). Larger values merge more.
        no_merge: If True, skip unary_union entirely. Individual polygons are
                  used with STRtree spatial index for grid computation.
        even_odd: If True (default), apply even-odd fill rule per cell.
                  Odd overlaps = filled, even overlaps = empty (hollow interior).
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
    Returns:
        dict: {layer_name: TraceGridMapper}
    """
    if not filepaths:
        print("No files to process.")
        return {}

    # Resolve custom grid edges (CSV paths take precedence over arrays).
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
        print(f"Custom grid enabled: {nx}x{ny} cells from provided edges "
              f"(X: {x_edges[0]:.4f}..{x_edges[-1]:.4f}, "
              f"Y: {y_edges[0]:.4f}..{y_edges[-1]:.4f})")

    skip_merge = no_merge or even_odd
    cache_enabled = cache and not interactive

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
            meta_p_hash = _raster_params_hash({**poly_params, 'kind': 'meta'})
            meta_path = _raster_cache_path(fp, f_hash, meta_p_hash, is_meta=True)
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
    if custom_grid:
        custom_b = (float(x_edges[0]), float(y_edges[0]),
                    float(x_edges[-1]), float(y_edges[-1]))
        effective = {fp: custom_b for fp in own_bounds}
        print(f"\nUsing custom-grid bounds: {custom_b}")
    elif bounds is not None:
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
        if custom_grid:
            # Non-uniform rasters are sized by the smallest cell, so the
            # bitmap shape differs from any uniform-grid cache for the same
            # bounds/params.  Tag the cache key with the edge signature.
            raster_params_fp['x_edges'] = [round(float(v), 9) for v in x_edges]
            raster_params_fp['y_edges'] = [round(float(v), 9) for v in y_edges]
        bitmap = cached_bounds = None
        cached_sub = 0
        r_path = None
        if cache_enabled:
            f_hash = file_hashes.get(fp) or _file_identity_hash(fp)
            r_p_hash = _raster_params_hash(raster_params_fp)
            r_path = _raster_cache_path(fp, f_hash, r_p_hash,
                                        min_display_pixels=min_display_pixels)
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

            grid_kw = dict(nx=nx, ny=ny, bounds=eff_b,
                           min_display_pixels=min_display_pixels)
            if custom_grid:
                grid_kw['x_edges'] = x_edges
                grid_kw['y_edges'] = y_edges

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
            grid_kw = dict(nx=nx, ny=ny, bounds=cached_bounds,
                           even_odd=even_odd,
                           min_display_pixels=min_display_pixels)
            if custom_grid:
                grid_kw['x_edges'] = x_edges
                grid_kw['y_edges'] = y_edges
            mapper = TraceGridMapper(**grid_kw)
            mapper._raster_bitmap = bitmap
            mapper._raster_sub = cached_sub
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


class CacheMissError(FileNotFoundError):
    """Raised by load_cached_mappers when any required cache file is absent."""
    def __init__(self, missing: List[str], kind: str = "cache"):
        self.missing = list(missing)
        self.kind = kind
        names = ", ".join(Path(m).name for m in missing) or "(none)"
        super().__init__(
            f"Missing {kind} for: {names}. "
            "Run the full processing once first to generate the cache files."
        )


def load_cached_mappers(filepaths: List[str], nx=20, ny=20,
                        bounds=None, shared_bounds=True,
                        merge_tolerance=0.0, no_merge=False,
                        even_odd=True, exclude_largest=0,
                        min_display_pixels=600,
                        x_coords_csv: Optional[str] = None,
                        y_coords_csv: Optional[str] = None,
                        x_edges: Optional[np.ndarray] = None,
                        y_edges: Optional[np.ndarray] = None):
    """Reconstruct TraceGridMapper objects purely from saved cache files.

    Mirrors the cache lookup used by process_layers but never parses a
    Gerber file. Raises CacheMissError when any requested file lacks its
    meta or raster cache so the caller can prompt the user to run once.

    Returns: {layer_name: TraceGridMapper}
    """
    if not filepaths:
        raise CacheMissError([], kind="input files")

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

    skip_merge = no_merge or even_odd
    poly_params = {
        'merge_tolerance': merge_tolerance,
        'no_merge': skip_merge,
        'even_odd': even_odd,
        'exclude_largest': exclude_largest,
    }
    raster_params = {**poly_params, 'min_display_pixels': min_display_pixels}
    meta_p_hash = _raster_params_hash({**poly_params, 'kind': 'meta'})

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
        meta_path = _raster_cache_path(fp, f_hash, meta_p_hash, is_meta=True)
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

    # Step 2: effective bounds (same rules as process_layers).
    if custom_grid:
        custom_b = (float(x_edges[0]), float(y_edges[0]),
                    float(x_edges[-1]), float(y_edges[-1]))
        effective = {fp: custom_b for fp in own_bounds}
    elif bounds is not None:
        effective = {fp: bounds for fp in own_bounds}
    elif shared_bounds and len(own_bounds) > 1:
        all_b = list(own_bounds.values())
        sb = (min(b[0] for b in all_b), min(b[1] for b in all_b),
              max(b[2] for b in all_b), max(b[3] for b in all_b))
        effective = {fp: sb for fp in own_bounds}
    else:
        effective = dict(own_bounds)

    # Step 3: raster cache -> mapper (fast path: compute() just derives fractions).
    results = {}
    missing_raster: List[str] = []
    for fp, eff_b in effective.items():
        raster_params_fp = {**raster_params,
                            'bounds': tuple(round(v, 9) for v in eff_b)}
        if custom_grid:
            raster_params_fp['x_edges'] = [round(float(v), 9) for v in x_edges]
            raster_params_fp['y_edges'] = [round(float(v), 9) for v in y_edges]

        r_p_hash = _raster_params_hash(raster_params_fp)
        r_path = _raster_cache_path(fp, file_hashes[fp], r_p_hash,
                                    min_display_pixels=min_display_pixels)
        loaded = _load_raster_cache(r_path)
        if loaded is None:
            missing_raster.append(fp)
            continue
        bitmap, cached_bounds, cached_sub = loaded

        grid_kw = dict(nx=nx, ny=ny, bounds=cached_bounds,
                       even_odd=even_odd,
                       min_display_pixels=min_display_pixels)
        if custom_grid:
            grid_kw['x_edges'] = x_edges
            grid_kw['y_edges'] = y_edges
        mapper = TraceGridMapper(**grid_kw)
        mapper._raster_bitmap = bitmap
        mapper._raster_sub = cached_sub
        mapper.compute()
        results[Path(fp).stem] = mapper

    if missing_raster:
        raise CacheMissError(missing_raster, kind="raster cache")

    return results
