"""
Gerber Trace Mapping Tool
=========================
Parses Gerber/Artwork (.art, .gbr) files and computes copper area fraction
on an N x M grid for ANSYS APDL trace mapping.

Dependencies:
    pip install gerber shapely numpy matplotlib

Usage:
    python trace_mapper.py layer1.art --nx 20 --ny 20
    python trace_mapper.py .          --nx 50 --ny 50   (scan directory)
    python trace_mapper.py            (launch GUI)

Codebase layout -- read only what you need:
  gerber_layer.py  GerberLayer class + Gerber primitive -> Shapely conversion
  trace_grid.py    TraceGridMapper (sub-pixel rasterisation, per-cell fractions)
  cache.py         Persistent raster cache (named .npz files next to art files)
  plot.py          Visualization: plot_copper, plot_fraction_map, plot_comparison
  process.py       Batch orchestration: process_layers, collect_art_files
  gui.py           tkinter GUI (gui_main)
  trace_mapper.py  CLI entry point + public API re-exports (this file)
"""

import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Public API re-exports (backward compatibility for `from trace_mapper import X`)
from gerber_layer import GerberLayer, pick_exclude_polygons          # noqa: F401
from trace_grid import TraceGridMapper                               # noqa: F401
from cache import CACHE_VERSION, CACHE_DIRNAME                       # noqa: F401
from plot import (plot_copper, plot_fraction_map,                    # noqa: F401
                  plot_evenodd_copper, plot_comparison)
from process import process_layers, collect_art_files, compute_shared_bounds  # noqa: F401
from gui import gui_main                                             # noqa: F401


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

    merge_group = parser.add_argument_group(
        'Merge Resolution',
        'Control how copper polygons are merged. Fine traces may be lost '
        'during unary_union due to coordinate snapping. Use these options '
        'to preserve fine trace geometry.')
    merge_group.add_argument(
        '--merge-tolerance', type=float, default=0.0,
        help='Coordinate grid size for Shapely set_precision before union. '
             '0 = full precision (default). '
             'Typical values: 1e-6 (very fine), 1e-4 (fine), 1e-2 (coarse).')
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
             'before running.')

    args = parser.parse_args()

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
        # Cache is always stored beside the art file, not in outdir.
        roots = {Path(p).parent if Path(p).is_file() else Path(p)
                 for p in args.paths}
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

    print("\n=== Summary ===")
    for name, mapper in results.items():
        info = mapper.grid_info
        print(f"  {name}: {info['nx']}x{info['ny']} grid, "
              f"avg Cu fraction = {mapper.fractions.mean():.4f}, "
              f"max = {mapper.fractions.max():.4f}")


if __name__ == '__main__':
    main()
