"""Visualization helpers for copper layers and grid fraction maps.

Functions:
  plot_copper(layer, ax)            -- outline of copper geometry
  plot_fraction_map(mapper, ax)     -- heatmap of per-cell fractions
  plot_evenodd_copper(mapper, ax)   -- two-color copper/empty view
  plot_comparison(layer, mapper)    -- side-by-side: bitmap + heatmap
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MplRect
from matplotlib.collections import PatchCollection


def plot_copper(layer, ax=None, color='darkorange', alpha=0.7):
    """Plot copper geometry outline."""
    from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if layer.copper is not None:
        geom = layer.copper
        polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
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


def plot_fraction_map(mapper, ax=None, cmap='YlOrRd', title=None, show_grid=True):
    """Plot copper fraction heatmap on the grid.

    Uses pcolormesh for non-uniform grids so each cell renders at its true
    size; falls back to the faster imshow for equal-division grids.
    """
    if mapper.fractions is None:
        mapper.compute()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    info = mapper.grid_info
    x_edges = info['x_edges']
    y_edges = info['y_edges']

    if info.get('custom'):
        im = ax.pcolormesh(x_edges, y_edges, mapper.fractions,
                           cmap=cmap, vmin=0, vmax=1, shading='flat')
        ax.set_aspect('equal')
    else:
        extent = [info['xmin'], info['xmax'], info['ymin'], info['ymax']]
        im = ax.imshow(mapper.fractions, origin='lower', extent=extent,
                       cmap=cmap, vmin=0, vmax=1, aspect='equal',
                       interpolation='nearest')

    if show_grid:
        for x in x_edges:
            ax.axvline(x, color='gray', lw=0.3, alpha=0.5)
        for y in y_edges:
            ax.axhline(y, color='gray', lw=0.3, alpha=0.5)

    plt.colorbar(im, ax=ax, label='Copper Fraction', shrink=0.8)
    ax.set_title(title or f"Trace Mapping ({mapper.nx}x{mapper.ny})")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return ax


def plot_evenodd_copper(mapper, layer_name="", ax=None, color='darkorange'):
    """Show even-odd copper result derived from already-computed fractions.

    Cells with fraction > 0 are copper; cells with fraction == 0 are empty
    (background or even-overlap hole).
    """
    import matplotlib.colors as mcolors

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    if mapper.fractions is None:
        mapper.compute()

    info = mapper.grid_info
    extent = [info['xmin'], info['xmax'], info['ymin'], info['ymax']]

    r, g, b, _ = mcolors.to_rgba(color)
    rgba = np.ones((*mapper.fractions.shape, 4))
    mask = mapper.fractions > 0
    rgba[mask] = [r, g, b, 0.85]
    rgba[~mask] = [1.0, 1.0, 1.0, 1.0]

    ax.imshow(rgba, origin='lower', extent=extent, aspect='equal',
              interpolation='nearest')
    ax.set_aspect('equal')
    ax.set_title(f"Cu even-odd: {layer_name}  ({mapper.nx}×{mapper.ny} grid)")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return ax


def plot_comparison(layer, mapper):
    """Side-by-side: custom-XOR copper bitmap with grid overlay vs fraction heatmap.

    Left panel : copper region after the custom XOR rule rendered from the
                 cached sub-pixel bitmap — no second polygon pass.
    Right panel: per-cell copper fraction (density) heat map.
    """
    import matplotlib.colors as mcolors

    if mapper.fractions is None:
        mapper.compute()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    info = mapper.grid_info
    extent = [info['xmin'], info['xmax'], info['ymin'], info['ymax']]

    # Left panel: render the cached XOR bitmap as an image.
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
        ax1.set_facecolor('white')

    for x in info['x_edges']:
        ax1.axvline(x, color='gray', lw=0.3, alpha=0.5)
    for y in info['y_edges']:
        ax1.axhline(y, color='gray', lw=0.3, alpha=0.5)

    ax1.set_xlim(info['xmin'], info['xmax'])
    ax1.set_ylim(info['ymin'], info['ymax'])
    ax1.set_aspect('equal')
    mode_str = "custom XOR" if mapper.even_odd else "coverage"
    ax1.set_title(f"Copper ({mode_str}): {layer.name}  ({mapper.nx}×{mapper.ny} grid)")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Right panel: fraction heatmap.
    plot_fraction_map(mapper, ax=ax2, title=f"{layer.name} -- Grid Mapping")

    fig.tight_layout()
    return fig
