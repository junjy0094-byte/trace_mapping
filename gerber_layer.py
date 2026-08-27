"""Gerber/Artwork file parsing -> Shapely polygon conversion.

Public API:
  GerberLayer(filepath, ...) -- load() -> to_polygons()
  pick_exclude_polygons(polys, title) -> list   (interactive UI helper)

Internal helpers (_line_to_shapely etc.) convert pcb-tools primitives to
Shapely geometries and are called by GerberLayer.to_polygons().

Fill/hole resolution: GerberLayer.copper (use_polarity=True, the default)
resolves the copper region straight from what the Gerber file encodes --
%LPD*%/%LPC*% level polarity and region contour nesting -- rather than
guessing via an even-odd raster overlap count. See
_resolve_copper_from_primitives() for the algorithm. The even-odd raster
heuristic (TraceGridMapper(even_odd=True)) remains available as a legacy
fallback.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
#  pcb-tools resilience patch
# ---------------------------------------------------------------------------
# pcb-tools (unmaintained since 2019) aborts an entire file parse with
# `KeyError: '<macro-name>'` whenever it meets a %ADDn,<macro>,...% that
# references an Aperture Macro it failed to register earlier -- typically
# because the macro uses an unsupported primitive code, arithmetic that
# eval_macro can't handle, or non-trivial line breaks in the AM body.
#
# We tolerate this by installing a zero-diameter Circle placeholder for the
# offending aperture.  Line/Arc/Flash primitives that later use the
# placeholder produce an empty Shapely geometry (buffer of 0), which is
# filtered out by `to_polygons()` alongside its other skip paths.  The
# result: every usable trace and pad still maps; only the shapes drawn with
# the broken aperture are silently omitted.

_PCB_TOOLS_PATCHED = False


def _install_pcb_tools_resilience():
    """Monkey-patch GerberParser so unresolvable macro apertures don't
    blow up the full file parse.  Idempotent across repeated calls."""
    global _PCB_TOOLS_PATCHED
    if _PCB_TOOLS_PATCHED:
        return
    try:
        from gerber.rs274x import GerberParser
        from gerber.primitives import Circle as _GCircle
    except Exception:
        # gerber package not importable yet; the caller will fail naturally.
        return

    _orig = GerberParser._define_aperture

    def _resilient_define_aperture(self, d, shape, modifiers):
        try:
            return _orig(self, d, shape, modifiers)
        except Exception as exc:
            # Standard shapes (C/R/O/P) parsing errors indicate a genuinely
            # malformed file -- propagate so the user sees it.
            if shape in ('C', 'R', 'O', 'P'):
                raise
            missing = getattr(self, '_missing_macro_apertures', None)
            if missing is None:
                missing = {}
                self._missing_macro_apertures = missing
            missing.setdefault(shape, []).append(d)
            units = getattr(self.settings, 'units', None)
            self.apertures[d] = _GCircle(position=None, diameter=0.0,
                                         units=units)

    GerberParser._define_aperture = _resilient_define_aperture
    _PCB_TOOLS_PATCHED = True


# ---------------------------------------------------------------------------
#  Gerber primitive -> Shapely conversion helpers
# ---------------------------------------------------------------------------

def _line_to_shapely(prim):
    """Trace line segment -> buffered LineString."""
    from shapely.geometry import LineString
    line = LineString([prim.start, prim.end])
    ap = prim.aperture
    if hasattr(ap, 'diameter'):
        return line.buffer(ap.diameter / 2.0, resolution=16, cap_style=1)
    elif hasattr(ap, 'width') and hasattr(ap, 'height'):
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
    return line.buffer(0.001)


def _region_to_shapely(prim):
    """Region (filled polygon) -> Shapely Polygon."""
    from shapely.geometry import Polygon
    import gerber.primitives as gp
    import math

    coords = []
    for seg in prim.primitives:
        if isinstance(seg, gp.Line):
            coords.append(seg.start)
        elif isinstance(seg, gp.Arc):
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


def _aperture_hole_shapely(prim, x, y):
    """Build the standard C/R/O/P aperture's own hole modifier as a Shapely
    geometry (circular via hole_diameter, or rectangular via hole_width /
    hole_height), or None if the aperture has no hole.

    This is real declarative information from the Gerber %ADn...% aperture
    definition (the hole is part of the template, same as the outer shape)
    -- not a guess. Ignoring it, as the previous flash conversion did,
    turns every annular/donut pad into a solid disc.
    """
    from shapely.geometry import Point, box

    d = getattr(prim, 'hole_diameter', None)
    if d:
        return Point(x, y).buffer(d / 2.0, resolution=32)
    hw = getattr(prim, 'hole_width', 0) or 0
    hh = getattr(prim, 'hole_height', 0) or 0
    if hw > 0 and hh > 0:
        return box(x - hw / 2.0, y - hh / 2.0, x + hw / 2.0, y + hh / 2.0)
    return None


def _flash_to_shapely(prim):
    """Flash (pad) primitive -> Shapely geometry at prim.position.

    Subtracts the aperture's own hole modifier, if any (see
    _aperture_hole_shapely), so annular/donut pads render with their hole.
    """
    from shapely.geometry import Point, box
    import gerber.primitives as gp

    x, y = prim.position

    if isinstance(prim, gp.Circle):
        outer = Point(x, y).buffer(prim.radius, resolution=32)
    elif isinstance(prim, gp.Rectangle):
        hw, hh = prim.width / 2.0, prim.height / 2.0
        outer = box(x - hw, y - hh, x + hw, y + hh)
    elif isinstance(prim, gp.Obround):
        from shapely.geometry import LineString
        hw, hh = prim.width / 2.0, prim.height / 2.0
        if prim.width >= prim.height:
            line = LineString([(x - hw + hh, y), (x + hw - hh, y)])
            outer = line.buffer(hh, resolution=32, cap_style=1)
        else:
            line = LineString([(x, y - hh + hw), (x, y + hh - hw)])
            outer = line.buffer(hw, resolution=32, cap_style=1)
    elif isinstance(prim, gp.Polygon):
        from shapely.geometry import Polygon
        if hasattr(prim, 'vertices') and prim.vertices:
            outer = Polygon(prim.vertices)
        else:
            return None
    else:
        return None

    hole = _aperture_hole_shapely(prim, x, y)
    return outer.difference(hole) if hole is not None else outer


# ---------------------------------------------------------------------------
#  Gerber polarity / region-nesting fill resolution
# ---------------------------------------------------------------------------
# Replaces the even-odd raster heuristic with the fill/hole information the
# Gerber file actually encodes:
#   - %LPD*% / %LPC*% (dark/clear level polarity) says whether a primitive
#     adds or knocks out copper, applied strictly in file order.
#   - A G36/G37 region with multiple D02-separated contours (pcb-tools
#     parses each contour as its own Region primitive) encodes holes via
#     nesting: a contour inside an already-drawn contour is a hole.
# See _resolve_copper_from_primitives() for how these combine.

def _primitive_polarity(prim):
    """'clear' if drawn under a Gerber %LPC*% level, else 'dark'.

    pcb-tools sets `level_polarity` on every Line/Arc/Region/flash
    primitive to the %LP*% state active when it was parsed.
    """
    return 'clear' if getattr(prim, 'level_polarity', 'dark') == 'clear' else 'dark'


def _resolve_region_group(polys):
    """Resolve one Gerber region block's sub-contours by nesting containment.

    pcb-tools splits a single multi-contour G36/G37 region (contours
    separated by D02 moves) into several consecutive Region primitives
    that share one polarity. Per the Gerber spec, a contour nested inside
    the already-accumulated result is a hole in it; anything else is
    additional solid area. Using real polygon containment -- rather than a
    rasterised even-odd guess -- resolves this exactly, including
    multi-level nesting (island-in-hole-in-solid).
    """
    ordered = sorted(polys, key=lambda p: p.area, reverse=True)
    result = ordered[0]
    for poly in ordered[1:]:
        try:
            is_hole = result.contains(poly.representative_point())
        except Exception:
            is_hole = False
        result = result.difference(poly) if is_hole else result.union(poly)
    return result


def _fold_polarity_runs(geoms, polarities):
    """Sequential dark=union / clear=difference fold, in file order.

    This is the painter's-algorithm rule the Gerber spec requires: later
    primitives act on whatever came before them. Consecutive primitives
    that share a polarity are batched into one unary_union before folding
    into the accumulator -- algebraically identical to folding one at a
    time (union/difference are associative within a single polarity run),
    but with far fewer boolean ops.
    """
    from shapely.ops import unary_union

    acc = None
    run, run_polarity = [], None

    def flush(acc):
        if not run:
            return acc
        merged = unary_union(run) if len(run) > 1 else run[0]
        if run_polarity == 'dark':
            return merged if acc is None else acc.union(merged)
        return acc if acc is None else acc.difference(merged)

    for geom, polarity in zip(geoms, polarities):
        if polarity != run_polarity:
            acc = flush(acc)
            run, run_polarity = [], polarity
        run.append(geom)
    return flush(acc)


def _resolve_copper_from_primitives(polys, polarities, kinds):
    """Resolve per-primitive (geometry, polarity, kind) triples into the
    final copper geometry, using only information the Gerber file itself
    encodes:

    1. Region nesting: consecutive same-polarity Region primitives (one
       original multi-contour region, split apart by the parser) are
       folded by containment so inner contours read as holes.
    2. Level polarity: the resulting primitives are folded across the
       whole file in order, 'dark' adding to the accumulated copper and
       'clear' cutting it out.
    """
    collapsed_polys, collapsed_pols = [], []
    i, n = 0, len(polys)
    while i < n:
        if kinds[i] == 'region':
            j = i
            group = []
            while j < n and kinds[j] == 'region' and polarities[j] == polarities[i]:
                group.append(polys[j])
                j += 1
            collapsed_polys.append(
                _resolve_region_group(group) if len(group) > 1 else group[0])
            collapsed_pols.append(polarities[i])
            i = j
        else:
            collapsed_polys.append(polys[i])
            collapsed_pols.append(polarities[i])
            i += 1

    return _fold_polarity_runs(collapsed_polys, collapsed_pols)


# ---------------------------------------------------------------------------
#  Interactive polygon exclusion picker
# ---------------------------------------------------------------------------

def pick_exclude_polygons(polys, title="Click polygons to exclude, then close window"):
    """Show all polygons interactively. Click to toggle exclude (red).
    Close the window to confirm. Returns list of non-excluded polygons."""
    from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.set_aspect('equal')
    ax.set_title(title)

    excluded = set()
    patch_map = {}

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
            excluded.discard(idx)
            for p, pidx in patch_map.items():
                if pidx == idx:
                    p.set_facecolor('darkorange')
                    p.set_alpha(0.7)
        else:
            excluded.add(idx)
            for p, pidx in patch_map.items():
                if pidx == idx:
                    p.set_facecolor('red')
                    p.set_alpha(0.3)
        ax.set_title(f"{title}  [excluded: {len(excluded)}/{len(polys)}]")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

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
    use_polarity: bool = True
    _parsed: object = field(default=None, repr=False)
    _copper: object = field(default=None, repr=False)
    _copper_polys: list = field(default_factory=list, repr=False)
    _copper_polarities: list = field(default_factory=list, repr=False)
    _copper_kinds: list = field(default_factory=list, repr=False)

    def load(self):
        """Parse the Gerber file using pcb-tools.

        Installs a resilience patch on pcb-tools' parser so a single
        unresolvable aperture macro does not abort the whole file.
        """
        _install_pcb_tools_resilience()

        # Capture any "missing macro" reports from the patched parser.
        # The patch stores them on the parser instance, but `read()` only
        # returns the GerberFile, so we intercept GerberParser.parse here.
        from gerber.rs274x import GerberParser
        print(f"Loading: {self.filepath}")
        parser = GerberParser()
        self._parsed = parser.parse(self.filepath)
        if not self.name:
            self.name = Path(self.filepath).stem

        missing = getattr(parser, '_missing_macro_apertures', None)
        if missing:
            total_d = sum(len(v) for v in missing.values())
            print(f"  WARNING: {len(missing)} aperture macro(s) could not be "
                  f"resolved (affects {total_d} aperture definition(s)). "
                  "Primitives drawn with these apertures will be skipped; "
                  "the rest of the layer is loaded normally.")
            for name, d_codes in missing.items():
                codes = ", ".join(f"D{d}" for d in d_codes)
                print(f"    - macro {name!r}: {codes}")

        print(f"  Bounds: {self._parsed.bounds}")
        print(f"  Primitives: {len(self._parsed.primitives)}")
        return self

    def to_polygons(self):
        """Convert all primitives to Shapely geometries.

        Builds self._copper_polys plus parallel self._copper_polarities /
        self._copper_kinds (one entry each, same order) used by the
        polarity-aware merge. The merge itself is NOT done here -- it is
        computed lazily by the `copper` property / _merge_copper_polys(),
        so that exclude_largest_polygons() and the interactive picker
        below (both of which run after this call, in process.py) still
        affect the final result.

        self.no_merge=True skips merging entirely; individual polygons are
        kept for even-odd / STRtree rasterisation instead.
        """
        import gerber.primitives as gp

        if self._parsed is None:
            self.load()

        polys, polarities, kinds = [], [], []
        skip_count = 0

        for prim in self._parsed.primitives:
            try:
                geom = None
                kind = 'other'
                if isinstance(prim, gp.Line):
                    geom = _line_to_shapely(prim)
                elif isinstance(prim, gp.Arc):
                    geom = _arc_to_shapely(prim)
                elif isinstance(prim, gp.Region):
                    geom = _region_to_shapely(prim)
                    kind = 'region'
                elif isinstance(prim, (gp.Circle, gp.Rectangle, gp.Obround, gp.Polygon)):
                    geom = _flash_to_shapely(prim)
                else:
                    skip_count += 1
                    continue

                if geom is not None and not geom.is_valid:
                    geom = geom.buffer(0)

                if geom is not None and geom.is_valid and not geom.is_empty:
                    polys.append(geom)
                    polarities.append(_primitive_polarity(prim))
                    kinds.append(kind)
            except Exception:
                skip_count += 1

        if skip_count > 0:
            print(f"  Skipped {skip_count} unsupported/failed primitives")

        if not polys:
            raise ValueError(f"No valid polygons from {self.filepath}")

        self._copper_polys = polys
        self._copper_polarities = polarities
        self._copper_kinds = kinds
        self._copper = None

        if self.interactive:
            print(f"  Opening interactive picker for {self.name} ...")
            print(f"  Click polygons to exclude (they turn red), then close the window.")
            before = self._copper_polys
            after = pick_exclude_polygons(
                before,
                title=f"{self.name}: click to exclude, then close window",
            )
            keep_ids = {id(g) for g in after}
            keep_idx = [i for i, g in enumerate(before) if id(g) in keep_ids]
            self._copper_polys = after
            self._copper_polarities = [self._copper_polarities[i] for i in keep_idx]
            self._copper_kinds = [self._copper_kinds[i] for i in keep_idx]
            if not self._copper_polys:
                raise ValueError(f"All polygons excluded for {self.filepath}")

        if self.no_merge:
            print(f"  No-merge mode: keeping {len(self._copper_polys)} individual polygons")

        return self._copper_polys

    def exclude_largest_polygons(self, n):
        """Remove the n largest primitive polygons (by area), keeping the
        polarity/kind tracking in sync and invalidating any cached merge.

        Useful for discarding outer board-outline polygons.
        """
        polys = self._copper_polys
        if n <= 0 or not polys:
            return
        if n >= len(polys):
            print(f"  Warning: exclude count ({n}) >= total polygons "
                  f"({len(polys)}), skipping exclusion.")
            return

        indexed = sorted(range(len(polys)), key=lambda i: polys[i].area, reverse=True)
        exclude_idx = set(indexed[:n])
        print(f"  Excluded {n} largest polygon(s) by area:")
        for i in indexed[:n]:
            print(f"    polygon #{i}: area = {polys[i].area:.6f}")

        self._copper_polys = [p for i, p in enumerate(polys) if i not in exclude_idx]
        if self._copper_polarities:
            self._copper_polarities = [p for i, p in enumerate(self._copper_polarities)
                                       if i not in exclude_idx]
        if self._copper_kinds:
            self._copper_kinds = [p for i, p in enumerate(self._copper_kinds)
                                  if i not in exclude_idx]
        self._copper = None

    def _merge_copper_polys(self):
        """Merge self._copper_polys into one geometry.

        use_polarity=True (default): resolve the Gerber file's own fill
        information -- %LPD*%/%LPC*% level polarity plus region nesting
        (see _resolve_copper_from_primitives) -- instead of guessing via
        an even-odd raster. For a typical trace layer with no clear-
        polarity primitives this degenerates to the same unary_union as
        the legacy path, at the same cost.

        use_polarity=False: blind unary_union of every primitive,
        ignoring polarity/nesting (legacy behavior; kept for comparison
        and for non-Gerber-polarity troubleshooting).
        """
        from shapely.ops import unary_union

        polys = self._copper_polys

        if self.use_polarity:
            print(f"  Resolving {len(polys)} polygons via Gerber polarity "
                  f"(LPD/LPC + region nesting) ... ", end="", flush=True)
            t0 = time.time()
            merged = _resolve_copper_from_primitives(
                polys, self._copper_polarities, self._copper_kinds)
            print(f"done ({time.time()-t0:.1f}s)")
            if merged is None or merged.is_empty:
                raise ValueError(
                    f"No copper remained after polarity resolution for "
                    f"{self.filepath} (check for a leading/only Clear layer).")
            return merged

        print(f"  Merging {len(polys)} polygons ", end="", flush=True)

        if self.merge_tolerance > 0:
            try:
                from shapely import set_precision
                print(f"(tolerance={self.merge_tolerance}) ... ", end="", flush=True)
                snapped = [set_precision(g, grid_size=self.merge_tolerance) for g in polys]
                snapped = [g for g in snapped if g.is_valid and not g.is_empty]
                if not snapped:
                    print("WARNING: all geometries became invalid after snapping, using originals")
                    snapped = polys
            except ImportError:
                print("(set_precision unavailable, using raw union) ... ", end="", flush=True)
                snapped = polys
        else:
            print("... ", end="", flush=True)
            snapped = polys

        t0 = time.time()
        merged = unary_union(snapped)
        print(f"done ({time.time()-t0:.1f}s)")
        return merged

    @property
    def copper(self):
        """Merged/resolved copper geometry. None in no_merge mode."""
        if self._copper is None and not self._copper_polys:
            self.to_polygons()
        if self._copper is None and not self.no_merge and self._copper_polys:
            self._copper = self._merge_copper_polys()
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
        if not self._copper_polys:
            self.to_polygons()
        all_bounds = [g.bounds for g in self._copper_polys]
        xmin = min(b[0] for b in all_bounds)
        ymin = min(b[1] for b in all_bounds)
        xmax = max(b[2] for b in all_bounds)
        ymax = max(b[3] for b in all_bounds)
        return (xmin, ymin, xmax, ymax)
