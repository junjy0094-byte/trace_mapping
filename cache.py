"""Persistent raster cache for sub-pixel bitmaps.

Cache files are stored in .trace_cache/ next to each art file so they
survive program restarts regardless of the --outdir used for CSV/PNG output.

File naming:
  raster: <stem>_mpx<min_pixels>_<file_hash8>_<params_hash6>.npz
  meta:   <stem>_meta_<file_hash8>_<params_hash6>.npz
"""

import os
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Tuple

CACHE_VERSION = 4
CACHE_DIRNAME = ".trace_cache"


def _file_identity_hash(filepath: str) -> str:
    """8-char hex derived from file size + partial content (no path, no mtime).
    Stable across directory moves and program restarts."""
    CHUNK = 32 * 1024
    st = os.stat(filepath)
    h = hashlib.sha256()
    h.update(str(st.st_size).encode())
    with open(filepath, 'rb') as f:
        h.update(f.read(CHUNK))
        if st.st_size > CHUNK * 2:
            f.seek(-CHUNK, 2)
            h.update(f.read(CHUNK))
    return h.hexdigest()[:8]


def _raster_params_hash(params: dict) -> str:
    """6-char hash covering only processing parameters (excludes file identity)."""
    canon = {'v': CACHE_VERSION, **params}
    blob = json.dumps(canon, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:6]


def _raster_cache_path(filepath: str, file_hash: str, params_hash: str,
                       min_display_pixels: int = 0, is_meta: bool = False) -> Path:
    """Cache is always stored next to the art file so it persists across
    restarts and regardless of which --outdir is used for CSV/PNG output."""
    stem = Path(filepath).stem
    root = Path(filepath).parent
    if is_meta:
        name = f"{stem}_meta_{file_hash}_{params_hash}.npz"
    else:
        mpx = f"_mpx{min_display_pixels}" if min_display_pixels > 0 else ""
        name = f"{stem}{mpx}_{file_hash}_{params_hash}.npz"
    return root / CACHE_DIRNAME / name


def _save_raster_cache(path: Path, bitmap: np.ndarray,
                       bounds: Tuple[float, float, float, float],
                       sub=(0, 0)) -> None:
    """`sub` is (sub_x, sub_y), the per-axis sub-pixels per grid cell.

    The two differ whenever the cells are not square, which is what keeps
    the sub-pixels themselves square. A scalar is accepted and taken to
    mean both axes.
    """
    if not isinstance(sub, (tuple, list, np.ndarray)):
        sub = (sub, sub)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.int32(CACHE_VERSION),
        bitmap=bitmap.astype(np.uint8),
        bounds=np.array(bounds, dtype=np.float64),
        sub=np.array([int(sub[0]), int(sub[1])], dtype=np.int32),
    )


def _load_raster_cache(path: Path):
    """Return (bitmap_bool, bounds_tuple, (sub_x, sub_y)) or None."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as d:
            if int(d['version']) != CACHE_VERSION:
                return None
            bitmap = d['bitmap'].astype(bool)
            bounds = tuple(d['bounds'].tolist())
            if 'sub' in d:
                raw = np.atleast_1d(d['sub']).astype(int).ravel()
                sub = (int(raw[0]), int(raw[1])) if raw.size >= 2 \
                    else (int(raw[0]), int(raw[0]))
            else:
                sub = (0, 0)
            return bitmap, bounds, sub
    except Exception as e:
        print(f"  Cache read failed ({path.name}): {e}")
        return None
