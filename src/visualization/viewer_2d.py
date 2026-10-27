"""Pure-NumPy helpers for the 2-D viewer (no Qt) so rendering logic is testable.

Volumes are in canonical RAS+ order (axis0→Right, axis1→Anterior, axis2→Superior).  Slices are shown in
radiological convention (patient Right on screen-left, anterior/superior at the top):
    axial    : rows = Y (A at top), cols = X (R at left)
    coronal  : rows = Z (S at top), cols = X (R at left)
    sagittal : rows = Z (S at top), cols = Y (A at left)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

PLANES = ("axial", "coronal", "sagittal")
# plane → (fixed axis, row axis, col axis)
_AX = {"axial": (2, 1, 0), "coronal": (1, 2, 0), "sagittal": (0, 2, 1)}
ORIENT_LABELS = {  # (left, right, top, bottom)
    "axial": ("R", "L", "A", "P"), "coronal": ("R", "L", "S", "I"), "sagittal": ("A", "P", "S", "I"),
}


def fixed_axis(plane: str) -> int:
    return _AX[plane][0]


def plane_image(vol: np.ndarray, plane: str, idx: int) -> np.ndarray:
    fa, ra, ca = _AX[plane]
    sl = np.take(vol, int(np.clip(idx, 0, vol.shape[fa] - 1)), axis=fa)       # remaining axes in ascending order
    rem = [a for a in range(3) if a != fa]
    if rem.index(ra) < rem.index(ca):                                         # make (rows, cols)
        pass
    else:
        sl = sl.T
    return np.ascontiguousarray(np.flip(sl, axis=(0, 1)))


def plane_shape(shape, plane: str) -> Tuple[int, int]:
    _, ra, ca = _AX[plane]
    return int(shape[ra]), int(shape[ca])


def plane_aspect(spacing, plane: str) -> float:
    """pixel height / pixel width in physical units."""
    _, ra, ca = _AX[plane]
    return float(spacing[ra]) / float(spacing[ca])


def voxel_to_pixel(plane: str, ijk, shape) -> Tuple[int, int]:
    _, ra, ca = _AX[plane]
    return int(shape[ra] - 1 - ijk[ra]), int(shape[ca] - 1 - ijk[ca])


def pixel_to_voxel(plane: str, row: int, col: int, ijk, shape):
    fa, ra, ca = _AX[plane]
    out = list(int(v) for v in ijk)
    out[ra] = int(np.clip(shape[ra] - 1 - row, 0, shape[ra] - 1))
    out[ca] = int(np.clip(shape[ca] - 1 - col, 0, shape[ca] - 1))
    return tuple(out)


def default_window(vol: np.ndarray) -> Tuple[float, float]:
    v = vol[vol > 0] if np.any(vol > 0) else vol.ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.5, 1.0
    if v.size > 500_000:
        v = v[:: v.size // 500_000]
    lo, hi = np.percentile(v, [1, 99.5])
    hi = max(hi, lo + 1e-6)
    return float((lo + hi) / 2), float(hi - lo)


def apply_window(img: np.ndarray, center: float, width: float) -> np.ndarray:
    lo = center - width / 2.0
    out = (img - lo) / max(width, 1e-9)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _blend(rgb: np.ndarray, color, alpha: np.ndarray) -> None:
    a = alpha[..., None].astype(np.float32)
    col = np.asarray(color, np.float32)
    rgb[...] = (rgb * (1 - a) + col * a)


SEG_COLOR = (255, 59, 107)
CONTOUR_COLOR = (255, 170, 190)
EDGE_COLOR = (110, 235, 255)


def uncertainty_colors(u: np.ndarray) -> np.ndarray:
    """0 → cyan (confident), 0.5 → amber, 1 → red/magenta (uncertain)."""
    u = np.clip(u, 0, 1)[..., None]
    c0, c1, c2 = np.array([0, 210, 255.]), np.array([255, 200, 40.]), np.array([255, 50, 100.])
    lo = c0 + (c1 - c0) * np.clip(u / 0.5, 0, 1)
    hi = c1 + (c2 - c1) * np.clip((u - 0.5) / 0.5, 0, 1)
    return np.where(u < 0.5, lo, hi).astype(np.float32)


def compose_rgba(base_u8: np.ndarray, seg: Optional[np.ndarray] = None, unc: Optional[np.ndarray] = None,
                 edge: Optional[np.ndarray] = None, boundary: Optional[np.ndarray] = None,
                 seg_opacity: float = 0.45, unc_opacity: float = 0.6, edge_opacity: float = 0.7,
                 contour: bool = True) -> np.ndarray:
    """Composite overlays on a windowed grey slice → (H, W, 4) uint8 RGBA.  Any overlay may be None (= off)."""
    rgb = np.repeat(base_u8[..., None].astype(np.float32), 3, axis=2)
    if edge is not None:
        _blend(rgb, EDGE_COLOR, np.clip(edge, 0, 1) ** 1.5 * edge_opacity)
    if unc is not None:
        a = np.clip(unc * 1.6, 0, 1) * unc_opacity
        rgb[...] = rgb * (1 - a[..., None]) + uncertainty_colors(unc) * a[..., None]
    if seg is not None and seg.any():
        s = seg.astype(bool)
        _blend(rgb, SEG_COLOR, s.astype(np.float32) * seg_opacity)
        if contour:
            edge_px = s & ~ndi.binary_erosion(s)
            _blend(rgb, CONTOUR_COLOR, edge_px.astype(np.float32) * min(1.0, 0.5 + seg_opacity))
    if boundary is not None and boundary.any():
        _blend(rgb, (255, 190, 60), boundary.astype(np.float32) * 0.85)
    out = np.empty(base_u8.shape + (4,), np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out
