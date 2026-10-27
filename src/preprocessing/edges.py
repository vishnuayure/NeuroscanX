"""Edge / boundary-strength representation.

Why 3-D Gaussian-derivative gradient magnitude?
  * It is isotropic and works on the full volume, so boundaries are consistent across slices
    (a 2-D Canny would flicker slice to slice).
  * Gaussian derivatives suppress noise at a physically meaningful scale (sigma in **mm**,
    converted to voxels with the real voxel spacing, so anisotropic data is handled correctly).
  * The output is a continuous, normalised strength in [0, 1] – exactly what the edge-aware
    fuzzy penalty needs (soft weights rather than a binary edge map).
Per-modality maps are normalised and combined (max by default) because a tumor boundary may be
visible in only one sequence (FLAIR/T2 oedema margin vs T1ce enhancing rim).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage import feature


def gradient_magnitude(vol: np.ndarray, spacing: Sequence[float], sigma_mm: float = 1.0) -> np.ndarray:
    sigma = [max(sigma_mm / float(s), 0.3) for s in spacing]
    return ndi.gaussian_gradient_magnitude(vol, sigma=sigma, mode="nearest").astype(np.float32, copy=False)


def sobel_magnitude(vol: np.ndarray, spacing: Sequence[float] = (1, 1, 1)) -> np.ndarray:
    g2 = np.zeros(vol.shape, dtype=np.float32)
    for ax in range(3):
        g = ndi.sobel(vol, axis=ax, mode="nearest") / float(spacing[ax])
        g2 += g * g
    return np.sqrt(g2)


def normalise_edge(e: np.ndarray, mask: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    vals = e[mask]
    ref = float(np.percentile(vals, percentile)) if vals.size else 1.0
    if ref <= 1e-9:
        ref = 1.0
    out = np.clip(e / ref, 0.0, 1.0)
    out[~mask] = 0.0
    return out.astype(np.float32, copy=False)


def compute_edges(vols: Dict[str, np.ndarray], mask: np.ndarray, spacing: Sequence[float], method: str = "gradient",
                  sigma_mm: float = 1.0, combine: str = "max", weights: Dict[str, float] | None = None,
                  percentile: float = 99.0) -> np.ndarray:
    maps = []
    for name, v in vols.items():
        e = gradient_magnitude(v, spacing, sigma_mm) if method == "gradient" else sobel_magnitude(v, spacing)
        e = normalise_edge(e, mask, percentile) * float((weights or {}).get(name, 1.0))
        maps.append(e)
    stack = np.stack(maps)
    if combine == "max":
        out = stack.max(axis=0)
    elif combine == "mean":
        out = stack.mean(axis=0)
    elif combine == "rms":
        out = np.sqrt((stack ** 2).mean(axis=0))
    else:
        raise ValueError(f"unknown combine {combine!r}")
    return normalise_edge(out, mask, percentile)


def canny_slice(slice2d: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """2-D Canny contour for visualisation only (not used by the clustering)."""
    return feature.canny(slice2d, sigma=sigma)
