"""Slice-wise CLAHE for local-contrast enhancement.

CLAHE is *not* applied to the clustering features by default: it is non-linear and would
destroy the quantitative relationship between modalities that the fuzzy clustering and the
tumor-cluster scoring rely on.  It is used to strengthen weak boundaries for edge extraction
and for display (see PreprocessConfig.clahe_apply_to).
"""
from __future__ import annotations

import numpy as np
from skimage import exposure


def clahe_volume(vol01: np.ndarray, mask: np.ndarray, clip_limit: float = 0.01, kernel_fraction: float = 0.125) -> np.ndarray:
    """vol01 must be in [0, 1].  Operates on axial slices that contain brain; background stays 0."""
    out = np.zeros_like(vol01)
    ks = max(8, int(round(min(vol01.shape[:2]) * kernel_fraction)))
    for k in range(vol01.shape[2]):
        m = mask[:, :, k]
        if m.sum() < 25:
            continue
        sl = exposure.equalize_adapthist(vol01[:, :, k], kernel_size=ks, clip_limit=clip_limit)
        out[:, :, k] = np.where(m, sl, 0.0)
    return out.astype(np.float32, copy=False)
