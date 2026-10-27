"""Brain/background handling and robust intensity normalisation."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def compute_brain_mask(volumes: Dict[str, np.ndarray], method: str = "auto", closing: int = 2) -> np.ndarray:
    """Foreground mask from all modalities.

    * ``nonzero`` – BraTS-style skull-stripped data (background exactly 0).
    * ``otsu``    – for data that is not skull-stripped (mean of normalised modalities).
    * ``auto``    – nonzero, unless nearly everything is non-zero (then otsu).
    """
    stack = [np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for v in volumes.values()]
    nz = np.zeros(stack[0].shape, dtype=bool)
    for v in stack:
        nz |= v != 0
    if method == "auto":
        method = "otsu" if nz.mean() > 0.85 else "nonzero"
    if method == "otsu":
        acc = np.zeros(stack[0].shape, dtype=np.float32)
        for v in stack:
            hi = np.percentile(v, 99.5) or 1.0
            acc += np.clip(v / hi, 0, 1)
        acc /= len(stack)
        mask = acc > threshold_otsu(acc)
    else:
        mask = nz
    if closing > 0:
        mask = ndi.binary_closing(mask, structure=ndi.generate_binary_structure(3, 1), iterations=closing)
    mask = ndi.binary_fill_holes(mask)
    mask = largest_component(mask)
    return mask.astype(bool)


def robust_normalize(vol: np.ndarray, mask: np.ndarray, lower: float = 1.0, upper: float = 99.5) -> np.ndarray:
    """Percentile window computed on brain voxels only → [0, 1]; background stays 0."""
    vals = vol[mask]
    lo, hi = np.percentile(vals, [lower, upper])
    if hi <= lo:
        hi = lo + 1.0
    out = (vol - lo) / (hi - lo)
    np.clip(out, 0.0, 1.0, out=out)
    out[~mask] = 0.0
    return out.astype(np.float32, copy=False)


def robust_stats(vol: np.ndarray, mask: np.ndarray, max_samples: int = 300_000, seed: int = 0):
    """Median and robust standard deviation (1.4826·MAD) of brain voxels."""
    vals = vol[mask]
    if vals.size > max_samples:
        vals = vals[np.random.default_rng(seed).choice(vals.size, max_samples, replace=False)]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    if mad < 1e-6:
        mad = float(vals.std()) + 1e-6
    return med, mad
