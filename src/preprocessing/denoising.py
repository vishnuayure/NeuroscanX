"""Light, edge-conserving denoising.  Kept deliberately mild so tumor boundaries survive."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def denoise(vol: np.ndarray, method: str = "gaussian", sigma: float = 0.6, median_size: int = 3) -> np.ndarray:
    if method == "none" or (method == "gaussian" and sigma <= 0):
        return vol
    if method == "gaussian":
        return ndi.gaussian_filter(vol, sigma=sigma, mode="nearest")
    if method == "median":
        return ndi.median_filter(vol, size=median_size, mode="nearest")
    raise ValueError(f"unknown denoise method {method!r}")
