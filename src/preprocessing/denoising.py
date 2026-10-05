from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from src.io.nifti_loader import NiftiVolume


def gaussian_denoise(
    data: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)

    if sigma <= 0:
        raise ValueError("sigma must be greater than 0.")

    return gaussian_filter(
        data,
        sigma=sigma,
    ).astype(np.float32)


def median_denoise(
    data: np.ndarray,
    size: int = 3,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)

    if size <= 0 or size % 2 == 0:
        raise ValueError(
            "size must be a positive odd integer."
        )

    return median_filter(
        data,
        size=size,
    ).astype(np.float32)


def denoise_volume(
    volume: NiftiVolume | np.ndarray,
    method: Literal["gaussian", "median"] = "gaussian",
    sigma: float = 1.0,
    size: int = 3,
) -> np.ndarray:
    if hasattr(volume, "get_data"):
        data = volume.get_data(dtype=np.float32)
    else:
        data = np.asarray(volume, dtype=np.float32)

    method = method.lower().strip()

    if method == "gaussian":
        return gaussian_denoise(
            data,
            sigma=sigma,
        )

    if method == "median":
        return median_denoise(
            data,
            size=size,
        )

    raise ValueError(
        f"Unknown denoising method: {method}. "
        "Use 'gaussian' or 'median'."
    )


__all__ = [
    "gaussian_denoise",
    "median_denoise",
    "denoise_volume",
]