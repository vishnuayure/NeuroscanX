from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.io.nifti_loader import NiftiVolume


def z_score_normalize(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)

    if mask is None:
        mask = np.isfinite(data) & (data != 0)

    else:
        mask = np.asarray(mask, dtype=bool)
        mask &= np.isfinite(data)

    if not np.any(mask):
        return np.zeros_like(data, dtype=np.float32)

    values = data[mask]
    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        return np.zeros_like(data, dtype=np.float32)

    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[mask] = (data[mask] - mean) / std

    return normalized


def min_max_normalize(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)

    if mask is None:
        mask = np.isfinite(data)

    else:
        mask = np.asarray(mask, dtype=bool)
        mask &= np.isfinite(data)

    if not np.any(mask):
        return np.zeros_like(data, dtype=np.float32)

    values = data[mask]
    minimum = np.min(values)
    maximum = np.max(values)

    if maximum == minimum:
        return np.zeros_like(data, dtype=np.float32)

    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[mask] = (
        (data[mask] - minimum)
        / (maximum - minimum)
    )

    return normalized


def normalize_volume(
    volume: Union[NiftiVolume, np.ndarray],
    method: str = "zscore",
) -> np.ndarray:
    if hasattr(volume, "get_data"):
        data = volume.get_data(dtype=np.float32)
    else:
        data = np.asarray(volume, dtype=np.float32)

    method = method.lower().strip()

    if method == "zscore":
        return z_score_normalize(data)

    if method == "minmax":
        return min_max_normalize(data)

    raise ValueError(
        f"Unknown normalization method: {method}. "
        "Use 'zscore' or 'minmax'."
    )


__all__ = [
    "z_score_normalize",
    "min_max_normalize",
    "normalize_volume",
]