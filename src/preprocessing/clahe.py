from __future__ import annotations

import numpy as np
from skimage.exposure import equalize_adapthist

from src.io.nifti_loader import NiftiVolume


def clahe_enhance(
    data: np.ndarray,
    clip_limit: float = 0.01,
    kernel_size: int | tuple[int, int, int] | None = None,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D volume, got shape {data.shape}."
        )

    if clip_limit <= 0:
        raise ValueError(
            "clip_limit must be greater than 0."
        )

    finite_mask = np.isfinite(data)

    if not np.any(finite_mask):
        return np.zeros_like(data, dtype=np.float32)

    values = data[finite_mask]
    minimum = np.min(values)
    maximum = np.max(values)

    if maximum == minimum:
        return np.zeros_like(data, dtype=np.float32)

    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[finite_mask] = (
        (data[finite_mask] - minimum)
        / (maximum - minimum)
    )

    enhanced = equalize_adapthist(
        normalized,
        kernel_size=kernel_size,
        clip_limit=clip_limit,
    )

    output = np.zeros_like(data, dtype=np.float32)
    output[finite_mask] = enhanced[finite_mask]

    return output


def clahe_volume(
    volume: NiftiVolume | np.ndarray,
    clip_limit: float = 0.01,
    kernel_size: int | tuple[int, int, int] | None = None,
) -> np.ndarray:
    if hasattr(volume, "get_data"):
        data = volume.get_data(dtype=np.float32)
    else:
        data = np.asarray(volume, dtype=np.float32)

    return clahe_enhance(
        data=data,
        clip_limit=clip_limit,
        kernel_size=kernel_size,
    )


__all__ = [
    "clahe_enhance",
    "clahe_volume",
]