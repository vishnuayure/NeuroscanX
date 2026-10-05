import numpy as np
from scipy.ndimage import sobel, gaussian_filter
from src.io.nifti_loader import NiftiVolume


def sobel_edges(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError("Input data must be 3D")

    gx = sobel(data, axis=0)
    gy = sobel(data, axis=1)
    gz = sobel(data, axis=2)

    magnitude = np.sqrt(gx**2 + gy**2 + gz**2)

    max_value = magnitude.max()
    if max_value > 0:
        magnitude /= max_value

    return magnitude.astype(np.float32)


def gradient_edges(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError("Input data must be 3D")

    gx, gy, gz = np.gradient(data)

    magnitude = np.sqrt(gx**2 + gy**2 + gz**2)

    max_value = magnitude.max()
    if max_value > 0:
        magnitude /= max_value

    return magnitude.astype(np.float32)


def smooth_edges(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    smoothed = gaussian_filter(
        np.asarray(data, dtype=np.float32),
        sigma=sigma
    )

    return sobel_edges(smoothed)


def extract_edges(
    volume: NiftiVolume | np.ndarray,
    method: str = "sobel"
) -> np.ndarray:
    if hasattr(volume, "get_data"):
        data = volume.get_data()
    else:
        data = np.asarray(volume)

    if method == "sobel":
        return sobel_edges(data)
    if method == "gradient":
        return gradient_edges(data)
    if method == "smooth":
        return smooth_edges(data)

    raise ValueError(f"Unknown edge detection method: {method}")


__all__ = [
    "sobel_edges",
    "gradient_edges",
    "smooth_edges",
    "extract_edges"
]