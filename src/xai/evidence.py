import numpy as np
from scipy.ndimage import gaussian_filter


def normalize_evidence(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)

    minimum = np.nanmin(data)
    maximum = np.nanmax(data)

    if maximum <= minimum:
        return np.zeros_like(data, dtype=np.float32)

    return ((data - minimum) / (maximum - minimum)).astype(np.float32)


def intensity_evidence(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)

    if volume.ndim != 3:
        raise ValueError("Volume must be 3D")

    return normalize_evidence(volume)


def edge_evidence(
    volume: np.ndarray,
    sigma: float = 1.0
) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)

    if volume.ndim != 3:
        raise ValueError("Volume must be 3D")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    smoothed = gaussian_filter(volume, sigma=sigma)

    gx, gy, gz = np.gradient(smoothed)

    edges = np.sqrt(
        gx ** 2 +
        gy ** 2 +
        gz ** 2
    )

    return normalize_evidence(edges)


def spatial_evidence(
    segmentation: np.ndarray,
    sigma: float = 2.0
) -> np.ndarray:
    segmentation = np.asarray(
        segmentation,
        dtype=np.float32
    )

    if segmentation.ndim != 3:
        raise ValueError("Segmentation must be 3D")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    spatial = gaussian_filter(
        segmentation > 0,
        sigma=sigma
    )

    return normalize_evidence(spatial)


def combined_evidence(
    intensity: np.ndarray,
    edges: np.ndarray,
    segmentation: np.ndarray,
    intensity_weight: float = 0.3,
    edge_weight: float = 0.4,
    spatial_weight: float = 0.3
) -> np.ndarray:
    if min(
        intensity_weight,
        edge_weight,
        spatial_weight
    ) < 0:
        raise ValueError("Evidence weights must be non-negative")

    total = (
        intensity_weight +
        edge_weight +
        spatial_weight
    )

    if total == 0:
        raise ValueError("At least one evidence weight must be positive")

    intensity_map = intensity_evidence(intensity)
    edge_map = normalize_evidence(edges)
    spatial_map = spatial_evidence(segmentation)

    evidence = (
        intensity_weight * intensity_map +
        edge_weight * edge_map +
        spatial_weight * spatial_map
    )

    return (evidence / total).astype(np.float32)


__all__ = [
    "normalize_evidence",
    "intensity_evidence",
    "edge_evidence",
    "spatial_evidence",
    "combined_evidence"
]