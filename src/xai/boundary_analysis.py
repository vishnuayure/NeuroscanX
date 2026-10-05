import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt


def extract_boundary(segmentation: np.ndarray) -> np.ndarray:
    segmentation = np.asarray(segmentation)

    if segmentation.ndim != 3:
        raise ValueError("Segmentation must be 3D")

    mask = segmentation > 0

    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    eroded = binary_erosion(mask)
    return mask & ~eroded


def boundary_thickness(segmentation: np.ndarray) -> np.ndarray:
    boundary = extract_boundary(segmentation)

    if not np.any(boundary):
        return np.zeros_like(segmentation, dtype=np.float32)

    distance = distance_transform_edt(~boundary)
    return distance.astype(np.float32)


def boundary_density(segmentation: np.ndarray) -> float:
    mask = np.asarray(segmentation) > 0

    if not np.any(mask):
        return 0.0

    boundary = extract_boundary(segmentation)

    return float(
        np.sum(boundary) / np.sum(mask)
    )


def boundary_irregularity(segmentation: np.ndarray) -> float:
    mask = np.asarray(segmentation) > 0

    if not np.any(mask):
        return 0.0

    boundary = extract_boundary(mask)

    dilated = binary_dilation(boundary)

    boundary_volume = np.sum(boundary)
    dilated_volume = np.sum(dilated)

    if boundary_volume == 0:
        return 0.0

    return float(
        dilated_volume / boundary_volume
    )


def analyze_boundary(
    segmentation: np.ndarray
) -> dict[str, float]:
    mask = np.asarray(segmentation) > 0

    if not np.any(mask):
        return {
            "tumor_volume": 0.0,
            "boundary_volume": 0.0,
            "boundary_density": 0.0,
            "boundary_irregularity": 0.0
        }

    boundary = extract_boundary(mask)

    return {
        "tumor_volume": float(np.sum(mask)),
        "boundary_volume": float(np.sum(boundary)),
        "boundary_density": boundary_density(mask),
        "boundary_irregularity": boundary_irregularity(mask)
    }


__all__ = [
    "extract_boundary",
    "boundary_thickness",
    "boundary_density",
    "boundary_irregularity",
    "analyze_boundary"
]