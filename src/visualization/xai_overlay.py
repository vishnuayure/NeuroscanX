import numpy as np
import matplotlib.pyplot as plt


def normalize_map(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)

    minimum = np.nanmin(data)
    maximum = np.nanmax(data)

    if maximum <= minimum:
        return np.zeros_like(data, dtype=np.float32)

    return ((data - minimum) / (maximum - minimum)).astype(np.float32)


def create_xai_overlay(
    image: np.ndarray,
    evidence: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "jet"
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    evidence = np.asarray(evidence, dtype=np.float32)

    if image.shape != evidence.shape:
        raise ValueError("Image and evidence must have the same shape")

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    image = normalize_map(image)
    evidence = normalize_map(evidence)

    base = np.stack([image, image, image], axis=-1)
    heatmap = plt.get_cmap(cmap)(evidence)[..., :3]

    mask = evidence > 0

    base[mask] = (
        (1 - alpha) * base[mask] +
        alpha * heatmap[mask]
    )

    return np.clip(base, 0, 1).astype(np.float32)


def xai_overlay_slice(
    image: np.ndarray,
    evidence: np.ndarray,
    plane: str,
    index: int,
    alpha: float = 0.5,
    cmap: str = "jet"
) -> np.ndarray:
    if image.ndim != 3 or evidence.ndim != 3:
        raise ValueError("Image and evidence must be 3D")

    if image.shape != evidence.shape:
        raise ValueError("Image and evidence must have the same shape")

    if plane == "axial":
        image_slice = image[:, :, index]
        evidence_slice = evidence[:, :, index]
    elif plane == "coronal":
        image_slice = image[:, index, :]
        evidence_slice = evidence[:, index, :]
    elif plane == "sagittal":
        image_slice = image[index, :, :]
        evidence_slice = evidence[index, :, :]
    else:
        raise ValueError("Plane must be axial, coronal, or sagittal")

    return create_xai_overlay(
        image_slice,
        evidence_slice,
        alpha=alpha,
        cmap=cmap
    )


def show_xai_overlay(
    image: np.ndarray,
    evidence: np.ndarray,
    plane: str = "axial",
    index: int | None = None,
    alpha: float = 0.5,
    cmap: str = "jet"
):
    if index is None:
        index = {
            "axial": image.shape[2] // 2,
            "coronal": image.shape[1] // 2,
            "sagittal": image.shape[0] // 2
        }[plane]

    overlay = xai_overlay_slice(
        image,
        evidence,
        plane,
        index,
        alpha,
        cmap
    )

    plt.figure(figsize=(7, 7))
    plt.imshow(np.rot90(overlay))
    plt.title(f"{plane.capitalize()} XAI Evidence")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


__all__ = [
    "normalize_map",
    "create_xai_overlay",
    "xai_overlay_slice",
    "show_xai_overlay"
]