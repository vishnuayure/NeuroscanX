import numpy as np
import matplotlib.pyplot as plt


def create_segmentation_overlay(
    image: np.ndarray,
    segmentation: np.ndarray,
    alpha: float = 0.4,
    cmap: str = "jet"
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    segmentation = np.asarray(segmentation)

    if image.shape != segmentation.shape:
        raise ValueError("Image and segmentation must have the same shape")

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    image_min = np.nanmin(image)
    image_max = np.nanmax(image)

    if image_max > image_min:
        normalized = (image - image_min) / (image_max - image_min)
    else:
        normalized = np.zeros_like(image)

    rgb = np.stack([normalized] * 3, axis=-1)

    if np.any(segmentation > 0):
        colormap = plt.get_cmap(cmap)
        labels = segmentation.astype(np.float32)
        labels /= max(labels.max(), 1)
        mask_rgb = colormap(labels)[..., :3]

        mask = segmentation > 0
        rgb[mask] = (
            (1 - alpha) * rgb[mask] +
            alpha * mask_rgb[mask]
        )

    return np.clip(rgb, 0, 1).astype(np.float32)


def overlay_slice(
    image: np.ndarray,
    segmentation: np.ndarray,
    plane: str,
    index: int,
    alpha: float = 0.4,
    cmap: str = "jet"
) -> np.ndarray:
    if image.ndim != 3 or segmentation.ndim != 3:
        raise ValueError("Image and segmentation must be 3D")

    if plane == "axial":
        image_slice = image[:, :, index]
        segmentation_slice = segmentation[:, :, index]
    elif plane == "coronal":
        image_slice = image[:, index, :]
        segmentation_slice = segmentation[:, index, :]
    elif plane == "sagittal":
        image_slice = image[index, :, :]
        segmentation_slice = segmentation[index, :, :]
    else:
        raise ValueError("Plane must be axial, coronal, or sagittal")

    return create_segmentation_overlay(
        image_slice,
        segmentation_slice,
        alpha=alpha,
        cmap=cmap
    )


def show_overlay(
    image: np.ndarray,
    segmentation: np.ndarray,
    plane: str = "axial",
    index: int | None = None,
    alpha: float = 0.4,
    cmap: str = "jet"
):
    if index is None:
        index = {
            "axial": image.shape[2] // 2,
            "coronal": image.shape[1] // 2,
            "sagittal": image.shape[0] // 2
        }[plane]

    overlay = overlay_slice(
        image,
        segmentation,
        plane,
        index,
        alpha,
        cmap
    )

    plt.figure(figsize=(7, 7))
    plt.imshow(np.rot90(overlay))
    plt.title(f"{plane.capitalize()} Segmentation Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


__all__ = [
    "create_segmentation_overlay",
    "overlay_slice",
    "show_overlay"
]