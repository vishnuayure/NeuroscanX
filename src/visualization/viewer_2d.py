import numpy as np
import matplotlib.pyplot as plt


class MRIViewer2D:
    def __init__(self, volume: np.ndarray):
        volume = np.asarray(volume)

        if volume.ndim != 3:
            raise ValueError("Volume must be 3D")

        self.volume = volume
        self.shape = volume.shape

    def get_slice(self, plane: str, index: int) -> np.ndarray:
        if plane == "axial":
            if not 0 <= index < self.shape[2]:
                raise IndexError("Invalid axial slice index")
            return self.volume[:, :, index]

        if plane == "coronal":
            if not 0 <= index < self.shape[1]:
                raise IndexError("Invalid coronal slice index")
            return self.volume[:, index, :]

        if plane == "sagittal":
            if not 0 <= index < self.shape[0]:
                raise IndexError("Invalid sagittal slice index")
            return self.volume[index, :, :]

        raise ValueError("Plane must be axial, coronal, or sagittal")

    def show_slice(
        self,
        plane: str,
        index: int,
        cmap: str = "gray"
    ):
        slice_data = self.get_slice(plane, index)

        plt.figure(figsize=(7, 7))
        plt.imshow(
            np.rot90(slice_data),
            cmap=cmap,
            interpolation="nearest"
        )
        plt.title(f"{plane.capitalize()} Slice {index}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def show_all_planes(self):
        indices = [
            self.shape[0] // 2,
            self.shape[1] // 2,
            self.shape[2] // 2
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        planes = ["sagittal", "coronal", "axial"]

        for ax, plane, index in zip(axes, planes, indices):
            slice_data = self.get_slice(plane, index)
            ax.imshow(
                np.rot90(slice_data),
                cmap="gray",
                interpolation="nearest"
            )
            ax.set_title(f"{plane.capitalize()} {index}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()


def view_volume(
    volume: np.ndarray,
    plane: str = "axial",
    index: int | None = None
):
    viewer = MRIViewer2D(volume)

    if index is None:
        index = {
            "axial": volume.shape[2] // 2,
            "coronal": volume.shape[1] // 2,
            "sagittal": volume.shape[0] // 2
        }[plane]

    viewer.show_slice(plane, index)


__all__ = ["MRIViewer2D", "view_volume"]