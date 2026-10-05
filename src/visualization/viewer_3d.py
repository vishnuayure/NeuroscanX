import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Brain3DViewer:
    def __init__(self, volume: np.ndarray, segmentation: np.ndarray | None = None):
        self.volume = np.asarray(volume, dtype=np.float32)

        if self.volume.ndim != 3:
            raise ValueError("Volume must be 3D")

        if segmentation is not None:
            self.segmentation = np.asarray(segmentation)

            if self.segmentation.shape != self.volume.shape:
                raise ValueError("Volume and segmentation must have the same shape")
        else:
            self.segmentation = None

    def _get_surface(self, data: np.ndarray, level: float = 0.5):
        if np.max(data) <= level:
            return None, None

        vertices, faces, _, _ = marching_cubes(
            data.astype(np.float32),
            level=level
        )

        return vertices, faces

    def show(self, brain_threshold: float = 0.2):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        brain = self.volume.copy()

        minimum = np.nanmin(brain)
        maximum = np.nanmax(brain)

        if maximum > minimum:
            brain = (brain - minimum) / (maximum - minimum)

        brain_mask = brain > brain_threshold

        vertices, faces = self._get_surface(
            brain_mask,
            level=0.5
        )

        if vertices is not None:
            brain_mesh = Poly3DCollection(
                vertices[faces],
                alpha=0.15
            )
            ax.add_collection3d(brain_mesh)

        if self.segmentation is not None:
            tumour_mask = self.segmentation > 0

            vertices, faces = self._get_surface(
                tumour_mask,
                level=0.5
            )

            if vertices is not None:
                tumour_mesh = Poly3DCollection(
                    vertices[faces],
                    alpha=0.8
                )
                ax.add_collection3d(tumour_mesh)

        ax.set_xlim(0, self.volume.shape[0])
        ax.set_ylim(0, self.volume.shape[1])
        ax.set_zlim(0, self.volume.shape[2])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Brain Volume and Tumour Mesh")

        plt.tight_layout()
        plt.show()


def show_brain_3d(
    volume: np.ndarray,
    segmentation: np.ndarray | None = None,
    brain_threshold: float = 0.2
):
    viewer = Brain3DViewer(
        volume,
        segmentation
    )

    viewer.show(brain_threshold)


__all__ = [
    "Brain3DViewer",
    "show_brain_3d"
]