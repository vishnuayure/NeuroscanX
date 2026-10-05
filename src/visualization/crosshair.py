import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class CrosshairViewer:
    def __init__(self, volume: np.ndarray):
        volume = np.asarray(volume, dtype=np.float32)

        if volume.ndim != 3:
            raise ValueError("Volume must be 3D")

        self.volume = volume
        self.shape = volume.shape

        self.x = self.shape[0] // 2
        self.y = self.shape[1] // 2
        self.z = self.shape[2] // 2

    def _draw(self):
        self.axial.clear()
        self.coronal.clear()
        self.sagittal.clear()

        self.axial.imshow(
            np.rot90(self.volume[:, :, self.z]),
            cmap="gray"
        )
        self.coronal.imshow(
            np.rot90(self.volume[:, self.y, :]),
            cmap="gray"
        )
        self.sagittal.imshow(
            np.rot90(self.volume[self.x, :, :]),
            cmap="gray"
        )

        self.axial.axvline(self.x, color="red")
        self.axial.axhline(self.y, color="red")

        self.coronal.axvline(self.x, color="red")
        self.coronal.axhline(self.z, color="red")

        self.sagittal.axvline(self.y, color="red")
        self.sagittal.axhline(self.z, color="red")

        self.axial.set_title(f"Axial Z={self.z}")
        self.coronal.set_title(f"Coronal Y={self.y}")
        self.sagittal.set_title(f"Sagittal X={self.x}")

        self.axial.axis("off")
        self.coronal.axis("off")
        self.sagittal.axis("off")

        self.fig.canvas.draw_idle()

    def show(self):
        self.fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5)
        )

        self.axial = axes[0]
        self.coronal = axes[1]
        self.sagittal = axes[2]

        self._draw()

        plt.show()

    def set_position(self, x: int, y: int, z: int):
        if not 0 <= x < self.shape[0]:
            raise IndexError("Invalid X coordinate")

        if not 0 <= y < self.shape[1]:
            raise IndexError("Invalid Y coordinate")

        if not 0 <= z < self.shape[2]:
            raise IndexError("Invalid Z coordinate")

        self.x = x
        self.y = y
        self.z = z

        if hasattr(self, "fig"):
            self._draw()


def show_crosshair(volume: np.ndarray):
    viewer = CrosshairViewer(volume)
    viewer.show()


__all__ = ["CrosshairViewer", "show_crosshair"]