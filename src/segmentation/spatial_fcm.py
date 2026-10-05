import numpy as np
from scipy.ndimage import uniform_filter
from src.segmentation.fcm import FCM


class SpatialFCM(FCM):
    def __init__(
        self,
        n_clusters: int = 4,
        m: float = 2.0,
        max_iter: int = 100,
        error: float = 1e-5,
        random_state: int = 42,
        spatial_weight: float = 0.5,
        window_size: int = 3
    ):
        super().__init__(
            n_clusters=n_clusters,
            m=m,
            max_iter=max_iter,
            error=error,
            random_state=random_state
        )

        if spatial_weight < 0:
            raise ValueError("spatial_weight must be non-negative")

        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd number")

        self.spatial_weight = spatial_weight
        self.window_size = window_size

    def _spatial_term(self, membership: np.ndarray) -> np.ndarray:
        spatial = np.zeros_like(membership)

        for cluster in range(self.n_clusters):
            spatial[..., cluster] = uniform_filter(
                membership[..., cluster],
                size=self.window_size,
                mode="nearest"
            )

        return spatial

    def _update_membership_spatial(
        self,
        data: np.ndarray,
        centers: np.ndarray,
        membership_shape: tuple
    ) -> np.ndarray:
        distances = np.linalg.norm(
            data[:, None, :] - centers[None, :, :],
            axis=2
        )

        distances = np.maximum(distances, 1e-12)

        power = 2.0 / (self.m - 1.0)

        ratios = (
            distances[:, :, None] /
            distances[:, None, :]
        ) ** power

        membership = 1.0 / ratios.sum(axis=2)

        spatial = self._spatial_term(
            membership.reshape(membership_shape)
        )

        spatial_flat = spatial.reshape(-1, self.n_clusters)

        membership = membership + self.spatial_weight * spatial_flat

        membership /= membership.sum(axis=1, keepdims=True)

        return membership

    def fit(self, data: np.ndarray):
        data = np.asarray(data, dtype=np.float32)

        if data.ndim != 3:
            raise ValueError("Spatial FCM requires 3D input data")

        if not np.all(np.isfinite(data)):
            raise ValueError("Input data contains non-finite values")

        original_shape = data.shape
        flat_data = data.reshape(-1, 1)

        membership = self._initialize_membership(len(flat_data))
        membership_shape = (*original_shape, self.n_clusters)

        for iteration in range(self.max_iter):
            centers = self._update_centers(flat_data, membership)

            new_membership = self._update_membership_spatial(
                flat_data,
                centers,
                membership_shape
            )

            change = np.max(
                np.abs(new_membership - membership)
            )

            membership = new_membership

            if change < self.error:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.centers = centers
        self.membership = membership.reshape(membership_shape)
        self.labels = np.argmax(
            self.membership,
            axis=-1
        )

        return self


def spatial_fuzzy_c_means(
    data: np.ndarray,
    n_clusters: int = 4,
    m: float = 2.0,
    spatial_weight: float = 0.5,
    window_size: int = 3,
    max_iter: int = 100,
    error: float = 1e-5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, SpatialFCM]:
    """
    Spatially Penalized Fuzzy C-Means segmentation.

    Returns:
        tuple: (labels, membership, model)
    """
    model = SpatialFCM(
        n_clusters=n_clusters,
        m=m,
        spatial_weight=spatial_weight,
        window_size=window_size,
        max_iter=max_iter,
        error=error,
        random_state=random_state,
    )
    model.fit(data)
    return model.labels, model.membership, model


__all__ = ["SpatialFCM", "spatial_fuzzy_c_means"]