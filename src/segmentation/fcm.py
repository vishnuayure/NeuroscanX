import numpy as np


class FCM:
    def __init__(
        self,
        n_clusters: int = 4,
        m: float = 2.0,
        max_iter: int = 100,
        error: float = 1e-5,
        random_state: int = 42
    ):
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if m <= 1:
            raise ValueError("m must be greater than 1")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if error <= 0:
            raise ValueError("error must be positive")

        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state
        self.centers = None
        self.membership = None
        self.labels = None
        self.n_iter_ = 0

    def _initialize_membership(self, n_samples: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        membership = rng.random((n_samples, self.n_clusters))
        membership /= membership.sum(axis=1, keepdims=True)
        return membership

    def _update_centers(
        self,
        data: np.ndarray,
        membership: np.ndarray
    ) -> np.ndarray:
        weights = membership ** self.m
        denominator = weights.sum(axis=0)

        centers = (
            weights.T @ data
        ) / np.maximum(denominator[:, None], 1e-12)

        return centers

    def _update_membership(
        self,
        data: np.ndarray,
        centers: np.ndarray
    ) -> np.ndarray:
        distances = np.linalg.norm(
            data[:, None, :] - centers[None, :, :],
            axis=2
        )

        zero_distance = distances == 0

        if np.any(zero_distance):
            membership = np.zeros_like(distances)
            rows = np.where(zero_distance.any(axis=1))[0]

            for row in rows:
                clusters = np.where(zero_distance[row])[0]
                membership[row, clusters] = 1.0 / len(clusters)

            nonzero_rows = np.where(~zero_distance.any(axis=1))[0]

            if len(nonzero_rows) > 0:
                d = distances[nonzero_rows]
                power = 2.0 / (self.m - 1.0)
                ratios = (d[:, :, None] / d[:, None, :]) ** power
                membership[nonzero_rows] = 1.0 / ratios.sum(axis=2)

            return membership

        power = 2.0 / (self.m - 1.0)
        ratios = (
            distances[:, :, None] /
            distances[:, None, :]
        ) ** power

        return 1.0 / ratios.sum(axis=2)

    def fit(self, data: np.ndarray):
        data = np.asarray(data, dtype=np.float32)

        if data.ndim == 3:
            original_shape = data.shape
            data = data.reshape(-1, 1)
        elif data.ndim == 2:
            original_shape = None
        else:
            raise ValueError("Input data must be 2D or 3D")

        if len(data) == 0:
            raise ValueError("Input data cannot be empty")

        if not np.all(np.isfinite(data)):
            raise ValueError("Input data contains non-finite values")

        membership = self._initialize_membership(len(data))

        for iteration in range(self.max_iter):
            centers = self._update_centers(data, membership)
            new_membership = self._update_membership(data, centers)

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
        self.membership = membership
        self.labels = np.argmax(membership, axis=1)

        if original_shape is not None:
            self.labels = self.labels.reshape(original_shape)
            self.membership = self.membership.reshape(
                *original_shape,
                self.n_clusters
            )

        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.centers is None:
            raise RuntimeError("FCM model has not been fitted")

        data = np.asarray(data, dtype=np.float32)
        original_shape = data.shape

        if data.ndim == 3:
            data = data.reshape(-1, 1)
        elif data.ndim != 2:
            raise ValueError("Input data must be 2D or 3D")

        if not np.all(np.isfinite(data)):
            raise ValueError("Input data contains non-finite values")

        membership = self._update_membership(data, self.centers)
        labels = np.argmax(membership, axis=1)

        if len(original_shape) == 3:
            labels = labels.reshape(original_shape)

        return labels

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.centers is None:
            raise RuntimeError("FCM model has not been fitted")

        data = np.asarray(data, dtype=np.float32)

        if data.ndim == 3:
            data = data.reshape(-1, 1)
        elif data.ndim != 2:
            raise ValueError("Input data must be 2D or 3D")

        if not np.all(np.isfinite(data)):
            raise ValueError("Input data contains non-finite values")

        membership = self._update_membership(data, self.centers)

        if len(data.shape) == 2:
            return membership

        return membership


def fuzzy_c_means(
    data: np.ndarray,
    n_clusters: int = 4,
    m: float = 2.0,
    max_iter: int = 100,
    error: float = 1e-5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, FCM]:
    """
    Standard Fuzzy C-Means segmentation.

    Returns:
        tuple: (labels, membership, model)
    """
    model = FCM(
        n_clusters=n_clusters,
        m=m,
        max_iter=max_iter,
        error=error,
        random_state=random_state,
    )
    model.fit(data)
    return model.labels, model.membership, model


__all__ = ["FCM", "fuzzy_c_means"]