"""
Edge-Aware Spatially Penalized Fuzzy C-Means (SP-FCM) for Brain Tumour Segmentation.

Integrates spatial neighborhood regularization with boundary-preserving edge gating:
Strong edges reduce spatial regularization across boundaries, preventing smoothing
of sharp tumour borders, while homogeneous tissue regions undergo stronger spatial smoothing.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.ndimage import uniform_filter

from src.segmentation.fcm import FCM
from src.features.edge_detection import sobel_edges


class EdgeSpatialFCM(FCM):
    """
    Edge-Aware Spatially Penalized Fuzzy C-Means (Edge-SP-FCM).
    """

    def __init__(
        self,
        n_clusters: int = 4,
        m: float = 2.0,
        max_iter: int = 100,
        error: float = 1e-5,
        random_state: int = 42,
        spatial_weight: float = 0.5,
        edge_weight: float = 1.0,
        window_size: int = 3,
    ):
        super().__init__(
            n_clusters=n_clusters,
            m=m,
            max_iter=max_iter,
            error=error,
            random_state=random_state,
        )

        if spatial_weight < 0:
            raise ValueError("spatial_weight must be non-negative")

        if edge_weight < 0:
            raise ValueError("edge_weight must be non-negative")

        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd number")

        self.spatial_weight = spatial_weight
        self.edge_weight = edge_weight
        self.window_size = window_size
        self.edge_map: Optional[np.ndarray] = None

    def _spatial_term(self, membership: np.ndarray) -> np.ndarray:
        spatial = np.zeros_like(membership)
        for cluster in range(self.n_clusters):
            spatial[..., cluster] = uniform_filter(
                membership[..., cluster],
                size=self.window_size,
                mode="nearest",
            )
        return spatial

    def _edge_term(self, edges: np.ndarray) -> np.ndarray:
        edge_strength = np.asarray(edges, dtype=np.float32)
        minimum = float(np.nanmin(edge_strength))
        maximum = float(np.nanmax(edge_strength))

        if maximum > minimum:
            edge_strength = (edge_strength - minimum) / (maximum - minimum)
        else:
            edge_strength = np.zeros_like(edge_strength)

        return np.clip(edge_strength, 0.0, 1.0)

    def _update_membership(
        self,
        data: np.ndarray,
        centers: np.ndarray,
        membership_shape: Tuple[int, ...],
        edge_map: np.ndarray,
    ) -> np.ndarray:
        distances = np.linalg.norm(
            data[:, None, :] - centers[None, :, :],
            axis=2,
        )
        distances = np.maximum(distances, 1e-12)

        power = 2.0 / (self.m - 1.0)
        ratios = (distances[:, :, None] / distances[:, None, :]) ** power
        membership = 1.0 / ratios.sum(axis=2)

        # Spatial neighborhood regularization
        spatial = self._spatial_term(membership.reshape(membership_shape))
        spatial_flat = spatial.reshape(-1, self.n_clusters)

        # Edge gating: strong edges reduce spatial smoothing across boundaries
        edge = self._edge_term(edge_map)
        edge_flat = edge.reshape(-1, 1)

        effective_spatial = self.spatial_weight / (1.0 + self.edge_weight * edge_flat)
        spatial_factor = 1.0 + effective_spatial * spatial_flat

        membership = membership * spatial_factor
        membership /= membership.sum(axis=1, keepdims=True)

        return membership

    def fit(
        self,
        data: np.ndarray,
        edge_map: Optional[np.ndarray] = None,
    ) -> EdgeSpatialFCM:
        data = np.asarray(data, dtype=np.float32)

        if data.ndim == 3:
            spatial_shape = data.shape
            flat_data = data.reshape(-1, 1)
            ref_channel = data
        elif data.ndim == 4:
            spatial_shape = data.shape[:3]
            n_features = data.shape[3]
            flat_data = data.reshape(-1, n_features)
            ref_channel = data[..., 0]
        else:
            raise ValueError("Edge Spatial FCM requires 3D or 4D input data")

        if not np.all(np.isfinite(flat_data)):
            raise ValueError("Input data contains non-finite values")

        if edge_map is None:
            self.edge_map = sobel_edges(ref_channel)
        else:
            self.edge_map = self._edge_term(edge_map)
            if self.edge_map.shape != spatial_shape:
                raise ValueError(
                    f"Edge map shape {self.edge_map.shape} does not match spatial data shape {spatial_shape}"
                )

        membership = self._initialize_membership(len(flat_data))
        membership_shape = (*spatial_shape, self.n_clusters)

        for iteration in range(self.max_iter):
            centers = self._update_centers(flat_data, membership)
            new_membership = self._update_membership(
                flat_data,
                centers,
                membership_shape,
                self.edge_map,
            )

            change = float(np.max(np.abs(new_membership - membership)))
            membership = new_membership

            if change < self.error:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.centers = centers
        self.membership = membership.reshape(membership_shape)
        self.labels = np.argmax(self.membership, axis=-1)

        return self


SPFCM = EdgeSpatialFCM


def edge_spatial_fuzzy_c_means(
    data: np.ndarray,
    edge_map: Optional[np.ndarray] = None,
    n_clusters: int = 4,
    m: float = 2.0,
    spatial_weight: float = 0.5,
    edge_weight: float = 1.0,
    window_size: int = 3,
    max_iter: int = 100,
    error: float = 1e-5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, EdgeSpatialFCM]:
    """
    Edge-Aware Spatially Penalized Fuzzy C-Means (SP-FCM).

    Returns:
        tuple: (labels, membership, model)
    """
    model = EdgeSpatialFCM(
        n_clusters=n_clusters,
        m=m,
        spatial_weight=spatial_weight,
        edge_weight=edge_weight,
        window_size=window_size,
        max_iter=max_iter,
        error=error,
        random_state=random_state,
    )
    model.fit(data, edge_map=edge_map)
    return model.labels, model.membership, model


sp_fuzzy_c_means = edge_spatial_fuzzy_c_means


def determine_tumour_cluster(
    membership: np.ndarray,
    modalities_or_features: Union[np.ndarray, Mapping[str, np.ndarray]],
    background_threshold: float = 0.05,
) -> int:
    """
    Determine the tumour cluster using reproducible MRI characteristics.
    In BraTS multimodal MRI, tumour tissue is hyperintense in FLAIR and T1ce,
    while background has near-zero intensity across all modalities.

    Parameters
    ----------
    membership : np.ndarray
        Fuzzy membership map of shape (X, Y, Z, C) or (C, X, Y, Z).
    modalities_or_features : dict or np.ndarray
        Either dict with 'flair' and 't1ce' volumes or 3D/4D feature array.
    background_threshold : float
        Intensity threshold below which clusters are considered background.

    Returns
    -------
    tumour_cluster_idx : int
        Index of the identified tumour cluster.
    """
    m = np.asarray(membership, dtype=np.float32)
    # Ensure shape (X, Y, Z, C)
    if m.ndim == 4 and m.shape[0] < m.shape[-1] and m.shape[0] <= 10:
        m = np.moveaxis(m, 0, -1)

    n_clusters = m.shape[-1]

    if isinstance(modalities_or_features, Mapping):
        norm_vols = {k.lower(): np.asarray(v, dtype=np.float32) for k, v in modalities_or_features.items()}
        flair = norm_vols.get("flair", None)
        t1ce = norm_vols.get("t1ce", None)
        t2 = norm_vols.get("t2", None)
        t1 = norm_vols.get("t1", None)
    elif isinstance(modalities_or_features, np.ndarray):
        arr = modalities_or_features
        if arr.ndim == 3:
            flair = arr
            t1ce = arr
            t2 = arr
            t1 = arr
        elif arr.ndim == 4:
            flair = arr[..., -1]
            t1ce = arr[..., 1] if arr.shape[-1] > 1 else arr[..., 0]
            t2 = arr[..., 0]
            t1 = arr[..., 0]
        else:
            raise ValueError(f"Unexpected features shape {arr.shape}")
    else:
        raise TypeError("modalities_or_features must be dict or ndarray")

    cluster_scores = []
    for k in range(n_clusters):
        weights = m[..., k]
        total_weight = float(np.sum(weights))
        if total_weight < 1.0:
            cluster_scores.append(-1.0)
            continue

        # Check background: average intensity across available modalities
        mean_intensity = 0.0
        n_avail = 0
        for vol in (flair, t1ce, t2, t1):
            if vol is not None:
                mean_intensity += float(np.sum(vol * weights) / total_weight)
                n_avail += 1
        mean_intensity = mean_intensity / max(n_avail, 1)

        if mean_intensity < background_threshold:
            # Background cluster
            cluster_scores.append(-1.0)
            continue

        # Score based on hyperintensity in FLAIR and T1ce
        flair_val = float(np.sum(flair * weights) / total_weight) if flair is not None else 0.0
        t1ce_val = float(np.sum(t1ce * weights) / total_weight) if t1ce is not None else 0.0

        # Tumour score: 0.6 FLAIR (whole tumour) + 0.4 T1ce (enhancing core)
        score = 0.6 * flair_val + 0.4 * t1ce_val
        cluster_scores.append(score)

    best_cluster = int(np.argmax(cluster_scores))
    return best_cluster


def generate_tumour_mask(
    membership: np.ndarray,
    tumour_cluster: Optional[int] = None,
    threshold: float = 0.5,
    modalities_or_features: Optional[Union[np.ndarray, Mapping[str, np.ndarray]]] = None,
) -> np.ndarray:
    """
    Convert fuzzy membership into a binary tumour mask.

    Parameters
    ----------
    membership : np.ndarray
        Fuzzy membership map of shape (X, Y, Z, C) or (C, X, Y, Z).
    tumour_cluster : int, optional
        Cluster index representing tumour. If None, determined automatically.
    threshold : float
        Membership cutoff threshold (default 0.5).
    modalities_or_features : dict or ndarray, optional
        Used to determine tumour cluster if tumour_cluster is None.

    Returns
    -------
    tumour_mask : np.ndarray
        Binary mask (uint8) with values 0 or 1.
    """
    m = np.asarray(membership, dtype=np.float32)
    if m.ndim == 4 and m.shape[0] < m.shape[-1] and m.shape[0] <= 10:
        m = np.moveaxis(m, 0, -1)

    if tumour_cluster is None:
        if modalities_or_features is None:
            # Fallback: cluster with highest max intensity or index 1
            tumour_cluster = 1 if m.shape[-1] > 1 else 0
        else:
            tumour_cluster = determine_tumour_cluster(m, modalities_or_features)

    tumour_membership = m[..., tumour_cluster]
    return (tumour_membership >= threshold).astype(np.uint8)


__all__ = [
    "EdgeSpatialFCM",
    "SPFCM",
    "edge_spatial_fuzzy_c_means",
    "sp_fuzzy_c_means",
    "determine_tumour_cluster",
    "generate_tumour_mask",
]
