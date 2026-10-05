"""
Multimodal feature representation for BraTS MRI volumes.
Constructs a feature representation combining T1, T1ce, T2, and FLAIR.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np


def build_multimodal_features(
    volumes: Mapping[str, np.ndarray],
    modalities: Sequence[str] = ("t1", "t1ce", "t2", "flair"),
    weights: Optional[Mapping[str, float]] = None,
    include_enhancing_contrast: bool = True,
    include_edges: bool = False,
    edges: Optional[Mapping[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Construct a multimodal voxel feature representation (X, Y, Z, D).

    Parameters
    ----------
    volumes : dict
        Mapping of modality name -> 3D volume array.
    modalities : sequence
        Modality names to include. Default: ('t1', 't1ce', 't2', 'flair').
    weights : dict, optional
        Scaling weights for each modality channel.
    include_enhancing_contrast : bool
        If True and both T1 and T1ce are present, includes (T1ce - T1) as a
        contrast enhancement feature.
    include_edges : bool
        If True and edge maps are provided, includes edge magnitude features.
    edges : dict, optional
        Mapping of modality name -> 3D edge array.

    Returns
    -------
    features : np.ndarray
        Array of shape (X, Y, Z, D) with float32 values.
    feature_names : list of str
        Names corresponding to each feature dimension.
    """
    norm_vols = {k.lower().strip(): np.asarray(v, dtype=np.float32) for k, v in volumes.items()}

    ref_key = next(iter(norm_vols))
    shape = norm_vols[ref_key].shape
    for k, v in norm_vols.items():
        if v.shape != shape:
            raise ValueError(f"Volume shape mismatch: {k} has {v.shape}, expected {shape}")

    feature_channels: List[np.ndarray] = []
    feature_names: List[str] = []

    for mod in modalities:
        m = mod.lower().strip()
        if m in norm_vols:
            w = 1.0 if weights is None else float(weights.get(mod, weights.get(m, 1.0)))
            channel = norm_vols[m] * w
            feature_channels.append(channel)
            feature_names.append(m.upper())

    # Contrast enhancement feature: T1ce - T1
    if include_enhancing_contrast and "t1ce" in norm_vols and "t1" in norm_vols:
        enhancement = np.maximum(0.0, norm_vols["t1ce"] - norm_vols["t1"])
        feature_channels.append(enhancement)
        feature_names.append("ENHANCEMENT(T1ce-T1)")

    # Edge features
    if include_edges and edges is not None:
        for mod, edge_map in edges.items():
            m = mod.lower().strip()
            arr = np.asarray(edge_map, dtype=np.float32)
            if arr.shape == shape:
                feature_channels.append(arr)
                feature_names.append(f"EDGE_{m.upper()}")

    if not feature_channels:
        raise ValueError("No feature channels could be constructed from provided inputs.")

    features = np.stack(feature_channels, axis=-1).astype(np.float32)
    return features, feature_names


__all__ = [
    "build_multimodal_features",
]
