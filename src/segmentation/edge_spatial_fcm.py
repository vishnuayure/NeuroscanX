"""Edge-Aware Spatially Penalised FCM (SP-FCM) – the main NeuroScanX algorithm.

Same penalised objective as Spatial FCM, but the neighbour weight w_ij is *modulated by evidence of a
boundary between voxels i and j*:

    w_ij = exp(−λ · e_ij)  ·  exp(−ρ · ||x_i − x_j||² / (2σ²))

  e_ij = ½ (E_i + E_j)    edge strength in [0,1] from the multimodal gradient map (preprocessing/edges.py)
  λ    = edge_penalty     how strongly boundaries block neighbourhood influence
  ρ    = range_weight     weight of the intensity-similarity (bilateral/range) factor, 0 disables it
  σ²   = range_scale × mean squared neighbour difference of the case (data driven, no magic constant)

Result: in homogeneous regions w≈1 and the penalty regularises noise exactly like Spatial FCM;
across strong boundaries w→0, so memberships on either side are *not* forced to agree and the
tumor margin is not smoothed away.
"""
from __future__ import annotations

import numpy as np

from .fcm import FCMResult
from .spatial_fcm import NeighbourPairs, neighbour_offsets, pair_slices, spatial_penalized_fcm


def edge_aware_weights(features: np.ndarray, mask: np.ndarray, edges: np.ndarray, connectivity: int = 6,
                       edge_penalty: float = 4.0, range_weight: float = 1.0, range_scale: float = 3.0) -> NeighbourPairs:
    offs = neighbour_offsets(connectivity)
    sigma2 = None
    pairs = []
    for off in offs:
        d, s = pair_slices(off, mask.shape)
        valid = mask[d] & mask[s]
        e_ij = 0.5 * (edges[d] + edges[s])
        w = np.exp(-edge_penalty * e_ij).astype(np.float32)
        if range_weight > 0:
            d2 = np.zeros(valid.shape, np.float32)
            for c in range(features.shape[0]):
                diff = features[c][d] - features[c][s]
                d2 += diff * diff
            if sigma2 is None:   # estimated once from the first (axis-aligned) offset
                sigma2 = max(range_scale * float(d2[valid].mean()) if valid.any() else 1.0, 1e-6)
            w *= np.exp(-range_weight * d2 / (2.0 * sigma2)).astype(np.float32)
        pairs.append((d, s, (w * valid).astype(np.float32)))
    return NeighbourPairs(pairs, 2 * len(offs))


def edge_spatial_fcm(features, mask, edges, centers0, cfg, ops=None, U0=None, max_iter=None) -> FCMResult:
    nb = edge_aware_weights(features, mask, edges, cfg.connectivity, cfg.edge_penalty, cfg.range_weight, cfg.range_scale)
    return spatial_penalized_fcm(features, mask, centers0, nb, cfg.m, cfg.spatial_weight, cfg.tol,
                                 max_iter or cfg.max_iter, cfg.relaxation, cfg.eps, ops, U0, "edge_spatial_fcm")
