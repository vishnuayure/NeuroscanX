"""Method dispatcher: runs FCM / Spatial FCM / Edge-Aware SP-FCM on a preprocessed case."""
from __future__ import annotations

import logging

import numpy as np

from ..config import FCMConfig, METHODS
from ..preprocessing.preprocess import PreprocessResult
from ..utils.backend import NumpyOps, get_ops
from .edge_spatial_fcm import edge_aware_weights
from .fcm import FCMResult, fcm_volume, fuzzy_cmeans
from .spatial_fcm import coarsen, spatial_penalized_fcm, uniform_weights, upsample_membership

log = logging.getLogger("neuroscanx")


def _run(method: str, pre: PreprocessResult, cfg: FCMConfig, ops) -> FCMResult:
    feats, mask = pre.features, pre.mask_box
    if method == "fcm":
        return fcm_volume(feats, mask, cfg, ops)

    # deterministic warm start: global FCM fit on a seeded voxel subsample
    X = feats[:, mask].T
    warm = fuzzy_cmeans(X, cfg.n_clusters, cfg.m, cfg.tol, min(cfg.max_iter, 60), cfg.seed, cfg.eps,
                        max_fit_samples=cfg.init_samples, return_membership=False, ops=ops)

    def build_nb(f, mk, e):
        if method == "spatial_fcm":
            return uniform_weights(mk, cfg.connectivity)
        return edge_aware_weights(f, mk, e, cfg.connectivity, cfg.edge_penalty, cfg.range_weight, cfg.range_scale)

    s = cfg.downsample
    if s > 1:
        log.info("Coarse-to-fine: clustering on 1/%d grid, then %d full-resolution refinement iterations", s, cfg.refine_iterations)
        fc, mc, ec = coarsen(feats, s), coarsen(mask, s, True), coarsen(pre.edges_box, s)
        coarse = spatial_penalized_fcm(fc, mc, warm.centers, build_nb(fc, mc, ec), cfg.m, cfg.spatial_weight, cfg.tol,
                                       cfg.max_iter, cfg.relaxation, cfg.eps, ops, None, method)
        U0 = upsample_membership(coarse.membership, mask.shape, mask)
        res = spatial_penalized_fcm(feats, mask, coarse.centers, build_nb(feats, mask, pre.edges_box), cfg.m,
                                    cfg.spatial_weight, cfg.tol, cfg.refine_iterations, cfg.relaxation, cfg.eps, ops, U0, method)
        res.n_iter += coarse.n_iter
        res.objective_history = coarse.objective_history + res.objective_history
        res.elapsed += coarse.elapsed + warm.elapsed
        return res
    res = spatial_penalized_fcm(feats, mask, warm.centers, build_nb(feats, mask, pre.edges_box), cfg.m,
                                cfg.spatial_weight, cfg.tol, cfg.max_iter, cfg.relaxation, cfg.eps, ops, None, method)
    res.elapsed += warm.elapsed
    return res


def segment(method: str, pre: PreprocessResult, cfg: FCMConfig, device: str = "cpu") -> FCMResult:
    """Run one clustering method; transparently falls back to CPU if the GPU path fails."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose one of {METHODS}")
    ops = get_ops(device)
    try:
        return _run(method, pre, cfg, ops)
    except Exception as e:
        if ops.name == "cuda":
            log.warning("GPU run failed (%s) → falling back to CPU", e)
            return _run(method, pre, cfg, NumpyOps())
        raise
