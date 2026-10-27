"""Genuine Fuzzy C-Means (Bezdek) – the baseline.

Minimises   J(U, V) = Σ_i Σ_k  u_ki^m · ||x_i − v_k||²     subject to  Σ_k u_ki = 1

Alternating updates:
    v_k  = Σ_i u_ki^m x_i / Σ_i u_ki^m
    u_ki = 1 / Σ_j ( d_ki / d_ji )^{2/(m−1)}   ≡   d_ki^{−2/(m−1)} / Σ_j d_ji^{−2/(m−1)}

Initialisation is deterministic (seeded k-means++ on a voxel subsample).  For very large volumes the
centres can be fitted on a seeded random subsample of brain voxels (`max_fit_samples`); the final
memberships are then always evaluated for **every** voxel.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..utils.backend import NumpyOps

log = logging.getLogger("neuroscanx")


@dataclass
class FCMResult:
    centers: np.ndarray                 # (C, D)
    membership: Optional[np.ndarray]    # (C, N) or (C, *box); None if not requested
    objective_history: List[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False
    final_delta: float = float("nan")
    method: str = "fcm"
    elapsed: float = 0.0
    params: dict = field(default_factory=dict)
    device: str = "cpu"

    @property
    def objective(self) -> float:
        return self.objective_history[-1] if self.objective_history else float("nan")


def kmeanspp_init(X: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(1, n_clusters):
        p = d2 / max(d2.sum(), 1e-12)
        centers.append(X[rng.choice(n, p=p)])
        d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(1))
    # order by mean intensity so cluster ids are reproducible (they remain *arbitrary* semantically)
    c = np.array(centers, dtype=np.float32)
    return c[np.argsort(c.mean(1))]


def _dist2_flat(ops, Xd, V):
    C = V.shape[0]
    D2 = ops.zeros((C, Xd.shape[0]))
    for k in range(C):
        D2[k] = ((Xd - ops.array(V[k])) ** 2).sum(1)
    return D2


def _membership_from_d2(ops, D2, m, eps):
    inv = ops.pow(ops.clip_min(D2, eps), -1.0 / (m - 1.0))
    return inv / inv.sum(0)


def memberships_from_centers(X: np.ndarray, V: np.ndarray, m: float = 2.0, eps: float = 1e-8, ops=None,
                             chunk: int = 1_000_000) -> np.ndarray:
    """Intensity-only fuzzy memberships for fixed centres (used for final FCM output and XAI)."""
    ops = ops or NumpyOps()
    out = np.empty((V.shape[0], X.shape[0]), dtype=np.float32)
    for s in range(0, X.shape[0], chunk):
        Xd = ops.array(X[s:s + chunk])
        out[:, s:s + chunk] = ops.host(_membership_from_d2(ops, _dist2_flat(ops, Xd, V), m, eps))
    return out


def fuzzy_cmeans(X: np.ndarray, n_clusters: int = 5, m: float = 2.0, tol: float = 1e-3, max_iter: int = 100,
                 seed: int = 42, eps: float = 1e-8, init_centers: Optional[np.ndarray] = None,
                 max_fit_samples: Optional[int] = None, return_membership: bool = True, ops=None) -> FCMResult:
    """Fuzzy C-Means on a (N, D) feature matrix.  Returns centres, memberships, objective history."""
    if m <= 1.0:
        raise ValueError("fuzziness m must be > 1")
    ops = ops or NumpyOps()
    t0 = time.time()
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0]
    if n < n_clusters:
        raise ValueError("fewer samples than clusters")
    rng = np.random.default_rng(seed)
    Xfit = X
    if max_fit_samples and n > max_fit_samples:
        Xfit = X[np.sort(rng.choice(n, max_fit_samples, replace=False))]
    V = (np.array(init_centers, dtype=np.float32) if init_centers is not None
         else kmeanspp_init(Xfit[rng.choice(Xfit.shape[0], min(20000, Xfit.shape[0]), replace=False)], n_clusters, rng))
    Xd = ops.array(Xfit)
    hist: List[float] = []
    delta, converged, U = float("inf"), False, None
    it = 0
    for it in range(1, max_iter + 1):
        D2 = _dist2_flat(ops, Xd, V)
        Unew = _membership_from_d2(ops, D2, m, eps)
        delta = float("inf") if U is None else ops.max_abs_diff(Unew, U)
        U = Unew
        Um = ops.pow(U, m)
        den = Um.sum(1)
        num = Um @ Xd
        V = (ops.host(num) / np.maximum(ops.host(den)[:, None], eps)).astype(np.float32)
        hist.append(ops.to_float((Um * D2).sum()))
        if delta < tol:
            converged = True
            break
    mem = None
    if return_membership:
        if Xfit is X and U is not None:
            mem = ops.host(_membership_from_d2(ops, _dist2_flat(ops, Xd, V), m, eps)).astype(np.float32)
        else:
            mem = memberships_from_centers(X, V, m, eps, ops)
        Um = mem.astype(np.float64) ** m
        d2 = np.zeros_like(Um)
        for k in range(V.shape[0]):
            d2[k] = ((X - V[k]) ** 2).sum(1)
        hist.append(float((Um * d2).sum()))   # objective on *all* samples
    log.info("FCM: %d iterations, converged=%s, J=%.4g", it, converged, hist[-1])
    return FCMResult(centers=V, membership=mem, objective_history=hist, n_iter=it, converged=converged,
                     final_delta=float(delta), method="fcm", elapsed=time.time() - t0,
                     params=dict(n_clusters=n_clusters, m=m, tol=tol, max_iter=max_iter, seed=seed), device=ops.name)


def fcm_volume(features: np.ndarray, mask_box: np.ndarray, cfg, ops=None, return_membership=True) -> FCMResult:
    """Run FCM on the brain voxels of a (D, *box) feature array; membership → (C, *box)."""
    D = features.shape[0]
    X = features[:, mask_box].T                      # (N, D)
    res = fuzzy_cmeans(X, cfg.n_clusters, cfg.m, cfg.tol, cfg.max_iter, cfg.seed, cfg.eps,
                       max_fit_samples=cfg.init_samples, return_membership=return_membership, ops=ops)
    if res.membership is not None:
        full = np.zeros((cfg.n_clusters,) + mask_box.shape, dtype=np.float32)
        full[:, mask_box] = res.membership
        res.membership = full
    return res
