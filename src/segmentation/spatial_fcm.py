"""Spatially penalised Fuzzy C-Means (shared solver) + the plain Spatial-FCM baseline.

Objective (Pham-style spatial penalty, extended with neighbour weights w_ij):

  J = Σ_i Σ_k u_ki^m d²_ki  +  (β / 2·n_nb) Σ_i Σ_k u_ki^m  Σ_{j∈N_i} w_ij  Σ_{l≠k} u_lj^m

The penalty term is large when voxel i is pulled towards cluster k while its neighbours
belong to *other* clusters, so neighbouring voxels are encouraged to have compatible
memberships.  Setting the Lagrangian derivative to zero gives the closed-form update

  P_ki = Σ_{j∈N_i} w_ij ( S_j − u_kj^m ),     S_j = Σ_l u_lj^m
  u_ki = A_ki^{−1/(m−1)} / Σ_l A_li^{−1/(m−1)},   A_ki = d²_ki + (β / n_nb) · P_ki

Cluster centres keep the standard FCM update (the penalty does not depend on V).

  * Spatial FCM        : w_ij = 1 for all in-brain neighbour pairs (uniform smoothing).
  * Edge-Aware SP-FCM  : w_ij derived from edge strength and intensity similarity
                         (see edge_spatial_fcm.py) – smoothing is suppressed across boundaries.

Pairs are stored once per undirected neighbour offset; the same weight array serves both
directions, which halves memory.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from ..utils.backend import NumpyOps
from .fcm import FCMResult

log = logging.getLogger("neuroscanx")

_OFFSETS_6 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
_OFFSETS_18 = _OFFSETS_6 + [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]
_OFFSETS_26 = _OFFSETS_18 + [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)]


def neighbour_offsets(connectivity: int):
    try:
        return {6: _OFFSETS_6, 18: _OFFSETS_18, 26: _OFFSETS_26}[connectivity]
    except KeyError:
        raise ValueError("connectivity must be 6, 18 or 26") from None


@dataclass
class NeighbourPairs:
    pairs: List[Tuple[tuple, tuple, np.ndarray]]   # (dst slices, src slices, weight array on dst region)
    n_nb: int                                      # neighbours per interior voxel (both directions)


def pair_slices(off, shape):
    dst, src = [], []
    for o, n in zip(off, shape):
        if o > 0:
            dst.append(slice(0, n - o)); src.append(slice(o, n))
        elif o < 0:
            dst.append(slice(-o, n)); src.append(slice(0, n + o))
        else:
            dst.append(slice(None)); src.append(slice(None))
    return tuple(dst), tuple(src)


def uniform_weights(mask: np.ndarray, connectivity: int = 6) -> NeighbourPairs:
    offs = neighbour_offsets(connectivity)
    pairs = []
    for off in offs:
        d, s = pair_slices(off, mask.shape)
        pairs.append((d, s, (mask[d] & mask[s]).astype(np.float32)))
    return NeighbourPairs(pairs, 2 * len(offs))


def _dist2(ops, Xd, V):
    C = V.shape[0]
    D = Xd.shape[0]
    D2 = ops.zeros((C,) + tuple(Xd.shape[1:]))
    for k in range(C):
        acc = (Xd[0] - float(V[k, 0])) ** 2
        for d in range(1, D):
            acc = acc + (Xd[d] - float(V[k, d])) ** 2
        D2[k] = acc
    return D2


def _update_u(ops, A, m, eps, M):
    inv = ops.pow(ops.clip_min(A, eps), -1.0 / (m - 1.0))
    return (inv / inv.sum(0)) * M


def spatial_penalized_fcm(features: np.ndarray, mask: np.ndarray, centers0: np.ndarray, nb: NeighbourPairs,
                          m: float = 2.0, beta: float = 0.6, tol: float = 1e-3, max_iter: int = 100,
                          relaxation: float = 1.0, eps: float = 1e-8, ops=None, U0: Optional[np.ndarray] = None,
                          method: str = "spatial_fcm") -> FCMResult:
    """Alternating optimisation of the spatially penalised objective on a (D, *box) feature volume."""
    ops = ops or NumpyOps()
    t0 = time.time()
    Xd = ops.array(features)
    M = ops.array(mask.astype(np.float32))
    V = np.array(centers0, dtype=np.float32)
    pairs = [(d, s, ops.array(w)) for d, s, w in nb.pairs]
    all_ = (slice(None),)
    bn = beta / nb.n_nb

    if U0 is None:
        U = _update_u(ops, _dist2(ops, Xd, V), m, eps, M)
    else:
        U = ops.array(U0)
    hist: List[float] = []
    delta, converged, it = float("inf"), False, 0
    for it in range(1, max_iter + 1):
        Um = ops.pow(U, m)
        S = Um.sum(0)
        Pen = ops.zeros_like(Um)
        for d, s, W in pairs:
            Pen[all_ + d] += W * (S[s] - Um[all_ + s])
            Pen[all_ + s] += W * (S[d] - Um[all_ + d])
        # centre update (standard FCM)
        C, D = V.shape
        for k in range(C):
            den = max(ops.to_float(Um[k].sum()), eps)
            for dd in range(D):
                V[k, dd] = ops.to_float((Um[k] * Xd[dd]).sum()) / den
        D2 = _dist2(ops, Xd, V)
        J = ops.to_float((Um * D2).sum()) + 0.5 * bn * ops.to_float((Um * Pen).sum())
        hist.append(J)
        Unew = _update_u(ops, D2 + bn * Pen, m, eps, M)
        if relaxation < 1.0:
            Unew = U + relaxation * (Unew - U)
        delta = ops.max_abs_diff(Unew, U)
        U = Unew
        if it % 10 == 0:
            log.info("  %s iter %d  J=%.5g  max|ΔU|=%.2e", method, it, J, delta)
        # converged when memberships stop moving, or the objective has plateaued (isolated voxels may keep
        # flip-flopping under synchronous updates without changing J)
        if delta < tol or (it >= 5 and abs(hist[-2] - hist[-1]) <= 0.1 * tol * abs(hist[-1])):
            converged = True
            break
    mem = ops.host(U).astype(np.float32)
    ops.free()
    log.info("%s: %d iterations, converged=%s, J=%.4g", method, it, converged, hist[-1] if hist else float("nan"))
    return FCMResult(centers=V, membership=mem, objective_history=hist, n_iter=it, converged=converged,
                     final_delta=float(delta), method=method, elapsed=time.time() - t0,
                     params=dict(m=m, beta=beta, tol=tol, max_iter=max_iter, n_nb=nb.n_nb), device=ops.name)


def spatial_fcm(features, mask, centers0, cfg, ops=None, U0=None, max_iter=None) -> FCMResult:
    """Plain Spatial FCM baseline: uniform neighbour weights."""
    return spatial_penalized_fcm(features, mask, centers0, uniform_weights(mask, cfg.connectivity), cfg.m,
                                 cfg.spatial_weight, cfg.tol, max_iter or cfg.max_iter, cfg.relaxation, cfg.eps,
                                 ops, U0, "spatial_fcm")


# ---- helpers for the optional coarse-to-fine mode ------------------------------------------
def coarsen(arr: np.ndarray, s: int, is_mask: bool = False):
    lead = arr.ndim - 3
    def one(a):
        f = ndi.uniform_filter(a.astype(np.float32), size=s, mode="nearest")[::s, ::s, ::s]
        return f > 0.5 if is_mask else f
    return one(arr) if lead == 0 else np.stack([one(a) for a in arr])


def upsample_membership(Uc: np.ndarray, shape, mask: np.ndarray) -> np.ndarray:
    out = np.empty((Uc.shape[0],) + tuple(shape), np.float32)
    for k in range(Uc.shape[0]):
        z = [t / c for t, c in zip(shape, Uc.shape[1:])]
        out[k] = ndi.zoom(Uc[k], z, order=1, mode="nearest")[:shape[0], :shape[1], :shape[2]]
    out = np.maximum(out, 1e-6)
    out /= out.sum(0, keepdims=True)
    return out * mask.astype(np.float32)
