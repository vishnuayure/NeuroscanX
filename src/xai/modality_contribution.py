"""Modality contribution – how much does each MRI sequence (T1, T1ce, T2, FLAIR) drive the segmentation?

Three computed quantities per modality (all from the real clustering outputs):
  effect_size     Cohen's d between tumor and non-tumor brain voxels
  distance_share  share of the squared feature-space distance between the mean tumor and the mean normal-tissue
                  vector contributed by this modality (this is exactly the FCM distance decomposition)
  ablation_drop   leave-one-out: drop in agreement (Dice) with the final mask when the modality is removed from the
                  intensity-only membership computation (fixed centres)
contribution = ½·distance_share + ½·ablation_share (shares normalised to sum to 1).
Levels are relative to the strongest modality (thresholds in XAIConfig).
"""
from __future__ import annotations

import numpy as np

from ..evaluation.metrics import compute_metrics
from ..segmentation.fcm import memberships_from_centers
from .context import XAIContext


def level_from_rel(rel: float, high: float = 0.75, moderate: float = 0.40) -> str:
    return "HIGH" if rel >= high else "MODERATE" if rel >= moderate else "LOW"


def compute_modality_contribution(ctx: XAIContext) -> dict:
    names = list(ctx.names)
    D = len(names)
    tumor, brain = ctx.tumor, ctx.brain
    normal = brain & ~tumor
    if not tumor.any() or not normal.any():
        return {n: {"contribution": 0.0, "level": "LOW"} for n in names}
    F = ctx.features
    mt = np.array([F[d][tumor].mean() for d in range(D)])
    mn = np.array([F[d][normal].mean() for d in range(D)])
    st = np.array([F[d][tumor].std() for d in range(D)])
    sn = np.array([F[d][normal].std() for d in range(D)])
    nt, nn = tumor.sum(), normal.sum()
    pooled = np.sqrt(((nt - 1) * st ** 2 + (nn - 1) * sn ** 2) / max(nt + nn - 2, 1)) + 1e-6
    delta = mt - mn
    effect = delta / pooled
    dist_share = delta ** 2 / max((delta ** 2).sum(), 1e-12)

    # leave-one-out ablation on a seeded voxel subsample
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(brain.ravel())
    if idx.size > ctx.cfg.ablation_max_voxels:
        idx = np.sort(rng.choice(idx, ctx.cfg.ablation_max_voxels, replace=False))
    X = F.reshape(D, -1)[:, idx].T
    ref = tumor.ravel()[idx]

    def seg(cols):
        U = memberships_from_centers(X[:, cols], ctx.centers[:, cols], ctx.m)
        return U[ctx.selected].sum(0) >= 0.5 if ctx.selected else np.zeros(X.shape[0], bool)

    full = seg(list(range(D)))
    dice_full = compute_metrics(full, ref)["dice"]
    drops = np.zeros(D)
    for d in range(D):
        cols = [c for c in range(D) if c != d]
        drops[d] = max(0.0, dice_full - compute_metrics(seg(cols), ref)["dice"])
    abl_share = drops / drops.sum() if drops.sum() > 1e-9 else dist_share
    contrib = 0.5 * dist_share + 0.5 * abl_share
    contrib = contrib / max(contrib.sum(), 1e-12)
    cmax = contrib.max()
    iT1, iT1c = (names.index("T1") if "T1" in names else None), (names.index("T1ce") if "T1ce" in names else None)
    z_t = (mt - ctx.feature_median) / np.maximum(ctx.feature_scale, 1e-6)
    out = {}
    for d, n in enumerate(names):
        direction = "elevated" if z_t[d] > 0.5 else "reduced" if z_t[d] < -0.5 else "similar to brain"
        out[n] = {
            "contribution": float(contrib[d]), "relative": float(contrib[d] / cmax),
            "level": level_from_rel(contrib[d] / cmax, ctx.cfg.high_level_rel, ctx.cfg.moderate_level_rel),
            "effect_size": float(effect[d]), "distance_share": float(dist_share[d]),
            "ablation_drop": float(drops[d]), "tumor_z": float(z_t[d]), "direction": direction,
            "mean_tumor": float(mt[d]), "mean_normal": float(mn[d]),
        }
    if iT1 is not None and iT1c is not None:
        out["T1ce"]["enhancement_z"] = float(z_t[iT1c] - z_t[iT1])
    out["_dice_intensity_only_vs_final"] = float(dice_full)
    return out
