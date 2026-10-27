"""Tumor-cluster identification and whole-tumor mask generation.

Cluster ids from FCM are arbitrary, so tumor clusters are *identified from multimodal evidence*:

  z_{k,m} = (centre_{k,m} − median_m) / robustSD_m           (signal relative to this brain)
  score_k = w_F·z_FLAIR + w_T2·z_T2 + w_enh·max(0, z_T1ce − z_T1) + w_T1·max(0, −z_T1)

A cluster is *eligible* when
  • FLAIR is clearly hyper-intense  (z_FLAIR ≥ min_flair_z)       – rejects CSF (T2-bright, FLAIR-dark)
  • T2 is not hypo-intense          (z_T2 ≥ min_t2_z)
  • it is not diffuse               (cluster ≤ max_cluster_fraction of the brain)
  • it is spatially concentrated    (largest connected component holds ≥ min_compactness of it)
The best eligible cluster is selected; further eligible clusters join when their score reaches
`relative_score` × the best (e.g. enhancing core next to oedema).  Every decision is recorded in
`cluster_report` so the XAI layer can explain it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import ball

from ..config import TumorConfig
from ..preprocessing.preprocess import PreprocessResult

log = logging.getLogger("neuroscanx")


@dataclass
class TumorExtraction:
    selected: List[int]
    cluster_report: List[dict]
    tumor_membership_box: np.ndarray       # Σ memberships of selected clusters
    mask_box: np.ndarray                   # bool
    warnings: List[str] = field(default_factory=list)
    postprocess: dict = field(default_factory=dict)


def _idx(names, n):
    return names.index(n) if n in names else None


def select_tumor_clusters(centers: np.ndarray, membership: np.ndarray, pre: PreprocessResult, cfg: TumorConfig):
    names = list(pre.names)
    iF, iT2, iT1c, iT1 = (_idx(names, n) for n in ("FLAIR", "T2", "T1ce", "T1"))
    z = (centers - pre.feature_median[None, :]) / np.maximum(pre.feature_scale[None, :], 1e-6)
    zg = lambda k, i: float(z[k, i]) if i is not None else 0.0
    C = centers.shape[0]
    labels = membership.argmax(0)
    brain_n = max(int(pre.mask_box.sum()), 1)
    report = []
    for k in range(C):
        zf, zt2, zt1c, zt1 = zg(k, iF), zg(k, iT2), zg(k, iT1c), zg(k, iT1)
        enh = max(0.0, zt1c - zt1) if (iT1c is not None and iT1 is not None) else 0.0
        score = cfg.w_flair * zf + cfg.w_t2 * zt2 + cfg.w_enh * enh + cfg.w_t1_hypo * max(0.0, -zt1)
        sel_vox = (labels == k) & pre.mask_box
        frac = float(sel_vox.sum()) / brain_n
        reasons = []
        if iF is not None and zf < cfg.min_flair_z:
            reasons.append(f"FLAIR z={zf:.2f} < {cfg.min_flair_z}")
        if iT2 is not None and zt2 < cfg.min_t2_z:
            reasons.append(f"T2 z={zt2:.2f} < {cfg.min_t2_z}")
        if frac > cfg.max_cluster_fraction:
            reasons.append(f"diffuse ({frac:.0%} of brain)")
        if frac == 0:
            reasons.append("empty cluster")
        compact = None
        if not reasons:
            lab, n = ndi.label(sel_vox)
            if n:
                sizes = ndi.sum(sel_vox, lab, index=np.arange(1, n + 1))
                compact = float(sizes.max() / sizes.sum())
                if compact < cfg.min_compactness:
                    reasons.append(f"spatially scattered (largest component {compact:.0%})")
        report.append(dict(cluster=k, z={nm: round(float(z[k, i]), 3) for i, nm in enumerate(names)},
                           enhancement_z=round(enh, 3), score=round(float(score), 4), volume_fraction=round(frac, 4),
                           compactness=None if compact is None else round(compact, 4), eligible=not reasons,
                           reject_reasons=reasons, selected=False))
    elig = sorted([r for r in report if r["eligible"]], key=lambda r: -r["score"])
    selected = []
    warnings = []
    if elig:
        best = elig[0]["score"]
        for r in elig:
            if r is elig[0] or r["score"] >= cfg.relative_score * best:
                r["selected"] = True
                selected.append(r["cluster"])
            else:
                r["reject_reasons"].append(f"score {r['score']:.2f} < {cfg.relative_score:.0%} of best ({best:.2f})")
    else:
        warnings.append("No cluster met the tumor-like abnormality criteria – empty segmentation returned.")
    return selected, report, warnings


def clean_mask(raw: np.ndarray, allowed: np.ndarray, cfg: TumorConfig) -> tuple[np.ndarray, dict]:
    stats = {"raw_voxels": int(raw.sum())}
    mask = raw.copy()

    def drop_small(m):
        lab, n = ndi.label(m)
        if n == 0:
            return m, 0
        sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
        keep = sizes >= max(cfg.min_component_voxels, cfg.min_component_fraction * sizes.max())
        out = np.isin(lab, 1 + np.where(keep)[0])
        return out, int(n - keep.sum())

    mask, removed = drop_small(mask)
    stats["components_removed"] = removed
    if cfg.closing_radius > 0 and mask.any():
        mask = ndi.binary_closing(np.pad(mask, cfg.closing_radius + 1), structure=ball(cfg.closing_radius))
        mask = mask[(slice(cfg.closing_radius + 1, -cfg.closing_radius - 1),) * 3] & allowed
    before = int(mask.sum())
    if cfg.fill_holes and mask.any():
        mask = ndi.binary_fill_holes(mask) & allowed
    stats["hole_voxels_filled"] = int(mask.sum()) - before
    mask, removed2 = drop_small(mask)
    stats["components_removed"] += removed2
    stats["final_voxels"] = int(mask.sum())
    return mask, stats


def extract_whole_tumor(centers: np.ndarray, membership: np.ndarray, pre: PreprocessResult,
                        cfg: TumorConfig) -> TumorExtraction:
    selected, report, warnings = select_tumor_clusters(centers, membership, pre, cfg)

    def build(sel):
        t = membership[sel].sum(0).astype(np.float32) if sel else np.zeros(pre.mask_box.shape, np.float32)
        t[~pre.mask_box] = 0
        raw = (t >= cfg.membership_threshold) & pre.mask_box
        m, st = clean_mask(raw, pre.mask_box, cfg)
        return t, m, st

    t, mask, stats = build(selected)
    # whole tumor = union of its compartments: clusters that are enclosed by the tumor (necrotic / enhancing core,
    # which differ from oedema in signal but are not abnormal in the same way) are added, with the reason recorded
    if selected and mask.any():
        shell = ndi.binary_dilation(ndi.binary_fill_holes(mask), iterations=1)
        labels = membership.argmax(0)
        added = []
        for r in report:
            k = r["cluster"]
            if r["selected"]:
                continue
            vox = (labels == k) & pre.mask_box
            n = int(vox.sum())
            if n < cfg.min_enclosed_voxels:
                continue
            frac = float((vox & shell).sum()) / n
            r["enclosed_fraction"] = round(frac, 4)
            if frac >= cfg.enclosed_fraction:
                r["selected"] = True
                r["reject_reasons"].append(f"re-included: {frac:.0%} of its voxels are enclosed by the tumor")
                selected.append(k); added.append(k)
        if added:
            t, mask, stats = build(selected)
            stats["enclosed_clusters_added"] = added
    frac = mask.sum() / max(pre.mask_box.sum(), 1)
    if frac > 0.5:
        warnings.append(f"Segmented region covers {frac:.0%} of the brain – unusually large, review inputs.")
    log.info("Tumor clusters %s → %d voxels (%.2f%% of brain)", selected, int(mask.sum()), 100 * frac)
    return TumorExtraction(selected, report, t, mask, warnings, stats)
