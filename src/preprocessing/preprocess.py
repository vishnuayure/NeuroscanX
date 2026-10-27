"""Preprocessing pipeline: MRI → brain ROI, normalised features, edge map.

All arrays returned for clustering live in the *brain bounding box* ("box") to keep memory low;
`bbox` maps them back to the full volume.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from ..config import PreprocessConfig
from ..io.nifti_loader import MultimodalMRI
from .clahe import clahe_volume
from .denoising import denoise
from .edges import compute_edges
from .normalization import compute_brain_mask, robust_normalize, robust_stats

log = logging.getLogger("neuroscanx")


@dataclass
class PreprocessResult:
    names: Tuple[str, ...]
    bbox: Tuple[slice, slice, slice]
    full_shape: Tuple[int, int, int]
    brain_mask: np.ndarray                 # full-size bool
    mask_box: np.ndarray                   # box bool
    features: np.ndarray                   # (D, *box) float32, normalised [0,1]
    edges_box: np.ndarray                  # box float32 [0,1]
    enhanced_box: Dict[str, np.ndarray]    # CLAHE-enhanced (display / edge input) – may be empty
    feature_median: np.ndarray             # (D,) robust brain median of the features
    feature_scale: np.ndarray              # (D,) robust brain SD of the features
    spacing: Tuple[float, float, float]
    config: PreprocessConfig = field(default_factory=PreprocessConfig)

    @property
    def box_shape(self):
        return tuple(self.mask_box.shape)

    def expand(self, arr_box: np.ndarray, fill=0) -> np.ndarray:
        """Place a box array (…, *box) back into the full volume."""
        lead = arr_box.shape[:-3]
        out = np.full(lead + tuple(self.full_shape), fill, dtype=arr_box.dtype)
        out[(Ellipsis,) + self.bbox] = arr_box
        return out

    @property
    def edges(self) -> np.ndarray:
        return self.expand(self.edges_box)


def _bbox(mask: np.ndarray, margin: int):
    sl = []
    for ax in range(3):
        other = tuple(a for a in range(3) if a != ax)
        idx = np.where(mask.any(axis=other))[0]
        lo = max(int(idx[0]) - margin, 0)
        hi = min(int(idx[-1]) + 1 + margin, mask.shape[ax])
        sl.append(slice(lo, hi))
    return tuple(sl)


def preprocess(mri: MultimodalMRI, cfg: PreprocessConfig | None = None) -> PreprocessResult:
    cfg = cfg or PreprocessConfig()
    names = tuple(n for n in cfg.feature_modalities if n in mri.modalities)
    if len(names) < 2:
        raise ValueError("At least two modalities are required")
    log.info("Preprocessing %s", ", ".join(names))

    # 1-3. float conversion + invalid values → 0 (kept out of the brain mask via nonzero logic)
    clean = {n: np.nan_to_num(mri.modalities[n].data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
             for n in names}
    # 4. brain / background handling
    mask = compute_brain_mask(clean, cfg.brain_mask_method, cfg.brain_mask_closing)
    if mask.sum() < 1000:
        raise ValueError("Brain mask is (almost) empty – check that the inputs are valid MRI volumes")
    bbox = _bbox(mask, cfg.crop_margin)
    mask_box = mask[bbox]
    spacing = tuple(float(s) for s in mri.voxel_spacing)
    log.info("Brain mask: %d voxels (%.1f%% of volume), ROI box %s", int(mask.sum()), 100 * mask.mean(),
             tuple(s.stop - s.start for s in bbox))

    feats, norm_box, enhanced = [], {}, {}
    for n in names:
        v = clean[n][bbox]
        # 6. denoise → 5. normalise (denoise first so percentile window is not driven by noise spikes)
        v = denoise(v, cfg.denoise_method, cfg.denoise_sigma, cfg.denoise_median_size)
        v = robust_normalize(v, mask_box, cfg.norm_lower_percentile, cfg.norm_upper_percentile)
        norm_box[n] = v
        f = v
        if cfg.clahe_enabled and cfg.clahe_apply_to != "none":
            enh = clahe_volume(v, mask_box, cfg.clahe_clip_limit, cfg.clahe_kernel_fraction)
            enhanced[n] = enh
            if cfg.clahe_apply_to == "features":
                f = enh
        feats.append(f)
    features = np.stack(feats).astype(np.float32, copy=False)

    # 8. edges (from CLAHE-enhanced images when requested, otherwise normalised images)
    edge_src = enhanced if (cfg.clahe_enabled and cfg.clahe_apply_to == "edges") else norm_box
    edges_box = compute_edges(edge_src, mask_box, spacing, cfg.edge_method, cfg.edge_sigma_mm, cfg.edge_combine,
                              cfg.edge_modality_weights, cfg.edge_normalise_percentile)

    med, sc = [], []
    for i in range(len(names)):
        m_, s_ = robust_stats(features[i], mask_box)
        med.append(m_)
        sc.append(s_)
    return PreprocessResult(
        names=names, bbox=bbox, full_shape=tuple(mri.shape), brain_mask=mask, mask_box=mask_box, features=features,
        edges_box=edges_box, enhanced_box=enhanced, feature_median=np.array(med, np.float32),
        feature_scale=np.array(sc, np.float32), spacing=spacing, config=cfg,
    )
