"""NeuroScanX pipeline – one clean interface between the UI and the algorithms.

    result = run(T1=..., T1ce=..., T2=..., FLAIR=..., method="edge_spatial_fcm")

MRI → validation → preprocessing → edge extraction → (Edge-Aware SP-)FCM → tumor-cluster identification →
whole-tumor mask → uncertainty → XAI → explanation (+ metrics if a ground truth is supplied).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from ..config import DISCLAIMER, METHODS, METHOD_LABELS, MODALITIES, PipelineConfig
from ..evaluation.metrics import compute_metrics, whole_tumor_from_brats
from ..io.nifti_loader import (MultimodalMRI, NiftiVolume, ValidationReport, load_case, load_ground_truth)
from ..preprocessing.preprocess import PreprocessResult, preprocess
from ..segmentation import extract_whole_tumor, segment
from ..uncertainty.fuzzy_uncertainty import tumor_uncertainty
from ..utils.backend import DeviceInfo, detect_device
from ..xai.context import XAIContext

log = logging.getLogger("neuroscanx")
Progress = Callable[[str, float, str], None]


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines: List[str] = []
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S"))

    def emit(self, record):
        self.lines.append(self.format(record))


@dataclass
class FuzzyMembership:
    cluster: np.ndarray            # (C, *box) float16 – fuzzy cluster memberships in the brain ROI
    bbox: tuple
    centers: np.ndarray            # (C, D)
    selected: List[int]            # tumor clusters
    tumor: np.ndarray              # full-size float32 – tumor membership Σ_{k∈selected} u_k
    feature_names: tuple = ()


@dataclass
class PreparedCase:
    mri: MultimodalMRI
    pre: PreprocessResult
    validation: Optional[ValidationReport]
    timings: Dict[str, float]
    ground_truth: Optional[np.ndarray] = None   # full-size bool whole-tumor GT


@dataclass
class PipelineResult:
    method: str
    mri: MultimodalMRI
    segmentation: np.ndarray                    # uint8 full volume, original (canonical) dimensions
    membership: FuzzyMembership
    uncertainty: np.ndarray                     # float32 full volume [0,1]
    edges: np.ndarray                           # float32 full volume [0,1]
    preprocessing: PreprocessResult
    modality_contribution: dict
    boundary_analysis: dict
    explanation: str
    explanation_source: str
    xai_summary: dict
    metrics: Optional[dict]
    timings: Dict[str, float]
    config: PipelineConfig
    device: DeviceInfo
    log: List[str]
    clustering: dict
    evidence: dict
    uncertainty_summary: dict
    validation: Optional[ValidationReport] = None
    boundary_band: Optional[np.ndarray] = None          # uint8 full volume
    uncertain_boundary: Optional[np.ndarray] = None     # uint8 full volume
    ground_truth: Optional[np.ndarray] = None
    xai_context: Optional[XAIContext] = None
    orchestrator: str = "sequential"
    meshes: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # convenience --------------------------------------------------------------------------
    @property
    def volume_mm3(self) -> float:
        return float(self.segmentation.sum()) * float(np.prod(self.mri.voxel_spacing))

    def report(self) -> dict:
        s = self.xai_summary
        return {
            "disclaimer": DISCLAIMER, "method": self.method, "method_label": METHOD_LABELS.get(self.method, self.method),
            "device": {"active": self.device.active, "name": self.device.name, "message": self.device.message},
            "volume_mm3": self.volume_mm3, "volume_ml": self.volume_mm3 / 1000.0,
            "confidence": s["segmentation"]["confidence"],
            "boundary_uncertainty": self.boundary_analysis.get("boundary_uncertainty"),
            "timings_sec": self.timings, "metrics": self.metrics, "clustering": self.clustering,
            "validation": self.validation.summary() if self.validation else None,
            "warnings": self.warnings, "explanation_source": self.explanation_source,
            "orchestrator": self.orchestrator, "config": self.config.to_dict(),
            "inputs": {k: v.path for k, v in self.mri.modalities.items()},
        }


class NeuroScanX:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------ stage 1
    def load(self, T1: str, T1ce: str, T2: str, FLAIR: str):
        return load_case({"T1": T1, "T1ce": T1ce, "T2": T2, "FLAIR": FLAIR}, self.config.reorient_to_ras)

    def prepare(self, mri: MultimodalMRI, validation: Optional[ValidationReport] = None,
                ground_truth: Union[None, str, NiftiVolume, np.ndarray] = None,
                progress: Optional[Progress] = None) -> PreparedCase:
        progress = progress or (lambda *a: None)
        timings: Dict[str, float] = {}
        gt_arr = None
        if ground_truth is not None:
            if isinstance(ground_truth, str):
                vol, info = load_ground_truth(ground_truth, mri, self.config.reorient_to_ras)
                if vol is None:
                    raise ValueError("Ground truth rejected: " + "; ".join(i.message for i in info.issues))
                gt_arr = whole_tumor_from_brats(vol.data)
            elif isinstance(ground_truth, NiftiVolume):
                gt_arr = whole_tumor_from_brats(ground_truth.data)
            else:
                gt_arr = whole_tumor_from_brats(np.asarray(ground_truth))
            if gt_arr.shape != tuple(mri.shape):
                raise ValueError(f"Ground-truth shape {gt_arr.shape} does not match MRI {tuple(mri.shape)}")
        progress("preprocessing", 0.05, "Preprocessing: brain mask, normalisation, denoising, edges")
        t = time.time()
        pre = preprocess(mri, self.config.preprocess)
        timings["preprocessing"] = time.time() - t
        return PreparedCase(mri, pre, validation, timings, gt_arr)

    # ------------------------------------------------------------------ stage 2
    def analyze(self, prep: PreparedCase, method: str = "edge_spatial_fcm", with_xai: bool = True,
                progress: Optional[Progress] = None) -> PipelineResult:
        from agents.graph import run_xai_graph   # local import keeps src/ importable without agents/

        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
        progress = progress or (lambda *a: None)
        cfg = self.config
        cap = _LogCapture()
        log.addHandler(cap)
        log.setLevel(logging.INFO)
        t_all = time.time()
        try:
            dev = detect_device(cfg.device)
            log.info(dev.message)
            log.info("Method: %s", METHOD_LABELS[method])
            pre, mri = prep.pre, prep.mri
            timings = dict(prep.timings)

            progress("segmentation", 0.25, f"Fuzzy clustering ({METHOD_LABELS[method]})")
            t = time.time()
            res = segment(method, pre, cfg.fcm, dev.active)
            progress("segmentation", 0.65, "Identifying tumor clusters and building the whole-tumor mask")
            ext = extract_whole_tumor(res.centers, res.membership, pre, cfg.tumor)
            t_box = ext.tumor_membership_box
            unc_box = tumor_uncertainty(t_box, pre.mask_box, cfg.xai.uncertainty_measure)
            timings["segmentation"] = time.time() - t

            seg_full = pre.expand(ext.mask_box.astype(np.uint8))
            fcm_info = {"n_clusters": int(res.centers.shape[0]), "iterations": res.n_iter, "converged": res.converged,
                        "final_delta": res.final_delta, "objective": res.objective, "objective_history": res.objective_history,
                        "device": res.device, "elapsed_sec": res.elapsed, "centers": res.centers.tolist(),
                        "feature_order": list(pre.names)}
            ctx = XAIContext(names=pre.names, features=pre.features, centers=res.centers, membership=res.membership,
                             selected=ext.selected, tumor_membership=t_box, tumor=ext.mask_box, brain=pre.mask_box,
                             edges=pre.edges_box, uncertainty=unc_box, feature_median=pre.feature_median,
                             feature_scale=pre.feature_scale, spacing=pre.spacing, m=cfg.fcm.m,
                             cluster_report=ext.cluster_report, cfg=cfg.xai)

            progress("xai", 0.75, "Explainability: evidence, modality contribution, boundary analysis")
            t = time.time()
            tumor_voxels = int(ext.mask_box.sum())
            state = {"ctx": ctx, "extraction": ext, "method": method, "fcm_info": fcm_info, "tumor_voxels": tumor_voxels}
            if with_xai:
                final, orch = run_xai_graph(state)
            else:
                final, orch = state, "skipped"
                final.update(evidence={"available": False}, uncertainty={}, modalities={}, boundary={"available": False},
                             boundary_arrays={}, summary={"segmentation": {"confidence": None}, "disclaimer": DISCLAIMER},
                             explanation="", explanation_source="skipped")
            timings["xai"] = time.time() - t

            metrics = None
            if prep.ground_truth is not None:
                metrics = compute_metrics(seg_full, prep.ground_truth, pre.brain_mask)
                metrics["evaluated_within"] = "brain mask"
                log.info("Dice %.4f  IoU %.4f  Sens %.4f  Spec %.4f", metrics["dice"], metrics["iou"],
                         metrics["sensitivity"], metrics["specificity"])
            timings["total"] = time.time() - t_all + timings.get("preprocessing", 0.0)
            ba = final.get("boundary_arrays", {})
            membership = FuzzyMembership(cluster=res.membership.astype(np.float16), bbox=pre.bbox, centers=res.centers,
                                         selected=ext.selected, tumor=pre.expand(t_box), feature_names=pre.names)
            progress("done", 1.0, "Complete")
            result = PipelineResult(
                method=method, mri=mri, segmentation=seg_full, membership=membership,
                uncertainty=pre.expand(unc_box), edges=pre.edges, preprocessing=pre,
                modality_contribution=final.get("modalities", {}), boundary_analysis=final.get("boundary", {}),
                explanation=final.get("explanation", ""), explanation_source=final.get("explanation_source", ""),
                xai_summary=final.get("summary", {}), metrics=metrics, timings=timings, config=cfg, device=dev,
                log=[], clustering=fcm_info, evidence=final.get("evidence", {}),
                uncertainty_summary=final.get("uncertainty", {}), validation=prep.validation,
                boundary_band=pre.expand(ba["band"].astype(np.uint8)) if ba else None,
                uncertain_boundary=pre.expand(ba["uncertain_band"].astype(np.uint8)) if ba else None,
                ground_truth=prep.ground_truth, xai_context=ctx, orchestrator=orch, warnings=list(ext.warnings))
            log.info("Done in %.1fs (pre %.1f · seg %.1f · xai %.1f)", timings["total"], timings.get("preprocessing", 0),
                     timings["segmentation"], timings["xai"])
        finally:
            log.removeHandler(cap)
        result.log = cap.lines
        return result

    # ------------------------------------------------------------------ convenience
    def run(self, T1: Optional[str] = None, T1ce: Optional[str] = None, T2: Optional[str] = None,
            FLAIR: Optional[str] = None, method: str = "edge_spatial_fcm",
            ground_truth: Union[None, str, NiftiVolume, np.ndarray] = None, mri: Optional[MultimodalMRI] = None,
            progress: Optional[Progress] = None, with_xai: bool = True) -> PipelineResult:
        validation = None
        if mri is None:
            mri, validation = self.load(T1, T1ce, T2, FLAIR)
            if not validation.ok or mri is None:
                raise ValueError("Case validation failed: " + "; ".join(i.message for i in validation.errors))
        prep = self.prepare(mri, validation, ground_truth, progress)
        return self.analyze(prep, method, with_xai, progress)


def run(**kwargs) -> PipelineResult:
    """Module-level shortcut:  result = run(T1=..., T1ce=..., T2=..., FLAIR=..., method="edge_spatial_fcm")."""
    config = kwargs.pop("config", None)
    return NeuroScanX(config).run(**kwargs)
