"""
metrics.py - NeuroScanX P3 (Evaluation & Uncertainty)

Binary segmentation metrics for 2D/3D masks, plus uncertainty-aware metrics.

Edge-case conventions (also used by the tests)
----------------------------------------------
* Masks: any non-zero value is foreground. Bool, int and float 0/1 are accepted.
  NaN/Inf or mismatched shapes raise ValueError.
* Both masks empty        -> Dice = IoU = Precision = Recall = F1 = 1.0
                             (nothing to find, nothing predicted: perfect).
* Exactly one mask empty  -> Dice = IoU = F1 = 0.0.
                             Empty prediction: Precision = 1.0 (no false alarms),
                             Recall = 0.0. Empty ground truth: Precision = 0.0,
                             Recall = 1.0 (nothing to miss).
* Specificity with no negative voxels at all -> 1.0.
* HD95: both empty -> 0.0; exactly one empty -> NaN (undefined).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "to_binary",
    "confusion_counts",
    "dice_coefficient",
    "iou_score",
    "precision_score",
    "recall_score",
    "sensitivity_score",
    "specificity_score",
    "accuracy_score",
    "f1_score",
    "volume_similarity",
    "relative_volume_error",
    "hausdorff95",
    "brats_region_masks",
    "boundary_mask",
    "boundary_vs_interior_uncertainty",
    "uncertainty_error_overlap",
    "evaluate_segmentation",
    "METRIC_DIRECTION",
]

# True = higher is better, False = lower is better
METRIC_DIRECTION = {
    "dice": True, "iou": True, "precision": True, "recall": True,
    "sensitivity": True, "specificity": True, "accuracy": True, "f1": True,
    "volume_similarity": True, "relative_volume_error": False, "hd95": False,
}


# --------------------------------------------------------------------------- #
# Validation & confusion matrix
# --------------------------------------------------------------------------- #
def to_binary(mask: np.ndarray, name: str = "mask") -> np.ndarray:
    """Convert bool/int/float mask to bool (non-zero = foreground)."""
    arr = np.asarray(mask)
    if arr.dtype != bool:
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"{name} must be numeric or boolean, got {arr.dtype}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or Inf")
        arr = arr != 0
    return arr


def _pair(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = to_binary(pred, "prediction")
    g = to_binary(gt, "ground_truth")
    if p.shape != g.shape:
        raise ValueError(f"Shape mismatch: prediction {p.shape} vs ground truth {g.shape}")
    return p, g


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> Dict[str, int]:
    """Return TP, FP, FN, TN voxel counts."""
    p, g = _pair(pred, gt)
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    tn = int(p.size - tp - fp - fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


# --------------------------------------------------------------------------- #
# Overlap metrics
# --------------------------------------------------------------------------- #
def dice_coefficient(pred, gt) -> float:
    """Dice = 2TP / (2TP + FP + FN). Both empty -> 1.0."""
    c = confusion_counts(pred, gt)
    d = 2 * c["tp"] + c["fp"] + c["fn"]
    return 1.0 if d == 0 else float(2 * c["tp"] / d)


def iou_score(pred, gt) -> float:
    """IoU (Jaccard) = TP / (TP + FP + FN). Both empty -> 1.0."""
    c = confusion_counts(pred, gt)
    d = c["tp"] + c["fp"] + c["fn"]
    return 1.0 if d == 0 else float(c["tp"] / d)


def precision_score(pred, gt) -> float:
    """Precision = TP / (TP + FP). Empty prediction -> 1.0 (no false alarms)."""
    c = confusion_counts(pred, gt)
    d = c["tp"] + c["fp"]
    return 1.0 if d == 0 else float(c["tp"] / d)


def recall_score(pred, gt) -> float:
    """Recall (= sensitivity) = TP / (TP + FN). Empty GT -> 1.0 (nothing to miss)."""
    c = confusion_counts(pred, gt)
    d = c["tp"] + c["fn"]
    return 1.0 if d == 0 else float(c["tp"] / d)


sensitivity_score = recall_score


def specificity_score(pred, gt) -> float:
    """Specificity = TN / (TN + FP). No negative voxels at all -> 1.0."""
    c = confusion_counts(pred, gt)
    d = c["tn"] + c["fp"]
    return 1.0 if d == 0 else float(c["tn"] / d)


def accuracy_score(pred, gt) -> float:
    """Accuracy = (TP + TN) / total. Note: inflated by the large background."""
    c = confusion_counts(pred, gt)
    total = sum(c.values())
    return 1.0 if total == 0 else float((c["tp"] + c["tn"]) / total)


def f1_score(pred, gt) -> float:
    """F1 score. For binary masks this is mathematically identical to Dice."""
    return dice_coefficient(pred, gt)


def volume_similarity(pred, gt) -> float:
    """VS = 1 - |Vp - Vg| / (Vp + Vg). Both empty -> 1.0."""
    p, g = _pair(pred, gt)
    vp, vg = int(p.sum()), int(g.sum())
    return 1.0 if vp + vg == 0 else float(1.0 - abs(vp - vg) / (vp + vg))


def relative_volume_error(pred, gt) -> float:
    """(Vp - Vg) / Vg. Positive = over-segmentation. GT empty -> NaN
    (0.0 if the prediction is also empty)."""
    p, g = _pair(pred, gt)
    vp, vg = int(p.sum()), int(g.sum())
    if vg == 0:
        return 0.0 if vp == 0 else float("nan")
    return float((vp - vg) / vg)


# --------------------------------------------------------------------------- #
# Boundary metric
# --------------------------------------------------------------------------- #
def boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Surface voxels: foreground voxels with at least one background neighbour."""
    from scipy.ndimage import binary_erosion

    m = to_binary(mask)
    if not m.any():
        return m.copy()
    # border_value=0 makes foreground touching the volume edge count as surface
    return m & ~binary_erosion(m, border_value=0)


def hausdorff95(pred, gt, spacing: Optional[Sequence[float]] = None) -> float:
    """95th-percentile symmetric Hausdorff distance between surfaces.

    Parameters
    ----------
    spacing : voxel size per axis (e.g. from ``nib_img.header.get_zooms()[:3]``).
              Result is then in the same unit (mm). Default: voxel units.
    """
    from scipy.ndimage import distance_transform_edt

    p, g = _pair(pred, gt)
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float("nan")
    if spacing is not None:
        spacing = tuple(float(s) for s in spacing)
        if len(spacing) != p.ndim:
            raise ValueError(f"spacing needs {p.ndim} values, got {len(spacing)}")

    sp, sg = boundary_mask(p), boundary_mask(g)
    d_to_g = distance_transform_edt(~sg, sampling=spacing)
    d_to_p = distance_transform_edt(~sp, sampling=spacing)
    dists = np.concatenate([d_to_g[sp], d_to_p[sg]])
    return float(np.percentile(dists, 95))


# --------------------------------------------------------------------------- #
# Multi-class helper (BraTS)
# --------------------------------------------------------------------------- #
def brats_region_masks(labels: np.ndarray) -> Dict[str, np.ndarray]:
    """Split a BraTS label volume into the standard evaluation regions.

    Labels: 1 = necrotic/non-enhancing core, 2 = edema, 4 = enhancing tumour.
    (Older BraTS releases use 4; if yours is relabelled to 3, map it first.)
    Returns ``whole`` (1,2,4), ``core`` (1,4) and ``enhancing`` (4).
    """
    lab = np.asarray(labels)
    return {
        "whole": np.isin(lab, (1, 2, 4)),
        "core": np.isin(lab, (1, 4)),
        "enhancing": lab == 4,
    }


# --------------------------------------------------------------------------- #
# Uncertainty-aware metrics
# --------------------------------------------------------------------------- #
def boundary_vs_interior_uncertainty(pred, uncertainty) -> Dict[str, float]:
    """Mean uncertainty on the predicted-tumour surface vs. its interior.

    A well-behaved model is more uncertain at the boundary than inside.
    NaN is returned for a region that has no voxels.
    """
    p = to_binary(pred, "prediction")
    u = np.asarray(uncertainty, dtype=np.float64)
    if u.shape != p.shape:
        raise ValueError(f"Shape mismatch: prediction {p.shape} vs uncertainty {u.shape}")
    b = boundary_mask(p)
    inner = p & ~b

    def _mean(region):
        vals = u[region]
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else float("nan")

    bm, im = _mean(b), _mean(inner)
    ratio = bm / im if np.isfinite(bm) and np.isfinite(im) and im > 1e-12 else float("nan")
    return {"boundary_mean": bm, "interior_mean": im, "boundary_interior_ratio": ratio}


def uncertainty_error_overlap(pred, gt, uncertainty, threshold: float = 0.5) -> Dict[str, float]:
    """How well does high uncertainty point at the actual errors?

    * ``error_mean_uncertainty`` / ``correct_mean_uncertainty``: mean uncertainty
      on wrongly vs. correctly labelled voxels (errors should be higher).
    * ``error_coverage``: fraction of error voxels flagged as high-uncertainty.
    * ``flag_precision``: fraction of high-uncertainty voxels that are errors.
    * ``dice_uncertainty_vs_error``: Dice between the high-uncertainty mask and
      the error mask.
    Values that cannot be computed (e.g. no errors) are NaN.
    """
    p, g = _pair(pred, gt)
    u = np.asarray(uncertainty, dtype=np.float64)
    if u.shape != p.shape:
        raise ValueError(f"Shape mismatch: prediction {p.shape} vs uncertainty {u.shape}")
    finite = np.isfinite(u)
    err = (p != g) & finite
    ok = (p == g) & finite
    high = (u >= threshold) & finite

    def _mean(region):
        return float(u[region].mean()) if region.any() else float("nan")

    n_err, n_high = int(err.sum()), int(high.sum())
    both = int((err & high).sum())
    return {
        "error_mean_uncertainty": _mean(err),
        "correct_mean_uncertainty": _mean(ok),
        "error_coverage": both / n_err if n_err else float("nan"),
        "flag_precision": both / n_high if n_high else float("nan"),
        "dice_uncertainty_vs_error": (2 * both / (n_err + n_high)) if (n_err + n_high) else float("nan"),
    }


# --------------------------------------------------------------------------- #
# One-call evaluation
# --------------------------------------------------------------------------- #
def evaluate_segmentation(
    prediction,
    ground_truth,
    spacing: Optional[Sequence[float]] = None,
    include_hd95: bool = True,
    uncertainty: Optional[np.ndarray] = None,
    uncertainty_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute every metric and return a flat dictionary.

    If ``uncertainty`` is given, the boundary/interior and uncertainty-vs-error
    metrics are added (keys prefixed ``unc_``).
    """
    c = confusion_counts(prediction, ground_truth)
    result: Dict[str, float] = {
        "dice": dice_coefficient(prediction, ground_truth),
        "iou": iou_score(prediction, ground_truth),
        "precision": precision_score(prediction, ground_truth),
        "recall": recall_score(prediction, ground_truth),
        "sensitivity": sensitivity_score(prediction, ground_truth),
        "specificity": specificity_score(prediction, ground_truth),
        "accuracy": accuracy_score(prediction, ground_truth),
        "f1": f1_score(prediction, ground_truth),
        "volume_similarity": volume_similarity(prediction, ground_truth),
        "relative_volume_error": relative_volume_error(prediction, ground_truth),
        "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"],
    }
    if include_hd95:
        result["hd95"] = hausdorff95(prediction, ground_truth, spacing)
    if uncertainty is not None:
        for k, v in boundary_vs_interior_uncertainty(prediction, uncertainty).items():
            result[f"unc_{k}"] = v
        for k, v in uncertainty_error_overlap(
            prediction, ground_truth, uncertainty, uncertainty_threshold
        ).items():
            result[f"unc_{k}"] = v
    return result
