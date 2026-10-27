"""Segmentation metrics for BraTS evaluation (binary whole-tumor masks)."""
from __future__ import annotations

from typing import Optional

import numpy as np


def whole_tumor_from_brats(gt: np.ndarray) -> np.ndarray:
    """BraTS labels (1 necrosis, 2 oedema, 4 enhancing [3 in older releases]) → whole tumor = any label > 0."""
    return np.asarray(gt) > 0


def confusion(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None):
    p, g = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {g.shape}")
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        p, g = p[m], g[m]
    tp = int(np.sum(p & g)); fp = int(np.sum(p & ~g)); fn = int(np.sum(~p & g)); tn = int(np.sum(~p & ~g))
    return tp, fp, fn, tn


def _div(a, b, empty=1.0):
    return float(a / b) if b > 0 else empty


def compute_metrics(pred, gt, mask=None) -> dict:
    """Dice, IoU, sensitivity, specificity, accuracy (+ precision, volumes).  `mask` restricts the evaluation domain."""
    tp, fp, fn, tn = confusion(pred, gt, mask)
    return {
        "dice": _div(2 * tp, 2 * tp + fp + fn),
        "iou": _div(tp, tp + fp + fn),
        "sensitivity": _div(tp, tp + fn),
        "specificity": _div(tn, tn + fp),
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
        "precision": _div(tp, tp + fp),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
