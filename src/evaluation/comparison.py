"""
comparison.py - NeuroScanX P3 (Evaluation & Uncertainty)

Compare segmentation methods (FCM, Spatial FCM, Edge-SP-FCM, ...) on the same
cases. Nothing is hard-coded: you pass in real results from P2.

Typical use
-----------
    from comparison import compare_methods

    res = compare_methods(
        ground_truths=[gt_case1, gt_case2, ...],          # binary tumour masks
        predictions={"FCM": [...], "Spatial FCM": [...], "Edge-SP-FCM": [...]},
        memberships={"FCM": [u1, u2, ...], ...},          # optional, (C,X,Y,Z)
        tumour_class=2,                                   # class index in P2 output
        spacing=(1, 1, 1),
        reference="Edge-SP-FCM",
    )
    print(res.format_table())
    df = res.to_dataframe()

If ``predictions`` has no entry for a method but ``memberships`` does, the
prediction is derived as ``argmax(membership, axis=0) == tumour_class``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from metrics import METRIC_DIRECTION, evaluate_segmentation
from uncertainty_map import compute_uncertainty, get_uncertainty_statistics, sanitize_membership

__all__ = ["segmentation_from_membership", "compare_methods", "ComparisonResult"]

DEFAULT_METRICS = ("dice", "iou", "precision", "recall", "specificity", "hd95")


def segmentation_from_membership(membership: np.ndarray, tumour_class: int) -> np.ndarray:
    """Hard tumour mask from a (C, ...) membership array."""
    m = sanitize_membership(membership)
    if not 0 <= tumour_class < m.shape[0]:
        raise ValueError(f"tumour_class {tumour_class} out of range for {m.shape[0]} classes")
    return np.argmax(m, axis=0) == tumour_class


def _nanmean_std(values: Sequence[float]):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0, int(arr.size)


def _wilcoxon(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired two-sided Wilcoxon signed-rank p-value; NaN if not computable."""
    from scipy.stats import wilcoxon

    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 1 or np.allclose(a, b):
        return float("nan")
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")


@dataclass
class ComparisonResult:
    """Container returned by :func:`compare_methods`."""

    methods: List[str]
    metrics: List[str]
    per_case: List[Dict]                          # one flat row per (method, case)
    summary: Dict[str, Dict[str, Dict[str, float]]]   # method -> metric -> mean/std/n
    best_method: Dict[str, Optional[str]]         # metric -> method name
    uncertainty_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    wilcoxon_p: Dict[str, Dict[str, float]] = field(default_factory=dict)  # method -> metric -> p
    reference: Optional[str] = None

    # ---- output helpers ------------------------------------------------- #
    def to_records(self) -> List[Dict]:
        """Per-case rows as a list of dicts (no pandas needed)."""
        return list(self.per_case)

    def to_dataframe(self, kind: str = "summary"):
        """``kind`` = "summary" (mean/std per method) or "per_case". Needs pandas."""
        import pandas as pd

        if kind == "per_case":
            return pd.DataFrame(self.per_case)
        if kind != "summary":
            raise ValueError("kind must be 'summary' or 'per_case'")
        rows = []
        for m in self.methods:
            row = {"method": m}
            for k in self.metrics:
                s = self.summary[m][k]
                row[f"{k}_mean"], row[f"{k}_std"] = s["mean"], s["std"]
            rows.append(row)
        return pd.DataFrame(rows).set_index("method")

    def format_table(self, precision: int = 3) -> str:
        """Plain-text table: mean ± std per metric, best value per column marked '*'."""
        head = ["Method"] + [k for k in self.metrics]
        lines = []
        for m in self.methods:
            cells = [m]
            for k in self.metrics:
                s = self.summary[m][k]
                txt = f"{s['mean']:.{precision}f} ± {s['std']:.{precision}f}"
                if self.best_method.get(k) == m:
                    txt += " *"
                cells.append(txt)
            lines.append(cells)
        widths = [max(len(r[i]) for r in [head] + lines) for i in range(len(head))]
        fmt = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths))
        out = [fmt(head), "-" * (sum(widths) + 2 * (len(widths) - 1))] + [fmt(r) for r in lines]
        out.append("* = best mean for that metric")
        return "\n".join(out)


def compare_methods(
    ground_truths: Sequence[np.ndarray],
    predictions: Optional[Mapping[str, Sequence[np.ndarray]]] = None,
    memberships: Optional[Mapping[str, Sequence[np.ndarray]]] = None,
    tumour_class: Optional[int] = None,
    spacing: Optional[Sequence[float]] = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    uncertainty_method: str = "entropy",
    uncertainty_threshold: float = 0.5,
    reference: Optional[str] = None,
) -> ComparisonResult:
    """Evaluate every method on every case.

    Parameters
    ----------
    ground_truths : list of binary masks, one per case.
    predictions   : {method: [mask per case]} (optional if memberships given).
    memberships   : {method: [(C, X, Y, Z) per case]} (optional). Used for the
                    uncertainty comparison and, with ``tumour_class``, to derive
                    predictions that are not supplied.
    reference     : method to test the others against (paired Wilcoxon).
                    Defaults to the last method listed (Edge-SP-FCM in P2's order).
    """
    predictions = predictions or {}
    memberships = memberships or {}
    methods = list(dict.fromkeys(list(predictions) + list(memberships)))
    if not methods:
        raise ValueError("Provide predictions and/or memberships")
    n = len(ground_truths)
    if n == 0:
        raise ValueError("ground_truths is empty")
    for name in metrics:
        if name not in METRIC_DIRECTION:
            raise ValueError(f"Unknown metric '{name}'. Choose from {sorted(METRIC_DIRECTION)}")
    if reference is not None and reference not in methods:
        raise ValueError(f"reference '{reference}' not among methods {methods}")
    reference = reference or methods[-1]

    per_case: List[Dict] = []
    values = {m: {k: [] for k in metrics} for m in methods}
    unc_values = {m: [] for m in methods}

    for m in methods:
        for src, label in ((predictions, "predictions"), (memberships, "memberships")):
            if m in src and len(src[m]) != n:
                raise ValueError(f"{label}['{m}'] has {len(src[m])} cases, expected {n}")
        for i in range(n):
            mem = memberships[m][i] if m in memberships else None
            if m in predictions:
                pred = predictions[m][i]
            elif mem is not None and tumour_class is not None:
                pred = segmentation_from_membership(mem, tumour_class)
            else:
                raise ValueError(f"No prediction for '{m}': pass predictions or memberships + tumour_class")

            unc = compute_uncertainty(mem, uncertainty_method) if mem is not None else None
            if unc is not None and unc.shape != np.asarray(ground_truths[i]).shape:
                raise ValueError(
                    f"{m}, case {i}: uncertainty shape {unc.shape} != ground truth "
                    f"{np.asarray(ground_truths[i]).shape}"
                )
            res = evaluate_segmentation(
                pred, ground_truths[i], spacing=spacing,
                include_hd95="hd95" in metrics, uncertainty=unc,
                uncertainty_threshold=uncertainty_threshold,
            )
            row = {"method": m, "case": i, **res}
            if unc is not None:
                st = get_uncertainty_statistics(unc, threshold=uncertainty_threshold)
                row.update({f"uncertainty_{k}": v for k, v in st.items() if k != "n_voxels"})
                unc_values[m].append({**st, **{k: res[k] for k in res if k.startswith("unc_")}})
            per_case.append(row)
            for k in metrics:
                values[m][k].append(res[k])

    summary: Dict = {}
    for m in methods:
        summary[m] = {}
        for k in metrics:
            mean, std, cnt = _nanmean_std(values[m][k])
            summary[m][k] = {"mean": mean, "std": std, "n": cnt}

    best: Dict[str, Optional[str]] = {}
    for k in metrics:
        cand = {m: summary[m][k]["mean"] for m in methods if np.isfinite(summary[m][k]["mean"])}
        if not cand:
            best[k] = None
        else:
            pick = max if METRIC_DIRECTION[k] else min
            best[k] = pick(cand, key=cand.get)

    unc_summary: Dict[str, Dict[str, float]] = {}
    for m in methods:
        if unc_values[m]:
            keys = unc_values[m][0].keys()
            unc_summary[m] = {k: _nanmean_std([d[k] for d in unc_values[m]])[0] for k in keys if k != "n_voxels"}

    wil: Dict[str, Dict[str, float]] = {}
    for m in methods:
        if m == reference:
            continue
        wil[m] = {k: _wilcoxon(values[reference][k], values[m][k]) for k in metrics}

    return ComparisonResult(
        methods=methods, metrics=list(metrics), per_case=per_case, summary=summary,
        best_method=best, uncertainty_summary=unc_summary, wilcoxon_p=wil, reference=reference,
    )
