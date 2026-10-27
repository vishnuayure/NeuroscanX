"""BraTS development mode: FCM vs Spatial FCM vs Edge-Aware SP-FCM on the same preprocessed case."""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional

from ..config import METHODS, METHOD_LABELS, PipelineConfig
from ..pipeline.pipeline import NeuroScanX, PreparedCase

METRIC_KEYS = ("dice", "iou", "sensitivity", "specificity", "accuracy")


def compare_methods(prep: PreparedCase, config: Optional[PipelineConfig] = None, methods=METHODS,
                    progress: Optional[Callable[[str, float, str], None]] = None) -> dict:
    """Preprocessing is shared; each method is clustered, post-processed and scored independently."""
    if prep.ground_truth is None:
        raise ValueError("A BraTS ground-truth segmentation is required for the comparison")
    nsx = NeuroScanX(config)
    rows: List[dict] = []
    results = {}
    for i, m in enumerate(methods):
        if progress:
            progress("benchmark", i / len(methods), f"Running {METHOD_LABELS[m]}")
        t = time.time()
        r = nsx.analyze(prep, m, with_xai=False)
        results[m] = r
        rows.append({"method": m, "label": METHOD_LABELS[m], **{k: r.metrics[k] for k in METRIC_KEYS},
                     "iterations": r.clustering["iterations"], "converged": r.clustering["converged"],
                     "segmentation_sec": r.timings["segmentation"], "volume_mm3": r.volume_mm3,
                     "elapsed_sec": time.time() - t})
    best = max(rows, key=lambda r: r["dice"])
    base = next((r for r in rows if r["method"] == "fcm"), rows[0])
    report = {"rows": rows, "best_by_dice": best["method"],
              "delta_vs_fcm": {r["method"]: {k: r[k] - base[k] for k in METRIC_KEYS} for r in rows},
              "note": "Metrics are evaluated inside the brain mask. A single case is anecdotal – use run_batch for cohort statistics."}
    if progress:
        progress("benchmark", 1.0, "Comparison complete")
    return {"report": report, "results": results}


def report_markdown(report: dict) -> str:
    lines = ["# NeuroScanX method comparison", "", "| Method | Dice | IoU | Sensitivity | Specificity | Accuracy | Iter | Time (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in report["rows"]:
        lines.append(f"| {r['label']} | {r['dice']:.4f} | {r['iou']:.4f} | {r['sensitivity']:.4f} | {r['specificity']:.4f} | "
                     f"{r['accuracy']:.4f} | {r['iterations']} | {r['segmentation_sec']:.1f} |")
    lines += ["", f"Best Dice: **{METHOD_LABELS[report['best_by_dice']]}**", "", report["note"]]
    return "\n".join(lines)


def save_report(report: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(report_markdown(report))
    return out_dir


def run_batch(cases: List[dict], config: Optional[PipelineConfig] = None, methods=METHODS) -> dict:
    """cases = [{"T1":..,"T1ce":..,"T2":..,"FLAIR":..,"seg":..}, …] → mean/std of every metric per method."""
    import numpy as np
    nsx = NeuroScanX(config)
    per = {m: [] for m in methods}
    for c in cases:
        mri, rep = nsx.load(c["T1"], c["T1ce"], c["T2"], c["FLAIR"])
        if not rep.ok:
            continue
        prep = nsx.prepare(mri, rep, c["seg"])
        for m in methods:
            per[m].append(nsx.analyze(prep, m, with_xai=False).metrics)
    return {m: {k: (float(np.mean([x[k] for x in v])), float(np.std([x[k] for x in v]))) for k in METRIC_KEYS}
            for m, v in per.items() if v}
