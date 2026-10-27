"""Background workers – the UI thread never runs the pipeline."""
from __future__ import annotations

import traceback
from typing import Dict, Optional

from PySide6.QtCore import QThread, Signal

from src.config import PipelineConfig
from src.io.nifti_loader import load_case, load_ground_truth
from src.pipeline.pipeline import NeuroScanX, PreparedCase


class ValidationWorker(QThread):
    done = Signal(object, object, object, object)   # mri, report, gt_volume, gt_info

    def __init__(self, paths: Dict[str, str], gt_path: Optional[str], parent=None):
        super().__init__(parent)
        self.paths, self.gt_path = paths, gt_path

    def run(self):
        try:
            mri, rep = load_case(self.paths)
            gt, info = (None, None)
            if self.gt_path and mri is not None and rep.ok:
                gt, info = load_ground_truth(self.gt_path, mri)
                rep.ground_truth = info
            self.done.emit(mri, rep, gt, info)
        except Exception:   # never let a worker die silently
            self.done.emit(None, None, None, traceback.format_exc())


class PipelineWorker(QThread):
    progress = Signal(str, float, str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, mri, report, gt_vol, method: str, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self.mri, self.report, self.gt_vol, self.method, self.config = mri, report, gt_vol, method, config

    def run(self):
        try:
            from src.visualization.viewer_3d import build_meshes
            cb = lambda stage, frac, msg: self.progress.emit(stage, frac, msg)
            nsx = NeuroScanX(self.config)
            prep = nsx.prepare(self.mri, self.report, self.gt_vol, cb)
            result = nsx.analyze(prep, self.method, True, cb)
            self.progress.emit("mesh", 0.95, "Generating 3D brain and tumor meshes from the segmentation")
            result.meshes = build_meshes(result)
            self.progress.emit("done", 1.0, "Case complete")
            self.finished_ok.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class BenchmarkWorker(QThread):
    progress = Signal(str, float, str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, result, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self.result, self.config = result, config

    def run(self):
        try:
            from src.evaluation.benchmark import compare_methods
            r = self.result
            prep = PreparedCase(r.mri, r.preprocessing, r.validation, dict(r.timings), r.ground_truth)
            out = compare_methods(prep, self.config, progress=lambda s, f, m: self.progress.emit(s, f, m))
            self.finished_ok.emit(out["report"])
        except Exception:
            self.failed.emit(traceback.format_exc())
