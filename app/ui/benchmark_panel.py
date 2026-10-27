"""BraTS development mode: FCM vs Spatial FCM vs Edge-Aware SP-FCM."""
from __future__ import annotations

import os

from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget, QHeaderView)

from app.controllers.workers import BenchmarkWorker
from app.state.app_state import AppState
from src.evaluation.benchmark import METRIC_KEYS, report_markdown, save_report
from . import theme


class BenchmarkPanel(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state, self.report, self.worker = state, None, None
        lay = QVBoxLayout(self)
        self.info = QLabel(""); self.info.setWordWrap(True); lay.addWidget(self.info)
        bar = QHBoxLayout()
        self.run = QPushButton("RUN COMPARISON · FCM vs SPATIAL FCM vs EDGE-AWARE SP-FCM"); self.run.setObjectName("primary"); self.run.clicked.connect(self._run)
        self.save = QPushButton("SAVE REPORT (JSON + MD)"); self.save.clicked.connect(self._save); self.save.setEnabled(False)
        bar.addWidget(self.run); bar.addWidget(self.save); bar.addStretch(); lay.addLayout(bar)
        self.status = QLabel(""); self.status.setObjectName("dim"); lay.addWidget(self.status)
        cols = ["METHOD", "DICE", "IoU", "SENSITIVITY", "SPECIFICITY", "ACCURACY", "ITER", "TIME (s)"]
        self.table = QTableWidget(0, len(cols)); self.table.setHorizontalHeaderLabels(cols)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide(); lay.addWidget(self.table)
        self.note = QLabel(""); self.note.setObjectName("dim"); self.note.setWordWrap(True); lay.addWidget(self.note); lay.addStretch()
        state.resultChanged.connect(self._on_result)

    def _on_result(self):
        r = self.state.result
        has = r.ground_truth is not None
        self.run.setEnabled(has)
        self.report = None; self.table.setRowCount(0); self.save.setEnabled(False)
        self.info.setText("Ground truth attached – you can compare the three clustering methods on this case (preprocessing is shared)."
                          if has else "Evaluation mode is off: select a BraTS ground-truth segmentation on the input screen to enable the comparison.")

    def _run(self):
        self.run.setEnabled(False); self.status.setText("Running…")
        self.worker = BenchmarkWorker(self.state.result, self.state.result.config, self)
        self.worker.progress.connect(lambda s, f, m: self.status.setText(m))
        self.worker.finished_ok.connect(self._done)
        self.worker.failed.connect(lambda tb: (self.status.setText("Failed – see console"), print(tb), self.run.setEnabled(True)))
        self.worker.start()

    def _done(self, report):
        self.report = report
        self.run.setEnabled(True); self.save.setEnabled(True); self.status.setText("Comparison complete")
        self.table.setRowCount(len(report["rows"]))
        for i, r in enumerate(report["rows"]):
            vals = [r["label"]] + [f"{r[k]:.4f}" for k in METRIC_KEYS] + [str(r["iterations"]), f"{r['segmentation_sec']:.1f}"]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if r["method"] == report["best_by_dice"] and j in (0, 1):
                    it.setForeground(__import__("PySide6.QtGui", fromlist=["QColor"]).QColor(theme.OK))
                self.table.setItem(i, j, it)
        self.note.setText(report["note"])

    def _save(self):
        d = QFileDialog.getExistingDirectory(self, "Save comparison report", os.path.abspath("data/outputs"))
        if d:
            save_report(self.report, d)
            self.status.setText(f"Saved comparison.json / comparison.md to {d}")
