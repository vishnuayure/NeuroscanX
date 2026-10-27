"""Main window: INPUT → PROCESSING → WORKSTATION (2D · 3D · DASHBOARD · BENCHMARK + XAI HUD)."""
from __future__ import annotations

import os
import traceback

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QSplitter, QStackedWidget, QTabWidget, QVBoxLayout, QWidget)

from app.controllers.workers import PipelineWorker
from app.state.app_state import AppState
from src.config import DISCLAIMER, METHOD_LABELS
from src.io.exporters import export_case, next_case_dir
from . import theme
from .benchmark_panel import BenchmarkPanel
from .dashboard import Dashboard
from .input_screen import InputScreen
from .viewer2d_panel import Viewer2DPanel
from .viewer3d_panel import Viewer3DPanel
from .xai_hud import XaiHud

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "outputs")


class ProcessingPage(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self); v.setContentsMargins(120, 80, 120, 40)
        t = QLabel("ANALYSING CASE"); t.setObjectName("title"); t.setStyleSheet("font-size:28px; letter-spacing:10px;")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter); v.addWidget(t)
        self.stage = QLabel("…"); self.stage.setObjectName("subtitle"); self.stage.setAlignment(Qt.AlignmentFlag.AlignCenter); v.addWidget(self.stage)
        self.bar = QProgressBar(); self.bar.setRange(0, 1000); v.addSpacing(16); v.addWidget(self.bar)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); v.addSpacing(10); v.addWidget(self.log, 1)

    def reset(self):
        self.bar.setValue(0); self.log.clear(); self.stage.setText("starting…")

    def on_progress(self, stage: str, frac: float, msg: str):
        self.bar.setValue(int(frac * 1000)); self.stage.setText(msg.upper())
        self.log.appendPlainText(f"[{stage:>13}] {msg}")


class Workstation(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        v = QVBoxLayout(self); v.setContentsMargins(8, 8, 8, 0)
        top = QHBoxLayout()
        self.head = QLabel("NEUROSCANX"); self.head.setStyleSheet(f"letter-spacing:6px; font-size:16px; color:#fff;")
        self.sub = QLabel(""); self.sub.setObjectName("dim")
        self.btn_new = QPushButton("NEW CASE"); self.btn_export = QPushButton("EXPORT CASE"); self.btn_shot = QPushButton("SCREENSHOT")
        top.addWidget(self.head); top.addWidget(self.sub); top.addStretch()
        for b in (self.btn_shot, self.btn_export, self.btn_new):
            top.addWidget(b)
        v.addLayout(top)
        split = QSplitter(Qt.Orientation.Horizontal); v.addWidget(split, 1)
        self.tabs = QTabWidget()
        self.v2d, self.v3d = Viewer2DPanel(state), Viewer3DPanel(state)
        self.dash, self.bench = Dashboard(state), BenchmarkPanel(state)
        self.tabs.addTab(self.v2d, "2D VIEWER"); self.tabs.addTab(self.v3d, "3D VIEW")
        self.tabs.addTab(self.dash, "DASHBOARD"); self.tabs.addTab(self.bench, "BENCHMARK")
        self.hud = XaiHud(state)
        split.addWidget(self.tabs); split.addWidget(self.hud)
        split.setStretchFactor(0, 1); split.setSizes([1000, 350])
        state.resultChanged.connect(self._on_result)

    def _on_result(self):
        r = self.state.result
        self.sub.setText(f"{METHOD_LABELS[r.method]}  ·  {tuple(r.mri.shape)}  ·  {r.device.name}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroScanX — Multimodal MRI · Uncertainty-Aware Tumor Segmentation")
        self.resize(1560, 940)
        self.state = AppState()
        self.worker = None
        root = QWidget(); root.setObjectName("root"); self.setCentralWidget(root)
        lay = QVBoxLayout(root); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        self.pages = QStackedWidget(); lay.addWidget(self.pages, 1)
        self.input = InputScreen(); self.proc = ProcessingPage(); self.ws = Workstation(self.state)
        for p in (self.input, self.proc, self.ws):
            self.pages.addWidget(p)
        foot = QLabel("⚠  " + DISCLAIMER); foot.setObjectName("disclaimer"); foot.setWordWrap(True); lay.addWidget(foot)
        self.input.runRequested.connect(self._start)
        self.ws.btn_new.clicked.connect(lambda: self.pages.setCurrentWidget(self.input))
        self.ws.btn_export.clicked.connect(self.export_case)
        self.ws.btn_shot.clicked.connect(self.save_current_screenshot)

    # ------------------------------------------------------------------ run
    def _start(self, mri, report, gt, method, cfg):
        self.proc.reset(); self.pages.setCurrentWidget(self.proc)
        self.worker = PipelineWorker(mri, report, gt, method, cfg, self)
        self.worker.progress.connect(self.proc.on_progress)
        self.worker.finished_ok.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _finished(self, result):
        self.state.set_result(result)
        self.ws.tabs.setCurrentIndex(0)
        self.pages.setCurrentWidget(self.ws)

    def _failed(self, tb: str):
        self.pages.setCurrentWidget(self.input)
        print(tb)
        QMessageBox.critical(self, "NeuroScanX – processing failed", tb.strip().splitlines()[-1] + "\n\nFull traceback printed to the console.")

    # ------------------------------------------------------------------ export
    def _grab_all(self, shots_dir: str):
        os.makedirs(shots_dir, exist_ok=True)
        cur = self.ws.tabs.currentIndex()
        names = ["viewer_2d", "viewer_3d", "dashboard", "benchmark"]
        for i, n in enumerate(names[:3]):
            self.ws.tabs.setCurrentIndex(i); QApplication.processEvents()
            if i == 1 and self.ws.v3d.plotter is not None:
                self.ws.v3d.screenshot(os.path.join(shots_dir, f"{n}.png"))
            else:
                self.ws.tabs.widget(i).grab().save(os.path.join(shots_dir, f"{n}.png"))
        self.ws.hud.grab().save(os.path.join(shots_dir, "xai_hud.png"))
        self.ws.tabs.setCurrentIndex(cur)

    def export_case(self):
        r = self.state.result
        if r is None:
            return
        try:
            out = export_case(r, base_dir=OUTPUT_ROOT)
            self._grab_all(os.path.join(out, "screenshots"))
            if getattr(self.ws.bench, "report", None):
                from src.evaluation.benchmark import save_report
                save_report(self.ws.bench.report, out)
            QMessageBox.information(self, "Export complete",
                                    f"Saved to:\n{out}\n\nsegmentation.nii.gz · uncertainty.nii.gz · tumor_membership.nii.gz\n"
                                    "xai.json · report.json · processing.log · screenshots/")
        except Exception:
            traceback.print_exc()
            QMessageBox.critical(self, "Export failed", traceback.format_exc().strip().splitlines()[-1])

    def save_current_screenshot(self):
        if self.state.result is None:
            return
        os.makedirs(os.path.join(OUTPUT_ROOT, "screenshots"), exist_ok=True)
        i = self.ws.tabs.currentIndex()
        path = os.path.join(OUTPUT_ROOT, "screenshots", f"screenshot_{len(os.listdir(os.path.join(OUTPUT_ROOT, 'screenshots'))) + 1:03d}.png")
        if i == 1 and self.ws.v3d.plotter is not None:
            self.ws.v3d.screenshot(path)
        else:
            self.ws.tabs.currentWidget().grab().save(path)
        self.statusBar().showMessage(f"Saved {path}", 5000)

    def closeEvent(self, e):
        self.ws.v3d.close()
        super().closeEvent(e)
