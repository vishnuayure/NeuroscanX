"""Results dashboard: case summary, evidence, timings, evaluation (only with ground truth), explanation."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QGridLayout, QLabel, QPlainTextEdit, QScrollArea, QVBoxLayout, QWidget)

from app.state.app_state import AppState
from src.config import DISCLAIMER
from . import theme
from .widgets import HudBar, header


def _card(title: str):
    f = QFrame(); f.setObjectName("card"); l = QVBoxLayout(f); l.setContentsMargins(16, 12, 16, 12)
    l.addWidget(header(title)); return f, l


class Dashboard(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.Shape.NoFrame); outer.addWidget(sc)
        self.body = QWidget(); sc.setWidget(self.body)
        self.grid = QGridLayout(self.body); self.grid.setContentsMargins(14, 14, 14, 14); self.grid.setSpacing(12)
        state.resultChanged.connect(self.rebuild)

    def _clear(self):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w:
                w.deleteLater()

    def rebuild(self):
        self._clear()
        r = self.state.result
        s = r.xai_summary
        t = QLabel("CASE COMPLETE"); t.setObjectName("big"); t.setStyleSheet(f"color:{theme.CYAN}; letter-spacing:6px; font-size:20px;")
        self.grid.addWidget(t, 0, 0, 1, 3)

        c, l = _card("Whole tumor")
        vol = QLabel(f"{r.volume_mm3:,.0f} mm³"); vol.setObjectName("big"); l.addWidget(vol)
        l.addWidget(QLabel(f"{r.volume_mm3 / 1000:.2f} mL · {int(r.segmentation.sum()):,} voxels"))
        self.grid.addWidget(c, 1, 0)

        c, l = _card("Segmentation")
        conf = s["segmentation"]["confidence"]
        lbl = QLabel("—" if conf is None else f"Confidence: {conf:.0%}"); lbl.setObjectName("big"); l.addWidget(lbl)
        if conf is not None:
            l.addWidget(QLabel(s["segmentation"]["confidence_label"]))
            u = r.uncertainty_summary
            l.addWidget(QLabel(f"{u['fraction_high_confidence']:.0%} high-confidence · {u['fraction_moderate']:.0%} moderate · {u['fraction_uncertain']:.0%} uncertain voxels"))
        self.grid.addWidget(c, 1, 1)

        c, l = _card("Uncertainty")
        b = r.boundary_analysis
        lbl = QLabel("—" if not b.get("available") else f"Boundary uncertainty: {b['boundary_uncertainty']:.0%}"); lbl.setObjectName("big"); l.addWidget(lbl)
        if b.get("available"):
            l.addWidget(QLabel(f"{b['fraction_uncertain_boundary']:.0%} of boundary voxels uncertain"))
            if b.get("most_uncertain_side"):
                l.addWidget(QLabel(f"Highest: {b['most_uncertain_side']} boundary ({b['sides'][b['most_uncertain_side']]['mean_uncertainty']:.0%})"))
        self.grid.addWidget(c, 1, 2)

        c, l = _card("Modality evidence")
        g = QGridLayout()
        mods = {k: v for k, v in r.modality_contribution.items() if not k.startswith("_")}
        for i, (n, m) in enumerate(sorted(mods.items(), key=lambda kv: -kv[1]["contribution"])):
            g.addWidget(QLabel(n), i, 0)
            g.addWidget(HudBar(m["relative"], theme.LEVEL_COLORS[m["level"]], segments=16), i, 1)
            g.addWidget(QLabel(f"{m['level']} · {m['contribution']:.0%} · z {m['tumor_z']:+.1f}"), i, 2)
        g.setColumnStretch(1, 1); l.addLayout(g)
        self.grid.addWidget(c, 2, 0, 1, 2)

        c, l = _card("Processing")
        tm = r.timings
        l.addWidget(QLabel(f"Preprocessing   {tm.get('preprocessing', 0):6.1f} s\nSegmentation    {tm.get('segmentation', 0):6.1f} s\n"
                           f"XAI             {tm.get('xai', 0):6.1f} s\nTotal           {tm.get('total', 0):6.1f} s"))
        l.itemAt(l.count() - 1).widget().setObjectName("mono")
        cl = r.clustering
        l.addWidget(QLabel(f"{r.method} · {cl['n_clusters']} clusters · {cl['iterations']} it · "
                           f"{'converged' if cl['converged'] else 'max-iter'} · {r.device.active.upper()}"))
        self.grid.addWidget(c, 2, 2)

        row = 3
        if r.metrics:
            c, l = _card("Evaluation (BraTS ground truth)")
            m = r.metrics
            g = QGridLayout()
            for i, (k, nm) in enumerate((("dice", "Dice"), ("iou", "IoU"), ("sensitivity", "Sensitivity"),
                                         ("specificity", "Specificity"), ("accuracy", "Accuracy"))):
                lab = QLabel(nm); lab.setObjectName("h"); val = QLabel(f"{m[k]:.4f}"); val.setObjectName("big")
                g.addWidget(lab, 0, i); g.addWidget(val, 1, i)
            l.addLayout(g)
            l.addWidget(QLabel("Evaluated inside the brain mask. Whole tumor = any BraTS label > 0."))
            self.grid.addWidget(c, row, 0, 1, 3); row += 1

        c, l = _card(f"Explanation · {r.explanation_source}")
        te = QPlainTextEdit(r.explanation); te.setReadOnly(True); te.setMinimumHeight(330); l.addWidget(te)
        self.grid.addWidget(c, row, 0, 1, 3); row += 1
        if r.warnings:
            c, l = _card("Notes")
            l.addWidget(QLabel("\n".join("⚠ " + w for w in r.warnings))); self.grid.addWidget(c, row, 0, 1, 3)
        self.grid.setRowStretch(row + 1, 1)
