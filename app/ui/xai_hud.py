"""Futuristic XAI HUD – every number is bound to the structured evidence computed by the pipeline."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from app.state.app_state import AppState
from src.visualization.xai_overlay import build_hud_model, build_local_model
from src.xai.evidence import LocalEvidenceExplainer
from . import theme
from .widgets import HudBar, header


def _col(label: str) -> str:
    return theme.LEVEL_COLORS.get(label.upper(), theme.CYAN)


class XaiHud(QFrame):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.explainer = None
        self.setObjectName("panel")
        self.setMinimumWidth(390); self.setMaximumWidth(430)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.Shape.NoFrame); sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); outer.addWidget(sc)
        body = QWidget(); sc.setWidget(body)
        v = QVBoxLayout(body); v.setContentsMargins(16, 14, 16, 14); v.setSpacing(6)

        t = QLabel("NEUROSCANX // XAI"); t.setStyleSheet(f"color:{theme.CYAN}; letter-spacing:3px; font-weight:600;"); v.addWidget(t)
        self.method = QLabel(""); self.method.setObjectName("dim"); v.addWidget(self.method)
        v.addSpacing(6)
        self.tumor_title = QLabel("WHOLE TUMOR"); self.tumor_title.setObjectName("cardTitle"); v.addWidget(self.tumor_title)
        self.conf_bar = HudBar(0, segments=28, height=12); v.addWidget(self.conf_bar)
        self.conf_lbl = QLabel("—"); self.conf_lbl.setObjectName("mono"); v.addWidget(self.conf_lbl)
        v.addSpacing(8); v.addWidget(header("Evidence"))
        self.ev_grid = QGridLayout(); self.ev_grid.setHorizontalSpacing(8); self.ev_grid.setVerticalSpacing(5)
        v.addLayout(self.ev_grid); self.ev_widgets = []
        v.addSpacing(8); v.addWidget(header("Boundary confidence"))
        self.b_bar = HudBar(0, segments=28, height=10); v.addWidget(self.b_bar)
        self.b_lbl = QLabel("—"); self.b_lbl.setObjectName("mono"); v.addWidget(self.b_lbl)
        v.addSpacing(8); v.addWidget(header("Why this region?"))
        self.why = QLabel("—"); self.why.setWordWrap(True); v.addWidget(self.why)
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine); sep.setStyleSheet(f"color:{theme.BORDER}"); v.addSpacing(8); v.addWidget(sep)
        v.addWidget(header("At crosshair"))
        self.loc_head = QLabel("—"); self.loc_head.setObjectName("mono"); v.addWidget(self.loc_head)
        self.loc_grid = QGridLayout(); self.loc_grid.setHorizontalSpacing(8); self.loc_grid.setVerticalSpacing(4)
        v.addLayout(self.loc_grid); self.loc_widgets = []
        self.loc_text = QLabel(""); self.loc_text.setWordWrap(True); self.loc_text.setObjectName("dim"); v.addWidget(self.loc_text)
        v.addStretch()
        state.resultChanged.connect(self._on_result)
        state.crosshairChanged.connect(self._on_cross)

    def _clear(self, grid, store):
        for w in store:
            grid.removeWidget(w); w.deleteLater()
        store.clear()

    def _on_result(self):
        r = self.state.result
        h = build_hud_model(r)
        self.explainer = LocalEvidenceExplainer(r.xai_context)
        self.method.setText(f"{h['method']} · explanation: {h['explanation_source']}")
        if h["confidence"] is None:
            self.conf_bar.set_value(0); self.conf_lbl.setText("NO REGION SEGMENTED")
        else:
            self.conf_bar.set_value(h["confidence"], theme.CYAN)
            self.conf_lbl.setText(f"{h['confidence']:.0%}   {h['confidence_label'].upper()}"
                                  + (f"   ·   {h['volume_ml']:.1f} mL" if h["volume_ml"] is not None else ""))
        self._clear(self.ev_grid, self.ev_widgets)
        for i, row in enumerate(h["evidence"]):
            n = QLabel(row["name"]); n.setObjectName("mono"); n.setMinimumWidth(52)
            b = HudBar(row["value"], _col(row["level"]), segments=10, height=9)
            l = QLabel(f"{row['level'][:4]} {row['detail']}"); l.setObjectName("mono"); l.setMinimumWidth(78); l.setStyleSheet(f"color:{_col(row['level'])}")
            for c, w in enumerate((n, b, l)):
                self.ev_grid.addWidget(w, i, c); self.ev_widgets.append(w)
            self.ev_grid.setColumnStretch(1, 1)
        if h["boundary_confidence"] is not None:
            self.b_bar.set_value(h["boundary_confidence"], theme.CYAN)
            side = f" · weakest: {h['most_uncertain_side']}" if h["most_uncertain_side"] else ""
            self.b_lbl.setText(f"{h['boundary_confidence']:.0%}   {h['boundary_label'].upper()}{side}")
        else:
            self.b_bar.set_value(0); self.b_lbl.setText("—")
        self.why.setText(h["why"] or "—")

    def _on_cross(self):
        if self.explainer is None or self.state.result is None:
            return
        r = self.state.result
        bx = tuple(c - s.start for c, s in zip(self.state.crosshair, r.preprocessing.bbox))
        loc = self.explainer.at(bx)
        m = build_local_model(loc)
        self._clear(self.loc_grid, self.loc_widgets)
        if not m["inside"]:
            self.loc_head.setText("OUTSIDE BRAIN MASK"); self.loc_text.setText(""); return
        self.loc_head.setText(f"{'INSIDE' if m['in_tumor'] else 'OUTSIDE'} REGION\nmembership {m['membership']:.2f} · uncertainty {m['uncertainty']:.2f}")
        for i, row in enumerate(m["rows"]):
            n = QLabel(row["name"]); n.setObjectName("mono"); n.setMinimumWidth(52)
            b = HudBar(row["value"], theme.CYAN, segments=10, height=8)
            l = QLabel(f"z {row['z']:+.1f}"); l.setObjectName("mono"); l.setMinimumWidth(78)
            for c, w in enumerate((n, b, l)):
                self.loc_grid.addWidget(w, i, c); self.loc_widgets.append(w)
        self.loc_grid.setColumnStretch(1, 1)
        self.loc_text.setText(m["text"] + f"  Edge strength {m['edge']:.2f}, neighbourhood support {m['support']:.2f}.")
