"""Synchronised axial / coronal / sagittal viewer with overlays, crosshair, window/level and modality switching."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget)

from app.state.app_state import AppState
from src.visualization import viewer_2d as v2
from .slice_view import SliceView


class Viewer2DPanel(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._vols = {}
        self._block = False
        root = QVBoxLayout(self); root.setContentsMargins(6, 6, 6, 6)

        bar = QHBoxLayout(); bar.setSpacing(10)
        self.mod = QComboBox(); self.mod.currentTextChanged.connect(self._on_modality)
        bar.addWidget(QLabel("MODALITY")); bar.addWidget(self.mod)
        self.cb_seg = self._cb("SEGMENTATION", True, bar)
        self.cb_unc = self._cb("UNCERTAINTY", False, bar)
        self.cb_edge = self._cb("EDGES", False, bar)
        self.cb_bnd = self._cb("UNCERTAIN BOUNDARY", False, bar)
        self.cb_cross = self._cb("CROSSHAIR", True, bar)
        bar.addWidget(QLabel("OPACITY")); self.op = QSlider(Qt.Orientation.Horizontal); self.op.setRange(0, 100)
        self.op.setValue(45); self.op.setFixedWidth(100); self.op.valueChanged.connect(self._on_display); bar.addWidget(self.op)
        bar.addStretch()
        b = QPushButton("RESET VIEW"); b.clicked.connect(self._reset); bar.addWidget(b)
        root.addLayout(bar)

        bar2 = QHBoxLayout(); bar2.setSpacing(10)
        bar2.addWidget(QLabel("LEVEL")); self.lvl = QSlider(Qt.Orientation.Horizontal); self.lvl.setRange(0, 1000); bar2.addWidget(self.lvl, 1)
        bar2.addWidget(QLabel("WINDOW")); self.win = QSlider(Qt.Orientation.Horizontal); self.win.setRange(1, 1000); bar2.addWidget(self.win, 1)
        self.lvl.valueChanged.connect(self._on_wl_slider); self.win.valueChanged.connect(self._on_wl_slider)
        self.legend = QLabel("Wheel: slice · Ctrl+wheel: zoom · middle-drag: pan · right-drag: window/level · double-click: reset")
        self.legend.setObjectName("dim"); bar2.addWidget(self.legend)
        root.addLayout(bar2)

        row = QHBoxLayout(); row.setSpacing(6)
        self.views = {p: SliceView(p) for p in v2.PLANES}
        for p, v in self.views.items():
            row.addWidget(v, 1)
            v.clicked.connect(lambda r, c, p=p: self._on_click(p, r, c))
            v.scrolled.connect(lambda d, p=p: self._on_scroll(p, d))
            v.wlDragged.connect(self._on_wl_drag)
        root.addLayout(row, 1)
        state.resultChanged.connect(self._on_result)
        state.crosshairChanged.connect(self.refresh)
        state.displayChanged.connect(self.refresh)

    def _cb(self, text, on, bar):
        c = QCheckBox(text); c.setChecked(on); c.toggled.connect(self._on_display); bar.addWidget(c); return c

    # ------------------------------------------------------------------ state → widgets
    def _on_result(self):
        r = self.state.result
        self._vols = {m: np.nan_to_num(v.data) for m, v in r.mri.modalities.items()}
        self._block = True
        self.mod.clear(); self.mod.addItems(list(self._vols))
        self.mod.setCurrentText(self.state.modality)
        self._block = False
        self._sync_sliders()
        for v in self.views.values():
            v.reset_view()

    def _sync_sliders(self):
        m = self.state.modality
        if m not in self._vols:
            return
        vmax = float(self._vols[m].max()) or 1.0
        c, w = self.state.wl[m]
        self._block = True
        self.lvl.setValue(int(np.clip(c / vmax * 1000, 0, 1000))); self.win.setValue(int(np.clip(w / vmax * 1000, 1, 1000)))
        self._block = False

    def _on_modality(self, m):
        if self._block or not m:
            return
        self.state.set_modality(m); self._sync_sliders()

    def _on_display(self, *_):
        if self._block:
            return
        self.state.update_display(seg_on=self.cb_seg.isChecked(), unc_on=self.cb_unc.isChecked(),
                                  edge_on=self.cb_edge.isChecked(), boundary_on=self.cb_bnd.isChecked(),
                                  crosshair_on=self.cb_cross.isChecked(), opacity=self.op.value() / 100.0)

    def _on_wl_slider(self, *_):
        if self._block:
            return
        vmax = float(self._vols[self.state.modality].max()) or 1.0
        self.state.set_wl(self.lvl.value() / 1000 * vmax, max(self.win.value() / 1000 * vmax, 1e-6))

    def _on_wl_drag(self, dx, dy):
        m = self.state.modality
        c, w = self.state.wl[m]
        self.state.set_wl(c + dx * w / 400.0, w - dy * w / 400.0)
        self._sync_sliders()

    def _reset(self):
        for v in self.views.values():
            v.reset_view()

    # ------------------------------------------------------------------ interaction
    def _on_click(self, plane, row, col):
        shp = self.state.result.segmentation.shape
        self.state.set_crosshair(v2.pixel_to_voxel(plane, row, col, self.state.crosshair, shp))

    def _on_scroll(self, plane, d):
        ijk = list(self.state.crosshair)
        ijk[v2.fixed_axis(plane)] += d
        self.state.set_crosshair(ijk)

    # ------------------------------------------------------------------ rendering
    def refresh(self):
        r = self.state.result
        if r is None or not self._vols:
            return
        st = self.state
        vol = self._vols[st.modality]
        c, w = st.wl[st.modality]
        shp = vol.shape
        sp = r.mri.voxel_spacing
        for plane, view in self.views.items():
            idx = st.crosshair[v2.fixed_axis(plane)]
            base = v2.apply_window(v2.plane_image(vol, plane, idx), c, w)
            seg = v2.plane_image(r.segmentation, plane, idx) > 0 if st.seg_on else None
            unc = v2.plane_image(r.uncertainty, plane, idx) if st.unc_on else None
            edge = v2.plane_image(r.edges, plane, idx) if st.edge_on else None
            bnd = (v2.plane_image(r.uncertain_boundary, plane, idx) > 0) if (st.boundary_on and r.uncertain_boundary is not None) else None
            rgba = v2.compose_rgba(base, seg, unc, edge, bnd, seg_opacity=st.opacity, unc_opacity=min(1.0, st.opacity + 0.2),
                                   edge_opacity=min(1.0, st.opacity + 0.2))
            view.show_cross = st.crosshair_on
            n = shp[v2.fixed_axis(plane)]
            view.set_image(rgba, v2.plane_aspect(sp, plane), v2.voxel_to_pixel(plane, st.crosshair, shp),
                           f"{st.modality}  ·  slice {idx + 1}/{n}  ·  W {w:.0f} L {c:.0f}")
