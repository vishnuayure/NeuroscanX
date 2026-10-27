"""3-D viewer: translucent brain surface + tumor mesh generated from the segmentation (PyVista / VTK)."""
from __future__ import annotations

import logging
import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QHBoxLayout, QLabel, QPushButton, QRadioButton, QSlider,
                               QVBoxLayout, QWidget)

from app.state.app_state import AppState
from src.visualization.viewer_2d import uncertainty_colors
from src.visualization.viewer_3d import voxel_to_world
from . import theme

log = logging.getLogger("neuroscanx")
MODES = ("SEGMENTATION", "UNCERTAINTY", "BOUNDARY")


def _edge_colors(e: np.ndarray) -> np.ndarray:
    e = np.clip(e, 0, 1)[:, None]
    lo, hi = np.array([20, 60, 140.]), np.array([240, 250, 255.])
    return (lo + (hi - lo) * e).astype(np.uint8)


class Viewer3DPanel(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.plotter = None
        self.actors = {}
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout(); bar.setSpacing(10)
        self.cb_brain = self._cb("BRAIN", True, bar); self.cb_tumor = self._cb("TUMOR", True, bar)
        self.cb_marker = self._cb("CROSSHAIR MARKER", True, bar)
        bar.addSpacing(10)
        self.group = QButtonGroup(self); self.radios = {}
        for i, m in enumerate(MODES):
            r = QRadioButton(m); self.radios[m] = r; self.group.addButton(r); bar.addWidget(r)
            r.toggled.connect(self._mode_changed)
        self.radios["SEGMENTATION"].setChecked(True)
        bar.addSpacing(10)
        bar.addWidget(QLabel("TUMOR OPACITY")); self.s_tumor = self._slider(100, bar)
        bar.addWidget(QLabel("BRAIN OPACITY")); self.s_brain = self._slider(14, bar)
        bar.addStretch()
        b = QPushButton("RESET CAMERA"); b.clicked.connect(self.reset_camera); bar.addWidget(b)
        lay.addLayout(bar)
        self.legend = QLabel("Left-drag: rotate · wheel: zoom · middle/shift-drag: pan")
        self.legend.setObjectName("dim")
        try:
            from pyvistaqt import QtInteractor
            self.plotter = QtInteractor(self)
            self.plotter.set_background(theme.BG)
            lay.addWidget(self.plotter.interactor if hasattr(self.plotter, "interactor") and not isinstance(self.plotter, QWidget) else self.plotter, 1)
        except Exception as e:   # no GL / missing package: degrade gracefully
            self.plotter = None
            msg = QLabel(f"3D view unavailable on this system:\n{e}")
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter); msg.setObjectName("dim"); lay.addWidget(msg, 1)
        lay.addWidget(self.legend)
        state.resultChanged.connect(self.rebuild)
        state.crosshairChanged.connect(self._update_marker)

    def _cb(self, text, on, bar):
        c = QCheckBox(text); c.setChecked(on); c.toggled.connect(self._visibility); bar.addWidget(c); return c

    def _slider(self, v, bar):
        s = QSlider(Qt.Orientation.Horizontal); s.setRange(0, 100); s.setValue(v); s.setFixedWidth(90)
        s.valueChanged.connect(self._opacity); bar.addWidget(s); return s

    @property
    def mode(self):
        return next(m for m, r in self.radios.items() if r.isChecked())

    # ------------------------------------------------------------------
    def rebuild(self):
        if self.plotter is None or self.state.result is None:
            return
        import pyvista as pv
        r, p = self.state.result, self.plotter
        p.clear(); self.actors = {}
        meshes = r.meshes or {}
        brain, tumor, core = meshes.get("brain"), meshes.get("tumor"), meshes.get("core")
        if brain is not None:
            self.actors["brain"] = p.add_mesh(brain, color="#8fb4c9", opacity=self.s_brain.value() / 100, smooth_shading=True,
                                              specular=0.15, name="brain")
        self._draw_tumor()
        try:
            p.add_axes(color="#6f8a97", line_width=2)
        except Exception:
            pass
        n = tumor.n_cells if tumor is not None else 0
        p.add_text(f"TUMOR MESH · {n:,} faces" if tumor is not None else "NO TUMOR REGION SEGMENTED", font_size=9,
                   color=theme.CYAN, position="upper_left", name="hud")
        if os.environ.get("NEUROSCANX_DEPTH_PEELING") == "1":   # opt-in: blanks the scene on some drivers / software GL
            try:
                p.enable_depth_peeling(8)
            except Exception:
                pass
        self._update_marker(); self._visibility()
        self.reset_camera()

    def _draw_tumor(self):
        p, r = self.plotter, self.state.result
        if p is None or r is None:
            return
        meshes = r.meshes or {}
        tumor, core = meshes.get("tumor"), meshes.get("core")
        for nm in ("tumor", "core"):
            if nm in self.actors:
                p.remove_actor(self.actors.pop(nm))
        if tumor is None:
            return
        op = self.s_tumor.value() / 100
        mode = self.mode
        if mode == "SEGMENTATION":
            self.actors["tumor"] = p.add_mesh(tumor, color="#ff3b6b", opacity=op, smooth_shading=True, specular=0.5, name="tumor")
        elif mode == "UNCERTAINTY":
            m = tumor.copy()
            m.point_data["rgb"] = (uncertainty_colors(np.asarray(m.point_data["uncertainty"]) * 1.0)).astype(np.uint8)
            self.actors["tumor"] = p.add_mesh(m, scalars="rgb", rgb=True, opacity=min(op, 0.55), smooth_shading=True, name="tumor")
            if core is not None:   # opaque high-confidence core (membership ≥ 0.9) inside the translucent uncertain shell
                self.actors["core"] = p.add_mesh(core, color="#e6faff", opacity=0.95, smooth_shading=True, name="core")
        else:   # BOUNDARY: surface coloured by local edge (boundary) evidence
            m = tumor.copy()
            m.point_data["rgb"] = _edge_colors(np.asarray(m.point_data["edge"]))
            self.actors["tumor"] = p.add_mesh(m, scalars="rgb", rgb=True, opacity=op, smooth_shading=True, name="tumor")
        p.render()

    def _mode_changed(self, checked):
        if checked:
            self._draw_tumor(); self._visibility()

    def _update_marker(self):
        p, r = self.plotter, self.state.result
        if p is None or r is None:
            return
        import pyvista as pv
        c = voxel_to_world(self.state.crosshair, r.mri.voxel_spacing)
        if "marker" in self.actors:
            p.remove_actor(self.actors.pop("marker"))
        self.actors["marker"] = p.add_mesh(pv.Sphere(radius=2.2, center=c), color="#ffffff", name="marker", opacity=0.9)
        self.actors["marker"].SetVisibility(self.cb_marker.isChecked())
        p.render()

    def _visibility(self, *_):
        if self.plotter is None:
            return
        for k, cb in (("brain", self.cb_brain), ("tumor", self.cb_tumor), ("core", self.cb_tumor), ("marker", self.cb_marker)):
            if k in self.actors:
                self.actors[k].SetVisibility(cb.isChecked())
        self.plotter.render()

    def _opacity(self, *_):
        if self.plotter is None:
            return
        if "brain" in self.actors:
            self.actors["brain"].GetProperty().SetOpacity(self.s_brain.value() / 100)
        if "tumor" in self.actors:
            op = self.s_tumor.value() / 100
            self.actors["tumor"].GetProperty().SetOpacity(min(op, 0.55) if self.mode == "UNCERTAINTY" else op)
        self.plotter.render()

    def reset_camera(self):
        if self.plotter is None:
            return
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

    def screenshot(self, path: str) -> bool:
        if self.plotter is None:
            return False
        self.plotter.screenshot(path)
        return True

    def closeEvent(self, e):
        if self.plotter is not None:
            try:
                self.plotter.close()
            except Exception:
                pass
        super().closeEvent(e)
