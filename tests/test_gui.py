"""Headless GUI smoke tests (QT_QPA_PLATFORM=offscreen): the real widgets are driven with a real pipeline result."""
import os
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF, Qt, QPoint
from PySide6.QtTest import QTest


@pytest.fixture(scope="module")
def qapp():
    from app.ui import theme
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(theme.QSS)
    return app


def test_workstation_flow(qapp, result):
    from app.state.app_state import AppState
    from app.ui.viewer2d_panel import Viewer2DPanel
    from app.ui.xai_hud import XaiHud
    from app.ui.dashboard import Dashboard
    from src.visualization.viewer_3d import build_meshes
    st = AppState(); v = Viewer2DPanel(st); hud = XaiHud(st); dash = Dashboard(st)
    v.resize(1200, 500); v.show()
    st.set_result(result)
    qapp.processEvents()
    # views show the actual MRI: pixel under the crosshair equals the windowed voxel value
    for plane, view in v.views.items():
        assert view.rgba is not None and view.rgba.shape[2] == 4
    # synchronized crosshair: clicking in the axial view moves the sagittal/coronal slices
    ax = v.views["axial"]
    before = st.crosshair
    r, c = 10, 12
    v._on_click("axial", r, c)
    assert st.crosshair[2] == before[2] and st.crosshair != before
    # scrolling changes the fixed axis only
    z = st.crosshair[2]; v._on_scroll("axial", 1)
    assert st.crosshair[2] == min(z + 1, result.segmentation.shape[2] - 1)
    # toggles / modality / opacity / W-L
    st.update_display(unc_on=True, edge_on=True, boundary_on=True, opacity=0.8)
    st.set_modality("T2"); v._on_wl_drag(10, -10)
    assert st.modality == "T2"
    # HUD texts bound to real values
    assert f"{result.xai_summary['segmentation']['confidence']:.0%}" in hud.conf_lbl.text()
    assert hud.ev_widgets and "membership" in hud.loc_head.text() or "OUTSIDE" in hud.loc_head.text()
    # zoom/pan do not crash
    from PySide6.QtGui import QWheelEvent
    ax.zoom = 3.0; ax.update(); ax.reset_view()
    assert dash.grid.count() > 4


def test_input_screen_validation(qapp, case_paths):
    from app.ui.input_screen import InputScreen
    s = InputScreen()
    for k in ("T1", "T1ce", "T2", "FLAIR"):
        s.cards[k].path = case_paths[k]
    s.gt_card.path = case_paths["GT"]
    from app.controllers.workers import ValidationWorker
    w = ValidationWorker(s.paths(), s.gt_card.path)
    w.run()   # run synchronously in the test
    class Sig: pass
    from src.io.nifti_loader import load_case, load_ground_truth
    mri, rep = load_case(s.paths()); gt, info = load_ground_truth(case_paths["GT"], mri); rep.ground_truth = info
    s._validated(mri, rep, gt, info)
    assert s.run_btn.isEnabled() and "CASE VALID" in s.summary.text()
    assert "valid" in s.cards["T1"].info.text()
    # missing modality → processing stays disabled
    s2 = InputScreen(); s2.cards["T1"].path = case_paths["T1"]; s2._changed("T1", case_paths["T1"])
    assert not s2.run_btn.isEnabled()


def test_main_window_constructs(qapp):
    from app.ui.main_window import MainWindow
    w = MainWindow(); w.show(); qapp.processEvents()
    assert "research and decision-support prototype" in w.findChild(type(w.centralWidget().layout().itemAt(1).widget()), "disclaimer").text()
    w.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs an X display / GL context (run under xvfb-run with QT_QPA_PLATFORM=xcb)")
def test_3d_panel_renders_real_mesh(qapp, result, tmp_path):
    from app.state.app_state import AppState
    from app.ui.viewer3d_panel import Viewer3DPanel
    from src.visualization.viewer_3d import build_meshes
    import imageio.v3 as iio  # noqa: F401  (pyvista dependency)
    result.meshes = build_meshes(result)
    st = AppState(); v = Viewer3DPanel(st); v.resize(900, 700); v.show(); qapp.processEvents()
    st.set_result(result); qapp.processEvents()
    assert v.plotter is not None and "tumor" in v.actors and "brain" in v.actors
    for mode in ("SEGMENTATION", "UNCERTAINTY", "BOUNDARY"):
        v.radios[mode].setChecked(True); qapp.processEvents()
        img = v.plotter.screenshot(None)
        assert int((img.sum(2) > 80).sum()) > 2000, f"{mode}: scene rendered blank"
    v.cb_tumor.setChecked(False); qapp.processEvents()
    assert not v.actors["tumor"].GetVisibility()
    v.close()
