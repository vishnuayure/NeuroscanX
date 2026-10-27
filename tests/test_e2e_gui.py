"""End-to-end journey through the real threaded UI: select 4 files + GT → validate → run → workstation → export."""
import os
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtTest import QTest


def _wait(cond, timeout_ms=120000):
    waited = 0
    while not cond() and waited < timeout_ms:
        QTest.qWait(100); waited += 100
    assert cond(), "timed out"


def test_full_journey(case_paths, tmp_path, monkeypatch):
    from app.ui import theme
    import app.ui.main_window as mw
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(theme.QSS)
    monkeypatch.setattr(mw, "OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    w = mw.MainWindow(); w.show()
    s = w.input
    assert not s.run_btn.isEnabled()                              # nothing selected → cannot run
    for k in ("T1", "T1ce", "T2", "FLAIR"):
        s.cards[k].set_path(case_paths[k])
    _wait(lambda: s.report is not None)
    assert s.run_btn.isEnabled() and "CASE VALID" in s.summary.text()
    s.gt_card.set_path(case_paths["GT"])
    _wait(lambda: s.gt_vol is not None)                           # re-validated with ground truth
    s.res.setCurrentIndex(1)                                      # fast mode
    s.run_btn.click()
    _wait(lambda: w.pages.currentWidget() is w.ws)
    r = w.state.result
    assert r.metrics is not None and r.meshes.get("tumor") is not None
    assert w.ws.hud.conf_lbl.text() and w.ws.bench.run.isEnabled()
    w.export_case()
    case = os.path.join(str(tmp_path), "case_001")
    for f in ("segmentation.nii.gz", "uncertainty.nii.gz", "xai.json", "report.json", "processing.log"):
        assert os.path.exists(os.path.join(case, f)), f
    assert os.path.exists(os.path.join(case, "screenshots", "viewer_2d.png"))
    w.close()
