import numpy as np, pytest
from src.evaluation.metrics import compute_metrics, whole_tumor_from_brats
from src.evaluation.benchmark import compare_methods, report_markdown


def test_perfect_and_disjoint():
    g = np.zeros((10, 10, 10), bool); g[2:6, 2:6, 2:6] = True
    m = compute_metrics(g, g)
    assert m["dice"] == m["iou"] == m["sensitivity"] == m["specificity"] == m["accuracy"] == 1.0
    d = np.zeros_like(g); d[7:9, 7:9, 7:9] = True
    m = compute_metrics(d, g)
    assert m["dice"] == 0 and m["iou"] == 0 and m["sensitivity"] == 0 and m["specificity"] < 1


def test_known_values():
    gt = np.zeros(100, bool); gt[:40] = True
    pr = np.zeros(100, bool); pr[20:60] = True       # tp=20 fp=20 fn=20 tn=40
    m = compute_metrics(pr, gt)
    assert m["tp"] == 20 and m["fp"] == 20 and m["fn"] == 20 and m["tn"] == 40
    assert m["dice"] == pytest.approx(0.5) and m["iou"] == pytest.approx(1 / 3)
    assert m["sensitivity"] == pytest.approx(0.5) and m["specificity"] == pytest.approx(40 / 60)
    assert m["accuracy"] == pytest.approx(0.6)


def test_mask_restriction_and_empty():
    gt = np.zeros(10, bool); pr = np.zeros(10, bool)
    assert compute_metrics(pr, gt)["dice"] == 1.0
    with pytest.raises(ValueError):
        compute_metrics(np.zeros(3), np.zeros(4))


def test_brats_labels():
    gt = np.array([0, 1, 2, 4, 0]); assert whole_tumor_from_brats(gt).tolist() == [False, True, True, True, False]


def test_comparison_report(prepared, cfg):
    nsx, prep = prepared
    out = compare_methods(prep, cfg)
    rows = out["report"]["rows"]
    assert [r["method"] for r in rows] == ["fcm", "spatial_fcm", "edge_spatial_fcm"]
    assert all(0 <= r["dice"] <= 1 for r in rows) and "Edge-Aware SP-FCM" in report_markdown(out["report"])
