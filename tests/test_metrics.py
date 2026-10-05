"""Unit tests for metrics.py (hand-checkable binary examples)."""
import numpy as np
import pytest

try:
    from src.evaluation.metrics import (
        accuracy_score, boundary_mask, boundary_vs_interior_uncertainty, brats_region_masks, confusion_counts,
        dice_coefficient, evaluate_segmentation, f1_score, hausdorff95, iou_score,
        precision_score, recall_score, relative_volume_error, sensitivity_score,
        specificity_score, uncertainty_error_overlap, volume_similarity,
    )
except ImportError:
    from metrics import (
        accuracy_score, boundary_mask, boundary_vs_interior_uncertainty, brats_region_masks, confusion_counts,
        dice_coefficient, evaluate_segmentation, f1_score, hausdorff95, iou_score,
        precision_score, recall_score, relative_volume_error, sensitivity_score,
        specificity_score, uncertainty_error_overlap, volume_similarity,
    )

# 1-D example: TP=2, FP=1, FN=1, TN=4  (8 voxels)
GT = np.array([1, 1, 1, 0, 0, 0, 0, 0])
PR = np.array([0, 1, 1, 1, 0, 0, 0, 0])


def test_confusion_counts():
    assert confusion_counts(PR, GT) == {"tp": 2, "fp": 1, "fn": 1, "tn": 4}


def test_partial_overlap_known_values():
    assert dice_coefficient(PR, GT) == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))   # 0.6667
    assert iou_score(PR, GT) == pytest.approx(2 / 4)
    assert precision_score(PR, GT) == pytest.approx(2 / 3)
    assert recall_score(PR, GT) == pytest.approx(2 / 3)
    assert sensitivity_score(PR, GT) == pytest.approx(2 / 3)
    assert specificity_score(PR, GT) == pytest.approx(4 / 5)
    assert accuracy_score(PR, GT) == pytest.approx(6 / 8)
    assert f1_score(PR, GT) == pytest.approx(dice_coefficient(PR, GT))
    assert volume_similarity(PR, GT) == pytest.approx(1.0)       # same volume
    assert relative_volume_error(PR, GT) == pytest.approx(0.0)


def test_perfect_segmentation():
    r = evaluate_segmentation(GT, GT)
    for k in ("dice", "iou", "precision", "recall", "specificity", "accuracy", "f1"):
        assert r[k] == 1.0
    assert r["hd95"] == 0.0


def test_completely_wrong_segmentation():
    inv = 1 - GT
    assert dice_coefficient(inv, GT) == 0.0
    assert iou_score(inv, GT) == 0.0
    assert recall_score(inv, GT) == 0.0
    assert precision_score(inv, GT) == 0.0
    assert specificity_score(inv, GT) == 0.0


def test_empty_prediction():
    empty = np.zeros_like(GT)
    assert dice_coefficient(empty, GT) == 0.0
    assert iou_score(empty, GT) == 0.0
    assert recall_score(empty, GT) == 0.0
    assert precision_score(empty, GT) == 1.0        # no false alarms
    assert specificity_score(empty, GT) == 1.0
    assert relative_volume_error(empty, GT) == pytest.approx(-1.0)
    assert np.isnan(hausdorff95(empty, GT))


def test_empty_ground_truth():
    empty = np.zeros_like(GT)
    assert dice_coefficient(PR, empty) == 0.0
    assert precision_score(PR, empty) == 0.0
    assert recall_score(PR, empty) == 1.0           # nothing to miss
    assert np.isnan(relative_volume_error(PR, empty))
    assert np.isnan(hausdorff95(PR, empty))


def test_both_empty():
    e = np.zeros((4, 4, 4), bool)
    r = evaluate_segmentation(e, e)
    assert r["dice"] == 1.0 and r["iou"] == 1.0
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["specificity"] == 1.0
    assert r["hd95"] == 0.0
    assert r["relative_volume_error"] == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        dice_coefficient(np.zeros((3, 3)), np.zeros((3, 4)))
    with pytest.raises(ValueError):
        evaluate_segmentation(np.zeros(5), np.zeros(6))


def test_bool_int_float_masks_agree():
    a_bool, b_bool = PR.astype(bool), GT.astype(bool)
    for cast in (bool, np.uint8, np.int64, np.float32):
        a, b = PR.astype(cast), GT.astype(cast)
        assert dice_coefficient(a, b) == pytest.approx(dice_coefficient(a_bool, b_bool))


def test_nonzero_labels_count_as_foreground():
    assert dice_coefficient(GT * 4, GT) == 1.0


def test_nan_mask_raises():
    bad = GT.astype(float)
    bad[0] = np.nan
    with pytest.raises(ValueError):
        dice_coefficient(bad, GT)


def test_volume_metrics_over_segmentation():
    big = np.ones_like(GT)                          # volume 8 vs 3
    assert relative_volume_error(big, GT) == pytest.approx((8 - 3) / 3)
    assert volume_similarity(big, GT) == pytest.approx(1 - 5 / 11)


# ---- Hausdorff ------------------------------------------------------------ #
def _cube(offset, size=5, shape=(20, 20, 20)):
    m = np.zeros(shape, bool)
    m[offset:offset + size, 5:5 + size, 5:5 + size] = True
    return m


def test_hd95_identical_is_zero():
    assert hausdorff95(_cube(5), _cube(5)) == 0.0


def test_hd95_shifted_cube_equals_shift():
    # Shifting along x by 3 voxels: all surface distances are <= 3 and the
    # 95th percentile lands on 3.0 for this geometry.
    assert hausdorff95(_cube(5), _cube(8)) == pytest.approx(3.0, abs=1e-6)


def test_hd95_respects_spacing():
    d1 = hausdorff95(_cube(5), _cube(8))
    d2 = hausdorff95(_cube(5), _cube(8), spacing=(2.0, 1.0, 1.0))
    assert d2 == pytest.approx(2 * d1, abs=1e-6)
    with pytest.raises(ValueError):
        hausdorff95(_cube(5), _cube(8), spacing=(1.0, 1.0))


# ---- BraTS regions -------------------------------------------------------- #
def test_brats_regions():
    lab = np.array([0, 1, 2, 4])
    r = brats_region_masks(lab)
    assert r["whole"].tolist() == [False, True, True, True]
    assert r["core"].tolist() == [False, True, False, True]
    assert r["enhancing"].tolist() == [False, False, False, True]


# ---- Uncertainty-aware metrics ------------------------------------------- #
def test_boundary_vs_interior_uncertainty():
    pred = _cube(5, size=7)
    unc = np.zeros(pred.shape, np.float32)
    unc[boundary_mask(pred)] = 0.8
    r = boundary_vs_interior_uncertainty(pred, unc)
    assert r["boundary_mean"] == pytest.approx(0.8)
    assert r["interior_mean"] == pytest.approx(0.0)


def test_uncertainty_error_overlap_perfect_flagging():
    err_unc = (PR != GT).astype(float)              # uncertainty = 1 exactly on errors
    r = uncertainty_error_overlap(PR, GT, err_unc, threshold=0.5)
    assert r["error_coverage"] == 1.0
    assert r["flag_precision"] == 1.0
    assert r["dice_uncertainty_vs_error"] == 1.0
    assert r["error_mean_uncertainty"] == 1.0
    assert r["correct_mean_uncertainty"] == 0.0


def test_uncertainty_error_overlap_no_errors_gives_nan():
    r = uncertainty_error_overlap(GT, GT, np.zeros(GT.shape))
    assert np.isnan(r["error_coverage"])


def test_evaluate_with_uncertainty_adds_keys():
    r = evaluate_segmentation(PR, GT, uncertainty=np.zeros(GT.shape), include_hd95=False)
    assert "unc_error_coverage" in r and "unc_boundary_mean" in r
    assert "hd95" not in r
    with pytest.raises(ValueError):
        evaluate_segmentation(PR, GT, uncertainty=np.zeros(3))
