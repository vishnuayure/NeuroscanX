"""Unit tests for uncertainty_map.py (deterministic, synthetic data only)."""
import numpy as np
import pytest

try:
    from src.uncertainty.uncertainty_map import (
        compute_uncertainty, confidence_map, entropy_uncertainty, get_high_uncertainty_mask,
        get_uncertainty_statistics, margin_uncertainty, max_membership_uncertainty,
        normalize_uncertainty, sanitize_membership,
    )
except ImportError:
    from uncertainty_map import (
        compute_uncertainty, confidence_map, entropy_uncertainty, get_high_uncertainty_mask,
        get_uncertainty_statistics, margin_uncertainty, max_membership_uncertainty,
        normalize_uncertainty, sanitize_membership,
    )


def _uniform(c=3, shape=(4, 4, 4)):
    return np.full((c,) + shape, 1.0 / c)


def _onehot(c=3, shape=(4, 4, 4), k=0):
    m = np.zeros((c,) + shape)
    m[k] = 1.0
    return m


def test_output_shape_dtype_range():
    rng = np.random.default_rng(0)
    raw = rng.random((3, 5, 5, 5))
    m = raw / raw.sum(axis=0, keepdims=True)
    for method in ("entropy", "max", "margin"):
        u = compute_uncertainty(m, method)
        assert u.shape == (5, 5, 5)
        assert u.dtype == np.float32
        assert u.min() >= 0.0 and u.max() <= 1.0


def test_entropy_uniform_is_one_onehot_is_zero():
    assert np.allclose(entropy_uncertainty(_uniform()), 1.0, atol=1e-6)
    assert np.allclose(entropy_uncertainty(_onehot()), 0.0, atol=1e-6)


def test_entropy_known_value_two_classes():
    m = np.zeros((2, 1, 1, 1))
    m[0], m[1] = 0.5, 0.5
    assert entropy_uncertainty(m)[0, 0, 0] == pytest.approx(1.0, abs=1e-6)
    m[0], m[1] = 0.9, 0.1
    expected = -(0.9 * np.log(0.9) + 0.1 * np.log(0.1)) / np.log(2)
    assert entropy_uncertainty(m)[0, 0, 0] == pytest.approx(expected, abs=1e-6)


def test_entropy_unnormalised_uniform_equals_log_c():
    assert entropy_uncertainty(_uniform(4), normalize=False)[0, 0, 0] == pytest.approx(np.log(4), abs=1e-6)


def test_high_uncertainty_near_half_low_near_extremes():
    mid = np.stack([np.full((2, 2), 0.5), np.full((2, 2), 0.5)])
    hi = np.stack([np.full((2, 2), 0.99), np.full((2, 2), 0.01)])
    assert entropy_uncertainty(mid).mean() > 0.99
    assert entropy_uncertainty(hi).mean() < 0.1
    assert entropy_uncertainty(mid).mean() > entropy_uncertainty(hi).mean()


def test_zero_and_one_probabilities_are_finite():
    u = entropy_uncertainty(_onehot())
    assert np.all(np.isfinite(u))
    assert np.allclose(u, 0.0, atol=1e-6)


def test_max_and_margin_extremes():
    assert np.allclose(max_membership_uncertainty(_uniform()), 1.0, atol=1e-6)
    assert np.allclose(max_membership_uncertainty(_onehot()), 0.0, atol=1e-6)
    assert np.allclose(margin_uncertainty(_uniform()), 1.0, atol=1e-6)
    assert np.allclose(margin_uncertainty(_onehot()), 0.0, atol=1e-6)


def test_margin_known_value():
    m = np.array([0.6, 0.3, 0.1]).reshape(3, 1, 1, 1)
    assert margin_uncertainty(m)[0, 0, 0] == pytest.approx(1 - 0.3, abs=1e-6)


def test_confidence_map():
    m = np.array([0.6, 0.3, 0.1]).reshape(3, 1, 1, 1)
    assert confidence_map(m)[0, 0, 0] == pytest.approx(0.6, abs=1e-6)


def test_nan_and_inf_are_handled():
    m = _uniform(2, (2, 2, 2)).copy()
    m[0, 0, 0, 0] = np.nan
    m[1, 1, 1, 1] = np.inf
    u = entropy_uncertainty(m)
    assert np.all(np.isfinite(u))
    assert u.min() >= 0 and u.max() <= 1


def test_nan_raises_when_requested():
    m = _uniform(2, (2, 2, 2)).copy()
    m[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        entropy_uncertainty(m, on_invalid="raise")
    with pytest.raises(ValueError):
        sanitize_membership(_uniform() * 4.0, on_invalid="raise")


def test_all_zero_membership_becomes_maximally_uncertain():
    m = np.zeros((3, 2, 2, 2))
    assert np.allclose(entropy_uncertainty(m), 1.0, atol=1e-6)


def test_unnormalised_membership_is_renormalised():
    m = _uniform(2, (2, 2, 2)) * 0.5   # sums to 0.5
    assert np.allclose(sanitize_membership(m).sum(axis=0), 1.0)


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        entropy_uncertainty(np.array([0.5, 0.5]))        # 1-D
    with pytest.raises(ValueError):
        entropy_uncertainty(np.ones((1, 3, 3, 3)))       # single class
    with pytest.raises(ValueError):
        compute_uncertainty(_uniform(), "nope")


def test_empty_spatial_volume_does_not_crash():
    u = entropy_uncertainty(np.zeros((3, 0, 4, 4)) + 1 / 3)
    assert u.shape == (0, 4, 4)


def test_normalize_uncertainty():
    u = np.array([2.0, 4.0, 6.0])
    n = normalize_uncertainty(u)
    assert n.tolist() == [0.0, 0.5, 1.0]
    assert n.dtype == np.float32
    assert np.all(normalize_uncertainty(np.full(5, 3.0)) == 0.0)      # constant
    assert np.all(np.isfinite(normalize_uncertainty(np.array([np.nan, 1.0, 2.0]))))
    assert normalize_uncertainty(np.array([0.5]), vmin=0, vmax=1)[0] == pytest.approx(0.5)


def test_high_uncertainty_mask_threshold():
    u = np.array([0.1, 0.5, 0.9, np.nan])
    assert get_high_uncertainty_mask(u, 0.5).tolist() == [False, True, True, False]
    assert get_high_uncertainty_mask(u, 0.95).sum() == 0


def test_statistics():
    u = np.array([0.0, 0.5, 1.0, 0.5])
    s = get_uncertainty_statistics(u, threshold=0.5)
    assert s["mean"] == pytest.approx(0.5)
    assert s["max"] == 1.0 and s["min"] == 0.0
    assert s["std"] == pytest.approx(np.std(u))
    assert s["pct_high"] == pytest.approx(75.0)
    assert s["n_voxels"] == 4


def test_statistics_with_mask_and_empty_mask():
    u = np.array([[0.0, 1.0], [0.5, 0.5]])
    mask = np.array([[False, True], [False, False]])
    assert get_uncertainty_statistics(u, mask)["mean"] == 1.0
    empty = get_uncertainty_statistics(u, np.zeros_like(mask))
    assert empty["n_voxels"] == 0 and np.isnan(empty["mean"])
    with pytest.raises(ValueError):
        get_uncertainty_statistics(u, np.zeros((3, 3), bool))


def test_statistics_ignore_non_finite():
    s = get_uncertainty_statistics(np.array([0.2, np.nan, np.inf, 0.4]))
    assert s["n_voxels"] == 2 and s["mean"] == pytest.approx(0.3)


def test_deterministic():
    rng = np.random.default_rng(1)
    raw = rng.random((3, 4, 4, 4))
    m = raw / raw.sum(axis=0)
    assert np.array_equal(entropy_uncertainty(m), entropy_uncertainty(m))
