import numpy as np
import pytest

from src.preprocessing.normalization import (
    z_score_normalize,
    min_max_normalize,
)
from src.preprocessing.denoising import (
    gaussian_denoise,
    median_denoise,
)
from src.preprocessing.clahe import (
    clahe_enhance,
)


def test_z_score_normalize():
    data = np.array(
        [[[1, 2], [3, 4]]],
        dtype=np.float32,
    )

    result = z_score_normalize(data)

    assert result.shape == data.shape
    assert result.dtype == np.float32
    assert np.isclose(np.mean(result), 0.0)
    assert np.isclose(np.std(result), 1.0)


def test_z_score_normalize_ignores_zero_background():
    data = np.array(
        [[[0, 0], [2, 4]]],
        dtype=np.float32,
    )

    result = z_score_normalize(data)

    assert result.shape == data.shape
    assert result[0, 0, 0] == 0
    assert result[0, 0, 1] == 0
    assert np.isclose(np.mean(result[data != 0]), 0.0)
    assert np.isclose(np.std(result[data != 0]), 1.0)


def test_min_max_normalize():
    data = np.array(
        [[[1, 2], [3, 4]]],
        dtype=np.float32,
    )

    result = min_max_normalize(data)

    assert result.shape == data.shape
    assert result.dtype == np.float32
    assert np.isclose(result.min(), 0.0)
    assert np.isclose(result.max(), 1.0)


def test_gaussian_denoise():
    data = np.zeros(
        (5, 5, 5),
        dtype=np.float32,
    )

    data[2, 2, 2] = 100.0

    result = gaussian_denoise(
        data,
        sigma=1.0,
    )

    assert result.shape == data.shape
    assert result.dtype == np.float32
    assert result[2, 2, 2] < 100.0
    assert result[2, 2, 2] > 0.0


def test_median_denoise():
    data = np.ones(
        (5, 5, 5),
        dtype=np.float32,
    )

    data[2, 2, 2] = 100.0

    result = median_denoise(
        data,
        size=3,
    )

    assert result.shape == data.shape
    assert result.dtype == np.float32
    assert result[2, 2, 2] == 1.0


def test_clahe_enhance():
    data = np.random.default_rng(42).random(
        (32, 32, 16),
        dtype=np.float32,
    )

    result = clahe_enhance(data)

    assert result.shape == data.shape
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_clahe_rejects_non_3d():
    data = np.ones(
        (32, 32),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        clahe_enhance(data)


def test_invalid_gaussian_sigma():
    data = np.ones(
        (5, 5, 5),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        gaussian_denoise(data, sigma=0)


def test_invalid_median_size():
    data = np.ones(
        (5, 5, 5),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        median_denoise(data, size=2)