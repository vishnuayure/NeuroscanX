import numpy as np
import pytest
from src.features.edge_detection import (
    sobel_edges,
    gradient_edges,
    smooth_edges,
)


def test_sobel_edges():
    data = np.random.rand(20, 20, 20).astype(np.float32)
    result = sobel_edges(data)

    assert result.shape == data.shape
    assert result.dtype == np.float32
    assert np.all(result >= 0)
    assert np.max(result) <= 1


def test_gradient_edges():
    data = np.random.rand(20, 20, 20).astype(np.float32)
    result = gradient_edges(data)

    assert result.shape == data.shape
    assert result.dtype == np.float32
    assert np.all(result >= 0)
    assert np.max(result) <= 1


def test_smooth_edges():
    data = np.random.rand(20, 20, 20).astype(np.float32)
    result = smooth_edges(data)

    assert result.shape == data.shape
    assert result.dtype == np.float32


def test_invalid_dimension():
    data = np.random.rand(20, 20).astype(np.float32)

    with pytest.raises(ValueError):
        sobel_edges(data)


def test_invalid_sigma():
    data = np.random.rand(20, 20, 20).astype(np.float32)

    with pytest.raises(ValueError):
        smooth_edges(data, sigma=0)