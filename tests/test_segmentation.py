import numpy as np
import pytest
from src.segmentation.fcm import FCM


def test_fcm_fit():
    data = np.random.default_rng(42).random((20, 20, 10)).astype(np.float32)

    model = FCM(n_clusters=4)
    result = model.fit(data)

    assert result is model
    assert model.centers.shape == (4, 1)
    assert model.membership.shape == (20, 20, 10, 4)
    assert model.labels.shape == data.shape
    assert model.n_iter_ > 0


def test_fcm_membership():
    data = np.random.default_rng(42).random((100, 1)).astype(np.float32)

    model = FCM(n_clusters=3)
    model.fit(data)

    membership = model.membership.reshape(-1, 3)

    assert np.allclose(membership.sum(axis=1), 1.0)
    assert np.all(membership >= 0)
    assert np.all(membership <= 1)


def test_fcm_predict():
    data = np.random.default_rng(42).random((20, 20, 5)).astype(np.float32)

    model = FCM(n_clusters=3)
    model.fit(data)

    labels = model.predict(data)

    assert labels.shape == data.shape
    assert np.all(labels >= 0)
    assert np.all(labels < 3)


def test_fcm_transform():
    data = np.random.default_rng(42).random((50, 1)).astype(np.float32)

    model = FCM(n_clusters=3)
    model.fit(data)

    membership = model.transform(data)

    assert membership.shape == (50, 3)
    assert np.allclose(membership.sum(axis=1), 1.0)


def test_invalid_clusters():
    with pytest.raises(ValueError):
        FCM(n_clusters=1)


def test_invalid_fuzziness():
    with pytest.raises(ValueError):
        FCM(m=1)


def test_invalid_input_dimension():
    model = FCM(n_clusters=3)
    data = np.random.default_rng(42).random((10, 10, 10, 2)).astype(np.float32)

    with pytest.raises(ValueError):
        model.fit(data)


def test_predict_before_fit():
    model = FCM(n_clusters=3)
    data = np.random.default_rng(42).random((10, 1)).astype(np.float32)

    with pytest.raises(RuntimeError):
        model.predict(data)