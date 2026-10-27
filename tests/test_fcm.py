import numpy as np
from src.config import FCMConfig
from src.segmentation.fcm import fuzzy_cmeans, memberships_from_centers, fcm_volume
from src.segmentation.spatial_fcm import spatial_fcm, uniform_weights, pair_slices
from src.segmentation.edge_spatial_fcm import edge_spatial_fcm, edge_aware_weights


def _blobs(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    c = np.array([[0.1, 0.1], [0.5, 0.9], [0.9, 0.2]], np.float32)
    X = np.concatenate([c[i] + 0.03 * rng.standard_normal((n, 2)) for i in range(3)]).astype(np.float32)
    return X, c


def test_membership_shape_and_sum():
    X, _ = _blobs()
    r = fuzzy_cmeans(X, 3, seed=1)
    assert r.membership.shape == (3, X.shape[0]) and r.centers.shape == (3, 2)
    np.testing.assert_allclose(r.membership.sum(0), 1.0, atol=1e-4)
    assert r.membership.min() >= 0 and r.membership.max() <= 1 + 1e-6


def test_convergence_and_centers():
    X, c = _blobs()
    r = fuzzy_cmeans(X, 3, seed=1, tol=1e-5, max_iter=300)
    assert r.converged and r.n_iter < 300
    d = np.linalg.norm(r.centers[:, None] - c[None], axis=2).min(1)   # each centre close to a true blob centre
    assert d.max() < 0.02
    assert r.objective_history[-1] <= r.objective_history[0] + 1e-6   # objective decreases


def test_deterministic_seed():
    X, _ = _blobs()
    a, b = fuzzy_cmeans(X, 3, seed=7), fuzzy_cmeans(X, 3, seed=7)
    np.testing.assert_array_equal(a.centers, b.centers)
    np.testing.assert_array_equal(a.membership, b.membership)


def test_intensity_memberships_at_centers():
    V = np.array([[0, 0], [1, 1]], np.float32)
    U = memberships_from_centers(np.array([[0, 0], [1, 1], [.5, .5]], np.float32), V)
    assert U[0, 0] > 0.999 and U[1, 1] > 0.999 and abs(U[0, 2] - 0.5) < 1e-4


def _vol(seed=0, shape=(24, 24, 20), noise=0.12):
    rng = np.random.default_rng(seed)
    lab = np.zeros(shape, int); lab[8:16, 8:16, 6:14] = 1
    f = np.stack([lab * 0.5 + 0.2 + noise * rng.standard_normal(shape) for _ in range(3)]).astype(np.float32)
    return f, np.ones(shape, bool), lab


def test_spatial_solvers_valid_and_denoise():
    f, mask, lab = _vol()
    cfg = FCMConfig(n_clusters=2, max_iter=60)
    w = fcm_volume(f, mask, cfg)
    s = spatial_fcm(f, mask, w.centers, cfg)
    e = edge_spatial_fcm(f, mask, np.zeros(mask.shape, np.float32), w.centers, cfg)
    for r in (s, e):
        assert r.membership.shape == (2,) + mask.shape
        np.testing.assert_allclose(r.membership.sum(0), 1.0, atol=1e-4)
        assert r.converged
    err = lambda r: (r.membership.argmax(0) != (lab if r.centers[1].mean() > r.centers[0].mean() else 1 - lab)).mean()
    assert err(s) < err(w)            # spatial penalty reduces isolated mis-assignments on noisy data


def test_edge_weights_suppressed_across_edges():
    f, mask, lab = _vol(noise=0.01)
    edges = np.zeros(mask.shape, np.float32); edges[7:9, :, :] = 1.0     # strong edge at x≈8
    uni = uniform_weights(mask, 6)
    nb = edge_aware_weights(f, mask, edges, 6, edge_penalty=4.0, range_weight=1.0)
    d, s, W = nb.pairs[0]               # x-direction pairs
    assert W[7, 10, 10] < 0.1 * uni.pairs[0][2][7, 10, 10]
    assert W[2, 2, 2] > 0.9             # homogeneous region keeps full weight


def test_pair_slices_cover_neighbours():
    d, s = pair_slices((1, -1, 0), (5, 5, 5))
    a = np.arange(125).reshape(5, 5, 5)
    assert a[d].shape == a[s].shape == (4, 4, 5)
