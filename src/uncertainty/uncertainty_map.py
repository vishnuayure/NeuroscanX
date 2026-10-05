"""
uncertainty_map.py - NeuroScanX P3 (Evaluation & Uncertainty)

Voxel-wise uncertainty from fuzzy membership maps produced by P2
(FCM / Spatial FCM / Edge-SP-FCM).

Conventions
-----------
* ``membership`` has shape ``(C, ...)``: axis 0 is the cluster/class axis and
  the remaining axes are spatial, typically ``(C, X, Y, Z)``.
  Values are fuzzy memberships in [0, 1] that sum to 1 over axis 0.
* Every uncertainty map returned has the spatial shape ``(X, Y, Z)``,
  dtype float32, and values in [0, 1] (0 = certain, 1 = maximally ambiguous).
  That is the form P4 expects for overlaying on the MRI.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

EPS = 1e-12

__all__ = [
    "sanitize_membership",
    "entropy_uncertainty",
    "max_membership_uncertainty",
    "margin_uncertainty",
    "compute_uncertainty",
    "confidence_map",
    "normalize_uncertainty",
    "get_high_uncertainty_mask",
    "get_uncertainty_statistics",
    "save_uncertainty_nifti",
]


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def sanitize_membership(membership: np.ndarray, on_invalid: str = "fix") -> np.ndarray:
    """Validate a membership array and return a clean float64 copy.

    Parameters
    ----------
    membership : array of shape (C, ...), C >= 2.
    on_invalid : "fix" or "raise".
        * "fix":   NaN/Inf -> 0, values clipped to [0, 1], and every voxel
                   renormalised to sum to 1 over axis 0. A voxel whose
                   memberships are all zero becomes uniform (1/C), i.e.
                   maximally uncertain, because nothing is known about it.
        * "raise": raise ValueError if any value is non-finite or outside
                   [0, 1] (tolerance 1e-6).
    """
    if on_invalid not in ("fix", "raise"):
        raise ValueError("on_invalid must be 'fix' or 'raise'")
    m = np.asarray(membership, dtype=np.float64)
    if m.ndim < 2:
        raise ValueError(
            f"membership must have shape (C, ...) with at least 2 dims, got {m.shape}"
        )
    if m.shape[0] < 2:
        raise ValueError("membership needs at least 2 classes on axis 0")

    bad = ~np.isfinite(m)
    out_of_range = (m < -1e-6) | (m > 1 + 1e-6)
    if on_invalid == "raise":
        if bad.any():
            raise ValueError("membership contains NaN or Inf")
        if out_of_range.any():
            raise ValueError("membership values must lie in [0, 1]")

    m = np.where(bad, 0.0, m)
    m = np.clip(m, 0.0, 1.0)
    total = m.sum(axis=0, keepdims=True)
    zero = total <= EPS
    m = np.where(zero, 1.0 / m.shape[0], m)
    total = np.where(zero, 1.0, m.sum(axis=0, keepdims=True))
    return m / total


def _finish(u: np.ndarray) -> np.ndarray:
    return np.clip(u, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Uncertainty measures
# --------------------------------------------------------------------------- #
def entropy_uncertainty(
    membership: np.ndarray, normalize: bool = True, on_invalid: str = "fix"
) -> np.ndarray:
    """Shannon entropy ``-sum(u * log u)`` per voxel.

    With ``normalize=True`` (default) the entropy is divided by ``log(C)`` so the
    result lies in [0, 1]: one-hot memberships give 0, uniform give 1.
    Uses ``xlogy`` so memberships of exactly 0 contribute 0 (no log(0) issues).
    """
    from scipy.special import xlogy

    m = sanitize_membership(membership, on_invalid)
    h = -xlogy(m, m).sum(axis=0)
    if normalize:
        return _finish(h / np.log(m.shape[0]))
    return np.maximum(h, 0.0).astype(np.float32)   # raw nats, range [0, log C]


def max_membership_uncertainty(membership: np.ndarray, on_invalid: str = "fix") -> np.ndarray:
    """``1 - max_c u_c``, rescaled so that uniform memberships give 1.

    The raw value ranges from 0 to ``1 - 1/C``; dividing by ``1 - 1/C`` makes it
    comparable across different numbers of clusters.
    """
    m = sanitize_membership(membership, on_invalid)
    c = m.shape[0]
    return _finish((1.0 - m.max(axis=0)) / (1.0 - 1.0 / c))


def margin_uncertainty(membership: np.ndarray, on_invalid: str = "fix") -> np.ndarray:
    """``1 - (top1 - top2)``: small gap between the two best classes = ambiguous."""
    m = sanitize_membership(membership, on_invalid)
    s = np.sort(m, axis=0)
    return _finish(1.0 - (s[-1] - s[-2]))


_METHODS = {
    "entropy": entropy_uncertainty,
    "max": max_membership_uncertainty,
    "margin": margin_uncertainty,
}


def compute_uncertainty(
    membership: np.ndarray, method: str = "entropy", on_invalid: str = "fix"
) -> np.ndarray:
    """Dispatch to one of ``"entropy"``, ``"max"`` or ``"margin"``."""
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {sorted(_METHODS)}")
    fn = _METHODS[method]
    return fn(membership, on_invalid=on_invalid)


def confidence_map(membership: np.ndarray, on_invalid: str = "fix") -> np.ndarray:
    """Per-voxel confidence = highest membership value (in [1/C, 1])."""
    m = sanitize_membership(membership, on_invalid)
    return m.max(axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Post-processing & statistics
# --------------------------------------------------------------------------- #
def normalize_uncertainty(
    uncertainty: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Min-max scale to [0, 1] (float32).

    NaN/Inf are treated as 0. If the map is constant (``vmax == vmin``) the
    result is all zeros. Pass fixed ``vmin``/``vmax`` to put several maps on
    the same scale.
    """
    u = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(u.min()) if vmin is None else float(vmin)
    hi = float(u.max()) if vmax is None else float(vmax)
    if hi - lo <= EPS:
        return np.zeros(u.shape, dtype=np.float32)
    return _finish((u - lo) / (hi - lo))


def get_high_uncertainty_mask(uncertainty: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Boolean mask of voxels with ``uncertainty >= threshold``. NaN -> False."""
    u = np.asarray(uncertainty, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(u) & (u >= threshold)


def get_uncertainty_statistics(
    uncertainty: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Summary statistics, optionally restricted to a boolean ``mask``.

    Returns mean, std, min, max, median, the percentage of voxels with
    uncertainty >= ``threshold``, and the number of voxels used. An empty
    region yields NaN statistics and ``n_voxels = 0`` (not an exception).
    Non-finite voxels are ignored.
    """
    u = np.asarray(uncertainty, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        if mask.shape != u.shape:
            raise ValueError(f"mask shape {mask.shape} != uncertainty shape {u.shape}")
        u = u[mask]
    else:
        u = u.ravel()
    u = u[np.isfinite(u)]

    if u.size == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "min": nan, "max": nan, "median": nan,
                "pct_high": nan, "n_voxels": 0}
    return {
        "mean": float(u.mean()),
        "std": float(u.std()),
        "min": float(u.min()),
        "max": float(u.max()),
        "median": float(np.median(u)),
        "pct_high": float(100.0 * np.mean(u >= threshold)),
        "n_voxels": int(u.size),
    }


# --------------------------------------------------------------------------- #
# Hand-off to P4
# --------------------------------------------------------------------------- #
def save_uncertainty_nifti(uncertainty: np.ndarray, affine: np.ndarray, path: str) -> None:
    """Save an uncertainty volume as float32 NIfTI using the MRI's affine.

    Requires ``nibabel`` (already needed by P1's NIfTI loader).
    """
    import nibabel as nib

    u = np.nan_to_num(np.asarray(uncertainty, dtype=np.float32))
    nib.save(nib.Nifti1Image(u, np.asarray(affine)), path)
