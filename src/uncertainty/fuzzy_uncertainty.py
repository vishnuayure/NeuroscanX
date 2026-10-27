"""Voxel-level uncertainty derived from fuzzy memberships.

  tumor membership  t_i = Σ_{k ∈ tumor clusters} u_ki
  uncertainty       linear index of fuzziness  1 − |2t − 1|  ∈ [0,1]      (tumor-vs-background ambiguity, default)
                    or binary entropy H_b(t) (option "entropy"; saturates early for soft m=2 memberships)
  cluster entropy   H(u_i)/log C  ∈ [0,1]                              (ambiguity between *all* clusters)
  margin            u_(1) − u_(2)                                       (top-2 membership gap)

`uncertainty` is the primary map: it is 0 where the tumor/non-tumor decision is
unambiguous, 1 where t = 0.5.  Confidence classes:
  HIGH CONFIDENCE  u < confident_threshold   |  MODERATE  |  UNCERTAIN  u ≥ uncertain_threshold
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-7


def binary_entropy(t: np.ndarray) -> np.ndarray:
    t = np.clip(t.astype(np.float32), _EPS, 1 - _EPS)
    return (-(t * np.log2(t) + (1 - t) * np.log2(1 - t))).astype(np.float32)


def cluster_entropy(U: np.ndarray) -> np.ndarray:
    """Normalised Shannon entropy of the membership vector, U shape (C, ...)."""
    C = U.shape[0]
    u = np.clip(U.astype(np.float32), _EPS, 1.0)
    return (-(u * np.log(u)).sum(0) / np.log(C)).astype(np.float32)


def membership_margin(U: np.ndarray) -> np.ndarray:
    s = np.sort(U, axis=0)
    return (s[-1] - s[-2]).astype(np.float32)


def linear_fuzziness(t: np.ndarray) -> np.ndarray:
    """Linear index of fuzziness of the tumor fuzzy set: 0 at t∈{0,1}, 1 at t=0.5."""
    return (1.0 - np.abs(2.0 * np.clip(t.astype(np.float32), 0.0, 1.0) - 1.0)).astype(np.float32)


def tumor_uncertainty(tumor_membership: np.ndarray, brain_mask: np.ndarray, measure: str = "linear") -> np.ndarray:
    u = linear_fuzziness(tumor_membership) if measure == "linear" else binary_entropy(tumor_membership)
    u[~brain_mask] = 0.0
    return u


def confidence_classes(u: np.ndarray, confident: float = 0.2, uncertain: float = 0.5) -> np.ndarray:
    """0 = high confidence, 1 = moderate, 2 = uncertain."""
    out = np.ones(u.shape, np.uint8)
    out[u < confident] = 0
    out[u >= uncertain] = 2
    return out


def summarize_uncertainty(u: np.ndarray, tumor: np.ndarray, confident: float = 0.2, uncertain: float = 0.5) -> dict:
    if not tumor.any():
        return {"mean_uncertainty_tumor": None, "confidence": None, "fraction_high_confidence": None,
                "fraction_moderate": None, "fraction_uncertain": None}
    v = u[tumor]
    return {
        "mean_uncertainty_tumor": float(v.mean()),
        "confidence": float(1.0 - v.mean()),
        "fraction_high_confidence": float((v < confident).mean()),
        "fraction_moderate": float(((v >= confident) & (v < uncertain)).mean()),
        "fraction_uncertain": float((v >= uncertain).mean()),
    }
