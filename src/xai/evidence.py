"""Evidence extraction: *why* is a voxel / region part of the tumor segmentation?

Four kinds of evidence, every one computed from the actual clustering outputs:
  intensity   – tumor membership obtained from the final centres using intensities only (no neighbours)
  spatial     – 3×3×3 neighbourhood agreement + connected-component compactness; and how much the spatial
                penalty / post-processing changed the intensity-only decision
  edge        – boundary-strength support along the segmentation margin
  membership  – strength of the final fuzzy tumor membership
All reported scores are in [0, 1].
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..segmentation.fcm import memberships_from_centers
from .context import XAIContext


def intensity_tumor_membership(ctx: XAIContext) -> np.ndarray:
    """Tumor membership (box) from final centres + intensities only (cached)."""
    if "t_int" in ctx.cache:
        return ctx.cache["t_int"]
    X = ctx.features[:, ctx.brain].T
    U = memberships_from_centers(X, ctx.centers, ctx.m)
    t = np.zeros(ctx.brain.shape, np.float32)
    if ctx.selected:
        t[ctx.brain] = U[ctx.selected].sum(0)
    ctx.cache["t_int"] = t
    return t


def boundary_band(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask)
    st = ndi.generate_binary_structure(3, 1)
    dil = ndi.binary_dilation(mask, st, iterations=thickness)
    ero = ndi.binary_erosion(mask, st, iterations=thickness)
    return dil & ~ero


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask)
    return mask & ~ndi.binary_erosion(mask, ndi.generate_binary_structure(3, 1))


def compute_global_evidence(ctx: XAIContext) -> dict:
    tumor, brain = ctx.tumor, ctx.brain
    if not tumor.any():
        return {"available": False}
    t_int = intensity_tumor_membership(ctx)
    int_mask = (t_int >= 0.5) & brain
    ev = {"available": True}
    ev["intensity_evidence"] = float(np.clip(t_int[tumor].mean(), 0, 1))
    ev["membership_strength"] = float(np.clip(ctx.tumor_membership[tumor].mean(), 0, 1))
    # spatial coherence
    agree = ndi.uniform_filter(tumor.astype(np.float32), size=3, mode="constant")
    ev["spatial_coherence"] = float(agree[tumor].mean())
    lab, n = ndi.label(tumor)
    sizes = ndi.sum(tumor, lab, index=np.arange(1, n + 1)) if n else np.array([0])
    ev["n_components"] = int(n)
    ev["largest_component_fraction"] = float(sizes.max() / sizes.sum())
    union = int((int_mask | tumor).sum())
    ev["spatial_changed_voxels"] = int((int_mask ^ tumor).sum())
    ev["spatial_changed_fraction"] = float(ev["spatial_changed_voxels"] / max(union, 1))
    ev["spatial_evidence"] = float(np.clip(0.5 * ev["spatial_coherence"] + 0.5 * ev["largest_component_fraction"], 0, 1))
    # edge support along the boundary
    band = boundary_band(tumor, ctx.cfg.boundary_band_voxels)
    edges_b = ctx.edges[band]
    ref = float(np.percentile(ctx.edges[brain], 75))
    ev["boundary_edge_mean"] = float(edges_b.mean()) if edges_b.size else 0.0
    ev["interior_edge_mean"] = float(ctx.edges[tumor & ~band].mean()) if (tumor & ~band).any() else 0.0
    ev["edge_reference_p75"] = ref
    ev["edge_support"] = float((edges_b >= ref).mean()) if edges_b.size else 0.0
    ev["edge_evidence"] = ev["edge_support"]
    return ev


class LocalEvidenceExplainer:
    """'Why here?' for a single voxel (box coordinates).  Caches the distance transform."""

    def __init__(self, ctx: XAIContext):
        self.ctx = ctx
        self.t_int = intensity_tumor_membership(ctx)
        self.agree = ndi.uniform_filter(ctx.tumor_membership, size=3, mode="constant")
        self.dist_in = ndi.distance_transform_edt(ctx.tumor) if ctx.tumor.any() else np.zeros(ctx.tumor.shape)
        self.dist_out = ndi.distance_transform_edt(~ctx.tumor) if ctx.tumor.any() else np.zeros(ctx.tumor.shape)
        self.sp = ctx.spacing
        # reference vectors: tumor centroid vs. dominant normal tissue (volume-weighted)
        sel = ctx.selected
        lab = ctx.membership.argmax(0)
        vols = np.array([int(((lab == k) & ctx.brain).sum()) for k in range(ctx.centers.shape[0])], float)
        self.tumor_ref = (ctx.centers[sel] * vols[sel, None]).sum(0) / max(vols[sel].sum(), 1) if sel else None
        normal = [k for k in range(ctx.centers.shape[0]) if k not in sel]
        self.normal_centers = ctx.centers[normal] if normal else None

    def at(self, ijk) -> dict:
        c = self.ctx
        i, j, k = (int(v) for v in ijk)
        if not (0 <= i < c.brain.shape[0] and 0 <= j < c.brain.shape[1] and 0 <= k < c.brain.shape[2]):
            return {"inside_brain": False}
        if not c.brain[i, j, k]:
            return {"inside_brain": False}
        x = c.features[:, i, j, k]
        z = (x - c.feature_median) / np.maximum(c.feature_scale, 1e-6)
        out = {
            "inside_brain": True,
            "in_tumor": bool(c.tumor[i, j, k]),
            "tumor_membership": float(c.tumor_membership[i, j, k]),
            "intensity_membership": float(self.t_int[i, j, k]),
            "uncertainty": float(c.uncertainty[i, j, k]),
            "edge_strength": float(c.edges[i, j, k]),
            "neighbourhood_support": float(self.agree[i, j, k]),
            "z": {n: float(z[a]) for a, n in enumerate(c.names)},
            "distance_to_boundary_mm": float((self.dist_in[i, j, k] if c.tumor[i, j, k] else self.dist_out[i, j, k])
                                             * float(np.mean(self.sp))),
            "best_cluster": int(c.membership[:, i, j, k].argmax()),
        }
        # per-modality local contribution: how much each modality moves this voxel closer to the tumor
        # reference than to its nearest normal centre (positive part, normalised)
        if self.tumor_ref is not None and self.normal_centers is not None:
            d_norm = ((x[None, :] - self.normal_centers) ** 2)
            nearest = d_norm.sum(1).argmin()
            adv = d_norm[nearest] - (x - self.tumor_ref) ** 2
            pos = np.maximum(adv, 0)
            tot = pos.sum()
            share = pos / tot if tot > 1e-9 else np.zeros_like(pos)
            out["local_contribution"] = {n: float(share[a]) for a, n in enumerate(c.names)}
        else:
            out["local_contribution"] = {n: 0.0 for n in c.names}
        out["local_explanation"] = local_text(out)
        return out


def local_text(loc: dict) -> str:
    if not loc.get("inside_brain"):
        return "Outside the brain mask."
    lc = loc["local_contribution"]
    top = sorted(lc, key=lc.get, reverse=True)[:2]
    where = "Inside" if loc["in_tumor"] else "Outside"
    s = (f"{where} the segmented region · tumor membership {loc['tumor_membership']:.2f} · "
         f"{loc['distance_to_boundary_mm']:.1f} mm from the boundary.")
    if loc["in_tumor"] and loc["tumor_membership"] < 0.5:
        s += " Included by post-processing (hole filling / closing), not by fuzzy membership alone."
    if loc["in_tumor"] or loc["tumor_membership"] > 0.2:
        s += f" Strongest local evidence: {top[0]} (z={loc['z'][top[0]]:+.1f})" + (f", {top[1]} (z={loc['z'][top[1]]:+.1f})." if len(top) > 1 else ".")
    return s
