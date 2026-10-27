"""Boundary analysis: where is the tumor margin and how certain is it?"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .context import XAIContext
from .evidence import boundary_band

# canonical RAS+ array: axis 0 → Right, axis 1 → Anterior, axis 2 → Superior
_SIDES = {(0, 1): "right", (0, -1): "left", (1, 1): "anterior", (1, -1): "posterior", (2, 1): "superior", (2, -1): "inferior"}


def analyze_boundary(ctx: XAIContext):
    """Returns (dict summary, dict of arrays {band, uncertain_band})."""
    tumor = ctx.tumor
    if not tumor.any():
        return {"available": False}, {"band": np.zeros_like(tumor), "uncertain_band": np.zeros_like(tumor)}
    band = boundary_band(tumor, ctx.cfg.boundary_band_voxels)
    unc, edges = ctx.uncertainty, ctx.edges
    core = ndi.binary_erosion(tumor, ndi.generate_binary_structure(3, 1), iterations=ctx.cfg.boundary_band_voxels + 1)
    ub = unc[band]
    uncertain_band = band & (unc >= ctx.cfg.uncertain_threshold)
    # sharpness: magnitude of the tumor-membership gradient on the band (steep = well-defined margin)
    gm = np.sqrt(sum(g ** 2 for g in np.gradient(ctx.tumor_membership)))
    out = {
        "available": True,
        "boundary_voxels": int(band.sum()),
        "boundary_uncertainty": float(ub.mean()),
        "boundary_confidence": float(1.0 - ub.mean()),
        "fraction_uncertain_boundary": float((ub >= ctx.cfg.uncertain_threshold).mean()),
        "core_uncertainty": float(unc[core].mean()) if core.any() else None,
        "boundary_edge_mean": float(edges[band].mean()),
        "boundary_sharpness": float(gm[band].mean()),
    }
    # direction-resolved analysis relative to the tumor centroid (physical mm)
    coords = np.argwhere(band)
    cen = np.array(ndi.center_of_mass(tumor))
    off = (coords - cen) * np.array(ctx.spacing)[None, :]
    axis = np.abs(off).argmax(1)
    sign = np.sign(off[np.arange(len(off)), axis]).astype(int)
    sign[sign == 0] = 1
    sides = {}
    for (ax, sg), name in _SIDES.items():
        sel = (axis == ax) & (sign == sg)
        n = int(sel.sum())
        if n == 0:
            continue
        c = coords[sel]
        sides[name] = {"n": n, "mean_uncertainty": float(unc[c[:, 0], c[:, 1], c[:, 2]].mean()),
                       "mean_edge": float(edges[c[:, 0], c[:, 1], c[:, 2]].mean())}
    min_n = max(30, int(0.02 * len(coords)))
    valid = {k: v for k, v in sides.items() if v["n"] >= min_n}
    out["sides"] = sides
    if valid:
        out["most_uncertain_side"] = max(valid, key=lambda k: valid[k]["mean_uncertainty"])
        out["most_certain_side"] = min(valid, key=lambda k: valid[k]["mean_uncertainty"])
    out["centroid_voxel"] = [float(v) for v in cen]
    return out, {"band": band, "uncertain_band": uncertain_band}
