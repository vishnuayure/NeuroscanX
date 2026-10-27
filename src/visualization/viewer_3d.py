"""3-D surfaces generated from the *actual* volumes (marching cubes) – no placeholder geometry.

Coordinates are millimetres in the canonical RAS+ array frame (x→Right, y→Anterior, z→Superior).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

log = logging.getLogger("neuroscanx")


def _to_polydata(verts_vox: np.ndarray, faces: np.ndarray, spacing, scalars: dict):
    import pyvista as pv
    verts = verts_vox * np.asarray(spacing, float)[None, :]
    f = np.hstack([np.full((faces.shape[0], 1), 3, np.int64), faces.astype(np.int64)]).ravel()
    mesh = pv.PolyData(verts.astype(np.float32), f)
    for k, v in scalars.items():
        mesh.point_data[k] = v.astype(np.float32)
    return mesh


def mask_to_mesh(mask: np.ndarray, spacing, sigma: float = 1.0, step: int = 1, level: float = 0.5,
                 scalar_volumes: Optional[dict] = None, smooth_iters: int = 12):
    """Binary mask → smooth triangle surface (pyvista PolyData) or None if the mask is empty.

    scalar_volumes = {name: 3-D array}: sampled at the vertices (trilinear) and attached as point data.
    """
    if mask is None or not mask.any():
        return None
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - 3, 0)
    hi = np.minimum(idx.max(0) + 4, mask.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    sub = ndi.gaussian_filter(mask[sl].astype(np.float32), sigma) if sigma > 0 else mask[sl].astype(np.float32)
    sub = np.pad(sub, 1)
    if sub.max() <= level:
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(sub, level=level, step_size=step)
    except (ValueError, RuntimeError):
        return None
    verts = verts - 1 + lo[None, :]                                  # back to full-volume voxel coords
    scalars = {}
    for name, vol in (scalar_volumes or {}).items():
        scalars[name] = ndi.map_coordinates(vol, verts.T, order=1, mode="nearest")
    mesh = _to_polydata(verts, faces, spacing, scalars)
    if smooth_iters:
        mesh = mesh.smooth(n_iter=smooth_iters, relaxation_factor=0.1)
    return mesh.compute_normals(auto_orient_normals=False, split_vertices=False)


def build_meshes(result) -> dict:
    """Brain surface, tumor surface (+uncertainty/edge/membership scalars) and high-confidence core surface."""
    sp = result.mri.voxel_spacing
    seg = result.segmentation.astype(bool)
    unc, edges, t = result.uncertainty, result.edges, result.membership.tumor
    # local uncertainty / boundary-evidence around each surface vertex (5³ and 3³ windows)
    local_unc = ndi.uniform_filter(unc, size=5)
    local_edge = ndi.maximum_filter(edges, size=3)
    out = {}
    out["brain"] = mask_to_mesh(result.preprocessing.brain_mask, sp, sigma=1.5, step=2, smooth_iters=15)
    out["tumor"] = mask_to_mesh(seg, sp, sigma=0.8, step=1, scalar_volumes={
        "uncertainty": local_unc, "edge": local_edge, "membership": t})
    core = seg & (t >= 0.9)
    out["core"] = mask_to_mesh(core, sp, sigma=0.8, step=1) if core.sum() > 30 else None
    log.info("3D meshes: brain %s faces, tumor %s faces",
             out["brain"].n_cells if out["brain"] is not None else 0, out["tumor"].n_cells if out["tumor"] is not None else 0)
    return out


def voxel_to_world(ijk, spacing):
    return tuple(float(a) * float(s) for a, s in zip(ijk, spacing))
