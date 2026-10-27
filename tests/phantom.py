"""Synthetic multimodal glioma phantom for TESTS / developer smoke tests only.

Never used by the application UI – the app only ever shows files the user selects.
Tissue signal model (arbitrary units, BraTS-like contrasts): WM, GM, CSF ventricles,
tumor oedema (FLAIR/T2 bright) and a T1ce-enhancing core.
"""
from __future__ import annotations

import os

import nibabel as nib
import numpy as np

TISSUE = {  # T1, T1ce, T2, FLAIR
    "wm": (0.80, 0.80, 0.40, 0.40),
    "gm": (0.60, 0.60, 0.55, 0.50),
    "csf": (0.10, 0.10, 0.95, 0.10),
    "edema": (0.50, 0.50, 0.90, 0.95),
    "core": (0.40, 1.00, 0.80, 0.80),
}


def _ell(shape, c, r):
    g = np.indices(shape).astype(np.float32)
    return sum(((g[i] - c[i]) / r[i]) ** 2 for i in range(3)) <= 1.0


def make_phantom(shape=(64, 72, 48), noise=0.04, seed=0, spacing=(1.0, 1.0, 1.0), tumor_center=None):
    rng = np.random.default_rng(seed)
    cx = (shape[0] / 2, shape[1] / 2, shape[2] / 2)
    brain = _ell(shape, cx, (shape[0] * 0.42, shape[1] * 0.42, shape[2] * 0.42))
    wm = _ell(shape, cx, (shape[0] * 0.34, shape[1] * 0.34, shape[2] * 0.33))
    vent = _ell(shape, cx, (shape[0] * 0.08, shape[1] * 0.14, shape[2] * 0.10))
    tc = tumor_center or (cx[0] + shape[0] * 0.14, cx[1] + shape[1] * 0.08, cx[2])
    edema = _ell(shape, tc, (shape[0] * 0.17, shape[1] * 0.16, shape[2] * 0.20)) & brain
    core = _ell(shape, tc, (shape[0] * 0.08, shape[1] * 0.08, shape[2] * 0.10)) & brain

    lab = np.zeros(shape, np.int8)           # 0 bg 1 gm 2 wm 3 csf 4 edema 5 core
    lab[brain] = 1
    lab[wm & brain] = 2
    lab[vent & brain] = 3
    lab[edema] = 4
    lab[core] = 5
    mods = np.zeros((4,) + shape, np.float32)
    for code, key in enumerate(["gm", "wm", "csf", "edema", "core"], start=1):
        sel = lab == code
        for m in range(4):
            mods[m][sel] = TISSUE[key][m] * 1000.0
    for m in range(4):  # smooth bias field + noise (Rician-like magnitude noise)
        g = np.indices(shape).astype(np.float32)
        bias = 1.0 + 0.08 * np.sin(g[0] / shape[0] * np.pi) * np.cos(g[1] / shape[1] * np.pi)
        v = mods[m] * bias
        n1 = rng.normal(0, noise * 1000, shape)
        n2 = rng.normal(0, noise * 1000, shape)
        v = np.sqrt((v + n1) ** 2 + n2 ** 2)
        v[~brain] = 0
        mods[m] = v.astype(np.float32)
    gt = np.zeros(shape, np.uint8)
    gt[edema] = 2
    gt[core] = 4
    return {"T1": mods[0], "T1ce": mods[1], "T2": mods[2], "FLAIR": mods[3]}, gt, np.diag(list(spacing) + [1.0])


def write_case(folder: str, name="case", **kw) -> dict:
    os.makedirs(folder, exist_ok=True)
    mods, gt, aff = make_phantom(**kw)
    paths = {}
    for k, v in mods.items():
        p = os.path.join(folder, f"{name}_{k.lower()}.nii.gz")
        nib.save(nib.Nifti1Image(v, aff), p)
        paths[k] = p
    gp = os.path.join(folder, f"{name}_seg.nii.gz")
    nib.save(nib.Nifti1Image(gt, aff), gp)
    paths["GT"] = gp
    return paths
