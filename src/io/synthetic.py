"""
Synthetic BraTS case generator for testing, prototyping, and demonstration.
Generates 3D NIfTI volumes for T1, T1ce, T2, FLAIR, and ground-truth segmentation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import nibabel as nib
import numpy as np


def generate_synthetic_brats_data(
    shape: Tuple[int, int, int] = (64, 64, 40),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Generate synthetic multimodal MRI arrays with realistic BraTS intensity signatures.

    Returns dict mapping 't1', 't1ce', 't2', 'flair', 'ground_truth' to 3D arrays.
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape

    # Coordinate grid centered at origin
    x = np.linspace(-1, 1, nx)[:, None, None]
    y = np.linspace(-1, 1, ny)[None, :, None]
    z = np.linspace(-1, 1, nz)[None, None, :]

    # Ellipsoidal brain mask
    brain_radius = (x / 0.75) ** 2 + (y / 0.85) ** 2 + (z / 0.7) ** 2
    brain_mask = brain_radius <= 1.0

    # Internal brain structures: White Matter & Gray Matter
    wm_mask = brain_mask & (brain_radius <= 0.65)
    gm_mask = brain_mask & (~wm_mask)

    # Ventricles (CSF)
    ventricle_mask = (
        brain_mask
        & ((x / 0.15) ** 2 + (y / 0.4) ** 2 + (z / 0.2) ** 2 <= 1.0)
    )
    wm_mask = wm_mask & (~ventricle_mask)

    # Tumour region: centered in right hemisphere
    tx, ty, tz = 0.25, 0.1, 0.0
    r_tumour = ((x - tx) / 0.22) ** 2 + ((y - ty) / 0.25) ** 2 + ((z - tz) / 0.22) ** 2

    # Edema (outer shell) and core (inner)
    edema_mask = brain_mask & (r_tumour <= 1.0) & (r_tumour > 0.4)
    enhancing_core = brain_mask & (r_tumour <= 0.4) & (r_tumour > 0.15)
    necrotic_core = brain_mask & (r_tumour <= 0.15)
    whole_tumour = edema_mask | enhancing_core | necrotic_core

    # Standard healthy tissue baseline intensities
    # Modality: (WM, GM, CSF)
    # T1: (0.7, 0.5, 0.15)
    # T1ce: (0.7, 0.5, 0.15)
    # T2: (0.4, 0.6, 0.95)
    # FLAIR: (0.4, 0.5, 0.05)

    def base_volume(wm_val, gm_val, csf_val, noise_std=0.03):
        vol = np.zeros(shape, dtype=np.float32)
        vol[gm_mask] = gm_val
        vol[wm_mask] = wm_val
        vol[ventricle_mask] = csf_val
        noise = rng.normal(0, noise_std, shape).astype(np.float32)
        vol[brain_mask] += noise[brain_mask]
        return np.clip(vol, 0.0, 1.5)

    t1 = base_volume(0.70, 0.52, 0.15)
    t1ce = base_volume(0.70, 0.52, 0.15)
    t2 = base_volume(0.40, 0.60, 0.95)
    flair = base_volume(0.42, 0.50, 0.08)

    # Tumour contrast profiles:
    # T1: hypointense whole tumour
    t1[edema_mask] = 0.35 + rng.normal(0, 0.02, int(np.sum(edema_mask)))
    t1[enhancing_core] = 0.40 + rng.normal(0, 0.02, int(np.sum(enhancing_core)))
    t1[necrotic_core] = 0.25 + rng.normal(0, 0.02, int(np.sum(necrotic_core)))

    # T1ce: hyperintense enhancing core
    t1ce[edema_mask] = 0.38 + rng.normal(0, 0.02, int(np.sum(edema_mask)))
    t1ce[enhancing_core] = 0.95 + rng.normal(0, 0.02, int(np.sum(enhancing_core)))
    t1ce[necrotic_core] = 0.30 + rng.normal(0, 0.02, int(np.sum(necrotic_core)))

    # T2: hyperintense edema and core
    t2[edema_mask] = 0.85 + rng.normal(0, 0.02, int(np.sum(edema_mask)))
    t2[enhancing_core] = 0.78 + rng.normal(0, 0.02, int(np.sum(enhancing_core)))
    t2[necrotic_core] = 0.90 + rng.normal(0, 0.02, int(np.sum(necrotic_core)))

    # FLAIR: hyperintense whole tumour, CSF suppressed
    flair[edema_mask] = 0.92 + rng.normal(0, 0.02, int(np.sum(edema_mask)))
    flair[enhancing_core] = 0.88 + rng.normal(0, 0.02, int(np.sum(enhancing_core)))
    flair[necrotic_core] = 0.80 + rng.normal(0, 0.02, int(np.sum(necrotic_core)))

    # Ground truth: BraTS label convention
    # 1: necrotic core, 2: edema, 4: enhancing core
    gt = np.zeros(shape, dtype=np.int16)
    gt[necrotic_core] = 1
    gt[edema_mask] = 2
    gt[enhancing_core] = 4

    return {
        "t1": np.clip(t1, 0.0, 2.0).astype(np.float32),
        "t1ce": np.clip(t1ce, 0.0, 2.0).astype(np.float32),
        "t2": np.clip(t2, 0.0, 2.0).astype(np.float32),
        "flair": np.clip(flair, 0.0, 2.0).astype(np.float32),
        "ground_truth": gt,
    }


def generate_synthetic_brats_case(
    output_dir: Union[str, Path],
    case_name: str = "BraTS_sample",
    shape: Tuple[int, int, int] = (64, 64, 40),
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int = 42,
) -> dict[str, Path]:
    """
    Generate synthetic BraTS NIfTI files and save them to output_dir.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_brats_data(shape=shape, seed=seed)

    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])

    paths: dict[str, Path] = {}
    for modality, array in data.items():
        if modality == "ground_truth":
            file_name = f"{case_name}_seg.nii.gz"
        else:
            file_name = f"{case_name}_{modality}.nii.gz"

        file_path = out / file_name
        nii = nib.Nifti1Image(array, affine=affine)
        nib.save(nii, str(file_path))
        paths[modality] = file_path

    return paths


__all__ = [
    "generate_synthetic_brats_data",
    "generate_synthetic_brats_case",
]
