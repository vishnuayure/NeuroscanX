"""Case export: NIfTI volumes (in the *original* voxel grid), JSON reports, log, screenshots folder."""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import nibabel as nib
import numpy as np

from .nifti_loader import to_original_orientation


def next_case_dir(base_dir: str, prefix: str = "case_") -> str:
    os.makedirs(base_dir, exist_ok=True)
    nums = [int(m.group(1)) for d in os.listdir(base_dir) if (m := re.fullmatch(prefix + r"(\d+)", d))]
    path = os.path.join(base_dir, f"{prefix}{(max(nums) + 1 if nums else 1):03d}")
    os.makedirs(os.path.join(path, "screenshots"), exist_ok=True)
    return path


def save_nifti_like(array: np.ndarray, ref_vol, path: str, dtype=np.float32) -> str:
    """Write `array` (canonical orientation) with the affine/header of the reference input file."""
    arr = to_original_orientation(array, ref_vol).astype(dtype)
    img = nib.Nifti1Image(arr, ref_vol.original_affine)
    img.header.set_zooms(tuple(float(s) for s in np.sqrt((ref_vol.original_affine[:3, :3] ** 2).sum(0))))
    img.set_data_dtype(dtype)
    img.set_qform(ref_vol.original_affine, code=1)
    img.set_sform(ref_vol.original_affine, code=1)
    nib.save(img, path)
    return path


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def export_case(result, out_dir: Optional[str] = None, base_dir: str = "data/outputs") -> str:
    """Write all case artefacts.  Returns the case directory."""
    out_dir = out_dir or next_case_dir(base_dir)
    os.makedirs(os.path.join(out_dir, "screenshots"), exist_ok=True)
    ref = result.mri.reference
    save_nifti_like(result.segmentation, ref, os.path.join(out_dir, "segmentation.nii.gz"), np.uint8)
    save_nifti_like(result.uncertainty, ref, os.path.join(out_dir, "uncertainty.nii.gz"), np.float32)
    save_nifti_like(result.membership.tumor, ref, os.path.join(out_dir, "tumor_membership.nii.gz"), np.float32)
    with open(os.path.join(out_dir, "xai.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": result.xai_summary, "explanation": result.explanation,
                   "explanation_source": result.explanation_source}, f, indent=2, default=_json_default)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(result.report(), f, indent=2, default=_json_default)
    with open(os.path.join(out_dir, "processing.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(result.log))
    return out_dir
