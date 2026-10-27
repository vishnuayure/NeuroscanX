"""NIfTI loading, canonical orientation handling and robust case validation.

Volumes are reoriented to RAS+ on load (voxel axis 0 → patient Right, 1 → Anterior,
2 → Superior).  The original orientation is remembered so that exports can be written
back voxel-for-voxel in the same grid as the input files.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel import orientations as nio

from ..config import MODALITIES


class NiftiError(Exception):
    """Raised when a file cannot be interpreted as a usable 3-D NIfTI volume."""


# --------------------------------------------------------------------------- containers
@dataclass
class NiftiVolume:
    data: np.ndarray                   # float32, canonical (RAS+) orientation
    affine: np.ndarray                 # 4x4 affine of the canonical array
    voxel_spacing: Tuple[float, float, float]
    orientation: str                   # axis codes of the *canonical* array, e.g. "RAS"
    path: str
    modality: str
    original_affine: np.ndarray = field(default_factory=lambda: np.eye(4))
    original_orientation: str = "RAS"  # axis codes of the file on disk
    original_shape: Tuple[int, ...] = ()

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)


@dataclass
class MultimodalMRI:
    modalities: Dict[str, NiftiVolume]

    @property
    def reference(self) -> NiftiVolume:
        for m in MODALITIES:
            if m in self.modalities:
                return self.modalities[m]
        return next(iter(self.modalities.values()))

    @property
    def shape(self):
        return self.reference.shape

    @property
    def affine(self):
        return self.reference.affine

    @property
    def voxel_spacing(self):
        return self.reference.voxel_spacing

    @property
    def names(self) -> List[str]:
        return [m for m in MODALITIES if m in self.modalities]


@dataclass
class Issue:
    level: str                  # "error" | "warning" | "info"
    message: str
    modality: Optional[str] = None


@dataclass
class ModalityInfo:
    name: str
    filename: str = ""
    shape: Tuple[int, ...] = ()
    spacing: Tuple[float, ...] = ()
    orientation: str = ""
    original_orientation: str = ""
    affine_valid: bool = False
    detected_modality: Optional[str] = None
    status: str = "missing"     # ok | warning | error | missing
    issues: List[Issue] = field(default_factory=list)


@dataclass
class ValidationReport:
    modalities: Dict[str, ModalityInfo] = field(default_factory=dict)
    issues: List[Issue] = field(default_factory=list)
    ground_truth: Optional[ModalityInfo] = None

    @property
    def errors(self) -> List[Issue]:
        out = list(i for i in self.issues if i.level == "error")
        for mi in self.modalities.values():
            out += [i for i in mi.issues if i.level == "error"]
        if self.ground_truth:
            out += [i for i in self.ground_truth.issues if i.level == "error"]
        return out

    @property
    def warnings(self) -> List[Issue]:
        out = list(i for i in self.issues if i.level == "warning")
        for mi in self.modalities.values():
            out += [i for i in mi.issues if i.level == "warning"]
        return out

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        if self.ok:
            return "VALID" if not self.warnings else f"VALID ({len(self.warnings)} warning(s))"
        return f"INVALID: {self.errors[0].message}"


# --------------------------------------------------------------------------- helpers
_MOD_PATTERNS = [
    ("FLAIR", re.compile(r"flair", re.I)),
    ("T1ce", re.compile(r"t1[\W_]*(ce|c|gd|post|contrast)(?![a-z0-9])", re.I)),
    ("T2", re.compile(r"t2(?![a-z0-9])", re.I)),
    ("T1", re.compile(r"t1(?![a-z0-9])", re.I)),
]


def detect_modality(filename: str) -> Optional[str]:
    """Best-effort modality detection from a BraTS-style file name."""
    base = os.path.basename(filename)
    for name, pat in _MOD_PATTERNS:
        if pat.search(base):
            return name
    return None


def _spacing_from_affine(aff: np.ndarray) -> Tuple[float, float, float]:
    return tuple(float(x) for x in np.sqrt((aff[:3, :3] ** 2).sum(axis=0)))


def affine_is_valid(aff: np.ndarray) -> bool:
    aff = np.asarray(aff, dtype=float)
    if aff.shape != (4, 4) or not np.all(np.isfinite(aff)):
        return False
    if abs(np.linalg.det(aff[:3, :3])) < 1e-9:
        return False
    sp = _spacing_from_affine(aff)
    return all(0.05 < s < 50 for s in sp)


def load_nifti(path: str, modality: str = "", reorient: bool = True) -> NiftiVolume:
    """Load a NIfTI file as float32.  NaN/Inf are preserved so validation can report them."""
    if not path or not os.path.isfile(path):
        raise NiftiError(f"File not found: {path!r}")
    try:
        img = nib.load(path)
        data = np.asanyarray(img.dataobj)
        affine = np.array(img.affine, dtype=np.float64)
    except Exception as e:  # nibabel raises many types
        raise NiftiError(f"Not a readable NIfTI file ({e.__class__.__name__}: {e})") from e
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise NiftiError(f"Expected a 3-D volume, got shape {data.shape}")
    data = data.astype(np.float32, copy=False)
    orig_codes = "".join(nib.aff2axcodes(affine)) if affine_is_valid(affine) else "???"
    orig_shape = tuple(data.shape)

    canon_affine = affine
    if reorient and affine_is_valid(affine):
        ornt = nio.io_orientation(affine)
        data = nio.apply_orientation(data, ornt)
        canon_affine = affine @ nio.inv_ornt_aff(ornt, orig_shape)
    data = np.ascontiguousarray(data, dtype=np.float32)
    codes = "".join(nib.aff2axcodes(canon_affine)) if affine_is_valid(canon_affine) else "???"
    return NiftiVolume(
        data=data, affine=canon_affine, voxel_spacing=_spacing_from_affine(canon_affine),
        orientation=codes, path=path, modality=modality or (detect_modality(path) or ""),
        original_affine=affine, original_orientation=orig_codes, original_shape=orig_shape,
    )


def to_original_orientation(array: np.ndarray, vol: NiftiVolume) -> np.ndarray:
    """Map an array in canonical orientation back onto the voxel grid of the input file."""
    if "?" in vol.original_orientation or vol.original_orientation == vol.orientation:
        return array
    start = nio.axcodes2ornt(tuple(vol.orientation))
    end = nio.axcodes2ornt(tuple(vol.original_orientation))
    return nio.apply_orientation(array, nio.ornt_transform(start, end))


# --------------------------------------------------------------------------- validation
def _validate_volume(name: str, vol: NiftiVolume) -> ModalityInfo:
    info = ModalityInfo(
        name=name, filename=vol.filename, shape=vol.shape, spacing=tuple(round(s, 4) for s in vol.voxel_spacing),
        orientation=vol.orientation, original_orientation=vol.original_orientation,
        affine_valid=affine_is_valid(vol.affine), detected_modality=detect_modality(vol.filename),
    )
    add = lambda lvl, msg: info.issues.append(Issue(lvl, msg, name))
    d = vol.data
    if not info.affine_valid:
        add("error", "Invalid affine (non-finite, singular or implausible voxel size)")
    if min(d.shape) < 16:
        add("error", f"Volume too small for analysis: {d.shape}")
    nonfinite = ~np.isfinite(d)
    n_bad = int(nonfinite.sum())
    if n_bad:
        frac = n_bad / d.size
        add("error" if frac > 0.01 else "warning", f"{n_bad} NaN/Inf voxels ({frac:.2%}); they will be treated as background")
    fin = d[~nonfinite] if n_bad else d
    if fin.size == 0 or not np.any(fin != 0):
        add("error", "Volume is empty (all zero)")
    elif float(fin.max()) == float(fin.min()):
        add("error", "Volume is constant – no image contrast")
    else:
        if float(np.mean(fin != 0)) > 0.85:
            add("warning", "Almost every voxel is non-zero – the image may not be skull-stripped; an Otsu brain mask will be used")
    if info.detected_modality and info.detected_modality != name:
        add("warning", f"File name suggests {info.detected_modality} but it was supplied as {name}")
    if info.orientation not in ("RAS", "???") and vol.original_orientation != vol.orientation:
        add("info", f"Reoriented {vol.original_orientation} → {vol.orientation}")
    info.status = ("error" if any(i.level == "error" for i in info.issues)
                   else "warning" if any(i.level == "warning" for i in info.issues) else "ok")
    return info


def validate_case(modalities: Dict[str, NiftiVolume], required=MODALITIES) -> ValidationReport:
    """Validate a set of loaded volumes for use as one multimodal case."""
    rep = ValidationReport()
    for name in required:
        if name not in modalities:
            rep.modalities[name] = ModalityInfo(name=name, status="missing")
            rep.issues.append(Issue("error", f"Missing required modality: {name}", name))
        else:
            rep.modalities[name] = _validate_volume(name, modalities[name])
    present = [n for n in required if n in modalities]
    if len(present) >= 2:
        ref = modalities[present[0]]
        for n in present[1:]:
            v = modalities[n]
            if v.shape != ref.shape:
                rep.issues.append(Issue("error", f"Shape mismatch: {present[0]} {ref.shape} vs {n} {v.shape}", n))
                continue
            dmax = float(np.abs(v.affine - ref.affine).max())
            if dmax > 0.5:
                rep.issues.append(Issue("error", f"{n} is not spatially aligned with {present[0]} (affine differs by {dmax:.2f} mm)", n))
            elif dmax > 1e-3:
                rep.issues.append(Issue("warning", f"{n} affine differs slightly from {present[0]} ({dmax:.3f} mm)", n))
        # duplicate-file / compatibility check
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                va, vb = modalities[a], modalities[b]
                if va.shape == vb.shape and np.array_equal(np.nan_to_num(va.data), np.nan_to_num(vb.data)):
                    rep.issues.append(Issue("error", f"{a} and {b} contain identical data – wrong files selected?", b))
    return rep


def load_case(paths: Dict[str, str], reorient: bool = True) -> Tuple[Optional[MultimodalMRI], ValidationReport]:
    """Load + validate.  Returns (mri or None, report).  Never raises for bad input files."""
    vols: Dict[str, NiftiVolume] = {}
    load_issues: Dict[str, Issue] = {}
    for name in MODALITIES:
        p = paths.get(name)
        if not p:
            continue
        try:
            vols[name] = load_nifti(p, name, reorient=reorient)
        except NiftiError as e:
            load_issues[name] = Issue("error", str(e), name)
    rep = validate_case(vols)
    for name, iss in load_issues.items():
        rep.modalities[name] = ModalityInfo(name=name, filename=os.path.basename(paths[name]), status="error", issues=[iss])
        rep.issues = [i for i in rep.issues if not (i.modality == name and i.message.startswith("Missing"))]
    mri = MultimodalMRI(vols) if len(vols) == len(MODALITIES) else None
    return (mri if rep.ok else mri), rep


def load_ground_truth(path: str, mri: MultimodalMRI, reorient: bool = True) -> Tuple[Optional[NiftiVolume], ModalityInfo]:
    """Load a BraTS segmentation and check it matches the MRI grid."""
    info = ModalityInfo(name="GT", filename=os.path.basename(path or ""))
    try:
        vol = load_nifti(path, "GT", reorient=reorient)
    except NiftiError as e:
        info.status = "error"
        info.issues.append(Issue("error", str(e), "GT"))
        return None, info
    info.shape, info.orientation = vol.shape, vol.orientation
    info.original_orientation = vol.original_orientation
    info.spacing = tuple(round(s, 4) for s in vol.voxel_spacing)
    info.affine_valid = affine_is_valid(vol.affine)
    ref = mri.reference
    if vol.shape != ref.shape:
        info.issues.append(Issue("error", f"Ground-truth shape {vol.shape} does not match MRI {ref.shape}", "GT"))
    elif float(np.abs(vol.affine - ref.affine).max()) > 0.5:
        info.issues.append(Issue("error", "Ground truth is not spatially aligned with the MRI", "GT"))
    d = np.nan_to_num(vol.data)
    if not np.allclose(d, np.round(d)):
        info.issues.append(Issue("error", "Ground truth is not a label map (non-integer values)", "GT"))
    elif not np.any(d > 0):
        info.issues.append(Issue("warning", "Ground truth contains no tumor labels", "GT"))
    vol.data = np.round(d).astype(np.float32)
    info.status = "error" if any(i.level == "error" for i in info.issues) else ("warning" if info.issues else "ok")
    return (vol if info.status != "error" else None), info
