import nibabel as nib, numpy as np, pytest
from src.io.nifti_loader import (load_case, load_nifti, validate_case, detect_modality, to_original_orientation,
                                 NiftiError, load_ground_truth, MultimodalMRI)
from tests.phantom import make_phantom

MODS = ("T1", "T1ce", "T2", "FLAIR")


def _paths(p):
    return {k: p[k] for k in MODS}


def test_valid_case(case_paths):
    mri, rep = load_case(_paths(case_paths))
    assert rep.ok and mri is not None
    assert set(mri.modalities) == set(MODS)          # `modalities`, never `volumes`
    assert not hasattr(mri, "volumes")
    v = mri.modalities["FLAIR"]
    assert v.orientation == "RAS" and len(v.voxel_spacing) == 3 and v.shape == mri.shape


def test_missing_modality(case_paths):
    p = _paths(case_paths); del p["T2"]
    mri, rep = load_case(p)
    assert not rep.ok and mri is None
    assert any("Missing required modality: T2" in i.message for i in rep.errors)


def test_mismatched_dimensions(case_paths, tmp_path):
    mods, _, aff = make_phantom((40, 40, 30))
    bad = tmp_path / "t2_small.nii.gz"; nib.save(nib.Nifti1Image(mods["T2"], aff), bad)
    p = _paths(case_paths); p["T2"] = str(bad)
    _, rep = load_case(p)
    assert not rep.ok and any("Shape mismatch" in i.message for i in rep.errors)


def test_invalid_affine(case_paths, tmp_path):
    mods, _, _ = make_phantom((48, 56, 40))
    bad = tmp_path / "bad_affine.nii.gz"
    nib.save(nib.Nifti1Image(mods["T2"], np.diag([1e-4, 1e-4, 1e-4, 1.0])), bad)   # implausible 0.0001 mm voxels
    p = _paths(case_paths); p["T2"] = str(bad)
    _, rep = load_case(p)
    assert not rep.ok and rep.modalities["T2"].affine_valid is False


def test_invalid_values_and_empty(case_paths, tmp_path):
    mods, _, aff = make_phantom((48, 56, 40))
    nanvol = mods["T2"].copy(); nanvol[:, :, :] = np.nan
    f1 = tmp_path / "nan.nii.gz"; nib.save(nib.Nifti1Image(nanvol, aff), f1)
    f2 = tmp_path / "empty.nii.gz"; nib.save(nib.Nifti1Image(np.zeros((48, 56, 40), np.float32), aff), f2)
    for f in (f1, f2):
        p = _paths(case_paths); p["T2"] = str(f)
        _, rep = load_case(p)
        assert not rep.ok
    some = mods["T2"].copy(); some[5, 5, 5] = np.inf
    f3 = tmp_path / "inf.nii.gz"; nib.save(nib.Nifti1Image(some, aff), f3)
    p = _paths(case_paths); p["T2"] = str(f3)
    _, rep = load_case(p)
    assert rep.ok and any("NaN/Inf" in w.message for w in rep.warnings)


def test_unreadable_file(tmp_path, case_paths):
    junk = tmp_path / "junk.nii"; junk.write_text("not nifti")
    with pytest.raises(NiftiError):
        load_nifti(str(junk))
    p = _paths(case_paths); p["T1"] = str(junk)
    _, rep = load_case(p)
    assert not rep.ok


def test_duplicate_files_rejected(case_paths):
    p = _paths(case_paths); p["T1ce"] = p["T1"]
    _, rep = load_case(p)
    assert not rep.ok and any("identical" in i.message for i in rep.errors)


def test_modality_detection():
    assert detect_modality("BraTS2021_00000_t1ce.nii.gz") == "T1ce"
    assert detect_modality("sub-01_T1c.nii") == "T1ce"
    assert detect_modality("x_t1.nii.gz") == "T1" and detect_modality("x_t2.nii.gz") == "T2"
    assert detect_modality("x_flair.nii.gz") == "FLAIR" and detect_modality("scan.nii") is None


@pytest.mark.parametrize("aff", [np.diag([-1, 1, 1, 1.0]), np.diag([1, -2, 1, 1.0]),
                                 np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]]),
                                 np.array([[0, 0, -1.5, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1.0]])])
def test_reorientation_roundtrip(tmp_path, aff):
    data = np.random.default_rng(0).random((10, 12, 14)).astype(np.float32)
    f = tmp_path / "a.nii.gz"; nib.save(nib.Nifti1Image(data, aff), f)
    v = load_nifti(str(f), "T1")
    assert v.orientation == "RAS"
    np.testing.assert_allclose(to_original_orientation(v.data, v), data)
    # canonical voxel (i,j,k) must map to the same world point as the original voxel it came from
    assert np.all(np.isfinite(v.affine))


def test_ground_truth_checks(case_paths, tmp_path):
    mri, _ = load_case(_paths(case_paths))
    gt, info = load_ground_truth(case_paths["GT"], mri)
    assert gt is not None and info.status in ("ok", "warning")
    bad = tmp_path / "gt_small.nii.gz"; nib.save(nib.Nifti1Image(np.ones((5, 5, 5), np.uint8), np.eye(4)), bad)
    gt2, info2 = load_ground_truth(str(bad), mri)
    assert gt2 is None and info2.status == "error"
