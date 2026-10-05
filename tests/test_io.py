from pathlib import Path

import pytest

from src.io.nifti_loader import NiftiLoader

DATA_DIR = Path("data/raw")


def get_case_dir():
    case_dirs = [p for p in DATA_DIR.iterdir() if p.is_dir()]
    if not case_dirs:
        pytest.skip("No case directories found in data/raw")
    return case_dirs[0]


def test_load_volume():
    nifti_files = list(DATA_DIR.rglob("*.nii")) + list(DATA_DIR.rglob("*.nii.gz"))

    if not nifti_files:
        pytest.skip("No NIfTI files found in data/raw")

    loader = NiftiLoader()
    volume = loader.load_volume(nifti_files[0])

    assert volume is not None
    assert len(volume.get_shape()) == 3
    assert len(volume.get_voxel_spacing()) == 3
    assert volume.get_orientation() is not None


def test_load_case():
    loader = NiftiLoader()
    case = loader.load_case(get_case_dir())

    assert case is not None
    assert case.get_modality("T1") is not None
    assert case.get_modality("T1ce") is not None
    assert case.get_modality("T2") is not None
    assert case.get_modality("FLAIR") is not None


def test_shapes_match():
    loader = NiftiLoader()
    case = loader.load_case(get_case_dir())

    shapes = case.get_shapes()

    assert len(shapes) == 4
    assert len(set(shapes.values())) == 1


def test_voxel_spacing_available():
    loader = NiftiLoader()
    case = loader.load_case(get_case_dir())

    spacings = case.get_voxel_spacings()

    assert len(spacings) == 4

    for spacing in spacings.values():
        assert len(spacing) == 3
        assert all(value > 0 for value in spacing)


def test_orientation_available():
    loader = NiftiLoader()
    case = loader.load_case(get_case_dir())

    orientations = case.get_orientations()

    assert len(orientations) == 4

    for orientation in orientations.values():
        assert orientation is not None


def test_validate_case():
    loader = NiftiLoader()
    case = loader.load_case(get_case_dir())

    assert loader.validate_case(case) is True


def test_load_ground_truth():
    gt_files = (
        list(DATA_DIR.rglob("*seg*.nii"))
        + list(DATA_DIR.rglob("*seg*.nii.gz"))
    )

    if not gt_files:
        pytest.skip("No ground-truth segmentation file found")

    loader = NiftiLoader()
    ground_truth = loader.load_ground_truth(gt_files[0])

    assert ground_truth is not None
    assert len(ground_truth.get_shape()) == 3