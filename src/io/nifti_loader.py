from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np

PathLike = Union[str, Path]


class NiftiVolume:

    def __init__(
        self,
        image: nib.spatialimages.SpatialImage,
        path: Optional[PathLike] = None,
        modality: Optional[str] = None,
    ) -> None:
        if not isinstance(image, nib.spatialimages.SpatialImage):
            raise TypeError(
                "image must be a nibabel SpatialImage."
            )

        self.image = image
        self.path = Path(path) if path is not None else None
        self.modality = (
            modality.lower().strip()
            if modality is not None
            else None
        )

    def get_shape(self) -> Tuple[int, ...]:
        return tuple(self.image.shape)

    def get_voxel_spacing(self) -> Tuple[float, ...]:
        return tuple(
            float(value)
            for value in self.image.header.get_zooms()
        )

    def get_orientation(self) -> Tuple[str, ...]:
        return tuple(
            str(value)
            for value in nib.aff2axcodes(self.image.affine)
        )

    def get_data(
        self,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        data = np.asanyarray(self.image.dataobj)

        if dtype is not None:
            data = data.astype(dtype, copy=False)

        return data

    def get_affine(self) -> np.ndarray:
        return np.asarray(self.image.affine)

    def get_header(self):
        return self.image.header

    def get_filename(self) -> Optional[str]:
        if self.path is None:
            return None

        return self.path.name

    def is_3d(self) -> bool:
        return len(self.image.shape) == 3

    def voxel_count(self) -> int:
        return int(np.prod(self.image.shape))

    def __repr__(self) -> str:
        return (
            "NiftiVolume("
            f"modality={self.modality!r}, "
            f"shape={self.get_shape()}, "
            f"spacing={self.get_voxel_spacing()[:3]}, "
            f"orientation={self.get_orientation()}, "
            f"path={str(self.path)!r}"
            ")"
        )


@dataclass
class MultimodalMRI:

    modalities: Dict[str, NiftiVolume] = field(
        default_factory=dict
    )

    case_id: Optional[str] = None

    def __post_init__(self) -> None:
        normalized_modalities: Dict[str, NiftiVolume] = {}

        for name, volume in self.modalities.items():

            normalized_name = self._normalize_modality_name(name)

            if not isinstance(volume, NiftiVolume):
                raise TypeError(
                    f"Modality '{name}' must contain "
                    "a NiftiVolume object."
                )

            normalized_modalities[normalized_name] = volume

        self.modalities = normalized_modalities

    @staticmethod
    def _normalize_modality_name(
        modality: str,
    ) -> str:

        if not isinstance(modality, str):
            raise TypeError(
                "Modality name must be a string."
            )

        value = modality.strip().lower()

        aliases = {
            "t1w": "t1",
            "t1-weighted": "t1",
            "t1weighted": "t1",

            "t1ce": "t1ce",
            "t1-ce": "t1ce",
            "t1gd": "t1ce",
            "t1-gd": "t1ce",
            "t1_gd": "t1ce",
            "post": "t1ce",
            "postcontrast": "t1ce",
            "post-contrast": "t1ce",
            "contrast": "t1ce",

            "t2w": "t2",
            "t2-weighted": "t2",
            "t2weighted": "t2",

            "flair": "flair",
            "t2flair": "flair",
            "t2-flair": "flair",

            "segmentation": "ground_truth",
            "seg": "ground_truth",
            "label": "ground_truth",
            "labels": "ground_truth",
            "groundtruth": "ground_truth",
            "ground-truth": "ground_truth",
            "ground_truth": "ground_truth",
        }

        return aliases.get(value, value)

    def get_modality(
        self,
        modality: str,
    ) -> Optional[NiftiVolume]:

        normalized = self._normalize_modality_name(modality)

        return self.modalities.get(normalized)

    def validity(
        self,
        required_modalities: Optional[
            Sequence[str]
        ] = None,
        check_affine: bool = True,
        check_spacing: bool = True,
        check_orientation: bool = True,
    ) -> bool:

        if not self.modalities:
            return False

        if required_modalities is None:
            required_modalities = (
                "t1",
                "t1ce",
                "t2",
                "flair",
            )

        required = {
            self._normalize_modality_name(m)
            for m in required_modalities
        }

        available = set(self.modalities.keys())

        if not required.issubset(available):
            return False

        volumes = list(self.modalities.values())

        reference = volumes[0]

        for volume in volumes[1:]:

            if reference.get_shape() != volume.get_shape():
                return False

            if check_spacing:

                if not np.allclose(
                    reference.get_voxel_spacing()[:3],
                    volume.get_voxel_spacing()[:3],
                    rtol=1e-5,
                    atol=1e-5,
                ):
                    return False

            if check_orientation:

                if (
                    reference.get_orientation()
                    != volume.get_orientation()
                ):
                    return False

            if check_affine:

                if not np.allclose(
                    reference.get_affine(),
                    volume.get_affine(),
                    rtol=1e-5,
                    atol=1e-5,
                ):
                    return False

        return True

    def get_shapes(
        self,
    ) -> Dict[str, Tuple[int, ...]]:

        return {
            modality: volume.get_shape()
            for modality, volume in self.modalities.items()
        }

    def get_voxel_spacing(
        self,
    ) -> Dict[str, Tuple[float, ...]]:

        return {
            modality: volume.get_voxel_spacing()
            for modality, volume in self.modalities.items()
        }

    def get_voxel_spacings(
        self,
    ) -> Dict[str, Tuple[float, ...]]:
        return self.get_voxel_spacing()

    def get_orientation(
        self,
    ) -> Dict[str, Tuple[str, ...]]:

        return {
            modality: volume.get_orientation()
            for modality, volume in self.modalities.items()
        }

    def get_orientations(
        self,
    ) -> Dict[str, Tuple[str, ...]]:
        return self.get_orientation()

    def get_modalities(self) -> Tuple[str, ...]:
        return tuple(sorted(self.modalities.keys()))

    def has_modality(
        self,
        modality: str,
    ) -> bool:

        return self.get_modality(modality) is not None

    def __getitem__(
        self,
        modality: str,
    ) -> NiftiVolume:

        volume = self.get_modality(modality)

        if volume is None:
            raise KeyError(
                f"Modality '{modality}' is not available."
            )

        return volume

    def __contains__(
        self,
        modality: str,
    ) -> bool:

        return self.has_modality(modality)

    def __len__(self) -> int:
        return len(self.modalities)

    def __repr__(self) -> str:
        return (
            "MultimodalMRI("
            f"case_id={self.case_id!r}, "
            f"modalities={list(self.modalities.keys())}"
            ")"
        )


class NiftiLoader:

    DEFAULT_MODALITIES = (
        "t1",
        "t1ce",
        "t2",
        "flair",
    )

    VALID_EXTENSIONS = (
        ".nii",
        ".nii.gz",
    )

    def __init__(
        self,
        required_modalities: Optional[
            Sequence[str]
        ] = None,
        check_affine: bool = True,
        check_spacing: bool = True,
        check_orientation: bool = True,
        check_finite: bool = True,
    ) -> None:

        if required_modalities is None:
            required_modalities = self.DEFAULT_MODALITIES

        self.required_modalities = tuple(
            MultimodalMRI._normalize_modality_name(
                modality
            )
            for modality in required_modalities
        )

        self.check_affine = check_affine
        self.check_spacing = check_spacing
        self.check_orientation = check_orientation
        self.check_finite = check_finite

    @classmethod
    def _validate_path(
        cls,
        path: PathLike,
    ) -> Path:

        if path is None:
            raise ValueError(
                "NIfTI path cannot be None."
            )

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"NIfTI file does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Expected a file but received: {path}"
            )

        if not path.name.lower().endswith(
            cls.VALID_EXTENSIONS
        ):
            raise ValueError(
                f"Unsupported NIfTI file: {path}. "
                "Expected .nii or .nii.gz."
            )

        return path

    @staticmethod
    def _infer_case_id(
        path: Path,
    ) -> str:

        filename = path.name

        if filename.endswith(".nii.gz"):
            return filename[:-7]

        if filename.endswith(".nii"):
            return filename[:-4]

        return path.stem

    @staticmethod
    def _validate_dimensions(
        image: nib.spatialimages.SpatialImage,
        path: Path,
    ) -> None:

        shape = image.shape

        if len(shape) == 3:
            return

        if len(shape) == 4 and shape[-1] == 1:
            return

        raise ValueError(
            f"Expected a 3D NIfTI volume at '{path}', "
            f"but found shape {shape}."
        )

    def load_volume(
        self,
        path: PathLike,
        modality: Optional[str] = None,
        dtype: Optional[np.dtype] = None,
    ) -> NiftiVolume:

        nifti_path = self._validate_path(path)

        try:
            image = nib.load(str(nifti_path))

        except Exception as exc:
            raise IOError(
                f"Failed to load NIfTI file: {nifti_path}"
            ) from exc

        self._validate_dimensions(
            image,
            nifti_path,
        )

        normalized_modality = None

        if modality is not None:
            normalized_modality = (
                MultimodalMRI._normalize_modality_name(
                    modality
                )
            )

        if dtype is not None:

            data = np.asanyarray(
                image.dataobj
            ).astype(
                dtype,
                copy=False,
            )

            header = image.header.copy()

            image = nib.Nifti1Image(
                data,
                affine=image.affine,
                header=header,
            )

        return NiftiVolume(
            image=image,
            path=nifti_path,
            modality=normalized_modality,
        )

    @classmethod
    def discover_case_files(cls, directory: PathLike) -> Dict[str, Path]:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        candidates = list(dir_path.glob("*.nii")) + list(dir_path.glob("*.nii.gz"))
        if not candidates:
            for sub in dir_path.iterdir():
                if sub.is_dir():
                    sub_candidates = list(sub.glob("*.nii")) + list(sub.glob("*.nii.gz"))
                    if sub_candidates:
                        candidates = sub_candidates
                        break

        discovered: Dict[str, Path] = {}
        for p in candidates:
            name = p.name.lower()
            if "t1ce" in name or "t1-ce" in name or "t1_ce" in name or "t1gd" in name:
                discovered["t1ce"] = p
            elif "flair" in name:
                discovered["flair"] = p
            elif "t2" in name:
                discovered["t2"] = p
            elif "t1" in name:
                discovered["t1"] = p
            elif "seg" in name or "ground_truth" in name:
                discovered["ground_truth"] = p
        return discovered

    def load_case(
        self,
        modality_paths: Union[Mapping[str, PathLike], PathLike],
        case_id: Optional[str] = None,
        validate: bool = True,
        dtype: Optional[np.dtype] = None,
    ) -> MultimodalMRI:

        if isinstance(modality_paths, (str, Path)):
            dir_path = Path(modality_paths)
            if dir_path.is_dir():
                discovered = self.discover_case_files(dir_path)
                if not case_id:
                    case_id = dir_path.name
                modality_paths = {k: v for k, v in discovered.items() if k != "ground_truth"}
            else:
                raise ValueError(f"Path is not a directory: {modality_paths}")

        if not isinstance(
            modality_paths,
            Mapping,
        ):
            raise TypeError(
                "modality_paths must be a directory path or mapping "
                "of modality names to NIfTI paths."
            )

        if not modality_paths:
            raise ValueError(
                "modality_paths cannot be empty."
            )

        volumes: Dict[
            str,
            NiftiVolume
        ] = {}

        for modality, path in modality_paths.items():

            normalized_modality = (
                MultimodalMRI._normalize_modality_name(
                    modality
                )
            )

            if normalized_modality in volumes:
                raise ValueError(
                    "Duplicate modality detected: "
                    f"{normalized_modality}"
                )

            volumes[normalized_modality] = (
                self.load_volume(
                    path=path,
                    modality=normalized_modality,
                    dtype=dtype,
                )
            )

        if case_id is None:

            first_path = next(
                iter(modality_paths.values())
            )

            case_id = self._infer_case_id(
                Path(first_path)
            )

        case = MultimodalMRI(
            modalities=volumes,
            case_id=case_id,
        )

        if validate:

            report = self.get_validation_report(case)

            if not report["valid"]:

                error_text = "\n".join(
                    f"  - {error}"
                    for error in report["errors"]
                )

                raise ValueError(
                    f"Invalid MRI case "
                    f"'{case.case_id}':\n"
                    f"{error_text}"
                )

        return case

    def load_ground_truth(
        self,
        path: PathLike,
        dtype: np.dtype = np.int16,
    ) -> NiftiVolume:

        volume = self.load_volume(
            path=path,
            modality="ground_truth",
            dtype=dtype,
        )

        data = volume.get_data()

        if not np.all(
            np.isfinite(data)
        ):
            raise ValueError(
                "Ground-truth segmentation contains "
                "NaN or infinite values."
            )

        if not np.allclose(
            data,
            np.round(data),
        ):
            raise ValueError(
                "Ground-truth segmentation contains "
                "non-integer voxel labels."
            )

        return volume

    def get_validation_report(
        self,
        case: MultimodalMRI,
        required_modalities: Optional[
            Sequence[str]
        ] = None,
    ) -> Dict[str, object]:

        if not isinstance(
            case,
            MultimodalMRI,
        ):
            raise TypeError(
                "case must be a MultimodalMRI object."
            )

        if required_modalities is None:
            required_modalities = (
                self.required_modalities
            )

        required = {
            MultimodalMRI._normalize_modality_name(
                modality
            )
            for modality in required_modalities
        }

        available = set(
            case.modalities.keys()
        )

        errors = []
        warnings = []

        if not available:

            errors.append(
                "No MRI modalities are loaded."
            )

            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "modalities": [],
                "shapes": {},
                "voxel_spacing": {},
                "orientations": {},
            }

        missing = sorted(
            required - available
        )

        if missing:

            errors.append(
                "Missing required modalities: "
                + ", ".join(missing)
            )

        shapes = case.get_shapes()

        unique_shapes = {
            tuple(shape)
            for shape in shapes.values()
        }

        if len(unique_shapes) > 1:

            errors.append(
                "MRI modalities do not have "
                f"matching shapes: {shapes}"
            )

        voxel_spacing = (
            case.get_voxel_spacing()
        )

        if voxel_spacing:

            reference_modality = next(
                iter(voxel_spacing)
            )

            reference_spacing = (
                voxel_spacing[
                    reference_modality
                ]
            )

            for modality, spacing in (
                voxel_spacing.items()
            ):

                if not np.allclose(
                    reference_spacing[:3],
                    spacing[:3],
                    rtol=1e-5,
                    atol=1e-5,
                ):

                    message = (
                        "Voxel spacing mismatch "
                        f"between '{reference_modality}' "
                        f"and '{modality}': "
                        f"{reference_spacing} vs "
                        f"{spacing}"
                    )

                    if self.check_spacing:
                        errors.append(message)
                    else:
                        warnings.append(message)

        orientations = case.get_orientation()

        if orientations:

            reference_modality = next(
                iter(orientations)
            )

            reference_orientation = (
                orientations[
                    reference_modality
                ]
            )

            for modality, orientation in (
                orientations.items()
            ):

                if (
                    orientation
                    != reference_orientation
                ):

                    message = (
                        "Orientation mismatch "
                        f"between '{reference_modality}' "
                        f"and '{modality}': "
                        f"{reference_orientation} vs "
                        f"{orientation}"
                    )

                    if self.check_orientation:
                        errors.append(message)
                    else:
                        warnings.append(message)

        volume_items = list(
            case.modalities.items()
        )

        if volume_items:

            reference_modality, reference_volume = (
                volume_items[0]
            )

            reference_affine = (
                reference_volume.get_affine()
            )

            for modality, volume in (
                volume_items[1:]
            ):

                if not np.allclose(
                    reference_affine,
                    volume.get_affine(),
                    rtol=1e-5,
                    atol=1e-5,
                ):

                    message = (
                        "Affine mismatch between "
                        f"'{reference_modality}' "
                        f"and '{modality}'."
                    )

                    if self.check_affine:
                        errors.append(message)
                    else:
                        warnings.append(message)

        if self.check_finite:

            for modality, volume in (
                case.modalities.items()
            ):

                try:

                    data = volume.get_data()

                    if not np.all(
                        np.isfinite(data)
                    ):

                        errors.append(
                            f"Modality '{modality}' "
                            "contains NaN or "
                            "infinite voxel values."
                        )

                except Exception as exc:

                    errors.append(
                        f"Unable to inspect "
                        f"'{modality}': {exc}"
                    )

        for modality, spacing in (
            voxel_spacing.items()
        ):

            spatial_spacing = spacing[:3]

            if any(
                value <= 0
                for value in spatial_spacing
            ):

                errors.append(
                    f"Invalid voxel spacing for "
                    f"'{modality}': {spacing}"
                )

        for modality, shape in (
            shapes.items()
        ):

            if len(shape) != 3:

                errors.append(
                    f"Modality '{modality}' "
                    f"is not 3D: shape={shape}"
                )

            if any(
                dimension <= 0
                for dimension in shape
            ):

                errors.append(
                    f"Invalid dimensions for "
                    f"'{modality}': {shape}"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "modalities": sorted(available),
            "shapes": shapes,
            "voxel_spacing": voxel_spacing,
            "orientations": orientations,
        }

    def validate_case(
        self,
        case: MultimodalMRI,
        required_modalities: Optional[
            Sequence[str]
        ] = None,
        as_dict: bool = False,
    ) -> Union[bool, Dict[str, object]]:
        report = self.get_validation_report(
            case=case,
            required_modalities=required_modalities,
        )
        if as_dict:
            return report
        return bool(report["valid"])


def load_nifti(
    path: PathLike,
    modality: Optional[str] = None,
) -> NiftiVolume:

    loader = NiftiLoader()

    return loader.load_volume(
        path=path,
        modality=modality,
    )


# ============================================================
# MANUAL BRATS FILE SELECTION
# ============================================================

def select_brats_files() -> Dict[str, Path]:
    """
    Manually select the five BraTS NIfTI files from your computer.

    Select:
        1. T1
        2. T1ce
        3. T2
        4. FLAIR
        5. Segmentation
    """

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    file_types = [
        ("NIfTI files", "*.nii *.nii.gz"),
        ("All files", "*.*"),
    ]

    print("\nSelect the BraTS T1 file...")
    t1 = filedialog.askopenfilename(
        title="Select T1",
        filetypes=file_types,
    )

    print("Select the BraTS T1ce file...")
    t1ce = filedialog.askopenfilename(
        title="Select T1ce",
        filetypes=file_types,
    )

    print("Select the BraTS T2 file...")
    t2 = filedialog.askopenfilename(
        title="Select T2",
        filetypes=file_types,
    )

    print("Select the BraTS FLAIR file...")
    flair = filedialog.askopenfilename(
        title="Select FLAIR",
        filetypes=file_types,
    )

    print("Select the BraTS segmentation file...")
    seg = filedialog.askopenfilename(
        title="Select Segmentation",
        filetypes=file_types,
    )

    root.destroy()

    selected_files = {
        "t1": Path(t1),
        "t1ce": Path(t1ce),
        "t2": Path(t2),
        "flair": Path(flair),
        "ground_truth": Path(seg),
    }

    # Check that every file was selected
    for modality, path in selected_files.items():

        if not path:
            raise ValueError(
                f"No file selected for modality: {modality}"
            )

    return selected_files


def load_selected_brats_case(
    case_dir: Optional[PathLike] = None,
    use_gui: bool = False,
) -> Tuple[
    MultimodalMRI,
    NiftiVolume,
]:
    """
    Load a BraTS case and ground truth segmentation.
    If case_dir is given, loads from directory.
    If use_gui is True, opens GUI file-selection dialogs.
    Otherwise, checks data/raw for cases; if none exists, generates a synthetic sample case.
    """
    loader = NiftiLoader()

    if case_dir is not None:
        case_path = Path(case_dir)
        case = loader.load_case(case_path, validate=True)
        discovered = loader.discover_case_files(case_path)
        if "ground_truth" in discovered:
            ground_truth = loader.load_ground_truth(discovered["ground_truth"])
        else:
            ref_vol = next(iter(case.modalities.values()))
            empty_img = nib.Nifti1Image(
                np.zeros(ref_vol.get_shape(), dtype=np.int16),
                affine=ref_vol.get_affine()
            )
            ground_truth = NiftiVolume(empty_img, modality="ground_truth")
        return case, ground_truth

    if use_gui:
        files = select_brats_files()
        mri_files = {
            "t1": files["t1"],
            "t1ce": files["t1ce"],
            "t2": files["t2"],
            "flair": files["flair"],
        }
        case = loader.load_case(
            modality_paths=mri_files,
            validate=True,
        )
        ground_truth = loader.load_ground_truth(
            files["ground_truth"]
        )
        return case, ground_truth

    # Non-interactive / default: check data/raw
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    if raw_dir.is_dir():
        case_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]
        if case_dirs:
            return load_selected_brats_case(case_dirs[0], use_gui=False)

    # If no data exists, create synthetic BraTS sample
    from src.io.synthetic import generate_synthetic_brats_case
    sample_dir = raw_dir / "BraTS_sample"
    generate_synthetic_brats_case(sample_dir)
    return load_selected_brats_case(sample_dir, use_gui=False)


# ============================================================
# TEST / MANUAL RUN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NeuroScanX - BraTS NIfTI Loader")
    print("=" * 60)

    case, ground_truth = load_selected_brats_case()

    print("\nMRI CASE LOADED SUCCESSFULLY")
    print("-" * 60)

    print(f"Case ID: {case.case_id}")

    print("\nModalities:")
    for modality in case.get_modalities():
        print(f"  {modality}")

    print("\nShapes:")
    for modality, shape in case.get_shapes().items():
        print(f"  {modality}: {shape}")

    print("\nVoxel spacing:")
    for modality, spacing in case.get_voxel_spacing().items():
        print(f"  {modality}: {spacing}")

    print("\nOrientation:")
    for modality, orientation in case.get_orientation().items():
        print(f"  {modality}: {orientation}")

    print("\nGround Truth:")
    print(f"  File: {ground_truth.get_filename()}")
    print(f"  Shape: {ground_truth.get_shape()}")
    print(f"  Spacing: {ground_truth.get_voxel_spacing()}")
    print(f"  Orientation: {ground_truth.get_orientation()}")

    print("\nValidation:")
    print(f"  Valid: {case.validity()}")

    print("\nLoaded successfully.")
    print("=" * 60)