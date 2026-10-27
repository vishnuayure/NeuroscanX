import json, os, numpy as np, nibabel as nib, pytest
from src.pipeline import run
from src.io.exporters import export_case


def test_public_api(case_paths, cfg):
    r = run(T1=case_paths["T1"], T1ce=case_paths["T1ce"], T2=case_paths["T2"], FLAIR=case_paths["FLAIR"],
            method="fcm", config=cfg)
    for attr in ("segmentation", "membership", "uncertainty", "edges", "preprocessing", "modality_contribution",
                 "boundary_analysis", "explanation", "metrics"):
        assert hasattr(r, attr)
    assert r.metrics is None                    # no ground truth → no evaluation metrics


def test_missing_input_rejected(case_paths, cfg):
    with pytest.raises(ValueError):
        run(T1=case_paths["T1"], T1ce=None, T2=case_paths["T2"], FLAIR=case_paths["FLAIR"], config=cfg)


def test_all_methods_and_metrics(prepared):
    nsx, prep = prepared
    for m in ("fcm", "spatial_fcm", "edge_spatial_fcm"):
        r = nsx.analyze(prep, m)
        assert r.metrics["dice"] > 0.85 and r.clustering["converged"]
        assert r.segmentation.shape == tuple(prep.mri.shape)


def test_export_roundtrip(result, tmp_path):
    out = export_case(result, base_dir=str(tmp_path))
    for f in ("segmentation.nii.gz", "uncertainty.nii.gz", "xai.json", "report.json", "processing.log"):
        assert os.path.exists(os.path.join(out, f))
    img = nib.load(os.path.join(out, "segmentation.nii.gz"))
    ref = nib.load(result.mri.reference.path)
    assert img.shape == ref.shape and np.allclose(img.affine, ref.affine)
    assert int(np.asarray(img.dataobj).sum()) == int(result.segmentation.sum())
    rep = json.load(open(os.path.join(out, "report.json")))
    assert "not a diagnosis" in rep["disclaimer"]
    assert 0 <= rep["metrics"]["dice"] <= 1


def test_downsample_mode_maps_back_to_full_resolution(prepared, cfg):
    import copy
    nsx, prep = prepared
    c2 = copy.deepcopy(cfg); c2.fcm.downsample = 2
    from src.pipeline import NeuroScanX
    r = NeuroScanX(c2).analyze(prep, "edge_spatial_fcm")
    assert r.segmentation.shape == tuple(prep.mri.shape) and r.metrics["dice"] > 0.8


def test_device_fallback_message():
    from src.utils.backend import detect_device
    d = detect_device("auto")
    assert d.active in ("cpu", "cuda") and (d.active == "cuda" or "falling back to CPU" in d.message)
