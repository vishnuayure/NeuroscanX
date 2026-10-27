import numpy as np
from scipy import ndimage as ndi
from src.config import TumorConfig
from src.segmentation import extract_whole_tumor, segment
from src.segmentation.tumor_extraction import clean_mask


def test_valid_binary_mask(result, prepared):
    seg = result.segmentation
    assert seg.shape == tuple(result.mri.shape) and seg.dtype == np.uint8
    assert set(np.unique(seg)) <= {0, 1} and seg.sum() > 100
    assert not seg[~result.preprocessing.brain_mask].any()          # nothing outside the brain


def test_connected_component_filtering():
    cfg = TumorConfig(min_component_voxels=30, closing_radius=0, fill_holes=False, min_component_fraction=0.0)
    raw = np.zeros((40, 40, 40), bool); raw[5:20, 5:20, 5:20] = True; raw[30:32, 30:32, 30:32] = True   # tiny island
    out, st = clean_mask(raw, np.ones_like(raw), cfg)
    assert out[10, 10, 10] and not out[30, 30, 30] and st["components_removed"] == 1


def test_hole_filling_keeps_real_structure():
    cfg = TumorConfig(min_component_voxels=10, closing_radius=0, fill_holes=True)
    raw = np.zeros((30, 30, 30), bool); raw[5:25, 5:25, 5:25] = True; raw[12:18, 12:18, 12:18] = False
    out, st = clean_mask(raw, np.ones_like(raw), cfg)
    assert out[15, 15, 15] and st["hole_voxels_filled"] == 216


def test_tumor_cluster_choice_independent_of_cluster_ids(prepared, cfg):
    """Cluster ids are arbitrary – permuting them must not change the selected region."""
    _, prep = prepared
    pre = prep.pre
    r = segment("fcm", pre, cfg.fcm)
    base = extract_whole_tumor(r.centers, r.membership, pre, cfg.tumor).mask_box
    perm = np.random.default_rng(3).permutation(r.centers.shape[0])
    again = extract_whole_tumor(r.centers[perm], r.membership[perm], pre, cfg.tumor).mask_box
    np.testing.assert_array_equal(base, again)


def test_selection_report_is_transparent(prepared, cfg):
    _, prep = prepared
    r = segment("fcm", prep.pre, cfg.fcm)
    ext = extract_whole_tumor(r.centers, r.membership, prep.pre, cfg.tumor)
    assert len(ext.cluster_report) == cfg.fcm.n_clusters and ext.selected
    sel = [x for x in ext.cluster_report if x["selected"]]
    assert all(x["z"]["FLAIR"] >= cfg.tumor.min_flair_z for x in sel)
    assert all(not x["selected"] for x in ext.cluster_report if not x["eligible"])


def test_phantom_segmentation_quality(result):
    assert result.metrics["dice"] > 0.9
