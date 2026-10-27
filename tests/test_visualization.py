import numpy as np, pytest
from src.visualization import viewer_2d as v2
from src.visualization.xai_overlay import build_hud_model, build_local_model
from src.xai.evidence import LocalEvidenceExplainer


@pytest.mark.parametrize("plane", v2.PLANES)
def test_plane_roundtrip_and_shapes(plane):
    shape = (20, 30, 40)
    vol = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    ijk = (3, 11, 25)
    img = v2.plane_image(vol, plane, ijk[v2.fixed_axis(plane)])
    assert img.shape == v2.plane_shape(shape, plane)
    r, c = v2.voxel_to_pixel(plane, ijk, shape)
    assert img[r, c] == vol[ijk]                       # pixel under the crosshair is exactly the voxel
    assert v2.pixel_to_voxel(plane, r, c, ijk, shape) == ijk


def test_radiological_orientation():
    vol = np.zeros((10, 10, 10), np.float32); vol[9, 5, 5] = 1     # voxel at high x = patient Right
    img = v2.plane_image(vol, "axial", 5)
    assert img[:, :5].sum() == 1 and img[:, 5:].sum() == 0         # Right appears on screen-left


def test_compose_overlay_changes_pixels_only_where_segmented():
    base = np.full((20, 20), 100, np.uint8); seg = np.zeros((20, 20), bool); seg[5:10, 5:10] = True
    out = v2.compose_rgba(base, seg=seg, seg_opacity=0.5, contour=False)
    assert out.shape == (20, 20, 4) and (out[0, 0, :3] == 100).all() and out[7, 7, 0] > out[7, 7, 2]
    assert (v2.compose_rgba(base, seg=None) == v2.compose_rgba(base)).all()


def test_window_level():
    x = np.array([0, 50, 100], np.float32)
    assert v2.apply_window(x, 50, 100).tolist() == [0, 127, 255]


def test_hud_models_from_real_result(result):
    h = build_hud_model(result)
    assert 0 <= h["confidence"] <= 1 and h["evidence"] and all(0 <= r["value"] <= 1 for r in h["evidence"])
    ex = LocalEvidenceExplainer(result.xai_context)
    pt = tuple(np.argwhere(result.xai_context.tumor)[0])
    lm = build_local_model(ex.at(pt))
    assert lm["inside"] and 0 <= lm["membership"] <= 1


def test_meshes_come_from_segmentation(result):
    from src.visualization.viewer_3d import build_meshes
    m = build_meshes(result)
    t = m["tumor"]
    assert t is not None and t.n_cells > 100 and "uncertainty" in t.point_data.keys() and "edge" in t.point_data.keys()
    seg_idx = np.argwhere(result.segmentation > 0)
    sp = np.array(result.mri.voxel_spacing)
    lo, hi = seg_idx.min(0) * sp - 3 * sp, seg_idx.max(0) * sp + 4 * sp
    assert (t.points.min(0) >= lo - 1).all() and (t.points.max(0) <= hi + 1).all()   # mesh lies on the segmented region
    assert m["brain"] is not None and m["brain"].n_cells > 100
