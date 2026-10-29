"""End to end on the synthetic room: fit, bake, export, manifest, idempotency."""

import io
import json

import numpy as np
import pytest
from PIL import Image

from movable_reality.bake import gaussians as G
from movable_reality.bake import glb, planefit, synthetic
from movable_reality.bake.pipeline import BakeOptions, bake_scene, revert_scene
from movable_reality.scene import Scene

OPTS = BakeOptions(texel=0.02, image_format="png")  # coarse texels keep the test fast


@pytest.fixture(scope="module")
def baked(tmp_path_factory):
    d = tmp_path_factory.mktemp("bakedemo")
    gt = synthetic.make_room(d, floor_spacing=0.04, ceiling_spacing=0.06, wall_spacing=0.1)
    report = bake_scene(d, OPTS, log=lambda _: None)
    return d, gt, report


def world(gt, pts_recon):
    w = gt.world_from_recon
    return pts_recon @ w[:3, :3].T + w[:3, 3]


def test_planes_match_ground_truth(baked):
    d, gt, report = baked
    assert report.status == "baked"
    by = {p.label: p for p in report.planes}
    assert set(by) == {"floor", "ceiling"}
    for label, n_true, off_true in (
        ("floor", gt.floor_normal, gt.floor_offset),
        ("ceiling", gt.ceiling_normal, gt.ceiling_offset),
    ):
        p = by[label]
        assert planefit.angle_deg(np.array(p.normal), n_true) < 0.2, label
        assert np.dot(p.normal, n_true) > 0, f"{label} normal must point into the room"
        assert abs(p.offset - off_true) < 0.003, label
        assert p.rms < 0.006
        # Footprint: the room is 4 x 5 m; the doorway island must not stretch the floor.
        assert sorted(p.size_m) == pytest.approx([4.0, 5.0], abs=0.08), label


def test_manifest_and_files(baked):
    d, gt, report = baked
    scene = Scene.load(d / "scene.json")
    assert [p.id for p in scene.planes] == ["floor", "ceiling"]
    assert scene.plane_bake is not None
    assert scene.background_ply == "objects/background_baked.ply"
    assert scene.plane_bake.source_background_ply == "objects/background.ply"
    assert (d / "objects" / "background.ply").is_file()  # original kept
    n_src = len(G.vertices(G.read_ply(d / "objects" / "background.ply")))
    n_red = len(G.vertices(G.read_ply(d / scene.background_ply)))
    assert n_src - n_red == scene.plane_bake.removed_splats
    assert scene.plane_bake.removed_splats == sum(p.removed_splats for p in scene.planes)
    # Most labelled Gaussians go; the mislabelled ones in the room volume stay.
    assert scene.plane_bake.removed_splats > 0.9 * (gt.counts["floor"] + gt.counts["ceiling"])
    assert scene.plane_bake.removed_splats < gt.counts["floor"] + gt.counts["ceiling"]
    # Walls are untouched.
    assert n_red >= gt.counts["background"] - gt.counts["floor"] - gt.counts["ceiling"]
    for p in scene.planes:
        assert p.texel_m == pytest.approx(0.02)
        assert len(p.corners) == 4


def test_glb_quad_is_in_recon_frame_and_on_the_plane(baked):
    d, gt, _ = baked
    scene = Scene.load(d / "scene.json")
    for p in scene.planes:
        doc, blob = glb.read_glb((d / p.mesh).read_bytes())
        assert doc["materials"][0]["alphaMode"] == "OPAQUE"
        assert "KHR_materials_unlit" in doc["materials"][0]["extensions"]
        pos = glb.accessor(doc, blob, 0).astype(np.float64)
        w = world(gt, pos)
        assert np.allclose(w, np.asarray(p.corners), atol=1e-5)
        expected_y = 0.0 if p.label == "floor" else synthetic.ROOM_H
        assert np.allclose(w[:, 1], expected_y, atol=0.005)
        idx = glb.accessor(doc, blob, 2)
        assert idx.tolist() == [0, 2, 1, 0, 3, 2]
        img = Image.open(io.BytesIO(glb.image_bytes(doc, blob)))
        assert img.size == tuple(p.texture_px)


def _texture_and_rect(d, p):
    doc, blob = glb.read_glb((d / p.mesh).read_bytes())
    img = np.asarray(Image.open(io.BytesIO(glb.image_bytes(doc, blob))).convert("RGB")) / 255.0
    c = np.asarray(p.corners)
    return img, c[0], c[1] - c[0], c[3] - c[0]


def test_floor_texture_shows_the_pattern_in_the_right_place(baked):
    d, gt, _ = baked
    scene = Scene.load(d / "scene.json")
    floor = next(p for p in scene.planes if p.label == "floor")
    img, o, du, dv = _texture_and_rect(d, floor)
    h, w, _ = img.shape
    # Sample texel centres away from checker edges, compare with the ground truth colour.
    rng = np.random.default_rng(0)
    ij = rng.uniform(0, 1, (4000, 2))
    world_pts = o + ij[:, :1] * du + ij[:, 1:] * dv
    x, z = world_pts[:, 0], world_pts[:, 2]
    edge = np.minimum(np.abs(np.mod(x, 0.5) - 0.25), np.abs(np.mod(z, 0.5) - 0.25)) > 0.17
    inside = (np.abs(x) < 1.9) & (np.abs(z) < 2.4)
    sofa = synthetic.BOXES[0]
    under = (x > sofa.lo[0]) & (x < sofa.hi[0]) & (z > sofa.lo[2]) & (z < sofa.hi[2])
    rug = (x > synthetic.RUG_X[0] - 0.1) & (x < synthetic.RUG_X[1] + 0.1)
    rug &= (z > synthetic.RUG_Z[0] - 0.1) & (z < synthetic.RUG_Z[1] + 0.1)
    sel = edge & inside & ~under & ~rug
    cols = np.minimum((ij[sel, 0] * w).astype(int), w - 1)
    rows = np.minimum((ij[sel, 1] * h).astype(int), h - 1)
    got = img[rows, cols]
    want = synthetic.floor_colour(x[sel], z[sel])
    assert np.median(np.abs(got - want).max(axis=1)) < 0.06
    # The hole under the sofa is filled with floor-like colour, not black.
    cols = np.minimum((ij[under, 0] * w).astype(int), w - 1)
    rows = np.minimum((ij[under, 1] * h).astype(int), h - 1)
    assert img[rows, cols].mean(axis=0) == pytest.approx(
        (synthetic.FLOOR_A + synthetic.FLOOR_B) / 2, abs=0.1
    )
    assert floor.filled_fraction > 0.03


def test_ceiling_texture_is_seen_from_inside(baked):
    """Lamp position checks the ceiling texture is not mirrored."""
    d, _, _ = baked
    scene = Scene.load(d / "scene.json")
    ceil = next(p for p in scene.planes if p.label == "ceiling")
    img, o, du, dv = _texture_and_rect(d, ceil)
    h, w, _ = img.shape
    lamp = np.all(img > [0.95, 0.9, 0.6], axis=2) & (img[..., 2] < 0.85)
    rows, cols = np.nonzero(lamp)
    assert len(rows) > 10
    p = o + (cols.mean() + 0.5) / w * du + (rows.mean() + 0.5) / h * dv
    assert np.hypot(p[0] - synthetic.LAMP_C[0], p[2] - synthetic.LAMP_C[1]) < 0.06


def test_idempotent_force_and_revert(baked, tmp_path):
    d, _, _ = baked
    mesh = d / "planes" / "floor.glb"
    before = mesh.stat().st_mtime_ns
    assert bake_scene(d, OPTS, log=lambda _: None).status == "skipped"
    assert mesh.stat().st_mtime_ns == before
    first = Scene.load(d / "scene.json")
    again = bake_scene(d, OPTS, force=True, log=lambda _: None)
    assert again.status == "baked"
    second = Scene.load(d / "scene.json")
    # Re-baking reads the original background, not the reduced one.
    assert second.plane_bake == first.plane_bake
    assert [p.removed_splats for p in second.planes] == [p.removed_splats for p in first.planes]
    assert revert_scene(d)
    s = Scene.load(d / "scene.json")
    assert s.background_ply == "objects/background.ply" and s.planes == [] and not s.plane_bake
    assert not revert_scene(d)


def test_missing_labels_is_nothing_to_bake(tmp_path):
    synthetic.make_room(tmp_path, floor_spacing=0.1, ceiling_spacing=0.1, wall_spacing=0.2)
    (tmp_path / "ids" / "labels.json").write_text(json.dumps({"2": "sofa#1"}))
    before = (tmp_path / "scene.json").read_text()
    r = bake_scene(tmp_path, OPTS, log=lambda _: None)
    assert r.status == "nothing to bake"
    assert all(p.status.startswith("skipped") for p in r.planes)
    assert (tmp_path / "scene.json").read_text() == before


def test_scene_without_ids_fails_clearly(tmp_path):
    Scene(scene_id="demo").save(tmp_path / "scene.json")
    with pytest.raises(FileNotFoundError, match="gaussian_ids.npy"):
        bake_scene(tmp_path, OPTS, log=lambda _: None)
