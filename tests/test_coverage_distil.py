"""C3: alignment check of repairs, seeds for the distil job, packing and the manifest update."""

import io
import json

import numpy as np
import pytest
from PIL import Image

from movable_reality.coverage import distil, repair
from movable_reality.coverage.cameras import look_at
from movable_reality.coverage.ply import read_generated, write_splats
from movable_reality.coverage.voxels import VoxelGrid
from movable_reality.scene import Coverage, Gap, Scene


def save(path, arr):
    Image.fromarray(np.asarray(arr, np.uint8)).save(path)
    return path


def test_alignment_error_ignores_the_mask_and_empty_pixels(tmp_path):
    view = np.full((20, 30, 3), 120, np.uint8)
    view[:, :5] = 0  # empty (black): Muse Image may paint it
    mask = np.zeros((20, 30), np.uint8)
    mask[5:15, 10:20] = 255
    fixed = view.copy()
    fixed[mask > 0] = (10, 220, 10)  # the repair itself
    fixed[:, :5] = (200, 200, 200)
    save(tmp_path / "v.png", view)
    save(tmp_path / "m.png", mask)
    save(tmp_path / "ok.png", fixed)
    ok = repair.alignment_error(tmp_path / "v.png", tmp_path / "ok.png", tmp_path / "m.png")
    assert ok < 1e-3
    moved = np.roll(view, 7, axis=0)
    moved[:7] = 250  # a redrawn room from another viewpoint
    save(tmp_path / "bad.png", moved)
    err = repair.alignment_error(tmp_path / "v.png", tmp_path / "bad.png", tmp_path / "m.png")
    assert err > repair.MAX_ALIGNMENT_L1


def test_edit_files_index_the_images_from_zero(tmp_path):
    a, b = (tmp_path / "view.png").open("wb"), (tmp_path / "ref.jpg").open("wb")
    parts = repair.edit_files([a, b])
    a.close(), b.close()
    assert [p[0] for p in parts] == ["image[0]", "image[1]"]
    assert parts[0][1][2] == "image/png" and parts[1][1][2] == "image/jpeg"


def test_images_edit_posts_indexed_parts_through_the_client():
    calls = []

    class Client:
        def post(self, path, **kw):
            calls.append((path, kw))
            return "resp"

    f = io.BytesIO(b"x")
    f.name = "view.png"
    assert repair.images_edit(Client(), [f], "fill", "1536x1024") == "resp"
    path, kw = calls[0]
    assert path == "/images/edits"
    assert kw["files"][0][0] == "image[0]"
    assert kw["body"]["size"] == "1536x1024" and kw["body"]["n"] == 1


# ---------------------------------------------------------------- distil inputs


def make_view(tmp_path, accepted=True):
    """One repaired view of gap_002 looking at the origin from +Z, mask over the centre."""
    cam = look_at(np.array([0.0, 0.5, 3.0]), np.array([0.0, 0.5, 0.0]), 40, 30, 60.0)
    d = tmp_path / "coverage" / "repair" / "gap_002"
    d.mkdir(parents=True)
    req = {"gap_id": "gap_002", "gap_center": [0, 0.5, 0], "gap_radius": 0.3}
    req |= {"camera": repair.camera_dict(cam), "mask_pixels": 0}
    (d / "request.json").write_text(json.dumps(req))
    gap = Gap(id="gap_002", center=(0, 0.5, 0), radius=0.3, score=1.0)
    Scene(scene_id="t", coverage=Coverage(gaps=[gap])).save(tmp_path / "scene.json")
    mask = np.zeros((30, 40), np.uint8)
    mask[8:22, 12:28] = 255
    save(d / "mask.png", mask)
    save(d / "view.png", np.full((30, 40, 3), 90))
    Image.new("RGB", (80, 60), (200, 40, 40)).save(d / "repaired.webp")  # other size: resized
    check = {"alignment_l1": 0.01, "max_l1": 0.08, "accepted": accepted}
    (d / "check.json").write_text(json.dumps(check))
    return cam


def gap_grid():
    """10 cm voxels over [-1, 1]^3 with label 2 on the voxels around the origin."""
    grid = VoxelGrid(np.array([-1.0, -1.0, -1.0]), 0.1, (20, 20, 20))
    labels = np.full(grid.n, -1, np.int32)
    c = grid.centres()
    labels[np.all(np.abs(c - [0, 0.5, 0]) < 0.25, axis=1)] = 2
    labels[np.all(np.abs(c - [0.8, -0.8, 0.8]) < 0.1, axis=1)] = 5  # another, unrepaired gap
    return grid, labels


def test_load_views_needs_an_accepted_check(tmp_path):
    make_view(tmp_path, accepted=False)
    assert distil.load_views(tmp_path) == []
    (tmp_path / "coverage" / "repair" / "gap_002" / "check.json").write_text(
        json.dumps({"accepted": True})
    )
    (v,) = distil.load_views(tmp_path)
    assert v.gap_id == "gap_002" and v.image.shape == (30, 40, 3) and v.mask.sum() == 14 * 16


def test_seeds_lie_in_the_repaired_gap_and_take_the_repaired_colour(tmp_path):
    make_view(tmp_path)
    views = distil.load_views(tmp_path)
    grid, labels = gap_grid()
    pts, rgb, lab = distil.seed_points(grid, labels, views, per_voxel=4)
    assert len(pts) > 0 and set(lab.tolist()) == {2}
    assert np.all(np.abs(pts - [0, 0.5, 0]) <= 0.25 + 0.05 + 1e-6)
    assert np.allclose(rgb, np.array([200, 40, 40]) / 255.0, atol=0.05)
    capped = distil.seed_points(grid, labels, views, per_voxel=4, max_seeds=10)[0]
    assert len(capped) == 10


def test_pack_inputs_moves_cameras_and_seeds_to_the_recon_frame(tmp_path):
    cam = make_view(tmp_path)
    views = distil.load_views(tmp_path)
    W = np.eye(4)
    W[:3, :3] = 2.0 * np.eye(3)  # world = 2 x recon
    W[:3, 3] = [1.0, 0.0, 0.0]
    seeds = np.array([[1.0, 0.5, 0.0]])
    z = np.load(io.BytesIO(distil.pack_inputs(views, seeds, np.ones((1, 3)), 0.1, W)))
    assert np.allclose(z["seeds"], [[0.0, 0.25, 0.0]])
    assert float(z["seed_scale"]) == pytest.approx(0.025)
    # The seed projects to the same pixel through the recon-frame camera.
    vm, K = z["viewmat_0"], z["K_0"]
    pc = vm[:3, :3] @ z["seeds"][0] + vm[:3, 3]
    uv = (K @ pc)[:2] / pc[2]
    assert np.allclose(uv, cam.project(seeds)[0][0], atol=1e-3)
    assert z["image_0"].shape == (30, 40, 3) and z["mask_0"].dtype == bool


def test_finish_records_the_ply_and_marks_gaps_generated(tmp_path):
    gaps = [
        Gap(id="gap_002", center=(0, 0.5, 0), radius=0.3, score=1.0),
        Gap(id="gap_005", center=(1, 0.5, 0), radius=0.3, score=1.0),
    ]
    Scene(scene_id="t", coverage=Coverage(gaps=gaps)).save(tmp_path / "scene.json")
    write_splats(
        tmp_path / distil.GENERATED_PLY,
        np.zeros((7, 3)),
        np.full((7, 3), 0.5),
        generated=np.ones(7, np.uint8),
        quats=np.tile([0.0, 0.0, 0.0, 2.0], (7, 1)),
    )
    s = distil.finish(tmp_path, ["gap_002"])
    assert s.coverage.generated_ply == distil.GENERATED_PLY
    assert s.coverage.generated_splats == 7
    assert [g.status for g in s.coverage.gaps] == ["generated", "open"]
    assert read_generated(tmp_path / distil.GENERATED_PLY).tolist() == [1] * 7


def test_load_views_skips_repairs_of_a_renumbered_gap(tmp_path):
    make_view(tmp_path)
    # A rescore gave id gap_002 to a gap elsewhere: the old repair must not seed it.
    moved = Gap(id="gap_002", center=(1.5, 0.5, 0), radius=0.3, score=1.0)
    Scene(scene_id="t", coverage=Coverage(gaps=[moved])).save(tmp_path / "scene.json")
    assert distil.load_views(tmp_path) == []
    Scene(scene_id="t", coverage=Coverage(gaps=[])).save(tmp_path / "scene.json")
    assert distil.load_views(tmp_path) == []


def test_pack_inputs_carries_the_seed_gap_labels(tmp_path):
    make_view(tmp_path)
    views = distil.load_views(tmp_path)
    pts = np.zeros((3, 3))
    packed = distil.pack_inputs(views, pts, pts, 0.1, np.eye(4), np.array([2, 2, 5]))
    z = np.load(io.BytesIO(packed))
    assert z["seed_label"].tolist() == [2, 2, 5]


def test_render_depth_is_scaled_to_metres_before_masking(tmp_path):
    # Recon frame = world / 2: gsplat reports half the metric depth.
    cam = look_at(np.array([0.0, 0.5, 3.0]), np.array([0.0, 0.5, 0.0]), 40, 30, 60.0)
    gap = Gap(id="g", center=(0, 0.5, 0), radius=0.3, score=1.0)
    depth_m = np.full((30, 40), 3.0, np.float32)  # a wall through the gap centre
    np.savez(tmp_path / "r.npz", rgb=np.zeros((30, 40, 3), np.uint8), depth=depth_m / 2,
             alpha=np.ones((30, 40), np.float16))  # fmt: skip
    rgb, depth, alpha = repair.load_render(tmp_path / "r.npz", 2.0)
    assert np.allclose(depth, depth_m)
    _, mask = repair.view_from_render(cam, gap, rgb, depth, alpha)
    assert mask.sum() > 0  # the wall lies inside the gap sphere
    _, wrong = repair.view_from_render(cam, gap, rgb, depth_m / 2, alpha)
    assert wrong.sum() == 0  # unscaled, the wall would sit 1.5 m in front of the gap
