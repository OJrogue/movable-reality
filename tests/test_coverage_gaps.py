import json
from dataclasses import replace

import numpy as np
import pytest

from movable_reality.coverage import nbv
from movable_reality.coverage.evaluate import evaluate_scene, precision_recall
from movable_reality.coverage.gaps import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    GapParams,
    bisect_wide,
    build_gaps,
    carry_over,
    gap_voxels,
    nearest_object,
    voxel_poor_share,
    voxel_states,
)
from movable_reality.coverage.pipeline import ScoreParams, score_scene, write_outputs
from movable_reality.coverage.synthetic import SynthConfig, make_capture, write_scene
from movable_reality.coverage.voxels import VoxelGrid, label_components
from movable_reality.scene import Gap, Scene, SceneObject


def test_voxel_states_priority():
    occ = np.array([0, 0, 3, 2])
    free = np.array([0, 2, 0, 5])
    assert voxel_states(occ, free).tolist() == [UNKNOWN, FREE, OCCUPIED, OCCUPIED]


def test_poor_share_sees_the_hidden_face_of_a_thin_panel():
    grid = VoxelGrid(np.zeros(3), 1.0, (2, 1, 1))
    gauss_voxel = np.array([0, 0, 0, 0, 1])
    cov = np.array([1.0, 1.0, 1.0, 0.0, 1.0])  # one face of voxel 0 never seen
    share = voxel_poor_share(grid, gauss_voxel, cov, np.ones(5), 0.35)
    assert share.tolist() == pytest.approx([0.25, 0.0])


def grid_and_mask(shape=(20, 6, 20), size=0.1):
    return VoxelGrid(np.zeros(3), size, shape), np.zeros(shape, bool)


def test_gap_voxels_grow_only_from_exposed_candidates_and_drop_small_ones():
    grid, cand = grid_and_mask()
    cand[2:5, 0:2, 2:5] = True  # 18 voxels, exposed on one corner
    cand[10, 0, 10] = True  # isolated, exposed, too small
    cand[15:18, 0:2, 15:18] = True  # never exposed
    exposed = np.zeros_like(cand)
    exposed[2, 1, 2] = exposed[10, 0, 10] = exposed[15, 1, 15] = False
    exposed[2, 1, 2] = True
    exposed[10, 0, 10] = True
    lab = gap_voxels(cand.ravel(), exposed.ravel(), grid, GapParams(grow=4, min_voxels=4))
    lab3 = lab.reshape(grid.dims)
    assert (lab3[2:5, 0:2, 2:5] == 0).all()
    assert lab3[10, 0, 10] == -1
    assert (lab3[15:18, 0:2, 15:18] == -1).all()


def test_bisect_wide_splits_long_components():
    grid, m = grid_and_mask((40, 2, 2))
    m[:, 0, 0] = True  # 4 m long bar of 0.1 m voxels
    lab = bisect_wide(grid, label_components(m), max_radius=0.6)
    pieces = np.unique(lab[lab >= 0])
    assert len(pieces) >= 4
    for g in pieces:
        c = grid.centres(np.nonzero(lab == g)[0])
        assert np.linalg.norm(c - c.mean(0), axis=1).max() + 0.05 <= 0.6 + 1e-6


def test_build_gaps_and_near_object():
    grid, m = grid_and_mask()
    m[2:6, 0:3, 2:6] = True
    lab = label_components(m)
    scene = Scene(
        scene_id="t",
        objects=[
            SceneObject(
                id="sofa#1",
                label="sofa",
                ply="x.ply",
                bbox_min=(0.6, 0.0, 0.2),
                bbox_max=(1.2, 0.5, 0.6),
            )
        ],
    )
    gaps = build_gaps(grid, lab, np.ones(grid.n, np.float32), scene, GapParams())
    assert len(gaps) == 1
    g = gaps[0]
    assert g.center == pytest.approx((0.4, 0.15, 0.4), abs=1e-6)
    assert g.near_object == "sofa#1"
    assert g.score == 1.0
    assert nearest_object(scene, np.array([5.0, 0.0, 5.0]), 0.2, 0.4) is None


def test_carry_over_keeps_label_and_status():
    old = [Gap(id="gap_000", label="behind the sofa", center=(0, 0, 0), radius=0.5, score=1.0,
               status="recapture_requested")]  # fmt: skip
    new = [
        Gap(id="gap_000", center=(0.1, 0, 0), radius=0.6, score=0.8),
        Gap(id="gap_001", center=(3, 0, 0), radius=0.6, score=0.8),
    ]
    out = carry_over(new, old)
    assert out[0].label == "behind the sofa" and out[0].status == "recapture_requested"
    assert out[1].label is None and out[1].status == "open"


def test_precision_recall_with_tolerance():
    grid, gap = grid_and_mask((10, 1, 1))
    truth = np.zeros(10, bool)
    scored = np.ones(10, bool)
    truth[[2, 3]] = True
    gap = np.zeros(10, bool)
    gap[[4, 8]] = True  # 4 is next to a truth voxel, 8 is not
    r = precision_recall(grid, gap, truth, scored, tol=1)
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(0.5)  # voxel 3 is matched, voxel 2 is not


# ---------------------------------------------------------------- next-best view


def box_room():
    """4 m x 2.5 m x 4 m room of 0.1 m voxels; a free interior; a wall panel at z = 1.0
    hiding an unknown pocket behind it."""
    grid = VoxelGrid.from_bounds(np.array([-2, 0, -2]), np.array([2, 2.5, 2]), 0.1)
    st = np.full(grid.dims, FREE, np.uint8)
    st[[0, -1], :, :] = OCCUPIED
    st[:, [0, -1], :] = OCCUPIED
    st[:, :, [0, -1]] = OCCUPIED
    c = grid.centres().reshape(*grid.dims, 3)
    panel = (np.abs(c[..., 2] - 1.0) < 0.05) & (np.abs(c[..., 0]) < 0.6) & (c[..., 1] < 1.0)
    pocket = (c[..., 2] > 1.05) & (c[..., 2] < 1.5) & (np.abs(c[..., 0]) < 0.55)
    pocket &= c[..., 1] < 0.9
    st[panel] = OCCUPIED
    st[pocket] = UNKNOWN
    return grid, st.ravel(), pocket.ravel()


def test_walkable_positions_avoid_obstacles():
    grid, state, _ = box_room()
    xz = nbv.walkable_positions(grid, state, nbv.WalkParams(spacing=0.25, clearance=0.3))
    assert len(xz) > 50
    assert (np.abs(xz) <= 2 - 0.3 + 1e-6).all()
    # Every standing position keeps the clearance from the panel (up to one voxel).
    c = grid.centres()
    panel = (state == OCCUPIED) & (np.abs(c[:, 2] - 1.0) < 0.05) & (np.abs(c[:, 0]) < 0.6)
    pxz = c[panel][:, [0, 2]]
    dist = np.linalg.norm(xz[:, None, :] - pxz[None, :, :], axis=2).min(axis=1)
    assert dist.min() >= 0.3 - 0.1


def test_exposure_passes_through_unknown_and_stops_at_occupied():
    grid, state, pocket = box_room()
    eyes = np.array([[0.0, 1.6, -1.0], [1.5, 1.6, 1.3]])
    exp = nbv.exposure(grid, state == OCCUPIED, state != FREE, eyes, n_dirs=4000)
    # From (1.5, 1.6, 1.3) the pocket behind the panel is in sight.
    assert exp[pocket].max() >= 1
    only_front = nbv.exposure(grid, state == OCCUPIED, state != FREE, eyes[:1], n_dirs=4000)
    c = grid.centres()
    deep = pocket & (c[:, 2] > 1.2) & (c[:, 1] < 0.6)
    assert only_front[deep].max() == 0  # hidden from the front side


def test_rank_views_prefers_the_side_that_sees_the_pocket():
    grid, state, pocket = box_room()
    labels = np.where(pocket, 0, -1).astype(np.int32)
    targets = grid.centres(np.nonzero(pocket)[0])
    cands = np.array([[0.0, -1.0], [1.4, 1.6], [-1.4, 1.6], [0.0, -1.7]])
    views = nbv.rank_views(grid, state, labels, 0, targets, cands, nbv.WalkParams(),
                           nbv.NbvParams(top_k=2))  # fmt: skip
    assert views, "expected at least one view"
    best = views[0]
    assert best.eye[2] > 1.0  # stands behind the panel
    assert best.eye[1] == pytest.approx(1.6)
    assert 0 < best.gain <= 1


# ---------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def synth_scene(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cov") / "synth"
    cfg = SynthConfig(spacing=0.05, n_views=40, write_frames=False, vr_spacing=0.5)
    write_scene(make_capture(cfg), folder, cfg, "synth")
    return folder


def test_synthetic_scene_layout(synth_scene):
    for rel in ("splat/scene.ply", "recon/sparse/0/cameras.bin", "recon/sparse/0/images.bin",
                "objects/background.ply", "scene.json", "synthetic/ground_truth.npz"):  # fmt: skip
        assert (synth_scene / rel).exists(), rel
    scene = Scene.load(synth_scene / "scene.json")
    assert {o.id for o in scene.objects} >= {"sofa#1", "desk#1"}
    assert scene.world_from_recon != [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def test_score_scene_finds_the_hidden_regions(synth_scene):
    params = replace(ScoreParams(), exposure_dirs=3000)
    res = score_scene(synth_scene, params)
    write_outputs(synth_scene, res)
    cov = Scene.load(synth_scene / "scene.json").coverage
    assert cov is not None and len(cov.gaps) >= 3
    assert any(g.near_object == "sofa#1" for g in cov.gaps)
    assert all(0 <= g.score <= 1 and g.radius > 0 for g in cov.gaps)
    assert any(g.next_best_views for g in cov.gaps)
    ev = evaluate_scene(synth_scene)
    assert ev["score"]["recall"] > 0.8
    assert ev["score"]["precision"] > 0.7
    assert ev["regions"]["behind_sofa"]["recall"] > 0.8
    report = json.loads((synth_scene / "coverage" / "report.json").read_text())
    assert report["gaussians"] > 0 and report["views"] == 40


def test_rescore_is_idempotent_and_keeps_gap_labels(synth_scene):
    path = synth_scene / "scene.json"
    params = replace(ScoreParams(), exposure_dirs=3000)
    s = Scene.load(path)
    assert s.coverage is not None
    s.coverage.gaps[0].label = "behind the sofa"
    s.save(path)
    write_outputs(synth_scene, score_scene(synth_scene, params))
    again = Scene.load(path).coverage
    assert again is not None
    assert again.gaps[0].label == "behind the sofa"
    assert [g.center for g in again.gaps] == [g.center for g in s.coverage.gaps]


def test_coarse_subset_takes_every_other_position():
    from movable_reality.coverage.pipeline import coarse_subset

    xs, zs = np.meshgrid(np.arange(6) * 0.25 - 1.954, np.arange(4) * 0.25 + 0.3, indexing="ij")
    xz = np.stack([xs.ravel(), zs.ravel()], 1)
    sub = coarse_subset(xz, 0.25, 0.5)
    assert len(sub) == 3 * 2
    assert len(coarse_subset(xz, 0.25, 0.25)) == len(xz)
