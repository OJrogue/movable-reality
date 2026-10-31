"""M4 geometry on CPU: PLY layout, transforms, SAM 3D pose, floor placement, ICP, flags.
No paid calls, no Modal."""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from movable_reality import splat_ops as so
from movable_reality.completion import best_frame, coarse_scale, coarse_starts, register
from movable_reality.insert import next_gen_index, parse_size, place_generated
from movable_reality.scene import Scene, SceneObject


def rot(axis: str, deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def sam3d_like(p: np.ndarray, extra: bool = True) -> np.ndarray:
    """SAM 3D Objects' save_ply layout (normals included) for centres p."""
    names = so.CORE + so.TAIL
    v = np.zeros(len(p), dtype=[(n, "f4") for n in names])
    v["x"], v["y"], v["z"] = p.T
    v["opacity"] = 3.0
    v["scale_0"] = v["scale_1"] = v["scale_2"] = math.log(0.01)
    v["rot_0"] = 1.0
    v["f_dc_0"] = 0.5
    return v


def box_points(size=(0.4, 0.5, 0.3), n=3000, seed=0) -> np.ndarray:
    """Points on the surface of an axis-aligned box centred at the origin."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(-0.5, 0.5, (n, 3))
    axis = rng.integers(0, 3, n)
    p[np.arange(n), axis] = np.sign(p[np.arange(n), axis]) * 0.5
    p *= np.asarray(size)
    p[:, 1] += 0.1 * p[:, 0] ** 2  # break the box symmetry so ICP has one answer
    return p


# ---------------------------------------------------------------- PLY layout and flag


def test_standard_layout_roundtrip_with_generated_flag(tmp_path):
    v = so.standard(sam3d_like(box_points(n=50)), generated=1)
    assert v.dtype.names[:3] == ("x", "y", "z")
    assert v.dtype.names[-1] == so.GENERATED and v[so.GENERATED].dtype == np.uint8
    for f in ("f_dc_0", "opacity", "scale_2", "rot_3"):
        assert f in v.dtype.names
    so.write_vertices(tmp_path / "a.ply", v)
    back = so.read_vertices(tmp_path / "a.ply")
    assert (back[so.GENERATED] == 1).all()
    np.testing.assert_allclose(so.xyz(back), so.xyz(v), atol=1e-6)
    # bytes input (the Modal result) reads the same
    again = so.read_vertices((tmp_path / "a.ply").read_bytes())
    assert len(again) == 50


def test_standard_keeps_existing_flag_and_rejects_missing_fields():
    v = so.standard(sam3d_like(box_points(n=10)), generated=0)
    v[so.GENERATED][:3] = 1
    kept = so.standard(v)
    assert kept[so.GENERATED].sum() == 3
    bad = np.zeros(3, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    with pytest.raises(ValueError):
        so.standard(bad)


# ---------------------------------------------------------------- rotations and transforms


def test_quaternion_roundtrip_and_product():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = so.quat_to_rotmat(q)
        q2 = so.rotmat_to_quat(R)
        np.testing.assert_allclose(so.quat_to_rotmat(q2), R, atol=1e-9)
    a, b = so.rotmat_to_quat(rot("x", 30)), so.rotmat_to_quat(rot("y", 50))
    np.testing.assert_allclose(
        so.quat_to_rotmat(so.quat_mul(a, b)), rot("x", 30) @ rot("y", 50), atol=1e-9
    )


def test_transform_moves_centres_scales_and_rotates_covariances():
    v = so.standard(sam3d_like(np.array([[1.0, 0, 0], [0, 2.0, 0]])))
    v["scale_0"] = math.log(0.1)  # elongated along local x
    R, t, s = rot("z", 90), np.array([1.0, 2.0, 3.0]), 2.0
    out = so.transform(v, R, t, s)
    np.testing.assert_allclose(so.xyz(out), [[1, 4, 3], [-3, 2, 3]], atol=1e-5)
    np.testing.assert_allclose(np.exp(out["scale_0"]), 0.2, rtol=1e-5)
    q = np.array([out[f"rot_{i}"][0] for i in range(4)])
    long_axis = so.quat_to_rotmat(q)[:, 0]  # the splat's long axis now points along +y
    np.testing.assert_allclose(np.abs(long_axis), [0, 1, 0], atol=1e-6)
    with pytest.raises(ValueError):
        so.transform(v, np.diag([1.0, 1.0, -1.0]), t)


def test_apply_matrix_matches_transform_and_inverse_restores():
    v = so.standard(sam3d_like(box_points(n=20)))
    M = so.similarity(rot("y", 40) @ rot("x", 10), [0.3, -1, 2], 1.7)
    out = so.apply_matrix(v, M)
    back = so.apply_matrix(out, np.linalg.inv(M))
    np.testing.assert_allclose(so.xyz(back), so.xyz(v), atol=1e-5)
    np.testing.assert_allclose(back["scale_1"], v["scale_1"], atol=1e-5)
    with pytest.raises(ValueError):
        so.decompose(np.diag([1.0, 1.0, -1.0, 1.0]))


def test_sam3d_pose_follows_row_vector_convention():
    """x_p3d = (s x) @ R(q) + t, then OpenCV flips x and y."""
    q = so.rotmat_to_quat(rot("y", 30))
    t = [0.1, 0.2, 2.0]
    R, tc, s = so.sam3d_cam_from_local(q.tolist(), t, [0.5, 0.5, 0.5])
    x = np.array([0.2, -0.1, 0.4])
    p3d = (0.5 * x) @ so.quat_to_rotmat(q) + np.array(t)
    np.testing.assert_allclose(s * R @ x + tc, p3d * [-1, -1, 1], atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


# ---------------------------------------------------------------- placement


def test_upright_rotation_snaps_to_local_axis_and_faces_camera():
    # Local frame: object up is local +Z. Camera looks at it level from the front, pitched
    # 15 degrees down: cam_from_local maps local +Z to camera up (OpenCV -Y) tilted by 15.
    level = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # local z -> cam -y
    A = rot("x", -15) @ level
    R, tilt = so.upright_rotation(A)
    assert tilt == pytest.approx(15, abs=1e-6)
    np.testing.assert_allclose(R @ [0, 0, 1], [0, 1, 0], atol=1e-9)  # local up -> world up
    assert np.linalg.det(R) == pytest.approx(1.0)
    # world +Z points from the object towards the camera (camera -Z), horizontally
    towards_cam_local = A.T @ [0, 0, -1]
    w = R @ towards_cam_local
    assert w[2] > 0.9 and abs(w[0]) < 1e-9


def test_upright_rotation_keeps_camera_up_when_no_axis_is_close():
    u = np.ones(3) / math.sqrt(3)  # local up direction, 54.7 degrees from every axis
    x = np.array([1.0, -1.0, 0.0]) / math.sqrt(2)
    A = np.stack([x, -u, np.cross(x, -u)])  # rows: camera axes in local coordinates
    R, tilt = so.upright_rotation(A)
    assert tilt > 40
    np.testing.assert_allclose(R @ (A.T @ [0, -1, 0]), [0, 1, 0], atol=1e-9)


def test_fit_scale_and_floor_placement():
    assert so.fit_scale(np.array([1.0, 2.0, 1.0]), (0.5, 1.0, 0.5)) == pytest.approx(0.5)
    # width and depth swapped still gives the same scale
    assert so.fit_scale(np.array([0.2, 0.5, 0.4]), (0.4, 0.5, 0.2)) == pytest.approx(1.0)
    assert so.fit_scale(np.array([1.0, 1.0, 1.0]), None) == 1.0
    p = box_points((1.0, 1.0, 1.0), n=4000)
    t = so.floor_placement(p, None, x=2.0, z=-1.0, scale=0.5)
    q = 0.5 * p + t
    lo, hi = so.robust_bounds(q)
    assert lo[1] == pytest.approx(0.0, abs=1e-9)
    assert (lo[0] + hi[0]) / 2 == pytest.approx(2.0, abs=1e-9)
    assert (lo[2] + hi[2]) / 2 == pytest.approx(-1.0, abs=1e-9)


def test_place_generated_stands_on_floor_in_recon_frame():
    # Canonical object: 0.4 wide, 0.5 tall (local +Z up), 0.3 deep; camera sees it level.
    local = sam3d_like(box_points((0.4, 0.5, 0.3)) @ np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))
    A = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    W = so.similarity(rot("x", 180), [0.0, 1.2, 0.0], 1.0)  # world_from_recon, OpenCV-ish
    v, info = place_generated(local, A, (0.8, 1.0, 0.6), 1.5, -0.5, W.tolist())
    assert (v[so.GENERATED] == 1).all()
    world = so.xyz(v) @ W[:3, :3].T + W[:3, 3]
    lo, hi = so.robust_bounds(world)
    assert lo[1] == pytest.approx(0.0, abs=0.02)
    assert hi[1] - lo[1] == pytest.approx(1.0, rel=0.05)  # height scaled to the estimate
    assert (lo[0] + hi[0]) / 2 == pytest.approx(1.5, abs=0.02)
    assert (lo[2] + hi[2]) / 2 == pytest.approx(-0.5, abs=0.02)
    assert info["bbox_min"][1] == pytest.approx(0.0, abs=0.02)


def test_parse_size_and_gen_index():
    d = parse_size('```json\n{"noun": "Stool", "width_m": 0.35, "height_m": 0.45, '
                   '"depth_m": 0.35}\n```')  # fmt: skip
    assert d == {"noun": "stool", "size": (0.35, 0.45, 0.35)}
    with pytest.raises(ValueError):
        parse_size('{"noun": "x", "width_m": 40, "height_m": 1, "depth_m": 1}')
    scene = Scene(scene_id="t")
    assert next_gen_index(scene) == 1
    for oid in ("chair#1", "gen#1", "gen#4"):
        scene.objects.append(
            SceneObject(id=oid, label="x", ply="p.ply", bbox_min=(0, 0, 0), bbox_max=(1, 1, 1))
        )
    assert next_gen_index(scene) == 5


def test_insert_history_is_undoable_by_hiding():
    scene = Scene(scene_id="t")
    scene.objects.append(
        SceneObject(
            id="gen#1", label="stool", ply="objects/gen_1.ply", bbox_min=(0, 0, 0),
            bbox_max=(0.4, 0.45, 0.4), source="generated", prompt="small wooden stool",
        )
    )  # fmt: skip
    scene._log("insert", "gen#1", {"prompt": "small wooden stool"}, None)
    scene.undo()
    assert scene.get("gen#1").visible is False


# ---------------------------------------------------------------- registration


def test_umeyama_recovers_similarity():
    p = box_points(n=500)
    R, t, s = rot("y", 25) @ rot("x", -10), np.array([0.5, 0.1, -2.0]), 1.3
    R2, t2, s2 = so.umeyama(p, s * p @ R.T + t)
    np.testing.assert_allclose(R2, R, atol=1e-9)
    np.testing.assert_allclose(t2, t, atol=1e-9)
    assert s2 == pytest.approx(s)


def test_icp_aligns_full_model_to_partial_scan():
    full = box_points((0.6, 0.8, 0.5), n=4000)
    R, t, s = rot("y", 6), np.array([0.04, -0.02, 0.03]), 1.08
    truth = s * full @ R.T + t
    partial = truth[truth[:, 2] > 0.0]  # the scan saw only the front half
    M, rms0, rms = so.icp(full, partial, iters=40)
    assert rms < rms0 and rms < 0.01
    aligned = full @ M[:3, :3].T + M[:3, 3]
    assert np.abs(aligned - truth).max() < 0.03


def test_coarse_starts_include_the_right_face_pairing():
    full = box_points((0.5, 1.0, 0.5), n=2000)
    truth = 2.0 * full + [1.0, 1.0, 0.0]
    scan = truth[truth[:, 2] > 0.0]  # partial in z
    s = coarse_scale(full, scan)
    assert s == pytest.approx(2.0, rel=0.05)
    errs = []
    for C in coarse_starts(full, scan, s):
        moved = full @ C[:3, :3].T + C[:3, 3]
        errs.append(np.abs(moved - truth).max())
    assert len(errs) == 9 and min(errs) < 0.12


def test_register_end_to_end_on_synthetic_object():
    # A scanned "chair" in the recon frame; SAM 3D returns it in a canonical frame with a
    # pose that is close but 10 % too small and 5 cm off.
    canon = box_points((0.5, 0.9, 0.5), n=5000)
    W = so.similarity(rot("x", 180), [0, 0, 0], 1.0)
    world_true = canon + [0.0, 0.45, 0.0]  # stands on the floor
    recon_true = world_true @ np.linalg.inv(W)[:3, :3].T
    scan = recon_true[recon_true[:, 2] < 0.0]  # partial: one side only
    guess = np.linalg.inv(W) @ so.similarity(rot("y", 3), [0.05, 0.45, 0.0], 0.9)
    v, info = register(so.standard(sam3d_like(canon)), guess, scan, W)
    assert (v[so.GENERATED] == 1).all()
    # RMS is bounded below by the spacing of 2000 random samples (about 1.5 cm here)
    assert info["icp_rms_after_m"] < info["icp_rms_before_m"] and info["icp_rms_after_m"] < 0.025
    assert np.abs(so.xyz(v) - recon_true).max() < 0.03


def test_best_frame_prefers_large_unclipped_masks(tmp_path):
    a = np.zeros((40, 60), np.uint8)
    a[10:30, 10:30] = 3  # 400 px, inside
    b = np.zeros((40, 60), np.uint8)
    b[0:25, 0:25] = 3  # 625 px but touches the border -> counts 312
    c = np.zeros((40, 60), np.uint8)
    c[5:10, 5:10] = 2
    for i, m in enumerate((a, b, c)):
        Image.fromarray(m).save(tmp_path / f"{i:06d}.png")
    assert best_frame(tmp_path, 3) == (0, 400)
    assert best_frame(tmp_path, 2) == (2, 25)
    with pytest.raises(ValueError):
        best_frame(tmp_path, 9)


def test_insert_tool_reports_failures_to_the_model():
    from movable_reality.agent.tools import run_tool
    from movable_reality.sam3d import Sam3DUnavailable

    def broken(*_a):
        raise Sam3DUnavailable("Modal app not deployed")

    scene = Scene(scene_id="t")
    args = {"prompt": "stool", "x": 0, "z": 0}
    out = run_tool(scene, "insert_object", args, None, inserter=broken)
    assert "Modal app not deployed" in out["error"]
    assert run_tool(scene, "insert_object", args, None)["error"]


def test_register_survives_floaters_noise_and_yaw_error():
    """Partial scan with 3 mm noise and 7 % floaters (sigma 0.6 m), a 15 degree yaw error and
    10 % scale error in the guess."""
    rng = np.random.default_rng(3)
    canon = box_points((0.5, 0.9, 0.5), n=5000)
    W = so.similarity(rot("x", 180), [0, 0, 0], 1.0)
    world_true = canon + [0.0, 0.45, 0.0]
    recon_true = world_true @ np.linalg.inv(W)[:3, :3].T
    scan = recon_true[recon_true[:, 2] < 0.0]
    scan = scan + rng.normal(0, 0.003, scan.shape)
    floaters = scan.mean(0) + rng.normal(0, 0.6, (int(0.07 * len(scan)), 3))
    scan = np.concatenate([scan, floaters])
    guess = np.linalg.inv(W) @ so.similarity(rot("y", 15), [0.05, 0.45, 0.03], 0.9)
    v, info = register(so.standard(sam3d_like(canon)), guess, scan, W)
    assert np.abs(so.xyz(v) - recon_true).max() < 0.06
    assert info["icp_rms_after_m"] < 0.03


def test_clean_scan_drops_floaters_and_transparent_splats():
    from movable_reality.completion import clean_scan

    rng = np.random.default_rng(0)
    body = rng.normal(0, 0.1, (1000, 3))
    p = np.concatenate([body, [[5.0, 0, 0], [0, -4.0, 0]]])
    alpha = np.ones(len(p))
    alpha[:200] = 0.1
    out = clean_scan(p, alpha)
    assert len(out) == 800 and np.abs(out).max() < 1.0


def test_upright_rotation_forced_axis():
    level = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    A = rot("x", -50) @ level  # camera pitched 50 degrees down onto a local +Z-up object
    R, tilt = so.upright_rotation(A, up_axis=2)
    assert tilt == pytest.approx(50, abs=1e-6)
    np.testing.assert_allclose(R @ [0, 0, 1], [0, 1, 0], atol=1e-9)


def test_register_with_scaled_world_from_recon():
    # lift_ids --scale 1.83: recon units are metres / 1.83. The guess comes from a metric
    # SAM 3D pose divided by the world scale, as complete_objects.py does.
    sw = 1.83
    canon = box_points((0.5, 0.9, 0.5), n=5000)
    W = so.similarity(rot("x", 180), [0, 0, 0], sw)
    world_true = canon + [0.0, 0.45, 0.0]
    recon_true = (np.linalg.inv(W)[:3, :3] @ world_true.T).T
    scan = recon_true[recon_true[:, 2] < 0.0]
    guess = np.linalg.inv(W) @ so.similarity(rot("y", 3), [0.05, 0.45, 0.0], 0.95)
    v, info = register(so.standard(sam3d_like(canon)), guess, scan, W)
    assert np.abs(so.xyz(v) - recon_true).max() < 0.03
