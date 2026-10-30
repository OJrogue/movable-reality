import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ev = load_script("blender_gt_eval")


def random_rotation(rng) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def rotmat_to_qvec(r: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(1 + np.trace(r), 1e-12)) / 2
    return np.array([w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w),
                     (r[1, 0] - r[0, 1]) / (4 * w)])  # fmt: skip


def test_umeyama_recovers_sim3():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    rot, s, t = random_rotation(rng), 0.37, np.array([1.0, -2.0, 0.5])
    dst = s * src @ rot.T + t
    s2, rot2, t2 = ev.umeyama(src, dst)
    assert s2 == pytest.approx(s)
    np.testing.assert_allclose(rot2, rot, atol=1e-9)
    np.testing.assert_allclose(t2, t, atol=1e-9)


def poses_on_path(rng, n=40):
    out = []
    for i in range(n):
        m = np.eye(4)
        m[:3, :3] = random_rotation(rng)
        m[:3, 3] = [np.cos(i / 6), 1.5 + 0.4 * np.sin(i / 3), np.sin(i / 5) * 2]
        out.append(m)
    return out


def test_pose_errors_zero_after_similarity_and_scale_reported():
    rng = np.random.default_rng(1)
    gt = poses_on_path(rng)
    rot, s, t = random_rotation(rng), 2.5, np.array([0.3, 0.1, -4.0])
    pred = []
    for m in gt:  # pred = inverse similarity of gt, so gt = s * rot @ pred + t
        p = np.eye(4)
        p[:3, :3] = rot.T @ m[:3, :3]
        p[:3, 3] = rot.T @ (m[:3, 3] - t) / s
        pred.append(p)
    r = ev.pose_errors(pred, gt)
    assert r["ate_rmse_m"] < 1e-9
    assert r["rot_err_max_deg"] < 1e-5
    assert r["sim3_scale"] == pytest.approx(s)


def test_pose_errors_report_a_rotation_offset():
    rng = np.random.default_rng(7)
    gt = poses_on_path(rng)
    off = ev.qvec_to_rotmat([np.cos(np.radians(1.0)), 0.0, np.sin(np.radians(1.0)), 0.0])  # 2 deg
    pred = [m.copy() for m in gt]
    for p in pred:
        p[:3, :3] = p[:3, :3] @ off  # each camera turned 2 deg about its own Y axis
    r = ev.pose_errors(pred, gt)
    assert r["ate_rmse_m"] < 1e-9
    assert r["rot_err_median_deg"] == pytest.approx(2.0, abs=1e-6)


def test_mirrored_reconstruction_is_not_aligned_away():
    rng = np.random.default_rng(8)
    gt = poses_on_path(rng)
    flip = np.diag([-1.0, 1.0, 1.0])
    pred = [m.copy() for m in gt]
    for p in pred:
        p[:3, 3] = flip @ p[:3, 3]  # a reflection is not a similarity
    assert ev.pose_errors(pred, gt)["ate_rmse_m"] > 0.1


def test_pose_errors_measure_noise():
    rng = np.random.default_rng(2)
    gt = poses_on_path(rng, 200)
    pred = [m.copy() for m in gt]
    for p in pred:
        p[:3, 3] += rng.normal(scale=0.01, size=3)
    r = ev.pose_errors(pred, gt)
    assert 0.012 < r["ate_rmse_m"] < 0.022  # sqrt(3) * 1 cm, minus what the fit absorbs


def write_images_bin(path: Path, entries):
    data = struct.pack("<Q", len(entries))
    for i, (name, q, t) in enumerate(entries, start=1):
        data += struct.pack("<I7dI", i, *q, *t, 1) + name.encode() + b"\x00"
        data += struct.pack("<Q", 2) + struct.pack("<ddq", 1.0, 2.0, -1) * 2
    path.write_bytes(data)


def test_read_colmap_bin_and_txt(tmp_path):
    rng = np.random.default_rng(3)
    entries, expected = [], {}
    for k in range(3):
        r_c2w = random_rotation(rng)
        c = rng.normal(size=3)
        r_w2c, t_w2c = r_c2w.T, -r_c2w.T @ c
        name = f"{k:06d}.jpg"
        entries.append((name, rotmat_to_qvec(r_w2c), t_w2c))
        expected[name] = (r_c2w, c)
    (tmp_path / "bin" / "0").mkdir(parents=True)
    write_images_bin(tmp_path / "bin" / "0" / "images.bin", entries)
    lines = ["# Image list", "# comment"]
    for i, (name, q, t) in enumerate(entries, start=1):
        lines.append(f"{i} {' '.join(map(str, q))} {' '.join(map(str, t))} 1 {name}")
        lines.append("" if i == 2 else "1.0 2.0 -1")  # an image with no 2D points
    (tmp_path / "images.txt").write_text("\n".join(lines) + "\n")
    for src in (tmp_path / "bin", tmp_path / "images.txt"):
        poses = ev.read_colmap_images(src)
        assert set(poses) == set(expected)
        for name, (r, c) in expected.items():
            np.testing.assert_allclose(poses[name][:3, :3], r, atol=1e-9)
            np.testing.assert_allclose(poses[name][:3, 3], c, atol=1e-9)


def test_gt_index_and_closest_frame():
    assert ev.gt_index(0.0, 100) == 0
    assert ev.gt_index(1.0, 100) == 30
    assert ev.gt_index(0.2501, 100) == 8
    assert ev.gt_index(99.0, 100) == 99
    rng = np.random.default_rng(4)
    cands = {i: rng.random((10, 10)).astype(np.float32) for i in range(5, 10)}
    assert ev.closest_frame(cands[7] + 0.01, cands) == 7


def test_map_frames_refines_with_renders(tmp_path):
    scene, rgb = tmp_path / "scene", tmp_path / "rgb"
    (scene / "frames").mkdir(parents=True)
    rgb.mkdir()
    rng = np.random.default_rng(5)
    imgs = [rng.integers(0, 255, (54, 96), dtype=np.uint8) for _ in range(12)]
    for j, im in enumerate(imgs):
        Image.fromarray(im).save(rgb / f"{j:06d}.png")
    # Pipeline frame 0 claims t = 0.2 s (GT 6) but is really GT frame 7.
    Image.fromarray(imgs[7]).save(scene / "frames" / "000000.jpg", quality=95)
    (scene / "frames.json").write_text(
        json.dumps({"fps": 4, "frames": [{"index": 0, "source_time_s": 0.2}]})
    )
    assert ev.map_frames(scene, 12, None) == {0: 6}
    assert ev.map_frames(scene, 12, rgb) == {0: 7}


def test_mask_matching_permuted_labels():
    gt = np.zeros((20, 20), np.int64)
    gt[:10, :10], gt[10:, 10:], gt[:5, 15:] = 1, 2, 101
    pred = np.zeros_like(gt)
    pred[:10, :10] = 7  # perfect
    pred[10:, 10:15] = 3  # half of object 2
    counts = ev.accumulate_overlap([(pred, gt), (pred, gt)])
    m = ev.match_objects(counts, {1, 2, 5, 101})
    assert m[1] == (7, 1.0)
    assert m[2][0] == 3 and m[2][1] == pytest.approx(0.5)
    assert m[5] == (None, 0.0)  # never seen
    assert m[101] == (None, 0.0)


def test_mask_matching_is_one_to_one():
    gt = np.zeros((4, 4), np.int64)
    gt[:, :2], gt[:, 2:] = 1, 2
    pred = np.full_like(gt, 9)  # one prediction covering both objects
    m = ev.match_objects(ev.accumulate_overlap([(pred, gt)]), {1, 2})
    assert [p for p, _ in m.values()].count(9) == 1


def test_chamfer():
    rng = np.random.default_rng(6)
    a = rng.random((500, 3))
    assert ev.chamfer(a, a) == 0.0
    shifted = a + np.array([0.0, 0.0, 5.0])  # far apart: distance ~ the shift
    assert ev.chamfer(a, shifted) == pytest.approx(5.0, abs=0.6)
