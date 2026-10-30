"""C2 recapture: merging top-up frames into a scene and re-aligning the new reconstruction."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from movable_reality.coverage.cameras import look_at
from movable_reality.coverage.topup import (
    align_world,
    fit_to_size,
    frame_indices,
    merge_frames,
    sam_clip_cmd,
    select_sharp,
    umeyama,
)


def make_scene(folder: Path, n: int, size=(64, 36)) -> None:
    (folder / "frames").mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", size, (i, 0, 0)).save(folder / "frames" / f"{i:06d}.jpg")
    meta = {"fps": 4.0, "frames": [{"index": i, "source_time_s": i / 4} for i in range(n)]}
    (folder / "frames.json").write_text(json.dumps(meta))


def make_topup(folder: Path, n: int, size=(128, 72)) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for k in range(n):
        p = folder / f"t{k}.jpg"
        Image.new("RGB", size, (0, 200, k)).save(p)
        out.append(p)
    return out


def test_select_sharp_takes_the_best_of_each_window_and_thins():
    scores = [1, 5, 2, 0, 9, 3, 4, 4, 7]
    assert select_sharp(scores, 3, 10) == [1, 4, 8]
    assert select_sharp(scores, 3, 2) == [1, 8]


def test_merge_appends_with_next_indices_and_keeps_the_originals(tmp_path):
    make_scene(tmp_path, 5)
    before = (tmp_path / "frames" / "000003.jpg").read_bytes()
    new = make_topup(tmp_path / "topup", 3)
    meta = [{"source_time_s": 0.25 * k, "sharpness": 1.0} for k in range(3)]
    idx = merge_frames(tmp_path, new, meta, "topup:abc")
    assert idx == [5, 6, 7]
    assert frame_indices(tmp_path / "frames") == list(range(8))
    assert (tmp_path / "frames" / "000003.jpg").read_bytes() == before
    # Top-up frames are resized to the original frame size (reconstruct.py needs one size).
    assert Image.open(tmp_path / "frames" / "000007.jpg").size == (64, 36)
    frames = json.loads((tmp_path / "frames.json").read_text())["frames"]
    assert [f["index"] for f in frames] == list(range(8))
    assert [f.get("source") for f in frames[5:]] == ["topup:abc"] * 3
    assert "source" not in frames[0]


def test_merge_is_idempotent_per_source(tmp_path):
    make_scene(tmp_path, 4)
    new = make_topup(tmp_path / "topup", 2)
    meta = [{}, {}]
    assert merge_frames(tmp_path, new, meta, "topup:x") == [4, 5]
    assert merge_frames(tmp_path, new, meta, "topup:x") == [4, 5]
    assert frame_indices(tmp_path / "frames") == list(range(6))
    assert merge_frames(tmp_path, new, meta, "topup:y") == [6, 7]


def test_merge_refuses_gaps_in_the_frame_indices(tmp_path):
    make_scene(tmp_path, 3)
    (tmp_path / "frames" / "000001.jpg").unlink()
    with pytest.raises(ValueError, match="contiguous"):
        merge_frames(tmp_path, make_topup(tmp_path / "t", 1), [{}], "topup:z")


def test_fit_to_size_refuses_another_orientation():
    with pytest.raises(ValueError, match="orientation"):
        fit_to_size(Image.new("RGB", (36, 64)), (64, 36))
    assert fit_to_size(Image.new("RGB", (128, 72)), (64, 36)).size == (64, 36)


def test_sam_clip_reads_every_frame_in_index_order(tmp_path):
    cmd = sam_clip_cmd(tmp_path, 4.0)
    assert cmd[cmd.index("-i") + 1] == str(tmp_path / "frames" / "%06d.jpg")
    assert cmd[-1] == str(tmp_path / "sam_input.mp4")
    assert cmd[cmd.index("-framerate") + 1] == "4.0"


def random_similarity(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    A = np.eye(4)
    A[:3, :3] = 1.7 * q
    A[:3, 3] = rng.normal(size=3)
    return A


def test_umeyama_recovers_a_similarity():
    rng = np.random.default_rng(0)
    A = random_similarity(rng)
    src = rng.normal(size=(20, 3))
    dst = src @ A[:3, :3].T + A[:3, 3]
    s, R, t = umeyama(src, dst)
    assert s == pytest.approx(1.7)
    assert np.allclose(s * R, A[:3, :3]) and np.allclose(t, A[:3, 3])


def test_align_world_maps_the_new_frame_onto_the_old_one():
    rng = np.random.default_rng(1)
    eyes = rng.uniform(-2, 2, size=(8, 3))
    prev = [look_at(e, np.zeros(3), 64, 48, name=f"{i:06d}.jpg") for i, e in enumerate(eyes)]
    # The rerun sees the same cameras in another arbitrary frame, plus top-up cameras.
    new_from_prev = random_similarity(rng)
    new = [c.in_frame(new_from_prev) for c in prev]
    new.append(look_at(np.array([3.0, 0, 0]), np.zeros(3), 64, 48, name="000008.jpg"))
    world_from_prev = random_similarity(rng)
    W, fit = align_world(prev, new, world_from_prev)
    assert fit["pairs"] == 8 and fit["rms"] < 1e-9
    assert np.allclose(W, world_from_prev @ np.linalg.inv(new_from_prev))
    # A camera of the old model lands at the same world position through either model.
    c_old = prev[3].in_frame(world_from_prev).center
    c_new = new[3].in_frame(W).center
    assert np.allclose(c_old, c_new)
