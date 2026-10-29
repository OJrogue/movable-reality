import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from movable_reality.config import Models
from movable_reality.coverage import repair
from movable_reality.coverage.cameras import look_at
from movable_reality.scene import Gap, NextBestView


class FakeImages:
    def __init__(self):
        self.calls = []

    def edit(self, **kwargs):
        names = [f.name for f in kwargs["image"]]
        self.calls.append({**kwargs, "image": names})
        buf = io.BytesIO()
        Image.new("RGB", (48, 32), (10, 200, 30)).save(buf, format="WEBP")
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(buf.getvalue()).decode())],
            usage=None,
        )


class FakeClient:
    def __init__(self):
        self.images = FakeImages()


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "cache"
    monkeypatch.setattr(repair, "CACHE_DIR", d)
    return d


def write_png(path, color):
    Image.new("RGB", (64, 40), color).save(path)
    return path


def test_repair_view_calls_muse_image_once_then_hits_the_cache(tmp_path, cache_dir):
    view = write_png(tmp_path / "view.png", (255, 0, 255))
    refs = (write_png(tmp_path / "a.jpg", (100, 90, 80)), write_png(tmp_path / "b.jpg", (1, 2, 3)))
    req = repair.RepairRequest(gap_id="gap_000", view=view, references=refs)
    client = FakeClient()
    r1 = repair.repair_view(req, tmp_path / "out1.webp", client=client)
    assert not r1.cached and r1.image.exists()
    (call,) = client.images.calls
    assert call["model"] == Models.IMAGE == "muse-image-1.0"
    assert call["image"] == [str(view), *map(str, refs)]  # rendered view first
    assert call["n"] == 1 and "never seen" in call["prompt"]
    r2 = repair.repair_view(req, tmp_path / "out2.webp", client=client)
    assert r2.cached and r2.cache_key == r1.cache_key
    assert len(client.images.calls) == 1  # no second call
    assert (tmp_path / "out2.webp").read_bytes() == (tmp_path / "out1.webp").read_bytes()
    assert list(cache_dir.glob("muse_image/edit_*.webp"))


def test_cache_key_changes_with_inputs(tmp_path, cache_dir):
    view = write_png(tmp_path / "view.png", (255, 0, 255))
    ref = write_png(tmp_path / "a.jpg", (100, 90, 80))
    a = repair.RepairRequest("g", view, (ref,))
    b = repair.RepairRequest("g", view, (ref,), prompt="other")
    assert repair.cache_path(a) != repair.cache_path(b)
    before = repair.cache_path(a)
    write_png(ref, (0, 0, 0))  # same path, new content: the key follows the bytes
    assert repair.cache_path(a) != before


def test_repair_camera_uses_the_next_best_view():
    nbv = NextBestView(eye=(1.0, 1.6, 0.5), target=(0.0, 0.3, -1.8), visible=10, gain=0.5)
    gap = Gap(id="g", center=(0.0, 0.3, -1.8), radius=0.4, score=1.0, next_best_views=[nbv])
    cam = repair.repair_camera(gap, 320, 200)
    assert cam.center == pytest.approx([1.0, 1.6, 0.5])
    uv, _ = cam.project(np.array([[0.0, 0.3, -1.8]]))
    assert uv[0] == pytest.approx([160, 100], abs=1e-3)
    no_view = repair.repair_camera(Gap(id="h", center=(0, 0.3, 0), radius=0.4, score=1.0))
    assert no_view.center[1] == pytest.approx(1.6)


def test_repair_camera_backs_off_inside_the_room():
    # Gap of radius 0.75 m 1.25 m from its view: it would fill the frame.
    nbv = NextBestView(eye=(0.0, 1.6, 1.25), target=(0.0, 1.6, 0.0), visible=10, gain=0.5)
    gap = Gap(id="g", center=(0.0, 1.6, 0.0), radius=0.75, score=1.0, next_best_views=[nbv])
    big = (np.array([-10.0, 0.0, -10.0]), np.array([10.0, 3.0, 10.0]))
    cam = repair.repair_camera(gap, 300, 200, 65.0, bounds=big, fill=0.5)
    vhalf = np.arctan(np.tan(np.radians(32.5)) * 200 / 300)
    dist = np.linalg.norm(cam.center - np.array(gap.center))
    assert np.arcsin(0.75 / dist) == pytest.approx(0.5 * vhalf)
    assert cam.center[[0, 1]] == pytest.approx([0.0, 1.6])  # moved along the view line only
    # A wall at z = 2.5 stops the back-off 0.15 m inside it.
    small = (np.array([-10.0, 0.0, -10.0]), np.array([10.0, 3.0, 2.5]))
    cam = repair.repair_camera(gap, 300, 200, 65.0, bounds=small, fill=0.5)
    assert cam.center[2] == pytest.approx(2.35)
    # A small gap needs no back-off; the view stays where coverage put it.
    tiny = gap.model_copy(update={"radius": 0.1})
    assert repair.repair_camera(tiny, 300, 200, 65.0, bounds=big).center == pytest.approx(
        [0.0, 1.6, 1.25]
    )
    # An eye outside the room (a bad next view) is left alone.
    outside = (np.array([-1.0, 0.0, -1.0]), np.array([1.0, 3.0, 1.0]))
    assert repair.repair_camera(gap, 300, 200, 65.0, bounds=outside).center == pytest.approx(
        [0.0, 1.6, 1.25]
    )


def test_gap_mask_covers_the_gap_and_empty_pixels_only():
    cam = look_at([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], 64, 48, hfov_deg=60)
    gap = Gap(id="g", center=(0.0, 1.0, 0.0), radius=0.3, score=1.0)
    # A wall at z = -1 fills the image; a hole in the wall's middle is empty.
    g = np.linspace(-3, 3, 241)
    gx, gy = np.meshgrid(g, g + 1)
    wall = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, -1.0)], 1)
    hole = (np.abs(wall[:, 0]) < 0.15) & (np.abs(wall[:, 1] - 1.0) < 0.15)
    wall = wall[~hole]
    n = len(wall)
    img, mask = repair.render_gap_view(
        cam, gap, wall, np.full((n, 3), 0.5), np.full(n, 0.03), np.ones(n)
    )
    assert mask[24, 32]  # centre: ray through the sphere hits nothing (the hole)
    assert not mask[2, 2]  # corner: wall far outside the sphere
    assert tuple(img[24, 32]) == repair.MASK_COLOR
    # Wall behind the sphere but outside it is not masked.
    _, d = repair.render_points(cam, wall, np.full((n, 3), 0.5), np.full(n, 0.03))
    m2 = repair.gap_mask(cam, gap, d)
    assert m2.sum() == mask.sum()


def test_nearest_frames_prefers_similar_poses():
    target = look_at([0, 1.6, 0], [0, 1.6, -1], 64, 48)
    cams = [
        look_at([0, 1.6, 3], [0, 1.6, -1], 64, 48),  # behind, same direction
        look_at([0.1, 1.6, 0.1], [0, 1.6, -1], 64, 48),  # almost the same pose
        look_at([0, 1.6, 0], [0, 1.6, 1], 64, 48),  # same spot, opposite direction
    ]
    assert repair.nearest_frames(cams, target, k=2) == [1, 0]
    assert repair.nearest_frames([], target) == []
    # (0, 1.6, 4) is behind cameras 0 and 1; only camera 2 (facing +z) has it in view.
    assert repair.nearest_frames(cams, target, k=1, focus=np.array([0, 1.6, 4.0])) == [2]


def test_prepare_request_writes_view_mask_and_references(tmp_path):
    (tmp_path / "frames").mkdir()
    cams = [look_at([x, 1.3, 1.0], [0, 0.5, -1], 64, 48, name=f"{i:06d}.jpg")
            for i, x in enumerate((-1.0, 0.0, 1.0))]  # fmt: skip
    for c in cams:
        write_png(tmp_path / "frames" / c.name, (90, 90, 90))
    nbv = NextBestView(eye=(0.0, 1.6, 1.0), target=(0.0, 0.5, -1.0), visible=5, gain=0.5)
    gap = Gap(id="gap_003", center=(0.0, 0.5, -1.0), radius=0.3, score=1.0, next_best_views=[nbv])
    pts = np.random.default_rng(0).uniform([-2, 0, -2], [2, 2, -1.5], (3000, 3))
    splats = (pts, np.full((3000, 3), 0.5), np.full(3000, 0.02), np.ones(3000), np.zeros(3000))
    req = repair.prepare_request(tmp_path, gap, cams, splats, k_refs=2)
    assert req.view.exists() and (req.view.parent / "mask.png").exists()
    assert len(req.references) == 2 and all(p.exists() for p in req.references)
    assert (tmp_path / "coverage" / "repair" / "gap_003" / "request.json").exists()
