"""Scene blender05, blender01's room with two windows (closed blinds) and two closed doors.
Opening geometry, placement against the furniture and visibility from the main walk. Pure
Python, no Blender."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


br = load_script("blender_room")
OPS = br.SCENES["blender05"]["openings"]


def _depth(op, pts) -> np.ndarray:
    """Distance of Blender points from the opening's wall plane, into the room."""
    o = np.array(br.wall_point(op["wall"], 0.0, 0.0, 0.0))
    n = np.array(br.wall_point(op["wall"], 0.0, 1.0, 0.0)) - o
    return (np.asarray(pts) - o) @ n


def _along(op, pts) -> np.ndarray:
    o = np.array(br.wall_point(op["wall"], 0.0, 0.0, 0.0))
    t = np.array(br.wall_point(op["wall"], 1.0, 0.0, 0.0)) - o
    return (np.asarray(pts) - o) @ t


def _normal(face) -> np.ndarray:
    p = np.asarray(face, dtype=float)
    return np.cross(p, np.roll(p, -1, axis=0)).sum(0)


def test_scene_is_blender01_plus_openings():
    cfg, ref = br.SCENES["blender05"], br.SCENES["blender01"]
    assert cfg["objects"] is ref["objects"] and cfg["outline"] is None
    assert cfg["room"] == ref["room"] and cfg["walks"] == ref["walks"]
    for walk in ("main", "test"):  # same cameras as blender01
        np.testing.assert_array_equal(
            br.camera_matrix(walk, 7.3, "blender05"), br.camera_matrix(walk, 7.3, "blender01")
        )
    kinds = [op["kind"] for op in OPS]
    assert kinds.count("window") == 2 and kinds.count("door") == 2
    assert len({op["wall"] for op in OPS if op["kind"] == "window"}) == 2
    assert len({op["wall"] for op in OPS}) == 4  # one opening per wall
    assert "openings" not in ref and br.STRUCTURE["window"] == 105 and br.STRUCTURE["door"] == 106
    assert len(set(br.STRUCTURE.values())) == len(br.STRUCTURE)


@pytest.mark.parametrize("op", OPS, ids=lambda op: f"{op['kind']}_{op['wall']}")
def test_opening_parts_are_closed_boxes_on_the_wall(op):
    parts = br.opening_parts(op)
    pts = np.array([p for faces in parts.values() for f in faces for p in f])
    d = _depth(op, pts)
    assert d.min() == pytest.approx(0.0, abs=1e-9)
    assert d.max() < (0.07 if op["kind"] == "window" else 0.085)  # sill 6.5 cm, lever 8 cm
    a = _along(op, pts)
    half = (br.ROOM_X if op["wall"] in ("north", "south") else br.ROOM_Y) / 2
    assert -half + 0.3 < a.min() and a.max() < half - 0.3  # clear of the room corners
    assert pts[:, 2].min() >= 0.0 and pts[:, 2].max() < br.ROOM_H - 0.2
    for key in ("casing", "blind", "door"):  # boxes of six quads, normals point outwards
        faces = parts.get(key, [])
        assert len(faces) % 6 == 0
        for i in range(0, len(faces), 6):
            box = np.array([p for f in faces[i : i + 6] for p in f])
            c = box.mean(0)
            for f in faces[i : i + 6]:
                assert _normal(f) @ (np.mean(f, axis=0) - c) > 0
    casing = np.array([p for f in parts["casing"] for p in f])
    assert _depth(op, casing).max() >= br.CASING_D - 1e-9  # 3.5 cm proud at least


@pytest.mark.parametrize(
    "op", [op for op in OPS if op["kind"] == "window"], ids=lambda o: o["wall"]
)
def test_blind_slats_close_the_window(op):
    blind = br.opening_parts(op)["blind"]
    boxes = [np.array([p for f in blind[i : i + 6] for p in f]) for i in range(0, len(blind), 6)]
    spans = sorted((b[:, 2].min(), b[:, 2].max()) for b in boxes)
    assert len(spans) >= 30  # real slats, ~32 at a 41.5 mm pitch
    for (_, hi), (lo, _) in zip(spans[:-1], spans[1:], strict=True):
        assert lo < hi  # each slat overlaps the next in height: no gap to look through
    assert spans[0][0] <= op["bottom"] + 0.005
    assert spans[-1][1] >= op["bottom"] + op["height"] - 0.005
    slat = boxes[
        2
    ]  # boxes 0 and 1 are the rails. A tilted slat spans more depth than its thickness
    tilt = math.degrees(math.atan2(np.ptp(slat[:, 2]), np.ptp(_depth(op, slat))))
    assert 60 < tilt < 80
    a = _along(op, np.vstack(boxes))
    assert a.min() > op["centre"] - op["width"] / 2 - 1e-9
    assert a.max() < op["centre"] + op["width"] / 2 + 1e-9


@pytest.mark.parametrize("op", [op for op in OPS if op["kind"] == "door"], ids=lambda o: o["wall"])
def test_door_slab_and_handle(op):
    parts = br.opening_parts(op)
    slab = np.array([p for f in parts["door"][:6] for p in f])
    assert np.ptp(_along(op, slab)) == pytest.approx(op["width"] - 0.006)
    assert slab[:, 2].max() == pytest.approx(op["height"] - 0.003)
    assert _depth(op, slab).max() == pytest.approx(br.DOOR_D)
    metal = np.array([p for f in parts["metal"] for p in f])
    assert 0.9 < metal[:, 2].min() and metal[:, 2].max() < 1.1  # lever at 1 m
    latch = op["centre"] + op["handle"] * op["width"] / 2
    assert np.abs(_along(op, metal) - latch).min() < 0.1  # on the latch side
    assert _depth(op, metal).max() < 0.085


def test_furniture_keeps_the_openings_clear():
    feet = br.furniture_footprints("blender05")
    heights = {o[0]: br.ASSET_SIZE[o[2]][2] for o in br.OBJECTS if o[5] is None}
    for op in OPS:
        lo, hi = op["centre"] - op["width"] / 2 - 0.1, op["centre"] + op["width"] / 2 + 0.1
        reach = 0.7 if op["kind"] == "door" else 0.4  # swing room, or space before the sill
        corners = [br.wall_point(op["wall"], a, d, 0.0)[:2] for a in (lo, hi) for d in (0, reach)]
        c = np.array(corners)
        zone = (*c.min(0), *c.max(0))
        for name, (x0, y0, x1, y1) in feet.items():
            overlap = x0 < zone[2] and zone[0] < x1 and y0 < zone[3] and zone[1] < y1
            if overlap:  # only furniture below a window sill may stand in front of it
                assert op["kind"] == "window" and heights[name] < op["bottom"], (name, op)


def test_main_walk_looks_at_every_opening():
    """The centre of every window and door is in the frame (in front, more than 0.1 m away)
    for at least 3 s of the main walk."""
    k = br.intrinsics()
    n = int(br.WALKS["main"] * br.FPS)
    w2c = [np.linalg.inv(br.cam_to_world_cv(br.camera_matrix("main", i / br.FPS, "blender05")))
           for i in range(0, n, 3)]  # fmt: skip
    for op in OPS:
        p = br.wall_point(op["wall"], op["centre"], 0.0, op["bottom"] + op["height"] / 2)
        pw = br.WORLD_FROM_BLENDER @ np.array([*p, 1.0])
        seen = 0
        for m in w2c:
            pc = (m @ pw)[:3]
            if pc[2] > 0.1:
                u = k @ pc
                seen += bool(0 <= u[0] / u[2] < br.WIDTH and 0 <= u[1] / u[2] < br.HEIGHT)
        assert seen * 3 / br.FPS > 3.0, (op, seen)


def test_world_openings():
    ops = br.world_openings("blender05")
    assert [o["index"] for o in ops] == [105, 105, 106, 106]
    for o, op in zip(ops, OPS, strict=True):
        (x0, z0), (x1, z1) = o["outline_xz"]
        assert math.hypot(x1 - x0, z1 - z0) == pytest.approx(op["width"])
        assert np.linalg.norm(o["normal_xz"]) == pytest.approx(1.0)
        mid = np.array([(x0 + x1) / 2, (z0 + z1) / 2])
        assert np.linalg.norm(mid + np.array(o["normal_xz"])) < np.linalg.norm(mid)  # inwards
        assert o["top"] == pytest.approx(op["bottom"] + op["height"])
    assert br.world_openings("blender01") == []


def test_capture_guide_draws_the_openings(tmp_path):
    cg = load_script("blender_capture_guide")
    assert cg.PHASES["blender05"] is cg.PHASES["blender01"]
    out = cg.scene_layout("blender05", tmp_path / "missing_gt", tmp_path / "l.png", size=500)
    from PIL import Image

    img = np.asarray(Image.open(out).convert("RGB"))
    plan = cg.Plan(cg.room_outline({"room_size_m": [4.0, 2.6, 5.0]}), [], size=500)
    x, y = (int(round(v)) for v in plan.px(2.0, -0.2))  # east window, world z = -0.2
    np.testing.assert_array_equal(img[y, x], [70, 200, 190])
    x, y = (int(round(v)) for v in plan.px(-2.0, 1.05))  # west door
    np.testing.assert_array_equal(img[y, x], [190, 130, 70])
