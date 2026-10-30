import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


br = load_script("blender_room")  # pure helpers import without bpy


def test_world_from_blender_is_a_rotation_to_y_up():
    m = br.WORLD_FROM_BLENDER[:3, :3]
    np.testing.assert_allclose(m @ m.T, np.eye(3))
    assert np.linalg.det(m) == pytest.approx(1.0)
    np.testing.assert_allclose(m @ [0, 0, 1], [0, 1, 0])  # Blender up -> world up


def test_cam_to_world_cv_looks_at_target():
    eye, target = np.array([1.0, -1.5, 1.5]), np.array([0.0, 0.5, 0.4])
    m = np.eye(4)
    m[:3, :3] = br.look_at(eye, target)
    m[:3, 3] = eye
    c2w = br.cam_to_world_cv(m)
    w_eye = (br.WORLD_FROM_BLENDER @ [*eye, 1])[:3]
    w_target = (br.WORLD_FROM_BLENDER @ [*target, 1])[:3]
    fwd = (w_target - w_eye) / np.linalg.norm(w_target - w_eye)
    np.testing.assert_allclose(c2w[:3, 3], w_eye)
    np.testing.assert_allclose(c2w[:3, 2], fwd, atol=1e-12)  # OpenCV +Z forward
    assert c2w[:3, 1] @ [0, 1, 0] < 0  # OpenCV +Y points down, world +Y is up
    assert c2w[:3, 0][1] == pytest.approx(0.0)  # no roll
    # The target projects to the principal point.
    k = br.intrinsics()
    p = k @ (np.linalg.inv(c2w) @ [*w_target, 1])[:3]
    np.testing.assert_allclose(p[:2] / p[2], [k[0, 2], k[1, 2]], atol=1e-9)


def test_intrinsics_phone_like():
    k = br.intrinsics()
    hfov = 2 * np.degrees(np.arctan(br.WIDTH / 2 / k[0, 0]))
    assert 65 < hfov < 75
    assert k[0, 2] == (br.WIDTH - 1) / 2


def test_rounded_rect_closed_and_inside():
    pts = np.array([br.rounded_rect(s, -1, 1, -2, 2, 0.5) for s in np.linspace(0, 1, 401)])
    np.testing.assert_allclose(pts[0], [0, -2])
    np.testing.assert_allclose(pts[-1], pts[0], atol=1e-9)
    assert pts[:, 0].min() >= -1 - 1e-9 and pts[:, 1].max() <= 2 + 1e-9
    assert pts[50, 0] > 0  # counter-clockwise: heads to +x first
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert step.max() < 2.5 * step.mean()  # roughly uniform in arc length


@pytest.mark.parametrize("walk", ["main", "test"])
def test_walks_are_smooth_and_inside_the_room(walk):
    n = int(br.WALKS[walk] * br.FPS)
    eyes = np.array([br.camera_matrix(walk, i / br.FPS)[:3, 3] for i in range(n)])
    assert np.abs(eyes[:, 0]).max() < br.ROOM_X / 2 - 0.3
    assert np.abs(eyes[:, 1]).max() < br.ROOM_Y / 2 - 0.3
    assert 0.4 < eyes[:, 2].min() and eyes[:, 2].max() < 1.7
    assert np.linalg.norm(np.diff(eyes, axis=0), axis=1).max() < 0.06  # < 1.8 m/s
    rots = [br.camera_matrix(walk, i / br.FPS)[:3, :3] for i in range(0, n, 3)]
    for r in rots:
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)


def test_main_walk_has_high_and_knee_parts():
    z = [br.camera_matrix("main", t)[2, 3] for t in (5.0, 20.0, 30.0, 36.0)]
    assert z[0] > 1.4 and z[1] > 1.4
    assert z[2] < 0.7 and z[3] < 0.7


def test_write_png_roundtrip(tmp_path):
    a8 = np.arange(12 * 7, dtype=np.uint8).reshape(7, 12)
    a16 = (np.arange(12 * 7, dtype=np.uint16) * 777).reshape(7, 12)
    br.write_png(tmp_path / "a8.png", a8)
    br.write_png(tmp_path / "a16.png", a16)
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "a8.png")), a8)
    np.testing.assert_array_equal(
        np.asarray(Image.open(tmp_path / "a16.png")).astype(np.uint16), a16
    )
    with pytest.raises(ValueError):
        br.write_png(tmp_path / "bad.png", a8.astype(np.float32))


def test_write_mesh_ply(tmp_path):
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    tris = np.array([[0, 1, 2], [0, 1, 3]])
    br.write_mesh_ply(tmp_path / "m.ply", verts, tris)
    ply = PlyData.read(str(tmp_path / "m.ply"))
    v = ply["vertex"]
    np.testing.assert_allclose(np.stack([v["x"], v["y"], v["z"]], 1), verts)
    np.testing.assert_array_equal(np.stack(ply["face"]["vertex_indices"]), tris)


def test_depth_to_mm():
    d = np.array([[0.0, 1.2344, np.inf], [70.0, np.nan, 0.0005]])
    np.testing.assert_array_equal(br.depth_to_mm(d), [[0, 1234, 0], [0, 0, 0]])


def test_parse_frames():
    assert br.parse_frames(None, 10) == range(10)
    assert br.parse_frames("2:5", 10) == range(2, 5)
    assert br.parse_frames("0:100:30", 50) == range(0, 50, 30)
    assert br.parse_frames(":3", 10) == range(0, 3)


def test_objects_table():
    names = [o[0] for o in br.OBJECTS]
    assert len(set(names)) == len(names)
    furniture = [o for o in br.OBJECTS if o[5] is None]
    items = [o for o in br.OBJECTS if o[5] is not None]
    assert 5 <= len(furniture) <= 8 and len(items) >= 2
    assert all(o[5] in names for o in items)
    assert len(br.OBJECTS) < min(br.STRUCTURE.values())


# ---------------------------------------------------------------- scene blender02


def _seg_dist(p, a, b):
    ab = b - a
    u = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + u * ab)))


def _inside(p, poly):
    """Even-odd point-in-polygon test."""
    inside = False
    for a, b in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        if (a[1] > p[1]) != (b[1] > p[1]):
            x = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0])
            inside ^= bool(p[0] < x)
    return inside


def _box_dist(p, box):
    x0, y0, x1, y1 = box
    return float(np.hypot(max(x0 - p[0], 0, p[0] - x1), max(y0 - p[1], 0, p[1] - y1)))


def _boxes_from_gt():
    """Furniture footprints (Blender x, y) from a built blender02 objects.json, if present."""
    path = ROOT / "data" / "raw" / "blender02_gt" / "objects.json"
    if not path.exists():
        return {}
    objs = json.loads(path.read_text())["objects"]
    # world (x, z) -> Blender (x, -z)
    return {
        o["name"]: (o["bbox_min"][0], -o["bbox_max"][2], o["bbox_max"][0], -o["bbox_min"][2])
        for o in objs
        if o["kind"] == "furniture"
    }


@pytest.mark.parametrize("walk", ["main", "test"])
def test_blender02_walks_keep_clearance(walk):
    """Every frame's camera centre (shake included) is inside the L and at least 0.35 m
    from every wall and every furniture footprint."""
    poly = np.array(br.ROOM2_OUTLINE)
    boxes = [*br.furniture_footprints("blender02").items(), *_boxes_from_gt().items()]
    n = int(br.WALKS2[walk] * br.FPS)
    for i in range(n):
        p = br.camera_matrix(walk, i / br.FPS, "blender02")[:2, 3]
        assert _inside(p, poly), (walk, i, p)
        wall = min(_seg_dist(p, poly[k], poly[(k + 1) % len(poly)]) for k in range(len(poly)))
        assert wall >= 0.35, (walk, i, p, wall)
        for name, box in boxes:
            assert _box_dist(p, box) >= 0.35, (walk, i, p, name)


@pytest.mark.parametrize("walk", ["main", "test"])
def test_blender02_walks_smooth(walk):
    n = int(br.WALKS2[walk] * br.FPS)
    ms = [br.camera_matrix(walk, i / br.FPS, "blender02") for i in range(n)]
    eyes = np.array([m[:3, 3] for m in ms])
    assert 0.4 < eyes[:, 2].min() and eyes[:, 2].max() < 1.7
    assert np.linalg.norm(np.diff(eyes, axis=0), axis=1).max() < 0.04  # < 1.2 m/s with the crouch
    fwd = np.array([-m[:3, 2] for m in ms])
    yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
    turn = np.degrees(np.abs(yaw[16:] - yaw[:-16])) / (16 / br.FPS)
    assert turn.max() < 45  # deg/s over ~0.5 s; the capture guide warns above 40
    assert (fwd[:, 2] < 0).all()  # always pointing a little down


def test_blender02_main_walk_parts_and_l_coverage():
    z = [br.camera_matrix("main", t, "blender02")[2, 3] for t in (5.0, 20.0, 30.0, 38.0)]
    assert z[0] > 1.4 and z[1] > 1.4
    assert z[2] < 0.7 and z[3] < 0.7
    ys = [br.camera_matrix("main", t, "blender02")[1, 3] for t in np.arange(0, 25, 0.5)]
    assert max(ys) > 1.0 and min(ys) < -1.5  # enters the bed alcove and the study's south
    np.testing.assert_array_equal(  # blender01 stays the default scene
        br.camera_matrix("main", 3.0), br.camera_matrix("main", 3.0, "blender01")
    )


def test_blender02_objects_table():
    objs = br.SCENES["blender02"]["objects"]
    names = [o[0] for o in objs]
    assert len(set(names)) == len(names)
    furniture = [o for o in objs if o[5] is None]
    items = [o for o in objs if o[5] is not None]
    assert 6 <= len(furniture) <= 8 and 3 <= len(items) <= 4
    assert all(o[5] in names for o in items)
    assert all(o[2] in br.FOOTPRINT2 for o in furniture)
    poly = np.array(br.ROOM2_OUTLINE)
    for name, (x0, y0, x1, y1) in br.furniture_footprints("blender02").items():
        for corner in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
            assert _inside(np.array(corner), poly), name  # every footprint inside the L


def test_outline_rects_cover_the_l():
    rects = br.outline_rects(br.ROOM2_OUTLINE)
    assert sorted(rects) == [(-2.25, -2.5, 2.25, 0.5), (-2.25, 0.5, 0.75, 2.5)]
    assert br.outline_rects([(-2, -2.5), (2, -2.5), (2, 2.5), (-2, 2.5)]) == [(-2, -2.5, 2, 2.5)]


def test_scene_paths():
    out, gt = br.scene_paths("blender02")
    assert out.name == "blender02" and out.parent.name == "raw"
    assert gt == out.parent / "blender02_gt"
    out, gt = br.scene_paths("blender02", Path("/tmp/x/room"))
    assert gt == Path("/tmp/x/room_gt").resolve()


def test_capture_guide_draws_the_l():
    cg = load_script("blender_capture_guide")
    box = cg.room_outline({"room_size_m": [4.0, 2.6, 5.0]})
    assert box == [(-2.0, -2.5), (2.0, -2.5), (2.0, 2.5), (-2.0, 2.5)]
    lshape = [[x, -y] for x, y in br.ROOM2_OUTLINE]
    outline = cg.room_outline({"room_size_m": [4.5, 2.6, 5.0], "room_outline_xz": lshape})
    assert outline == [tuple(p) for p in lshape]
    plan = cg.Plan(outline, [])
    img = np.asarray(plan.base(np.zeros((2, 3))))
    x, y = (int(v) for v in plan.px(1.5, -1.5))  # the cut-out north-east corner
    np.testing.assert_array_equal(img[y, x], [24, 26, 30])  # background
    x, y = (int(v) for v in plan.px(-1.0, 1.0))
    np.testing.assert_array_equal(img[y, x], [52, 48, 44])  # floor
