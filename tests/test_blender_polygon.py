"""Polygon rooms blender03 (angled wall, bay, column) and blender04 (attic): room shell,
furniture placement and walk clearance. Pure Python, no Blender."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
POLY_SCENES = ["blender03", "blender04"]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


br = load_script("blender_room")


def _normal(face):
    p = np.asarray(face, dtype=float)
    n = np.cross(p, np.roll(p, -1, axis=0)).sum(0)
    return n / np.linalg.norm(n)


def _perimeter(poly):
    p = np.asarray(poly, dtype=float)
    return float(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum())


# ---------------------------------------------------------------- polygon helpers


def test_polygon_helpers():
    square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert br.polygon_area(square) == 4.0
    assert br.polygon_area(square[::-1]) == -4.0
    assert br.point_in_polygon((1, 1), square) and not br.point_in_polygon((3, 1), square)
    assert br.wall_distance((0.5, 1.0), square) == pytest.approx(0.5)
    assert br.box_distance((3, 3), (0, 0, 2, 2)) == pytest.approx(math.sqrt(2))
    assert br.box_distance((1, 1), (0, 0, 2, 2)) == 0.0
    assert br.is_convex(square) and not br.is_convex([(0, 0), (2, 0), (1, 0.5), (2, 2), (0, 2)])
    assert br.footprint_box(0, 0, 90, 2.0, 1.0) == pytest.approx((-0.5, -1.0, 0.5, 1.0))
    assert br.footprint_box(1, 1, 0, 2.0, 1.0) == pytest.approx((0.0, 0.5, 2.0, 1.5))


@pytest.mark.parametrize("scene", POLY_SCENES)
def test_outline_is_non_convex_and_tiled_by_zones(scene):
    cfg = br.SCENES[scene]
    outline, zones = cfg["outline"], cfg["zones"]
    assert br.polygon_area(outline) > 0 and not br.is_convex(outline)
    assert all(br.is_convex(poly) and br.polygon_area(poly) > 0 for poly, _ in zones)
    assert sum(br.polygon_area(p) for p, _ in zones) == pytest.approx(br.polygon_area(outline))
    # Every interior grid point lies in exactly one zone, every exterior point in none.
    rng = np.random.default_rng(0)
    pts = np.asarray(outline)
    lo, hi = pts.min(0) - 0.2, pts.max(0) + 0.2
    for p in rng.uniform(lo, hi, size=(3000, 2)):
        n = sum(br.point_in_polygon(p, poly) for poly, _ in zones)
        assert n == int(br.point_in_polygon(p, outline)), p


def test_blender03_has_angled_wall_bay_and_column():
    out = np.asarray(br.ROOM3_OUTLINE)
    edges = np.roll(out, -1, axis=0) - out
    angles = np.degrees(np.arctan2(edges[:, 1], edges[:, 0])) % 90
    assert ((angles > 5) & (angles < 85)).sum() >= 3  # the SE wall and the two bay sides
    assert len(br.ROOM3_COLUMNS) == 1
    (x0, y0, x1, y1) = br.ROOM3_COLUMNS[0]
    assert br.wall_distance(((x0 + x1) / 2, (y0 + y1) / 2), out) > 1.5  # free-standing
    assert br.ceiling_height(br.ROOM3_ZONES, 3.0, 0.9) == pytest.approx(2.4)  # bay
    assert br.ceiling_height(br.ROOM3_ZONES, 0.0, 0.0) == pytest.approx(2.6)


def test_blender04_attic_ceiling():
    z = br.ROOM4_ZONES
    assert br.ceiling_height(z, -2.4, 0.0 + 1.5) == pytest.approx(1.0)  # west knee wall
    assert br.ceiling_height(z, 2.4, -2.0) == pytest.approx(1.0)  # east knee wall
    assert br.ceiling_height(z, 0.0, 0.0) == pytest.approx(2.6)  # ridge, flat collar
    assert br.ceiling_height(z, -1.6, 0.0) == pytest.approx(1.8)  # 45 deg slope
    assert br.ceiling_height(z, 2.7, 0.0) == pytest.approx(2.3)  # dormer
    assert br.ceiling_height(z, 1.8, -1.0) == pytest.approx(1.6)  # slope beside the dormer
    assert br.ceiling_height(z, 3.5, 0.0) is None
    assert max(br.SCENES["blender04"]["room"]) == pytest.approx(5.4)


# ---------------------------------------------------------------- room shell


@pytest.mark.parametrize("scene", POLY_SCENES)
def test_shell_faces_close_the_room(scene):
    cfg = br.SCENES[scene]
    outline, zones = cfg["outline"], cfg["zones"]
    faces = br.shell_faces(zones, cfg["columns"])
    area = br.polygon_area(outline)
    assert sum(br.face_area(f) for f in faces["floor"]) == pytest.approx(area)
    assert all(_normal(f)[2] == pytest.approx(1.0) for f in faces["floor"])
    for f in faces["ceiling"]:
        assert _normal(f)[2] < 0  # faces down into the room
        for x, y, h in f:  # on a step edge both sides touch: the top is the max
            assert h <= br.ceiling_height(zones, x, y) + 1e-9
        c = np.asarray(f).mean(0)
        assert br.ceiling_height(zones, c[0], c[1]) == pytest.approx(c[2])
    # Ceiling area in plan equals the floor area.
    plan = sum(br.face_area(f) * -_normal(f)[2] for f in faces["ceiling"])
    assert plan == pytest.approx(area)
    # Walls from the floor run the whole outline once.
    base = 0.0
    for f in faces["wall"]:
        p = np.asarray(f)
        bottom = p[np.abs(p[:, 2]) < 1e-9]
        if len(bottom) == 2:
            base += float(np.linalg.norm(bottom[1] - bottom[0]))
    assert base == pytest.approx(_perimeter(outline))
    # Every wall or step face is vertical and has air on its normal side, solid behind it.
    for f in faces["wall"]:
        n = _normal(f)
        assert abs(n[2]) < 1e-9
        c = np.asarray(f).mean(0)
        front, back = c[:2] + 0.02 * n[:2], c[:2] - 0.02 * n[:2]
        assert br.point_in_polygon(front, outline)
        assert br.ceiling_height(zones, *front) >= c[2] - 1e-9
        hb = br.ceiling_height(zones, *back) if br.point_in_polygon(back, outline) else None
        assert hb is None or hb <= c[2] + 1e-9, (f, hb)


def test_blender04_step_faces_are_the_dormer_cheeks():
    faces = br.shell_faces(br.ROOM4_ZONES)
    steps = [f for f in faces["wall"] if min(p[2] for p in f) > 1e-9]
    assert len(steps) == 2
    for f in steps:
        assert len(f) == 3  # triangle between the slope and the flat dormer ceiling
        assert br.face_area(f) == pytest.approx(0.5 * 1.3 * 1.3)
        assert abs(abs(f[0][1]) - 0.7) < 1e-9
        assert _normal(f)[1] * f[0][1] < 0  # faces into the dormer
    south = [f for f in faces["wall"] if all(abs(p[1] + 2.5) < 1e-9 for p in f)]
    assert sum(br.face_area(f) for f in south) == pytest.approx(2 * 1.6 * 1.8 + 1.6 * 2.6)


def test_blender03_bay_header_and_column():
    faces = br.shell_faces(br.ROOM3_ZONES, br.ROOM3_COLUMNS)
    steps = [f for f in faces["wall"] if min(p[2] for p in f) > 1e-9]
    assert len(steps) == 1
    assert br.face_area(steps[0]) == pytest.approx(1.8 * 0.2)  # header over the bay opening
    assert _normal(steps[0])[0] == pytest.approx(-1.0)  # faces the main room
    assert len(faces["column"]) == 4
    x0, y0, x1, y1 = br.ROOM3_COLUMNS[0]
    assert sum(br.face_area(f) for f in faces["column"]) == pytest.approx(
        2 * ((x1 - x0) + (y1 - y0)) * 2.6
    )
    for f in faces["column"]:  # outward, into the room
        n, c = _normal(f), np.asarray(f).mean(0)
        assert not (x0 < c[0] + 0.01 * n[0] < x1 and y0 < c[1] + 0.01 * n[1] < y1)


def test_shell_faces_match_the_blender02_outline():
    """The zone builder covers the axis-aligned case: blender02's bands as zones give the
    same floor area and wall run as its own builder."""
    zones = [
        ([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], (0.0, 0.0, 2.6))
        for x0, y0, x1, y1 in br.outline_rects(br.ROOM2_OUTLINE)
    ]
    faces = br.shell_faces(zones)
    walls = sum(br.face_area(f) for f in faces["wall"])
    assert walls == pytest.approx(_perimeter(br.ROOM2_OUTLINE) * 2.6)
    assert sum(br.face_area(f) for f in faces["floor"]) == pytest.approx(
        br.polygon_area(br.ROOM2_OUTLINE)
    )
    assert "column" not in faces


def test_world_room_extras():
    ex = br.world_room_extras("blender04")
    assert len(ex["ceiling_zones"]) == len(br.ROOM4_ZONES) and ex["columns_xz"] == []
    west = ex["ceiling_zones"][0]
    a, b, c = west["height"]
    x, z = west["outline_xz"][0]  # (-2.4, 2.5) in world x, z
    assert a * x + b * z + c == pytest.approx(1.0)
    ex3 = br.world_room_extras("blender03")
    x0, y0, x1, y1 = br.ROOM3_COLUMNS[0]
    assert ex3["columns_xz"] == [[x0, -y1, x1, -y0]]


# ---------------------------------------------------------------- furniture


@pytest.mark.parametrize("scene", POLY_SCENES)
def test_objects_table(scene):
    objs = br.SCENES[scene]["objects"]
    names = [o[0] for o in objs]
    assert len(set(names)) == len(names)
    furniture = [o for o in objs if o[5] is None]
    items = [o for o in objs if o[5] is not None]
    assert 6 <= len(furniture) <= 8 and 3 <= len(items) <= 4
    assert all(o[5] in names for o in items)
    assert all(o[2] in br.ASSET_SIZE for o in furniture)
    assert len(objs) < min(br.STRUCTURE.values())


@pytest.mark.parametrize("scene", POLY_SCENES)
def test_furniture_fits_the_room(scene):
    """Every footprint is inside the outline, clear of the columns and of each other, and
    every piece is at least 5 cm below the ceiling over its whole footprint."""
    cfg = br.SCENES[scene]
    obs = br.scene_obstacles(scene)
    for name, _, asset, _, _, support in cfg["objects"]:
        if support:
            continue
        x0, y0, x1, y1 = obs[name]
        h = br.ASSET_SIZE[asset][2]
        for gx in np.linspace(x0, x1, 5):
            for gy in np.linspace(y0, y1, 5):
                assert br.point_in_polygon((gx, gy), cfg["outline"]), (name, gx, gy)
                assert br.ceiling_height(cfg["zones"], gx, gy) >= h + 0.05, (name, gx, gy)
    names = sorted(obs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = obs[a], obs[b]
            assert ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0, (a, b)


def test_blender04_furniture_under_the_slopes():
    """The bed, nightstand, armchair and plant stand where the ceiling is below 2 m."""
    obs = br.furniture_footprints("blender04")
    for name in ("bed", "nightstand", "armchair", "plant"):
        x0, y0, x1, y1 = obs[name]
        low = min(br.ceiling_height(br.ROOM4_ZONES, x, y) for x in (x0, x1) for y in (y0, y1))
        assert low < 2.0, name


def _gt_boxes(scene):
    path = ROOT / "data" / "raw" / f"{scene}_gt" / "objects.json"
    if not path.exists():
        return {}
    objs = json.loads(path.read_text())["objects"]
    return {
        o["name"]: (o["bbox_min"][0], -o["bbox_max"][2], o["bbox_max"][0], -o["bbox_min"][2])
        for o in objs
        if o["kind"] == "furniture"
    }


@pytest.mark.parametrize("scene", POLY_SCENES)
def test_footprints_bound_the_built_meshes(scene):
    gt = _gt_boxes(scene)
    if not gt:
        pytest.skip(f"data/raw/{scene}_gt/objects.json not built")
    fp = br.furniture_footprints(scene)
    for name, (x0, y0, x1, y1) in gt.items():
        f = fp[name]
        assert f[0] - 1e-3 <= x0 and f[1] - 1e-3 <= y0 and x1 <= f[2] + 1e-3 and y1 <= f[3] + 1e-3


# ---------------------------------------------------------------- walks


def _eyes(scene, walk):
    n = int(br.SCENES[scene]["walks"][walk] * br.FPS)
    return [br.camera_matrix(walk, i / br.FPS, scene) for i in range(n)]


@pytest.mark.parametrize("scene", POLY_SCENES)
@pytest.mark.parametrize("walk", ["main", "test"])
def test_walks_keep_clearance(scene, walk):
    """Every frame's camera centre (shake included) is inside the outline, at least 0.35 m
    from every wall, furniture footprint and column, and 0.3 m below the lowest ceiling
    within 0.3 m of it."""
    cfg = br.SCENES[scene]
    obs = [*br.scene_obstacles(scene).items(), *_gt_boxes(scene).items()]
    for i, m in enumerate(_eyes(scene, walk)):
        p = m[:3, 3]
        assert br.point_in_polygon(p[:2], cfg["outline"]), (walk, i, p)
        assert br.wall_distance(p[:2], cfg["outline"]) >= 0.35, (walk, i, p)
        for name, box in obs:
            assert br.box_distance(p[:2], box) >= 0.35, (walk, i, p, name)
        assert br.min_ceiling_near(cfg["zones"], p[0], p[1], 0.3) - p[2] >= 0.3, (walk, i, p)


@pytest.mark.parametrize("scene", POLY_SCENES)
@pytest.mark.parametrize("walk", ["main", "test"])
def test_walks_smooth(scene, walk):
    ms = _eyes(scene, walk)
    eyes = np.array([m[:3, 3] for m in ms])
    assert 0.4 < eyes[:, 2].min() and eyes[:, 2].max() < 1.7
    assert np.linalg.norm(np.diff(eyes, axis=0), axis=1).max() < 0.04  # < 1.2 m/s
    fwd = np.array([-m[:3, 2] for m in ms])
    yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
    turn = np.degrees(np.abs(yaw[16:] - yaw[:-16])) / (16 / br.FPS)
    assert turn.max() < 40  # deg/s over ~0.5 s; the capture guide warns above 40
    assert (fwd[:, 2] < 0).all()


@pytest.mark.parametrize("scene", POLY_SCENES)
def test_main_walk_has_high_and_knee_parts(scene):
    z = [br.camera_matrix("main", t, scene)[2, 3] for t in (5.0, 20.0, 30.0, 38.0)]
    assert z[0] > 1.4 and z[1] > 1.4
    assert z[2] < 0.7 and z[3] < 0.7


def test_blender03_walk_circles_the_column_and_reaches_the_bay():
    x0, y0, x1, y1 = br.ROOM3_COLUMNS[0]
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    eyes = np.array([m[:3, 3] for m in _eyes("blender03", "main")])
    chest = eyes[: 25 * br.FPS]
    ang = np.degrees(np.arctan2(chest[:, 1] - c[1], chest[:, 0] - c[0])) % 360
    assert len(set((ang // 45).astype(int))) == 8  # all round, so behind it too
    behind = (chest[:, 1] > y1) & (chest[:, 0] > x0 - 0.3) & (chest[:, 0] < x1 + 0.3)
    assert behind.sum() > 30  # between the column and the north wall
    bay = (eyes[:, 0] > 1.8) & (eyes[:, 1] > 0.0) & (eyes[:, 1] < 1.8)
    assert bay.sum() > 30  # walks along the bay mouth
    se = br.ROOM3_OUTLINE[1:3]
    assert min(br.segment_distance(p[:2], *se) for p in eyes) < 1.3  # near the angled wall


def test_blender04_walks_enter_the_dormer_and_go_under_the_slopes():
    z = br.ROOM4_ZONES
    eyes = np.array([m[:3, 3] for m in _eyes("blender04", "main")])
    chest, knee = eyes[: 25 * br.FPS], eyes[27 * br.FPS :]
    dormer = z[-1][0]
    assert any(br.point_in_polygon(p[:2], dormer) for p in chest)
    low = np.array([br.ceiling_height(z, p[0], p[1]) for p in knee])
    assert (low < 1.7).any()  # under a slope, where you cannot stand
    assert ((knee[:, 0] < -1.4) & (low < 2.0)).any() and ((knee[:, 0] > 1.4) & (low < 2.0)).any()


# ---------------------------------------------------------------- capture guide


def test_capture_guide_phases_match_the_walks():
    cg = load_script("blender_capture_guide")
    for scene, cfg in br.SCENES.items():
        for walk, seconds in cfg["walks"].items():
            phases = cg.PHASES[scene][walk]
            assert phases[0][0] == 0.0 and phases[-1][1] == pytest.approx(seconds), (scene, walk)


def test_capture_guide_draws_polygon_and_column():
    cg = load_script("blender_capture_guide")
    extras = br.world_room_extras("blender03")
    outline = [(x, -y) for x, y in br.ROOM3_OUTLINE]
    plan = cg.Plan(outline, [], zones=extras["ceiling_zones"], columns=extras["columns_xz"])
    img = np.asarray(plan.base(np.zeros((2, 3))))
    x, y = (int(v) for v in plan.px(2.6, 2.3))  # outside the angled south-east wall
    np.testing.assert_array_equal(img[y, x], [24, 26, 30])
    x, y = (int(v) for v in plan.px(3.1, -0.9))  # inside the bay
    np.testing.assert_array_equal(img[y, x], [52, 48, 44])
    x, y = (int(v) for v in plan.px(0.5, -0.75))  # the column, world z = -y
    np.testing.assert_array_equal(img[y, x], [220, 210, 195])


def test_scene_layout_without_a_build(tmp_path):
    cg = load_script("blender_capture_guide")
    out = cg.scene_layout("blender04", tmp_path / "missing_gt", tmp_path / "l.png", size=500)
    from PIL import Image

    img = Image.open(out)
    assert img.size == (500, 500)
