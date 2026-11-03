"""Scene blender07, the modern living room (scripts/blender_living.py builds it): its entry in
blender_room.SCENES and the camera walks. CPU only, no Blender."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import blender_room as br  # noqa: E402

CFG = br.SCENES["blender07"]


def _eyes(walk: str, step: int = 3) -> np.ndarray:
    n = int(CFG["walks"][walk] * br.FPS)
    return np.array([br.camera_matrix(walk, i / br.FPS, "blender07")[:3, 3]
                     for i in range(0, n, step)])  # fmt: skip


def test_scene_entry():
    assert CFG["builder"] == "blender_living"
    assert CFG["room"] == (5.6, 4.8, 2.7)
    assert CFG["walks"] == {"main": 122.0, "test": 30.0}
    assert br.scene_paths("blender07")[0].name == "blender07"


@pytest.mark.parametrize("walk", ["main", "test"])
def test_walk_stays_inside_the_room(walk):
    e = _eyes(walk)
    h = CFG["room"][2]
    hx, hy = CFG["room"][0] / 2, CFG["room"][1] / 2
    assert (np.abs(e[:, 0]) < hx - 0.25).all()
    assert (np.abs(e[:, 1]) < hy - 0.25).all()
    assert e[:, 2].min() > 0.5 and e[:, 2].max() < h - 0.9


@pytest.mark.parametrize("walk", ["main", "test"])
def test_walk_keeps_clear_of_furniture(walk):
    """Below an obstacle's top (+10 cm), the eye stays 15 cm outside its footprint."""
    for x, y, z in _eyes(walk, step=1):
        for name, (x0, y0, x1, y1, top) in br.LIVING_OBSTACLES.items():
            if z > top + 0.1:
                continue
            d = np.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1))
            assert d >= 0.15, f"{walk}: eye ({x:.2f}, {y:.2f}, {z:.2f}) {d:.2f} m from {name}"


@pytest.mark.parametrize("walk", ["main", "test"])
def test_walker_is_never_over_the_furniture(walk):
    """The walker's body (the eye's x, y) stays 10 cm outside every footprint at any height,
    also at chest height over the coffee table and the pouf."""
    for x, y, _ in _eyes(walk, step=1):
        for name, (x0, y0, x1, y1, _) in br.LIVING_OBSTACLES.items():
            d = np.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1))
            assert d >= 0.1, f"{walk}: eye ({x:.2f}, {y:.2f}) {d:.2f} m from {name}"


@pytest.mark.parametrize("walk", ["main", "test"])
def test_camera_turns_like_a_handheld_phone(walk):
    """No whip pans: under 75 deg/s between frames (blender07 peaks at about 53)."""
    n = int(CFG["walks"][walk] * br.FPS)
    rot = [br.camera_matrix(walk, i / br.FPS, "blender07")[:3, :3] for i in range(n)]
    cos = [(np.trace(a.T @ b) - 1) / 2 for a, b in zip(rot, rot[1:], strict=False)]
    rate = np.degrees(np.arccos(np.clip(cos, -1, 1))) * br.FPS
    assert rate.max() < 75, f"{walk}: {rate.max():.0f} deg/s at {rate.argmax() / br.FPS:.1f} s"


# GT ids (objects.json) each obstacle box must hold; world (x, y up, z) is Blender (x, -z, y).
OBSTACLE_GT = {"media_console": ["tv_stand", "game_console", "succulent#5", "vase#6"],
               "floor_vase": ["vase#7"], "sofa": ["sofa", "pillow", "blanket"],
               "side_table": ["side_table", "table_lamp", "books#15"],
               "coffee_table": ["coffee_table", "tray", "candle#18", "remote", "books#20",
                                "bowl#21", "mug"],
               "pouf": ["pouf"], "armchair": ["armchair"], "floor_lamp": ["floor_lamp"],
               "bookshelf": ["bookshelf", "basket", "books#29", "books#30", "vase#31", "statue",
                             "books#33", "picture_frame", "books#35", "candle#36",
                             "succulent#37", "books#38", "vase#39", "bowl#40"],
               "tall_plant": ["potted_plant#41"], "corner_plant": ["potted_plant#42"],
               "radiators_curtains": ["radiator", "curtains"]}  # fmt: skip


def test_obstacles_hold_the_built_furniture():
    """LIVING_OBSTACLES are hand-copied: check them against the last build's objects.json."""
    path = Path(__file__).resolve().parents[1] / "data/raw/blender07_gt/objects.json"
    if not path.exists():
        pytest.skip("blender07 not built (blender_room.py --scene blender07 --stage build)")
    objs = json.loads(path.read_text(encoding="utf8"))["objects"]
    assert set(OBSTACLE_GT) == set(br.LIVING_OBSTACLES)
    for name, ids in OBSTACLE_GT.items():
        x0, y0, x1, y1, top = br.LIVING_OBSTACLES[name]
        for o in objs:
            if o["id"] in ids or o["id"].split("#")[0] in ids:
                (ax, ay, az), (bx, by, bz) = o["bbox_min"], o["bbox_max"]
                lo, hi = (ax, -bz, ay), (bx, -az, by)  # Blender x, y, z
                ok = (lo[0] >= x0 - 0.02 and lo[1] >= y0 - 0.02 and hi[0] <= x1 + 0.02
                      and hi[1] <= y1 + 0.02 and hi[2] <= top + 0.02)  # fmt: skip
                assert ok, f"{o['id']} ({lo}, {hi}) is outside obstacle {name}"


def test_main_walk_visits_every_part_of_the_room():
    """The chest tour reaches every pocket of free floor, not only the ring round the table."""
    e = _eyes("main", step=1)
    for name, (x, y) in {"south-west": (-1.9, -1.5), "TV wall": (-1.85, -0.4),
                         "north-west": (-1.5, 1.45), "north-east": (1.4, 1.55),
                         "bookshelf": (1.35, -1.55), "sofa": (1.5, 0.6)}.items():  # fmt: skip
        assert np.hypot(e[:, 0] - x, e[:, 1] - y).min() < 0.1, name


def test_main_walk_has_chest_knee_and_upward_parts():
    n = int(CFG["walks"]["main"] * br.FPS)
    z = np.array([br.camera_matrix("main", i / br.FPS, "blender07")[2, 3] for i in range(n)])
    assert (z[: 64 * br.FPS] > 1.4).all()  # chest tour
    assert (np.abs(z[68 * br.FPS : 102 * br.FPS] - 0.6) < 0.1).all()  # knee ring
    # Last part looks up: the camera's -Z axis (view direction) points above the horizon.
    view = -np.array([br.camera_matrix("main", t, "blender07")[:3, 2] for t in (108.0, 116.0)])
    assert (view[:, 2] > 0.15).all()


def test_walk_speed_is_handheld():
    for walk in ("main", "test"):
        e = _eyes(walk, step=1)
        v = np.linalg.norm(np.diff(e, axis=0), axis=1) * br.FPS
        assert v.max() < 1.6, f"{walk}: {v.max():.2f} m/s"
