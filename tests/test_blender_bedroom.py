"""Scene blender06, the realistic bedroom (scripts/blender_bedroom.py builds it): its entry in
blender_room.SCENES and the camera walks. CPU only, no Blender."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import blender_room as br  # noqa: E402

CFG = br.SCENES["blender06"]


def _eyes(walk: str, step: int = 3) -> np.ndarray:
    n = int(CFG["walks"][walk] * br.FPS)
    return np.array([br.camera_matrix(walk, i / br.FPS, "blender06")[:3, 3]
                     for i in range(0, n, step)])  # fmt: skip


def test_scene_entry():
    assert CFG["builder"] == "blender_bedroom"
    assert CFG["room"] == (3.6, 4.2, 2.6)
    assert CFG["walks"] == {"main": 54.0, "test": 20.0}
    assert br.scene_paths("blender06")[0].name == "blender06"


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
        for name, (x0, y0, x1, y1, top) in br.BEDROOM_OBSTACLES.items():
            if z > top + 0.1:
                continue
            d = np.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1))
            assert d >= 0.15, f"{walk}: eye ({x:.2f}, {y:.2f}, {z:.2f}) {d:.2f} m from {name}"


@pytest.mark.parametrize("walk", ["main", "test"])
def test_walker_is_never_over_the_bed_or_the_chair(walk):
    """The walker's body (the eye's x, y) stays 10 cm outside every footprint at any height:
    the chair closes the corridor by the bed foot, so nobody stands over it or the bed."""
    for x, y, _ in _eyes(walk, step=1):
        for name, (x0, y0, x1, y1, _) in br.BEDROOM_OBSTACLES.items():
            d = np.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1))
            assert d >= 0.1, f"{walk}: eye ({x:.2f}, {y:.2f}) {d:.2f} m from {name}"


@pytest.mark.parametrize("walk", ["main", "test"])
def test_camera_turns_like_a_handheld_phone(walk):
    """No whip pans: under 75 deg/s between frames (blender02 peaks at about 41)."""
    n = int(CFG["walks"][walk] * br.FPS)
    rot = [br.camera_matrix(walk, i / br.FPS, "blender06")[:3, :3] for i in range(n)]
    cos = [(np.trace(a.T @ b) - 1) / 2 for a, b in zip(rot, rot[1:], strict=False)]
    rate = np.degrees(np.arccos(np.clip(cos, -1, 1))) * br.FPS
    assert rate.max() < 75, f"{walk}: {rate.max():.0f} deg/s at {rate.argmax() / br.FPS:.1f} s"


# GT ids (objects.json) each obstacle box must hold; world (x, y up, z) is Blender (x, -z, y).
OBSTACLE_GT = {"bed": ["bed", "duvet", "blanket", "pillow"], "nightstand": ["nightstand"],
               "plant": ["potted_plant"], "desk": ["desk"], "desk_lamp": ["desk_lamp"],
               "chair": ["chair", "sweater"], "backpack": ["backpack"], "bin": ["waste_bin"],
               "shelf": ["shelf"], "dresser": ["chest_of_drawers"], "basket": ["laundry_basket"],
               "ukulele": ["ukulele"], "radiator_curtains": ["radiator", "curtains"]}  # fmt: skip


def test_obstacles_hold_the_built_furniture():
    """BEDROOM_OBSTACLES are hand-copied: check them against the last build's objects.json."""
    path = Path(__file__).resolve().parents[1] / "data/raw/blender06_gt/objects.json"
    if not path.exists():
        pytest.skip("blender06 not built (blender_room.py --scene blender06 --stage build)")
    objs = json.loads(path.read_text(encoding="utf8"))["objects"]
    assert set(OBSTACLE_GT) == set(br.BEDROOM_OBSTACLES)
    for name, ids in OBSTACLE_GT.items():
        x0, y0, x1, y1, top = br.BEDROOM_OBSTACLES[name]
        for o in objs:
            if o["id"].split("#")[0] in ids:
                (ax, ay, az), (bx, by, bz) = o["bbox_min"], o["bbox_max"]
                lo, hi = (ax, -bz, ay), (bx, -az, by)  # Blender x, y, z
                ok = (lo[0] >= x0 - 0.02 and lo[1] >= y0 - 0.02 and hi[0] <= x1 + 0.02
                      and hi[1] <= y1 + 0.02 and hi[2] <= top + 0.02)  # fmt: skip
                assert ok, f"{o['id']} ({lo}, {hi}) is outside obstacle {name}"


def test_main_walk_has_chest_knee_and_upward_parts():
    n = int(CFG["walks"]["main"] * br.FPS)
    z = np.array([br.camera_matrix("main", i / br.FPS, "blender06")[2, 3] for i in range(n)])
    assert (z[: 24 * br.FPS] > 1.4).all()  # chest loop
    assert (np.abs(z[27 * br.FPS : 37 * br.FPS] - 0.6) < 0.1).all()  # knee loop
    # Last part looks up: the camera's -Z axis (view direction) points above the horizon.
    view = -np.array([br.camera_matrix("main", t, "blender06")[:3, 2] for t in (45.0, 50.0)])
    assert (view[:, 2] > 0.15).all()


def test_walk_speed_is_handheld():
    for walk in ("main", "test"):
        e = _eyes(walk, step=1)
        v = np.linalg.norm(np.diff(e, axis=0), axis=1) * br.FPS
        assert v.max() < 1.6, f"{walk}: {v.max():.2f} m/s"
