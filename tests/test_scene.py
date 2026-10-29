import json
import math

import pytest

from movable_reality.scene import BakedPlane, Coverage, Gap, Scene, SceneObject


def make_scene() -> Scene:
    return Scene(
        scene_id="t",
        objects=[
            SceneObject(
                id="chair#1",
                label="chair",
                ply="objects/chair_1.ply",
                bbox_min=(0.0, 0.0, 0.0),
                bbox_max=(0.5, 1.0, 0.5),
            ),
            SceneObject(
                id="table#1",
                label="table",
                ply="objects/table_1.ply",
                bbox_min=(2.0, 0.0, 0.0),
                bbox_max=(3.0, 0.75, 2.0),
            ),
        ],
    )


def test_roundtrip(tmp_path):
    s = make_scene()
    s.move("chair#1", 1.0, 0.0, 0.0, instruction="move the chair right")
    path = tmp_path / "scene.json"
    s.save(path)
    loaded = Scene.load(path)
    assert loaded == s
    assert loaded.history[0].op == "move"
    assert not path.with_suffix(".json.tmp").exists()


def test_move_and_aabb():
    s = make_scene()
    s.move("chair#1", 1.0, 0.0, 2.0)
    mn, mx = s.get("chair#1").world_aabb()
    assert mn == pytest.approx((1.0, 0.0, 2.0))
    assert mx == pytest.approx((1.5, 1.0, 2.5))


def test_yaw_grows_aabb_footprint():
    s = make_scene()
    s.rotate("table#1", 45.0)
    mn, mx = s.get("table#1").world_aabb()
    width_x = mx[0] - mn[0]
    expected = (1.0 / 2 + 2.0 / 2) * math.sqrt(2)  # (hx + hz) * cos45 * 2
    assert width_x == pytest.approx(expected)


def test_overlap():
    s = make_scene()
    assert not s.overlaps("chair#1", "table#1")
    s.move("chair#1", 1.8, 0.0, 0.5)
    assert s.overlaps("chair#1", "table#1")


def test_remove_restore_keeps_object():
    s = make_scene()
    s.set_visible("chair#1", False)
    assert not s.get("chair#1").visible
    s.set_visible("chair#1", True)
    assert [h.op for h in s.history] == ["remove", "restore"]
    assert len(s.objects) == 2


def test_unknown_object_lists_known_ids():
    with pytest.raises(KeyError, match="chair#1"):
        make_scene().get("sofa#9")


def test_lane_sections_roundtrip(tmp_path):
    s = make_scene()
    s.background_lod = "lod/background.rad"
    s.get("chair#1").lod = "lod/obj_chair_1.rad"
    s.planes.append(BakedPlane(id="floor", label="floor", mesh="planes/floor.glb"))
    s.coverage = Coverage(
        gaps=[Gap(id="g1", near_object="table#1", center=(2.5, 0.2, 1.0), radius=0.4, score=0.9)]
    )
    path = tmp_path / "scene.json"
    s.save(path)
    loaded = Scene.load(path)
    assert loaded == s
    assert loaded.coverage is not None and loaded.coverage.gaps[0].status == "open"


def test_manifest_without_lane_sections_still_loads(tmp_path):
    path = tmp_path / "scene.json"
    data = make_scene().model_dump()
    for key in ("background_lod", "planes", "coverage"):
        data.pop(key)
    for obj in data["objects"]:
        obj.pop("lod")
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = Scene.load(path)
    assert loaded.planes == [] and loaded.coverage is None and loaded.background_lod is None
