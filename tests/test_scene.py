import json
import math

import pytest

from movable_reality.scene import BakedPlane, Coverage, Gap, PlaneBake, Scene, SceneObject


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
    s.get("chair#1").completed_ply = "objects/obj_chair_1_completed.ply"
    s.get("chair#1").completed_lod = "lod/obj_chair_1_completed.rad"
    s.planes.append(BakedPlane(id="floor", label="floor", mesh="planes/floor.glb"))
    s.coverage = Coverage(
        gaps=[Gap(id="g1", near_object="table#1", center=(2.5, 0.2, 1.0), radius=0.4, score=0.9)]
    )
    path = tmp_path / "scene.json"
    s.save(path)
    loaded = Scene.load(path)
    assert loaded == s
    assert loaded.coverage is not None and loaded.coverage.gaps[0].status == "open"


def test_plane_bake_section_roundtrip(tmp_path):
    s = make_scene()
    s.planes.append(
        BakedPlane(
            id="floor",
            label="floor",
            mesh="planes/floor.glb",
            removed_splats=120,
            normal=(0.0, 1.0, 0.0),
            offset=0.0,
            corners=[(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 0.0, -5.0), (0.0, 0.0, -5.0)],
            texel_m=0.005,
            texture_px=(800, 1000),
            fit_rms_m=0.003,
            filled_fraction=0.05,
        )
    )
    s.plane_bake = PlaneBake(
        source_background_ply="objects/background.ply",
        background_ply="objects/background_baked.ply",
        removed_splats=120,
    )
    s.background_ply = s.plane_bake.background_ply
    path = tmp_path / "scene.json"
    s.save(path)
    loaded = Scene.load(path)
    assert loaded == s
    assert loaded.plane_bake is not None and loaded.plane_bake.ids == "ids/gaussian_ids.npy"
    # The minimal plane written before these fields existed still loads.
    old = BakedPlane.model_validate({"id": "f", "label": "floor", "mesh": "planes/f.glb"})
    assert old.corners == [] and old.normal is None and old.filled_fraction == 0.0


def test_manifest_without_lane_sections_still_loads(tmp_path):
    path = tmp_path / "scene.json"
    data = make_scene().model_dump()
    for key in ("background_lod", "planes", "plane_bake", "coverage"):
        data.pop(key)
    for obj in data["objects"]:
        obj.pop("lod")
        obj.pop("completed_lod")
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = Scene.load(path)
    assert loaded.planes == [] and loaded.coverage is None and loaded.background_lod is None
    assert loaded.plane_bake is None
    assert all(o.completed_lod is None for o in loaded.objects)


def test_place_sets_centre_and_undo_restores():
    s = make_scene()
    s.move("chair#1", 0.1, 0.0, 0.0)
    s.apply_edit("place", "chair#1", {"x": 2.0, "z": -1.0})
    mn, mx = s.get("chair#1").world_aabb()
    assert math.isclose((mn[0] + mx[0]) / 2, 2.0)
    assert math.isclose((mn[2] + mx[2]) / 2, -1.0)
    s.apply_edit("undo")
    assert s.get("chair#1").transform.translation == pytest.approx((0.1, 0.0, 0.0))


def test_undo_walks_back_through_history():
    s = make_scene()
    s.apply_edit("move", "chair#1", {"dx": 1.0})
    s.apply_edit("rotate", "table#1", {"yaw_deg": 90})
    s.apply_edit("remove", "chair#1")
    s.apply_edit("undo")
    assert s.get("chair#1").visible
    s.apply_edit("undo")
    assert s.get("table#1").transform.yaw_deg == 0
    s.apply_edit("undo")
    assert s.get("chair#1").transform.translation == (0.0, 0.0, 0.0)
    assert s.undo() is None  # nothing left
    assert [e.op for e in s.history].count("undo") == 3


def test_apply_edit_rejects_unknown_op():
    with pytest.raises(ValueError):
        make_scene().apply_edit("explode", "chair#1")


def test_undo_remove_twice_restores_previous_visibility():
    s = make_scene()
    s.apply_edit("remove", "chair#1")
    s.apply_edit("remove", "chair#1")
    s.apply_edit("undo")
    assert not s.get("chair#1").visible


def test_apply_edit_rejects_nan():
    with pytest.raises(ValueError):
        make_scene().apply_edit("move", "chair#1", {"dx": float("nan")})
