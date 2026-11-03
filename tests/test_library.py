"""The mesh library: packing a glTF, adding items to a scene, the agent tool, the server route."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from movable_reality import library, server
from movable_reality.agent import run_tool
from movable_reality.agent.loop import system_prompt
from movable_reality.agent.tools import RoomBounds, fits, free_spot
from movable_reality.agent.verify import edit_focus
from movable_reality.library import (
    LibraryItem,
    add_to_scene,
    pack_model,
    read_glb_json,
    scene_bounds,
)
from movable_reality.publish import without_edits
from movable_reality.scene import Scene, SceneObject

BOUNDS = RoomBounds((-3.0, 3.0), (-3.0, 3.0), 2.5)


def _item(id_: str = "sofa", size=(2.0, 0.8, 1.0), on_surface=False) -> LibraryItem:
    return LibraryItem(id=id_, label=id_, category="Seating", on_surface=on_surface, size=size,
                       glb=f"{id_}.glb")  # fmt: skip


def _scene() -> Scene:
    table = SceneObject(id="table#1", label="table", ply="objects/t.ply",
                        bbox_min=(1.0, 0.0, 1.0), bbox_max=(2.0, 0.7, 2.0))  # fmt: skip
    return Scene(scene_id="s", objects=[table])


@pytest.fixture
def catalog_dir(tmp_path: Path, monkeypatch) -> Path:
    items = [_item(), _item("vase", (0.2, 0.4, 0.2), True)]
    (tmp_path / "catalog.json").write_text(json.dumps([i.model_dump(mode="json") for i in items]))
    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path)
    return tmp_path


def _gltf(folder: Path) -> Path:
    """One 1 x 2 x 3 m box-shaped accessor under a node moved to (5, 1, -2)."""
    positions = np.array([[-0.5, -1, -1.5], [0.5, 1, 1.5], [0.5, -1, 1.5]], dtype=np.float32)
    (folder / "m.bin").write_bytes(positions.tobytes())
    Image.new("RGB", (64, 64), (200, 10, 10)).save(folder / "c.png")
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "translation": [5, 1, -2]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "textures": [{"source": 0}],
        "images": [{"uri": "c.png"}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                       "min": [-0.5, -1, -1.5], "max": [0.5, 1, 1.5]}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes}],
        "buffers": [{"uri": "m.bin", "byteLength": positions.nbytes}],
    }  # fmt: skip
    path = folder / "m.gltf"
    path.write_text(json.dumps(gltf))
    return path


def test_pack_model_stands_it_on_the_floor_and_embeds_everything(tmp_path: Path):
    out = tmp_path / "m.glb"
    size, tris = pack_model(_gltf(tmp_path), out, colour_px=16)
    assert size == pytest.approx((1, 2, 3))
    assert tris == 1
    gltf = read_glb_json(out)
    lo, hi = scene_bounds(gltf)
    assert lo == pytest.approx([-0.5, 0, -1.5])
    assert hi == pytest.approx([0.5, 2, 1.5])
    assert "uri" not in gltf["buffers"][0]
    assert "uri" not in gltf["images"][0] and gltf["images"][0]["mimeType"] == "image/jpeg"
    assert gltf["bufferViews"][gltf["images"][0]["bufferView"]]["byteLength"] > 0


def test_library_object_goes_through_the_normal_edits():
    scene = _scene()
    obj = add_to_scene(scene, _item(), 0.0, -1.0, yaw_deg=-90.0)
    assert (obj.id, obj.source, obj.library_id, obj.ply) == ("lib#1", "library", "sofa", "")
    assert obj.bbox_min == (-1.0, 0.0, -1.5)
    assert obj.bbox_max == (1.0, 0.8, -0.5)
    assert obj.transform.yaw_deg == 270.0
    assert add_to_scene(scene, _item(), 2.0, -2.0).id == "lib#2"
    scene.apply_edit("place", "lib#1", {"x": 1.0, "z": 1.0})
    assert scene.get("lib#1").world_aabb()[0][1] == 0.0
    scene.apply_edit("remove", "lib#1")
    assert not scene.get("lib#1").visible
    scene.undo()
    scene.undo()
    assert scene.get("lib#1").visible
    assert scene.get("lib#1").transform.translation == (0, 0, 0)


def test_undo_of_an_add_hides_it_and_publishing_drops_it():
    scene = _scene()
    add_to_scene(scene, _item(), 0.0, 0.0)
    assert scene.undo().op == "insert"
    assert not scene.get("lib#1").visible
    add_to_scene(scene, _item(), 0.0, 0.0)
    assert [o.id for o in without_edits(scene).objects] == ["table#1"]


def test_add_on_top_of_an_object_uses_its_edited_top():
    scene = _scene()
    scene.apply_edit("move", "table#1", {"dx": 1.0, "dz": 0.0})
    obj = add_to_scene(scene, _item("vase", (0.2, 0.4, 0.2)), on_object_id="table#1")
    assert obj.bbox_min[1] == pytest.approx(0.7)
    assert obj.center[0] == pytest.approx(2.5)
    assert obj.center[2] == pytest.approx(1.5)
    with pytest.raises(ValueError):
        add_to_scene(scene, _item())


def test_fits_gives_an_added_object_no_exemption_for_its_first_pose():
    scene = _scene()
    add_to_scene(scene, _item(), 1.5, 1.5)
    hits = fits(scene, "lib#1", None, None, BOUNDS)["collides_with"]
    assert [c["id"] for c in hits] == ["table#1"]


def test_free_spot_keeps_a_free_point_and_avoids_walls_and_objects():
    scene = _scene()
    size = (1.0, 0.8, 1.0)
    assert free_spot(scene, size, 0.0, -1.0, -1.0, BOUNDS) == (-1.0, -1.0)
    # past the wall: back inside, half a size from it
    assert free_spot(scene, size, 0.0, -9.0, 0.0, BOUNDS) == (-2.5, 0.0)
    x, z = free_spot(scene, size, 0.0, 1.5, 1.5, BOUNDS)
    assert (x, z) != (1.5, 1.5)
    assert x + 0.5 < 1.0 or x - 0.5 > 2.0 or z + 0.5 < 1.0 or z - 0.5 > 2.0  # clear of the table
    # 90 degrees swaps the footprint: 2 m deep no longer fits 0.5 m from the z wall
    assert free_spot(scene, (2.0, 0.8, 0.5), 90.0, -2.0, -2.9, BOUNDS)[1] == pytest.approx(-2.0)


def test_agent_tool_and_prompt(catalog_dir: Path):
    scene = _scene()
    def add(**args):
        return run_tool(scene, "add_from_library", args, BOUNDS, "add a sofa")

    out = add(library_id="sofa", x=-1.0, z=-1.0)
    assert out["ok"] and out["object"]["source"] == "library" and "warning" not in out
    assert scene.history[-1].instruction == "add a sofa"
    # no x and z, null yaw, on a table
    out = add(library_id="vase", on_object_id="table#1", yaw_deg=None)
    assert out["object"]["footprint"]["x"] == pytest.approx([1.4, 1.6])
    assert "error" in add(library_id="throne", x=0, z=0)
    assert "error" in add(library_id="sofa")
    # colliding with the table: added, and the model hears about it
    assert "warning" in add(library_id="sofa", x=1.5, z=1.5)
    text = system_prompt(scene, BOUNDS, None)
    assert "sofa: sofa, 2.0 x 0.8 x 1.0" in text
    assert "vase: vase, 0.2 x 0.4 x 0.2, small" in text
    tool = "add_from_library"
    assert edit_focus(scene, tool, {"library_id": "sofa", "x": 1, "z": 2})[0] == [1.0, 0.4, 2.0]
    assert edit_focus(scene, tool, {"library_id": "vase", "on_object_id": "table#1"}) is not None


def test_server_route_adds_and_nudges(catalog_dir: Path, tmp_path: Path, monkeypatch):
    scenes = tmp_path / "scenes"
    _scene().save(scenes / "s" / "scene.json")
    monkeypatch.setattr(server, "SCENES_DIR", scenes)
    monkeypatch.setattr(server, "_hubs", {})
    client = TestClient(server.app)

    def post(**body):
        return client.post("/api/scenes/s/library", json=body)

    # asked for the middle of the table: on the floor it moves clear of it
    r = post(library_id="vase", x=1.5, z=1.5)
    assert r.status_code == 200 and r.json()["id"] == "lib#1"
    saved = Scene.load(scenes / "s" / "scene.json")
    assert saved.get("lib#1").center[0] != 1.5 or saved.get("lib#1").center[2] != 1.5
    assert post(library_id="vase", on_object_id="table#1").status_code == 200
    assert post(library_id="vase", x=1e308, z=0, on_object_id="table#1").status_code == 400
    assert Scene.load(scenes / "s" / "scene.json").get("lib#2").bbox_min[1] == pytest.approx(0.7)
    assert post(library_id="throne", x=0, z=0).status_code == 400
    assert post(library_id="sofa").status_code == 400
    assert post(library_id="vase", on_object_id="nope").status_code == 400


def test_add_on_top_stays_over_the_object():
    scene, vase = _scene(), _item("vase", (0.2, 0.4, 0.2))
    # where a view from the door would hit the floor, far in front of the table
    obj = add_to_scene(scene, vase, 1.5, 0.23, on_object_id="table#1")
    assert obj.center[0] == pytest.approx(1.5) and obj.center[2] == pytest.approx(1.1)
    assert obj.bbox_min[1] == pytest.approx(0.7)
    # a long item turned 90 deg: its width runs along z
    long = _item("runner", (0.6, 0.1, 0.2))
    obj = add_to_scene(scene, long, 0.0, 1.0, 90.0, "table#1")
    assert obj.center[0] == pytest.approx(1.1) and obj.center[2] == pytest.approx(1.3)
    # wider than the table: centred
    obj = add_to_scene(scene, _item(), 9.0, 9.0, on_object_id="table#1")
    assert obj.center[0] == pytest.approx(1.5) and obj.center[2] == pytest.approx(1.5)


def test_add_on_a_removed_object_is_refused():
    scene = _scene()
    scene.apply_edit("remove", "table#1", {})
    with pytest.raises(ValueError):
        add_to_scene(scene, _item("vase", (0.2, 0.4, 0.2)), on_object_id="table#1")


def test_a_scanned_object_moved_back_onto_an_added_one_collides():
    scene = _scene()
    scene.apply_edit("move", "table#1", {"dx": -3.0, "dz": -3.0})
    sofa = add_to_scene(scene, _item(), 1.5, 1.5)
    scene.apply_edit("move", "table#1", {"dx": 3.0, "dz": 3.0})
    check = fits(scene, "table#1", None, None, BOUNDS)
    assert [c["id"] for c in check["collides_with"]] == [sofa.id]


def test_every_curated_item_has_a_licence_record():
    """blender_assets.py records the licence; build_library.py refuses an item without one."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from blender_assets import MODELS

    assert not [c.asset for c in library.CURATED if c.asset not in MODELS]
