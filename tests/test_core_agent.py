import asyncio
import io
import json
import wave

import pytest

from movable_reality.agent import llm, loop
from movable_reality.agent.loop import run_command, view_directions
from movable_reality.agent.tools import (
    RoomBounds,
    fits,
    measure,
    room_bounds,
    room_bounds_from_points,
    run_tool,
    step_text,
)
from movable_reality.agent.verify import Verdict, edit_focus, parse_verdict, verify_camera
from movable_reality.scene import Scene, SceneObject
from movable_reality.voice import check_wav

ROOM = RoomBounds((-2.0, 2.0), (-2.5, 2.5), 2.6)


def make_scene() -> Scene:
    return Scene(
        scene_id="t",
        objects=[
            SceneObject(id="sofa#1", label="sofa", ply="a.ply",
                        bbox_min=(-1.0, 0.0, -2.4), bbox_max=(1.0, 0.9, -1.5)),
            SceneObject(id="table#1", label="coffee table", ply="b.ply",
                        bbox_min=(-0.5, 0.0, -0.8), bbox_max=(0.5, 0.45, -0.2)),
            SceneObject(id="vase#1", label="vase", ply="c.ply",
                        bbox_min=(-0.1, 0.45, -0.6), bbox_max=(0.1, 0.75, -0.4)),
        ],
    )  # fmt: skip


def test_fits_reports_collision_and_walls():
    s = make_scene()
    assert fits(s, "table#1", 0.0, 1.0, ROOM)["fits"]
    hit = fits(s, "table#1", 0.0, -1.8, ROOM)
    assert not hit["fits"] and hit["collides_with"][0]["id"] == "sofa#1"
    out = fits(s, "table#1", 1.9, 1.0, ROOM)
    assert not out["fits"] and out["outside_room"]


def test_fits_ignores_items_stacked_on_top():
    # the vase stands on the table: its footprint is inside, but it is above the table top
    assert fits(make_scene(), "table#1", None, None, ROOM)["fits"]


def test_fits_allows_overlaps_that_were_scanned():
    s = make_scene()
    s.objects.append(SceneObject(id="chair#1", label="chair", ply="d.ply",
                                 bbox_min=(0.3, 0.0, -1.0), bbox_max=(0.8, 0.9, -0.5)))  # fmt: skip
    # tucked against the table in the scan: turning it in place is fine
    assert fits(s, "chair#1", None, None, ROOM, yaw_deg=45)["fits"]
    # moved onto the sofa it collides
    assert not fits(s, "chair#1", 0.0, -1.9, ROOM)["fits"]
    # pushed further into the table it collides with the table
    hit = fits(s, "chair#1", 0.0, -0.5, ROOM)
    assert "table#1" in [c["id"] for c in hit["collides_with"]]


def test_fits_scan_overlap_needs_both_objects_at_their_scan_spots():
    s = make_scene()
    # the sofa moved onto the table's spot: it never overlapped there in the scan
    run_tool(s, "place_object", {"object_id": "sofa#1", "x": 0.0, "z": -0.5}, ROOM)
    assert any(c["id"] == "sofa#1" for c in fits(s, "table#1", None, None, ROOM)["collides_with"])


def test_measure_reports_top_above_floor():
    s = make_scene()
    s.objects[0].bbox_min = (-1.0, 0.1, -2.4)  # legs cut from the box, as on blender01
    a = measure(s, "sofa#1")["a"]
    assert a["size_m"][1] == pytest.approx(0.8)
    assert a["top_above_floor_m"] == pytest.approx(0.9)


def test_measure_gap():
    m = measure(make_scene(), "sofa#1", "table#1")
    assert m["floor_gap_m"] == pytest.approx(0.7)


def test_run_tool_edits_and_errors():
    s = make_scene()
    out = run_tool(s, "move_object", {"object_id": "table#1", "dx": 0.5, "dz": 0}, ROOM, "go")
    assert out["ok"] and s.history[-1].instruction == "go"
    assert "error" in run_tool(s, "remove_object", {"object_id": "nope#9"}, ROOM)
    assert run_tool(s, "undo", {}, ROOM)["ok"]
    assert s.get("table#1").transform.translation == (0.0, 0.0, 0.0)


def test_room_bounds_drops_floaters():
    import numpy as np

    rng = np.random.default_rng(0)
    pts = rng.uniform([-2, 0, -2.5], [2, 2.6, 2.5], size=(20000, 3))
    pts[:10] = 50.0  # floaters
    b = room_bounds_from_points(pts)
    assert -2.1 < b.x[0] < -1.9 and 1.9 < b.x[1] < 2.1


def test_view_directions_right_hand():
    _, fwd, right = view_directions({"eye": [0, 1.6, 0], "forward": [0, -0.3, -1]})
    assert fwd == [0.0, 0.0, -1.0] and right == [1.0, 0.0, 0.0]


def test_parse_verdict():
    assert parse_verdict("YES\nThe chair moved left.").ok
    assert not parse_verdict("No, the lamp is still there.").ok


class FakeClient:
    """Stands in for the OpenAI client: replays scripted assistant messages."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.chat = self
        self.completions = self

    async def create(self, **payload):
        msg = self.scripted.pop(0)

        class R:
            def model_dump(self, exclude_none=True):
                return {"choices": [{"message": msg}]}

        return R()


def test_agent_loop_runs_tools_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CHAT_CACHE", tmp_path)
    args = json.dumps({"object_id": "table#1", "dx": -0.5, "dz": 0})
    call = {"id": "c1", "type": "function", "function": {"name": "move_object", "arguments": args}}
    script = [{"role": "assistant", "content": "", "tool_calls": [call]},
              {"role": "assistant", "content": "Moved the table left."}]  # fmt: skip
    s = make_scene()
    res = asyncio.run(run_command(s, "move the table left", ROOM, client=FakeClient(script)))
    assert res.reply == "Moved the table left." and res.edited
    assert s.get("table#1").transform.translation == (-0.5, 0.0, 0.0)
    # the same command again is served from the cache: an empty script would raise otherwise
    s2 = make_scene()
    asyncio.run(run_command(s2, "move the table left", ROOM, client=FakeClient([])))
    assert s2.get("table#1").transform.translation == (-0.5, 0.0, 0.0)


def test_edit_focus_covers_the_move():
    s = make_scene()
    c, r = edit_focus(s, "move_object", {"object_id": "table#1", "dx": 1.0, "dz": 0})
    assert c[0] == pytest.approx(0.5) and r > 0.5  # midway, radius reaches both ends
    run_tool(s, "rotate_object", {"object_id": "sofa#1", "yaw_deg": 90}, ROOM)
    assert edit_focus(s, "undo", {})[0][2] == pytest.approx(-1.95)  # undo frames the sofa
    assert edit_focus(s, "measure", {}) is None
    assert edit_focus(s, "move_object", {"object_id": "table#1", "dx": None}) is None
    assert edit_focus(s, "move_object", {"object_id": "nope#9", "dx": 1}) is None


def test_edit_focus_undo_covers_where_it_goes_back_to():
    s = make_scene()
    run_tool(s, "move_object", {"object_id": "sofa#1", "dx": 3.0, "dz": 0}, ROOM)
    c, r = edit_focus(s, "undo", {})
    assert c[0] == pytest.approx(1.5) and r > 1.5
    s2 = make_scene()
    run_tool(s2, "place_object", {"object_id": "sofa#1", "x": 3.0, "z": -1.95}, ROOM)
    assert edit_focus(s2, "undo", {})[0][0] == pytest.approx(1.5)


def test_verify_camera_stays_in_the_room_and_looks_down():
    cam = verify_camera([0.0, 0.4, -1.95], 1.0, ROOM, {"eye": [0, 1.6, 0], "forward": [0, 0, -1]})
    ex, ey, ez = cam["eye"]
    assert ROOM.x[0] < ex < ROOM.x[1] and ROOM.z[0] < ez < ROOM.z[1]
    assert 0.4 < ey < ROOM.height and ez > -1.95  # above it, on the user's side
    assert cam["target"] == [0.0, 0.4, -1.95]


def test_step_text_names_the_object():
    s = make_scene()
    assert step_text(s, "move_object", {"object_id": "table#1"}) == "Moving the coffee table"
    assert step_text(s, "measure", {"id_a": "nope"}) == "Measuring an object"


def _call(name, cid="c1", **args):
    fn = {"name": name, "arguments": json.dumps(args)}
    return {"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function",
                                                                "function": fn}]}  # fmt: skip


def test_agent_loop_frames_the_edit_before_and_after(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CHAT_CACHE", tmp_path)
    script = [_call("move_object", object_id="sofa#1", dx=0.5, dz=0),
              {"role": "assistant", "content": "Moved it."}]  # fmt: skip
    shots = []
    seen = []

    async def capture(cam):
        shots.append(cam)
        seen.append(make_edit_state(s))
        return "data:image/jpeg;base64,AA"

    async def verdict(*a, **k):
        return Verdict(True, "fine")

    def make_edit_state(scene):
        return scene.get("sofa#1").transform.translation[0]

    monkeypatch.setattr(loop, "verify_edit", verdict)
    s = make_scene()
    res = asyncio.run(
        run_command(s, "sofa right", ROOM, capture=capture, client=FakeClient(script))
    )
    assert res.verified is True
    assert len(shots) == 2 and shots[0] == shots[1]  # same viewpoint before and after
    assert shots[0]["target"][0] == pytest.approx(0.25)  # between both positions
    assert seen == [0.0, 0.5]  # the first shot is taken before the edit


def test_agent_loop_bad_id_first_still_frames_the_real_edit(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CHAT_CACHE", tmp_path)
    script = [_call("move_object", object_id="nope#9", dx=1, dz=0),
              _call("move_object", cid="c2", object_id="table#1", dx=0.2, dz=0),
              {"role": "assistant", "content": "Moved it."}]  # fmt: skip
    shots = []

    async def capture(cam):
        shots.append(cam)
        return None

    asyncio.run(run_command(make_scene(), "table right", ROOM, capture=capture,
                            client=FakeClient(script)))  # fmt: skip
    assert len(shots) == 1 and shots[0]["target"][0] == pytest.approx(0.1)


def test_agent_loop_survives_null_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CHAT_CACHE", tmp_path)
    script = [_call("move_object", object_id="table#1", dx=None, dz=0),
              {"role": "assistant", "content": "Could not."}]  # fmt: skip

    async def capture(cam):
        return None

    res = asyncio.run(run_command(make_scene(), "move", ROOM, capture=capture,
                                  client=FakeClient(script)))  # fmt: skip
    assert res.reply == "Could not." and not res.edited


def test_room_bounds_reads_the_background_ply(tmp_path):
    import numpy as np
    from plyfile import PlyData, PlyElement

    rng = np.random.default_rng(0)
    pts = rng.uniform([-2, 0, -2.5], [2, 2.6, 2.5], size=(5000, 3)).astype("f4")
    v = np.empty(5000, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    v["x"], v["y"], v["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    PlyData([PlyElement.describe(v, "vertex")]).write(str(tmp_path / "bg.ply"))
    s = make_scene()
    s.background_ply = "bg.ply"
    b = room_bounds(s, tmp_path)
    assert -2.1 < b.x[0] < -1.9 and 2.4 < b.z[1] < 2.6


def _wav(rate=16000, ch=1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * (ch * rate // 10))
    return buf.getvalue()


def test_check_wav():
    assert check_wav(_wav())[0] == 16000
    with pytest.raises(ValueError):
        check_wav(_wav(44100))
    with pytest.raises(ValueError):
        check_wav(b"nope")


def test_server_edit_endpoint(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from movable_reality import server

    monkeypatch.setattr(server, "SCENES_DIR", tmp_path)
    server._hubs.clear()
    make_scene().save(tmp_path / "t" / "scene.json")
    c = TestClient(server.app)
    r = c.post("/api/scenes/t/edit", json={"op": "rotate", "object_id": "sofa#1",
                                            "args": {"yaw_deg": 90}})  # fmt: skip
    assert r.status_code == 200 and r.json()["scene"]["objects"][0]["transform"]["yaw_deg"] == 90
    assert c.post("/api/scenes/t/edit", json={"op": "undo"}).status_code == 200
    assert Scene.load(tmp_path / "t" / "scene.json").get("sofa#1").transform.yaw_deg == 0
    assert c.post("/api/scenes/t/edit", json={"op": "move", "object_id": "x#1"}).status_code == 400
    with c.websocket_connect("/ws/scenes/t") as ws:
        assert ws.receive_json()["type"] == "scene"


def test_parse_verdict_empty_is_unverified():
    assert parse_verdict("").ok is None
    assert parse_verdict("Maybe.").ok is None


def test_sam_cache_packs_decoded_masks(tmp_path, monkeypatch):
    import json as _json

    import numpy as _np

    from movable_reality import sam

    rng = _np.random.default_rng(0)
    masks = [rng.random((5, 7)) > 0.5, rng.random((3, 11)) > 0.5]
    calls = []

    def fake(d):
        calls.append(d)
        return sam.Instance(d["object_id"], d["frame_index"], tuple(d["box"]),
                            masks[d["frame_index"]], d.get("confidence"))  # fmt: skip

    monkeypatch.setattr(sam, "_instance_from_json", fake)
    cache = tmp_path / "video_x.json"
    recs = [{"object_id": "1", "frame_index": i, "box": [0, 0, 7, 5], "confidence": 0.9}
            for i in range(2)]  # fmt: skip
    cache.write_text(_json.dumps(recs))
    first = sam._load_cached(cache)
    second = sam._load_cached(cache)  # from the .npz, no decoding
    assert len(calls) == 2
    for a, b, m in zip(first, second, masks, strict=True):
        assert _np.array_equal(b.mask, m) and b.box == a.box and b.frame_index == a.frame_index
