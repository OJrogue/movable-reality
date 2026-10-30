import asyncio
import io
import json
import wave

import pytest

from movable_reality.agent import llm
from movable_reality.agent.loop import run_command, view_directions
from movable_reality.agent.tools import RoomBounds, fits, measure, room_bounds_from_points, run_tool
from movable_reality.agent.verify import parse_verdict
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
