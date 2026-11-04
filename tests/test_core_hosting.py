"""Hosted demo: which files go up, and that visitors cannot touch the shared scene."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect

from movable_reality import server
from movable_reality.publish import missing_lod, referenced_files, without_edits
from movable_reality.scene import Scene, SceneObject


def _scene() -> Scene:
    return Scene(
        scene_id="s",
        background_ply="objects/background.ply",
        objects=[SceneObject(id="chair#1", label="chair", ply="objects/chair.ply",
                             bbox_min=(0, 0, 0), bbox_max=(1, 1, 1))],
    )  # fmt: skip


def test_without_edits_undoes_and_leaves_the_original():
    scene = _scene()
    scene.apply_edit("move", "chair#1", {"dx": 1, "dy": 0, "dz": 2})
    scene.apply_edit("remove", "chair#1", {})
    clean = without_edits(scene)
    assert clean.history == []
    assert clean.get("chair#1").transform.translation == (0, 0, 0)
    assert clean.get("chair#1").visible
    assert len(scene.history) == 2


def test_missing_lod_names_the_meshes_without_a_tree():
    scene = _scene()
    assert missing_lod(scene) == ["background", "chair#1"]
    scene.background_lod = "lod/background.rad"
    scene.get("chair#1").lod = "lod/chair.rad"
    assert missing_lod(scene) == []


def test_referenced_files_takes_only_named_files_inside_the_folder(tmp_path: Path):
    folder = tmp_path / "scene"
    for name in ("objects/a.ply", "lod/a.rad", "ids/ids.npy", "frames/000000.jpg"):
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text("x")
    (tmp_path / "secret.txt").write_text("x")
    manifest = {
        "background_ply": "objects/a.ply",
        "objects": [{"lod": "lod/a.rad", "mesh": "objects/missing.glb", "label": "a/b"}],
        "plane_bake": {"ids": "ids/ids.npy"},
        "escape": "../secret.txt",
    }
    assert referenced_files(manifest, folder) == ["lod/a.rad", "objects/a.ply"]
    (folder / "room").mkdir()
    (folder / "room" / "shell.json").write_text("{}")
    assert "room/shell.json" in referenced_files(manifest, folder)


@pytest.fixture
def hosted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    folder = tmp_path / "scenes" / "s"
    folder.mkdir(parents=True)
    _scene().save(folder / "scene.json")
    monkeypatch.setattr(server, "HOSTED", True)
    monkeypatch.setattr(server, "SCENES_DIR", tmp_path / "scenes")
    monkeypatch.setattr(server, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(server, "_usage", {})
    monkeypatch.setattr(server, "_hubs", {})
    return folder


MOVE = {"op": "move", "object_id": "chair#1", "args": {"dx": 1, "dy": 0, "dz": 0}}


def test_splat_files_ask_proxies_not_to_compress_them(hosted: Path):
    with TestClient(server.app) as client:
        assert "no-transform" in client.get("/scenes/s/lod/x.rad").headers["cache-control"]
        manifest = client.get("/scenes/s/scene.json").headers
        assert "no-transform" not in manifest.get("cache-control", "")


def test_edit_without_a_session_is_refused(hosted: Path):
    with TestClient(server.app) as client:
        assert client.post("/api/scenes/s/edit", json=MOVE).status_code == 403
    assert Scene.load(hosted / "scene.json").history == []


def test_edit_with_a_session_stays_in_its_copy(hosted: Path):
    with TestClient(server.app) as client:
        client.get("/api/health")  # sets the cookie
        assert client.post("/api/scenes/s/edit", json=MOVE).status_code == 200
        assert client.get("/scenes/s/scene.json").json()["history"]
    assert Scene.load(hosted / "scene.json").history == []


def test_reading_leaves_nothing_on_disk(hosted: Path):
    with TestClient(server.app) as client:
        client.get("/api/health")
        assert client.get("/api/scenes/s").status_code == 200
        assert client.get("/api/scenes/made-up").status_code == 404
    assert not server.SESSIONS_DIR.exists()
    assert len(server._hubs) == 1


def test_provider_error_text_never_reaches_the_visitor(hosted: Path, monkeypatch):
    secret = "Incorrect API key provided: sk-secret"

    async def boom(*_a, **_k):
        raise RuntimeError(secret)

    monkeypatch.setattr(server, "run_command", boom)
    with TestClient(server.app) as client:
        client.get("/api/health")
        r = client.post("/api/scenes/s/command", json={"text": "move the chair"})
    assert r.status_code == 502
    assert "sk-secret" not in r.text


def test_address_cap_stops_fresh_cookies(hosted: Path, monkeypatch):
    monkeypatch.setattr(server, "IP_COMMANDS", 1)
    server._charge("one", "addr")
    with pytest.raises(server.HTTPException) as err:
        server._charge("two", "addr")
    assert err.value.status_code == 429
    server._charge("two", "other")


def test_address_uses_the_entry_the_proxy_added(hosted: Path):
    def request(forwarded: str) -> Request:
        return Request({"type": "http", "client": None,
                        "headers": [(b"x-forwarded-for", forwarded.encode())]})  # fmt: skip

    assert server._address(request("1.1.1.1, 9.9.9.9")) == server._address(request("9.9.9.9"))
    assert server._address(request("1.1.1.1, 9.9.9.9")) != server._address(request("1.1.1.1"))


def test_oversized_body_and_bad_numbers_are_refused(hosted: Path):
    with TestClient(server.app) as client:
        client.get("/api/health")
        big = client.post("/api/scenes/s/command", content=b"x" * (server.MAX_BODY_BYTES + 1))
        assert big.status_code == 413
        bad = {**MOVE, "args": {"dx": float("inf"), "dy": 0, "dz": 0}}
        r = client.post("/api/scenes/s/edit", content=json.dumps(bad),
                        headers={"content-type": "application/json"})  # fmt: skip
        assert r.status_code == 400
    assert not server.SESSIONS_DIR.exists()


def test_chunked_body_past_the_limit_is_cut_off(hosted: Path):
    def chunks():
        for _ in range(4):
            yield b"x" * (1024 * 1024)

    with TestClient(server.app, raise_server_exceptions=False) as client:
        client.get("/api/health")
        r = client.post("/api/scenes/s/command", content=chunks(),
                        headers={"content-type": "application/json"})  # fmt: skip
    assert r.status_code not in (200, 422)  # never parsed


def test_camera_must_be_two_vectors_of_three(hosted: Path, monkeypatch):
    async def never(*_a, **_k):
        raise AssertionError("the agent must not run")

    monkeypatch.setattr(server, "run_command", never)
    with TestClient(server.app) as client:
        client.get("/api/health")
        for camera in ({"eye": [0.0] * 1000, "forward": [0, 0, 1]},
                       {"eye": "x", "forward": [0, 0, 1]}):  # fmt: skip
            r = client.post("/api/scenes/s/command", json={"text": "hi", "camera": camera})
            assert r.status_code == 422
        r = client.post("/api/scenes/s/voice?camera=%7Bbad", files={"audio": ("a.wav", b"x")})
        assert r.status_code == 400
    assert server._usage == {}  # nothing charged


def test_camera_reaches_the_agent_as_a_plain_dict(hosted: Path, monkeypatch):
    seen = {}

    class Result:
        reply, steps, verified, verdicts = "ok", [], None, []

    async def fake(_scene, _text, **kw):
        seen.update(kw)
        return Result()

    monkeypatch.setattr(server, "run_command", fake)
    with TestClient(server.app) as client:
        client.get("/api/health")
        pose = {"eye": [1, 2, 3], "forward": [0, 0, -1]}
        r = client.post("/api/scenes/s/command", json={"text": "hi", "camera": pose})
        assert r.status_code == 200
    assert seen["camera"]["eye"] == (1, 2, 3)


def test_huge_edit_values_are_refused(hosted: Path):
    with TestClient(server.app) as client:
        client.get("/api/health")
        bad = {**MOVE, "args": {"dx": 1e300, "dy": 0, "dz": 0}}
        assert client.post("/api/scenes/s/edit", json=bad).status_code == 400


def test_commands_without_a_session_are_refused_and_not_charged(hosted: Path, monkeypatch):
    async def never(*_a, **_k):
        raise AssertionError("the agent must not run")

    monkeypatch.setattr(server, "run_command", never)
    requests = (
        lambda c: c.post("/api/scenes/s/edit", json=MOVE),
        lambda c: c.post("/api/scenes/s/command", json={"text": "hi"}),
        lambda c: c.post("/api/scenes/s/voice", files={"audio": ("a.wav", b"x")}),
    )
    for send in requests:
        for cookie in (None, "bad/value"):  # a fresh client: each answer sets a valid cookie
            with TestClient(server.app) as client:
                if cookie:
                    client.cookies.set(server.COOKIE, cookie)
                assert send(client).status_code == 403
    assert server._usage == {}
    assert Scene.load(hosted / "scene.json").history == []


def test_two_visitors_do_not_see_each_others_edits(hosted: Path):
    with TestClient(server.app) as one, TestClient(server.app) as two:
        one.get("/api/health")
        two.get("/api/health")
        assert one.post("/api/scenes/s/edit", json=MOVE).status_code == 200
        assert two.get("/scenes/s/scene.json").json()["history"] == []


def test_socket_needs_a_session_and_is_capped_per_address(hosted: Path, monkeypatch):
    with TestClient(server.app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/scenes/s"):
                pass
        client.get("/api/health")
        with client.websocket_connect("/ws/scenes/s") as ws:
            assert ws.receive_json()["type"] == "scene"
            assert len(server._address_sockets) == 1
            monkeypatch.setattr(server, "MAX_ADDRESS_SOCKETS", 1)
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/ws/scenes/s"):
                    pass
    assert server._address_sockets == {}


@pytest.mark.parametrize("text", ["", "{broken", "[1, 2]", '{"a": "x"}'])
def test_bad_usage_file_counts_from_zero(hosted: Path, text: str):
    server.SESSIONS_DIR.mkdir(parents=True)
    (server.SESSIONS_DIR / "usage.json").write_text(text)
    assert server._load_usage() == {}


def test_usage_survives_a_restart(hosted: Path, monkeypatch: pytest.MonkeyPatch):
    server._charge("abc")
    saved = json.loads((server.SESSIONS_DIR / "usage.json").read_text())
    assert saved["s:abc"] == 1
    assert server._load_usage() == saved
    monkeypatch.setattr(server, "_usage", server._load_usage())
    monkeypatch.setattr(server, "MAX_COMMANDS", 1)
    with pytest.raises(server.HTTPException) as err:
        server._charge("abc")
    assert err.value.status_code == 429
