"""Scene server (ARCHITECTURE stage 8): edit API, agent commands, voice, WebSocket pushes.

    uv run python -m movable_reality.server            # http://127.0.0.1:8000
    uv run python -m movable_reality.server --port 8000 --host 0.0.0.0

API: docs/SERVER_API.md. scene.json on disk stays the source of truth: every request loads
it, applies the edit through Scene.apply_edit and saves it atomically, so scripts and the
viewer's poll see the same state. When web/dist exists (npm run build) the server also
serves the viewer and the scene folders, which is the hosted-demo layout (U3).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import RoomBounds, room_bounds, run_command
from .config import REPO_ROOT, SCENES_DIR, Models
from .scene import Scene
from .voice import transcribe

CAPTURE_TIMEOUT_S = 8.0


@dataclass
class SceneHub:
    """Per-scene server state: one lock for edits, connected viewers, pending screenshots."""

    scene_id: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    sockets: set[WebSocket] = field(default_factory=set)
    captures: dict[int, asyncio.Future] = field(default_factory=dict)
    bounds: RoomBounds | None = None
    bounds_loaded: bool = False

    @property
    def folder(self) -> Path:
        return SCENES_DIR / self.scene_id

    def load(self) -> Scene:
        path = self.folder / "scene.json"
        if not path.exists():
            raise HTTPException(404, f"no scene '{self.scene_id}'")
        return Scene.load(path)

    def save(self, scene: Scene) -> None:
        scene.save(self.folder / "scene.json")

    def room(self, scene: Scene) -> RoomBounds | None:
        if not self.bounds_loaded:
            self.bounds = room_bounds(scene, self.folder)
            self.bounds_loaded = True
        return self.bounds

    async def broadcast(self, msg: dict[str, Any]) -> None:
        for ws in list(self.sockets):
            try:
                await ws.send_json(msg)
            except Exception:  # noqa: BLE001 - a closed socket is dropped
                self.sockets.discard(ws)

    async def capture(self, camera: dict | None = None) -> str | None:
        """Ask one connected viewer for a JPEG data URL of its current view."""
        if not self.sockets:
            return None
        rid = next(_ids)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self.captures[rid] = fut
        msg: dict[str, Any] = {"type": "capture_request", "id": rid}
        if camera:
            msg["camera"] = camera
        ws = next(iter(self.sockets))
        try:
            await ws.send_json(msg)
            return await asyncio.wait_for(fut, CAPTURE_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - no screenshot means no verification, not a failure
            return None
        finally:
            self.captures.pop(rid, None)


_ids = itertools.count(1)
_hubs: dict[str, SceneHub] = {}


def hub(scene_id: str) -> SceneHub:
    if "/" in scene_id or "\\" in scene_id or scene_id.startswith("."):
        raise HTTPException(400, "bad scene id")
    if scene_id not in _hubs:
        _hubs[scene_id] = SceneHub(scene_id)
    return _hubs[scene_id]


class EditBody(BaseModel):
    op: str
    object_id: str | None = None
    args: dict[str, float] = {}
    instruction: str | None = None


class CommandBody(BaseModel):
    text: str
    camera: dict[str, Any] | None = None  # {"eye": [x, y, z], "forward": [x, y, z]}
    verify: bool = True


app = FastAPI(title="Movable Reality scene server")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "models": {"agent": Models.AGENT, "verifier": Models.VERIFIER_OPEN,
                                   "transcribe": Models.TRANSCRIBE}}  # fmt: skip


@app.get("/api/scenes/{scene_id}")
async def get_scene(scene_id: str) -> dict[str, Any]:
    return hub(scene_id).load().model_dump(mode="json")


@app.post("/api/scenes/{scene_id}/edit")
async def edit(scene_id: str, body: EditBody) -> dict[str, Any]:
    h = hub(scene_id)
    async with h.lock:
        scene = h.load()
        try:
            scene.apply_edit(body.op, body.object_id, body.args, body.instruction)
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc).strip("'\"")) from exc
        h.save(scene)
        data = scene.model_dump(mode="json")
    await h.broadcast({"type": "scene", "scene": data})
    return {"ok": True, "scene": data}


async def _command(h: SceneHub, text: str, camera: dict | None, verify: bool) -> dict[str, Any]:
    async def emit(event: dict[str, Any]) -> None:
        await h.broadcast({"type": "agent", **{k: v for k, v in event.items() if k != "step"}})
        if event.get("stage") == "tool":  # show each edit as it happens
            await h.broadcast({"type": "scene", "scene": live.model_dump(mode="json")})

    async def capture() -> str | None:
        return await h.capture()

    async with h.lock:
        live = h.load()
        try:
            res = await run_command(
                live, text, bounds=h.room(live), camera=camera, capture=capture, emit=emit,
                inserter=_inserter(h), verify=verify,
            )  # fmt: skip
        except Exception as exc:  # noqa: BLE001 - report API failures to the command bar
            await h.broadcast({"type": "agent", "stage": "done", "text": f"error: {exc}"})
            raise HTTPException(502, f"agent failed: {exc}") from exc
        h.save(live)
        data = live.model_dump(mode="json")
    await h.broadcast({"type": "scene", "scene": data})
    return {"reply": res.reply, "steps": res.steps, "verified": res.verified,
            "verdicts": res.verdicts, "scene": data}  # fmt: skip


def _inserter(h: SceneHub):
    """M4 insert path, when its dependencies are importable on this machine."""
    try:
        from .insert import insert_object  # noqa: PLC0415 - optional, heavy
    except ImportError:
        return None

    def run(scene: Scene, prompt: str, x: float, z: float, instruction: str | None):
        return insert_object(scene, h.folder, prompt, x, z, instruction)

    return run


@app.post("/api/scenes/{scene_id}/command")
async def command(scene_id: str, body: CommandBody) -> dict[str, Any]:
    return await _command(hub(scene_id), body.text, body.camera, body.verify)


@app.post("/api/scenes/{scene_id}/voice")
async def voice(scene_id: str, audio: UploadFile, camera: str | None = None) -> dict[str, Any]:
    h = hub(scene_id)
    data = await audio.read()
    await h.broadcast({"type": "agent", "stage": "transcript", "text": "transcribing"})
    try:
        text = (await transcribe(data)).strip()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    await h.broadcast({"type": "agent", "stage": "transcript", "text": text})
    if not text:
        return {"transcript": "", "reply": "I did not catch that.", "steps": [], "verified": None}
    import json  # noqa: PLC0415

    cam = json.loads(camera) if camera else None
    out = await _command(h, text, cam, True)
    return {"transcript": text, **out}


@app.websocket("/ws/scenes/{scene_id}")
async def scene_socket(ws: WebSocket, scene_id: str) -> None:
    h = hub(scene_id)
    await ws.accept()
    h.sockets.add(ws)
    try:
        try:
            await ws.send_json({"type": "scene", "scene": h.load().model_dump(mode="json")})
        except HTTPException:
            pass
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "capture":
                fut = h.captures.get(int(msg.get("id", -1)))
                if fut and not fut.done():
                    fut.set_result(msg.get("jpeg"))
    except WebSocketDisconnect:
        pass
    finally:
        h.sockets.discard(ws)


def _mount_static() -> None:
    dist = REPO_ROOT / "web" / "dist"
    if SCENES_DIR.is_dir():
        app.mount("/scenes", StaticFiles(directory=SCENES_DIR), name="scenes")
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="viewer")


_mount_static()


def main() -> None:
    import uvicorn  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
