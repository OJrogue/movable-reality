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
import json
import os
import secrets
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import RoomBounds, room_bounds, run_command
from .config import DATA_DIR, REPO_ROOT, SCENES_DIR, Models
from .scene import Scene
from .voice import check_wav, transcribe

CAPTURE_TIMEOUT_S = 8.0

# Hosted demo (U3). MR_SESSIONS=1 gives every visitor (cookie mr_session) a private copy of
# scene.json, so edits never leak between visitors, and caps paid calls per visitor and per
# day. MR_CACHE_ONLY=1 (agent/llm.py) answers only from cached replies, with no key at all.
HOSTED = os.environ.get("MR_SESSIONS", "") == "1"
SESSIONS_DIR = Path(os.environ.get("MR_SESSIONS_DIR", DATA_DIR / "sessions"))
MAX_COMMANDS = int(os.environ.get("MR_MAX_COMMANDS", "20"))  # per visitor
DAILY_COMMANDS = int(os.environ.get("MR_DAILY_COMMANDS", "300"))  # all visitors, per UTC day
MAX_VOICE_S = float(os.environ.get("MR_MAX_VOICE_S", "15"))
COOKIE = "mr_session"
_usage: dict[str, int] = {}


@dataclass
class SceneHub:
    """Per-scene server state: one lock for edits, connected viewers, pending screenshots."""

    scene_id: str
    session: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    sockets: set[WebSocket] = field(default_factory=set)
    captures: dict[int, asyncio.Future] = field(default_factory=dict)
    bounds: RoomBounds | None = None
    bounds_loaded: bool = False

    @property
    def folder(self) -> Path:
        """The scene folder (PLYs, trees, planes). Never written in hosted mode."""
        return SCENES_DIR / self.scene_id

    @property
    def manifest_path(self) -> Path:
        if self.session is None:
            return self.folder / "scene.json"
        return SESSIONS_DIR / self.session / self.scene_id / "scene.json"

    def load(self) -> Scene:
        path = self.manifest_path
        if not path.exists():
            src = self.folder / "scene.json"
            if self.session is None or not src.exists():
                raise HTTPException(404, f"no scene '{self.scene_id}'")
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, path)
        return Scene.load(path)

    def save(self, scene: Scene) -> None:
        scene.save(self.manifest_path)

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


def hub(scene_id: str, session: str | None = None) -> SceneHub:
    if "/" in scene_id or "\\" in scene_id or scene_id.startswith("."):
        raise HTTPException(400, "bad scene id")
    session = session if HOSTED else None
    key = f"{scene_id}@{session}"
    if key not in _hubs:
        _hubs[key] = SceneHub(scene_id, session)
    return _hubs[key]


def _session(cookies) -> str | None:
    sid = cookies.get(COOKIE)
    return sid if sid and sid.isalnum() and len(sid) <= 64 else None


def _charge(session: str | None) -> None:
    """Count one paid command; HTTP 429 past the per-visitor or daily cap (hosted only)."""
    if not HOSTED:
        return
    day = time.strftime("%Y-%m-%d", time.gmtime())
    if _usage.get(f"day:{day}", 0) >= DAILY_COMMANDS:
        raise HTTPException(429, "the demo's daily command budget is used up; try tomorrow")
    if _usage.get(f"s:{session}", 0) >= MAX_COMMANDS:
        raise HTTPException(429, f"limit of {MAX_COMMANDS} commands per visitor reached")
    _usage[f"day:{day}"] = _usage.get(f"day:{day}", 0) + 1
    _usage[f"s:{session}"] = _usage.get(f"s:{session}", 0) + 1


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


@app.middleware("http")
async def session_cookie(request: Request, call_next):
    """Hosted mode: give every visitor a session cookie on their first request."""
    response = await call_next(request)
    if HOSTED and not _session(request.cookies):
        response.set_cookie(COOKIE, secrets.token_hex(16), httponly=True, samesite="lax",
                            max_age=7 * 24 * 3600)  # fmt: skip
    return response


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "hosted": HOSTED,
            "models": {"agent": Models.AGENT, "verifier": Models.VERIFIER_OPEN,
                       "transcribe": Models.TRANSCRIBE}}  # fmt: skip


@app.get("/api/scenes/{scene_id}")
async def get_scene(scene_id: str, request: Request) -> dict[str, Any]:
    return hub(scene_id, _session(request.cookies)).load().model_dump(mode="json")


@app.get("/scenes/{scene_id}/scene.json", response_model=None)
async def scene_json(scene_id: str, request: Request):
    """In hosted mode the viewer's poll must see the visitor's copy, not the original."""
    h = hub(scene_id, _session(request.cookies))
    if h.session is None:
        path = h.folder / "scene.json"
        if not path.exists():
            raise HTTPException(404, "not found")
        return FileResponse(path, media_type="application/json",
                            headers={"Cache-Control": "no-store"})  # fmt: skip
    return h.load().model_dump(mode="json")


@app.post("/api/scenes/{scene_id}/edit")
async def edit(scene_id: str, body: EditBody, request: Request) -> dict[str, Any]:
    h = hub(scene_id, _session(request.cookies))
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
            # tool steps were pushed live but never saved: show the saved state again
            await h.broadcast({"type": "scene", "scene": h.load().model_dump(mode="json")})
            await h.broadcast({"type": "agent", "stage": "done", "text": f"error: {exc}"})
            raise HTTPException(502, f"agent failed: {exc}") from exc
        h.save(live)
        if any(st["tool"] == "insert_object" for st in res.steps):
            h.bounds_loaded = False  # new splats may change the room box
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
async def command(scene_id: str, body: CommandBody, request: Request) -> dict[str, Any]:
    session = _session(request.cookies)
    h = hub(scene_id, session)
    h.load()  # 404 before charging for an unknown scene
    _charge(session)
    return await _command(h, body.text[:500], body.camera, body.verify)


@app.post("/api/scenes/{scene_id}/voice")
async def voice(
    scene_id: str, audio: UploadFile, request: Request, camera: str | None = None
) -> dict[str, Any]:
    session = _session(request.cookies)
    h = hub(scene_id, session)
    data = await audio.read()
    if HOSTED:
        try:
            if check_wav(data)[2] > MAX_VOICE_S:
                raise HTTPException(400, f"voice commands are limited to {MAX_VOICE_S:.0f} s")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    _charge(session)
    await h.broadcast({"type": "agent", "stage": "transcript", "text": "transcribing"})
    try:
        text = (await transcribe(data)).strip()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    await h.broadcast({"type": "agent", "stage": "transcript", "text": text})
    if not text:
        return {"transcript": "", "reply": "I did not catch that.", "steps": [], "verified": None}
    cam = json.loads(camera) if camera else None
    out = await _command(h, text, cam, True)
    return {"transcript": text, **out}


@app.websocket("/ws/scenes/{scene_id}")
async def scene_socket(ws: WebSocket, scene_id: str) -> None:
    h = hub(scene_id, _session(ws.cookies))
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
