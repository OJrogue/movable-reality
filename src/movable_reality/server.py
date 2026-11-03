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
import hashlib
import itertools
import json
import math
import mimetypes
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from starlette.requests import HTTPConnection

from .agent import RoomBounds, room_bounds, run_command
from .agent.tools import free_spot
from .config import DATA_DIR, LIBRARY_DIR, REPO_ROOT, SCENES_DIR, Models
from .fsutil import replace_file
from .library import add_to_scene, get_item
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
IP_COMMANDS = int(os.environ.get("MR_IP_COMMANDS", "60"))  # one address per UTC day: new cookies
MAX_BODY_BYTES = 2 * 1024 * 1024  # 15 s of 24 kHz voice is 720 KB
MAX_HISTORY = 1000  # edits in one visitor's copy
MAX_EDIT_VALUE = 1e4  # metres or degrees in one edit; repeated huge moves would overflow to inf
MAX_SOCKETS = 150  # of modal_jobs/demo.py's 200 inputs, so requests always have room
MAX_ADDRESS_SOCKETS = 8
MAX_VOICE_S = float(os.environ.get("MR_MAX_VOICE_S", "15"))
COOKIE = "mr_session"


def _load_usage() -> dict[str, int]:
    """The caps must survive a restart: the hosted container scales to zero when idle."""
    path = SESSIONS_DIR / "usage.json"
    if not HOSTED or not path.exists():
        return {}
    try:
        usage = json.loads(path.read_text())
        return {str(k): int(v) for k, v in usage.items()}
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        print(f"usage.json unreadable, counting from zero: {type(exc).__name__}", flush=True)
        return {}


_usage: dict[str, int] = _load_usage()


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
        """The visitor's copy once they edited, the original until then (save writes the copy),
        so requests that only read leave nothing on disk."""
        path = self.manifest_path
        if not path.exists():
            path = self.folder / "scene.json"
            if self.session is None or not path.exists():
                raise HTTPException(404, f"no scene '{self.scene_id}'")
        return Scene.load(path)

    def save(self, scene: Scene) -> None:
        scene.save(self.manifest_path)

    async def room(self, scene: Scene) -> RoomBounds | None:
        if not self.bounds_loaded:  # reads the background PLY: keep the event loop free
            self.bounds = await asyncio.to_thread(room_bounds, scene, self.folder)
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
_address_sockets: dict[str, int] = {}


def hub(scene_id: str, session: str | None = None) -> SceneHub:
    if "/" in scene_id or "\\" in scene_id or scene_id.startswith("."):
        raise HTTPException(400, "bad scene id")
    session = session if HOSTED else None
    if HOSTED and not (SCENES_DIR / scene_id / "scene.json").exists():
        raise HTTPException(404, f"no scene '{scene_id}'")  # no hub per made-up id
    key = f"{scene_id}@{session}"
    if key not in _hubs:
        _hubs[key] = SceneHub(scene_id, session)
    return _hubs[key]


def _session(cookies) -> str | None:
    sid = cookies.get(COOKIE)
    return sid if sid and sid.isalnum() and len(sid) <= 64 else None


def _address(request: HTTPConnection) -> str:
    """Hash of the client address. The platform's proxy appends the address it saw to
    X-Forwarded-For, so only the last entry cannot be set by the client."""
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[-1].strip()
    ip = forwarded or (request.client.host if request.client else "")
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def _charge(session: str | None, address: str = "") -> None:
    """Count one paid command; HTTP 429 past the per-visitor, per-address or daily cap (hosted
    only). The address cap keeps one client with fresh cookies from using up the daily budget."""
    if not HOSTED:
        return
    day = time.strftime("%Y-%m-%d", time.gmtime())
    if _usage.get(f"day:{day}", 0) >= DAILY_COMMANDS:
        raise HTTPException(429, "the demo's daily command budget is used up; try tomorrow")
    if _usage.get(f"s:{session}", 0) >= MAX_COMMANDS:
        raise HTTPException(429, f"limit of {MAX_COMMANDS} commands per visitor reached")
    if _usage.get(f"ip:{day}:{address}", 0) >= IP_COMMANDS:
        raise HTTPException(429, "too many commands from this network today; try tomorrow")
    _usage[f"ip:{day}:{address}"] = _usage.get(f"ip:{day}:{address}", 0) + 1
    _usage[f"day:{day}"] = _usage.get(f"day:{day}", 0) + 1
    _usage[f"s:{session}"] = _usage.get(f"s:{session}", 0) + 1
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SESSIONS_DIR / "usage.json.tmp"
    tmp.write_text(json.dumps(_usage))
    replace_file(tmp, SESSIONS_DIR / "usage.json")


def _writable(h: SceneHub) -> None:
    """Hosted mode: without a session an edit would land in the scene every visitor shares."""
    if HOSTED and h.session is None:
        raise HTTPException(403, "this demo needs cookies to keep your edits apart from others'")


class EditBody(BaseModel):
    op: str
    object_id: str | None = None
    args: dict[str, float] = {}
    instruction: str | None = Field(default=None, max_length=500)


class LibraryBody(BaseModel):
    library_id: str
    x: float | None = None  # floor position; the server nudges it to a free spot in the room
    z: float | None = None
    yaw_deg: float = 0.0
    on_object_id: str | None = None  # stand on this object instead of the floor
    instruction: str | None = Field(default=None, max_length=500)


_Coordinate = Annotated[float, Field(allow_inf_nan=False)]


class Camera(BaseModel):
    """The viewer's pose. Typed, since it goes into every paid prompt."""

    eye: tuple[_Coordinate, _Coordinate, _Coordinate]
    forward: tuple[_Coordinate, _Coordinate, _Coordinate]


class CommandBody(BaseModel):
    text: str
    camera: Camera | None = None
    verify: bool = True


# No API browser on the public deployment.
_docs = {"docs_url": None, "redoc_url": None, "openapi_url": None} if HOSTED else {}
app = FastAPI(title="Movable Reality scene server", **_docs)


class BodyLimit:
    """Hosted mode: end a request whose body passes the limit, also without a Content-Length."""

    def __init__(self, inner) -> None:
        self.inner = inner

    async def __call__(self, scope, receive, send) -> None:
        if not HOSTED or scope["type"] != "http":
            return await self.inner(scope, receive, send)
        seen = 0

        async def limited():
            nonlocal seen
            message = await receive()
            seen += len(message.get("body", b""))
            return {"type": "http.disconnect"} if seen > MAX_BODY_BYTES else message

        await self.inner(scope, limited, send)


app.add_middleware(BodyLimit)


@app.middleware("http")
async def session_cookie(request: Request, call_next):
    """Hosted mode: a session cookie on the first request, a body limit, security headers."""
    if HOSTED and int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
        return JSONResponse({"detail": "request too large"}, status_code=413)
    response = await call_next(request)
    if HOSTED:
        https = "https" in (request.url.scheme, request.headers.get("x-forwarded-proto"))
        if not _session(request.cookies):
            response.set_cookie(COOKIE, secrets.token_hex(16), httponly=True, samesite="lax",
                                secure=https, max_age=7 * 24 * 3600)  # fmt: skip
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "same-origin"
        if https:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
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
    _writable(h)
    if not all(math.isfinite(v) and abs(v) <= MAX_EDIT_VALUE for v in body.args.values()):
        raise HTTPException(400, f"edit values must be numbers up to {MAX_EDIT_VALUE:.0f}")
    async with h.lock:
        scene = h.load()
        if HOSTED and len(scene.history) >= MAX_HISTORY:
            raise HTTPException(429, "this visitor's scene has too many edits")
        try:
            scene.apply_edit(body.op, body.object_id, body.args, body.instruction)
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc).strip("'\"")) from exc
        h.save(scene)
        data = scene.model_dump(mode="json")
    await h.broadcast({"type": "scene", "scene": data})
    return {"ok": True, "scene": data}


@app.post("/api/scenes/{scene_id}/library")
async def add_library_object(scene_id: str, body: LibraryBody, request: Request) -> dict[str, Any]:
    """Add a premade mesh. Nothing is written to the scene folder, so hosted visitors can too."""
    h = hub(scene_id, _session(request.cookies))
    _writable(h)
    numbers = [v for v in (body.x, body.z, body.yaw_deg) if v is not None]
    if not all(math.isfinite(v) and abs(v) <= MAX_EDIT_VALUE for v in numbers):
        raise HTTPException(400, f"values must be numbers up to {MAX_EDIT_VALUE:.0f}")
    async with h.lock:
        scene = h.load()
        if HOSTED and len(scene.history) >= MAX_HISTORY:
            raise HTTPException(429, "this visitor's scene has too many edits")
        try:
            item = get_item(body.library_id)
            x, z = body.x, body.z
            if not body.on_object_id:
                if x is None or z is None:
                    raise ValueError("give x and z, or on_object_id")
                x, z = free_spot(scene, item.size, body.yaw_deg, x, z, await h.room(scene))
            obj = add_to_scene(scene, item, x, z, body.yaw_deg, body.on_object_id, body.instruction)
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc).strip("'\"")) from exc
        h.save(scene)
        data = scene.model_dump(mode="json")
    await h.broadcast({"type": "scene", "scene": data})
    return {"ok": True, "id": obj.id, "scene": data}


async def _command(h: SceneHub, text: str, camera: dict | None, verify: bool) -> dict[str, Any]:
    async def emit(event: dict[str, Any]) -> None:
        await h.broadcast({"type": "agent", **{k: v for k, v in event.items() if k != "step"}})
        if event.get("stage") == "tool":  # show each edit as it happens
            await h.broadcast({"type": "scene", "scene": live.model_dump(mode="json")})

    async def capture(camera: dict | None = None) -> str | None:
        return await h.capture(camera)

    async with h.lock:
        live = h.load()
        try:
            res = await run_command(
                live, text, bounds=await h.room(live), camera=camera, capture=capture, emit=emit,
                inserter=_inserter(h), verify=verify,
            )  # fmt: skip
        except Exception as exc:  # noqa: BLE001 - report API failures to the command bar
            # tool steps were pushed live but never saved: show the saved state again
            await h.broadcast({"type": "scene", "scene": h.load().model_dump(mode="json")})
            # Provider errors can quote request details: the public demo never shows them.
            detail = "the assistant is unavailable, please try again" if HOSTED else str(exc)
            print(f"agent failed: {type(exc).__name__}", flush=True)
            await h.broadcast({"type": "agent", "stage": "done", "text": f"error: {detail}"})
            raise HTTPException(502, f"agent failed: {detail}") from exc
        h.save(live)
        if any(st["tool"] == "insert_object" for st in res.steps):
            h.bounds_loaded = False  # new splats may change the room box
        data = live.model_dump(mode="json")
    await h.broadcast({"type": "scene", "scene": data})
    return {"reply": res.reply, "steps": res.steps, "verified": res.verified,
            "verdicts": res.verdicts, "scene": data}  # fmt: skip


def _inserter(h: SceneHub):
    """M4 insert path, when its dependencies are importable on this machine. Off in hosted
    mode: it writes PLYs into the shared scene folder, which visitors' copies must not touch."""
    if HOSTED:
        return None
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
    _writable(h)
    h.load()  # 404 before charging for an unknown scene
    _charge(session, _address(request))
    camera = body.camera.model_dump() if body.camera else None
    return await _command(h, body.text[:500], camera, body.verify)


@app.post("/api/scenes/{scene_id}/voice")
async def voice(
    scene_id: str, audio: UploadFile, request: Request, camera: str | None = None
) -> dict[str, Any]:
    session = _session(request.cookies)
    h = hub(scene_id, session)
    _writable(h)
    h.load()  # 404 before charging for an unknown scene
    try:
        cam = Camera.model_validate_json(camera).model_dump() if camera else None
    except ValidationError as exc:
        raise HTTPException(400, "bad camera") from exc
    data = await audio.read()
    if HOSTED:
        try:
            if check_wav(data)[2] > MAX_VOICE_S:
                raise HTTPException(400, f"voice commands are limited to {MAX_VOICE_S:.0f} s")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    _charge(session, _address(request))
    await h.broadcast({"type": "agent", "stage": "transcript", "text": "transcribing"})
    try:
        text = (await transcribe(data)).strip()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    await h.broadcast({"type": "agent", "stage": "transcript", "text": text})
    if not text:
        return {"transcript": "", "reply": "I did not catch that.", "steps": [], "verified": None}
    out = await _command(h, text, cam, True)
    return {"transcript": text, **out}


@app.websocket("/ws/scenes/{scene_id}")
async def scene_socket(ws: WebSocket, scene_id: str) -> None:
    address = _address(ws)
    session = _session(ws.cookies)
    full = (
        (session is None and HOSTED)
        or sum(_address_sockets.values()) >= MAX_SOCKETS
        or _address_sockets.get(address, 0) >= MAX_ADDRESS_SOCKETS
    )
    if HOSTED and full:
        await ws.close(code=1008)
        return
    h = hub(scene_id, session)
    await ws.accept()
    h.sockets.add(ws)
    _address_sockets[address] = _address_sockets.get(address, 0) + 1
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
                    jpeg = msg.get("jpeg")  # goes to the verifier: never a URL to fetch
                    ok = isinstance(jpeg, str) and jpeg.startswith("data:image/jpeg;base64,")
                    fut.set_result(jpeg if ok else None)
    except WebSocketDisconnect:
        pass
    finally:
        h.sockets.discard(ws)
        _address_sockets[address] -= 1
        if not _address_sockets[address]:
            del _address_sockets[address]


def _mount_static() -> None:
    # The Windows registry can map .js to text/plain, and browsers refuse such a module script.
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("model/gltf-binary", ".glb")
    dist = REPO_ROOT / "web" / "dist"
    if SCENES_DIR.is_dir():
        app.mount("/scenes", StaticFiles(directory=SCENES_DIR), name="scenes")
    if LIBRARY_DIR.is_dir():
        app.mount("/library", StaticFiles(directory=LIBRARY_DIR), name="library")
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
