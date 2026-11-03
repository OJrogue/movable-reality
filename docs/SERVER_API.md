# Scene server API (core lane, M3)

The server is `src/movable_reality/server.py` (FastAPI). Start it with
`uv run python -m movable_reality.server` (port 8000). The Vite dev server proxies `/api` and
`/ws` to it, so the viewer uses relative URLs. Without the server the viewer still loads
scenes from `/scenes` and applies edits locally (they are lost on reload).

## HTTP

| Method and path | Body | Answer |
|---|---|---|
| `GET /api/health` | none | `{"ok": true, "models": {...}}` |
| `GET /api/scenes/{id}` | none | the manifest (`scene.json`) |
| `POST /api/scenes/{id}/edit` | `{"op", "object_id", "args", "instruction"?}` | `{"ok": true, "scene": manifest}` or HTTP 400 `{"detail"}` |
| `POST /api/scenes/{id}/command` | `{"text", "camera"?, "verify"?}` | `{"reply", "steps": [...], "verified", "verdicts", "scene"}` |
| `POST /api/scenes/{id}/voice` | multipart `audio` = WAV, mono, 16-bit, 16 kHz, optional query `?camera=<json>` | `{"transcript", "reply", "steps", "verified", "scene"}` |
| `GET /scenes/{id}/scene.json` | none | the manifest (the visitor's copy in hosted mode) |

`camera` in a command is the viewer's pose, `{"eye": [x, y, z], "forward": [x, y, z]}` in world
coordinates, so "left" and "towards me" mean the user's left and the user's position. `verify`
(default true) turns the screenshot check off. `verified` is true, false, or null when no
viewer was connected or the verifier gave no usable answer.

Edit ops (same as `Scene.apply_edit` in `scene.py`):

| op | args | effect |
|---|---|---|
| `move` | `dx, dy, dz` (metres, world frame) | adds to `transform.translation` |
| `place` | `x, z` | puts the bbox centre at world (x, z), height unchanged |
| `rotate` | `yaw_deg` | adds to `transform.yaw_deg` (about world +Y, bbox centre) |
| `remove` / `restore` | none | sets `visible` |
| `undo` | none, `object_id` null | reverts the newest edit not yet undone, appends `{"op": "undo", "args": {"index": i, "of": op}}` |

Every edit is appended to `history`. Undo never deletes history.

## WebSocket `/ws/scenes/{id}`

Server to viewer:
- `{"type": "scene", "scene": manifest}` after every change (and once on connect).
- `{"type": "agent", "stage": "transcript"|"thinking"|"tool"|"verify"|"done", "text": ...}` progress
  of a command, for the command bar.
- `{"type": "capture_request", "id": n, "camera"?: {"eye": [x,y,z], "target": [x,y,z]}}` asks
  the viewer for a screenshot (the verifier uses it).

Viewer to server:
- `{"type": "capture", "id": n, "jpeg": "data:image/jpeg;base64,..."}` answers a capture request.

## Hosted mode (U3)

`MR_SESSIONS=1` gives each visitor a cookie (`mr_session`) and a private copy of `scene.json`
under `data/sessions/<session>/<scene>/`, so edits stay local to that visitor. Paid calls are
capped with `MR_MAX_COMMANDS` per visitor (default 20) and `MR_DAILY_COMMANDS` for all
visitors per UTC day (default 300); past a cap the server answers HTTP 429. Voice is limited to
`MR_MAX_VOICE_S` seconds (default 15). `MR_CACHE_ONLY=1` answers commands only from cached
replies and turns voice off, for a deployment without keys.

In hosted mode an edit, command or voice request without the session cookie gets HTTP 403, so it
cannot change the scene every visitor shares. The counters live in `usage.json` in the sessions
folder and survive a restart. One address gets `MR_IP_COMMANDS` commands per UTC day (default 60).
A body over 2 MiB is refused (HTTP 413, or the connection ends when it has no Content-Length), a
visitor's copy holds at most 1000 edits (HTTP 429), one edit value is at most 10000 and `camera` must
be `{"eye", "forward"}` with three finite numbers each (HTTP 422, or 400 on voice). A WebSocket
without the cookie, or past 8 per address or 150 in all, is closed with code 1008. Visitors' copies
never expire, so one made before a republish keeps the old manifest. Not limited: new copies from
fresh cookies, at about 170 KB each on the volume.

The public deployment is `modal_jobs/demo.py`: one CPU container that scales to zero, with the
scenes from `scripts/publish_demo.py` and the visitors' copies on a Modal volume. It has no second
container because WebSockets and scene locks live in process memory. A running container does not
see new volume files, so deploy again after publishing. With `max_containers=1` the new container
waits for the old one's open viewers, up to 10 min (seen on 3 Nov), so do not redeploy while someone
is presenting. Keep it warm for an event with
`modal.Function.from_name("movable-reality-demo", "web").update_autoscaler(min_containers=1)`.
