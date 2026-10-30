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
