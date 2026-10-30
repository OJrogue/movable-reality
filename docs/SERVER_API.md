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
| `POST /api/scenes/{id}/command` | `{"text"}` | `{"reply", "steps": [...], "verified", "scene"}` |
| `POST /api/scenes/{id}/voice` | multipart `audio` = WAV, mono, 16-bit, 16 kHz | `{"transcript", "reply", "steps", "verified", "scene"}` |

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
