"""The hosted demo on Modal (U3): viewer, scene server and agent in one small CPU container.

    cd web && npm run build                     # the viewer goes into the image
    uv run python scripts/publish_demo.py       # scenes -> the movable-reality-demo volume
    uv run modal deploy modal_jobs/demo.py      # prints the public URL

One container on purpose: open WebSockets and the per-scene locks live in process memory
(server.py), so a second container would not see the visitors of the first. Scales to zero
when idle. Visitors' scene copies, the command counters and the reply cache stay on the volume
across cold starts. MODEL_API_KEY and BASETEN_API_KEY go in as a Modal secret built from the
local .env at deploy time and are never written to the repo.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "movable-reality-demo"
VOLUME_NAME = "movable-reality-demo"
REPO = Path(__file__).resolve().parents[1]
DATA = "/data"

secret_env = {"MODEL_API_KEY": "", "BASETEN_API_KEY": ""}
if modal.is_local():
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    secret_env = {k: os.environ.get(k, "") for k in secret_env}
    if not all(secret_env.values()):
        raise SystemExit("MODEL_API_KEY and BASETEN_API_KEY must be in .env for the demo agent")
    if not (REPO / "web" / "dist" / "index.html").exists():
        raise SystemExit("web/dist is missing: cd web && npm run build")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(  # versions from uv.lock; only what the server and the agent import
        "fastapi==0.162.0",
        "starlette==1.10.0",
        "uvicorn[standard]==0.54.0",
        "python-multipart==0.0.32",
        "openai==3.39.0",
        "httpx==0.28.1",
        "pydantic==2.14.1",
        "python-dotenv==1.2.4",
        "numpy==2.5.4",
        "pillow==13.0.0",
        "plyfile==1.1.5",
    )
    .env({"PYTHONPATH": "/app/src", "MR_DATA_DIR": DATA, "MR_SESSIONS": "1"})
    .add_local_dir(
        REPO / "src" / "movable_reality", "/app/src/movable_reality", ignore=["**/__pycache__/**"]
    )
    .add_local_dir(REPO / "web" / "dist", "/app/web/dist")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    volumes={DATA: volume},
    secrets=[modal.Secret.from_dict(secret_env)],
    cpu=1.0,
    memory=2048,  # room_bounds memory-maps the background PLY on the first command
    max_containers=1,
    scaledown_window=300,
    # A WebSocket is one input and the viewer reconnects when it closes. A deploy starts the new
    # container only after the old one's inputs end, so this also bounds a deploy's outage.
    timeout=600,
)
@modal.concurrent(max_inputs=200)
@modal.asgi_app(label=APP_NAME)
def web():
    from movable_reality.server import app as server  # noqa: PLC0415 - needs the volume mounted

    return server
