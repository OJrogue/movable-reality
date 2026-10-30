"""Hosted demo (U3): viewer + scene server as one public Modal ASGI app.

    cd web && npm run build && cd ..                       # web/dist
    uv run modal serve modal_jobs/hosted.py                # temporary URL while it runs
    uv run modal deploy modal_jobs/hosted.py               # permanent URL (OJ decides)
    MR_HOSTED_SCENES=blender01,demo uv run modal deploy modal_jobs/hosted.py

The image carries web/dist and the listed scene folders (read-only). Visitors get their own
copy of scene.json (MR_SESSIONS=1, cookie), and paid calls are capped per visitor and per day
(src/movable_reality/server.py). Keys come from the Modal secret "movable-reality-keys"
(MODEL_API_KEY, BASETEN_API_KEY), created once with
    uv run modal secret create movable-reality-keys MODEL_API_KEY=... BASETEN_API_KEY=...
Without that secret, set MR_HOSTED_CACHE_ONLY=1: commands are then answered only from the
cached replies in data/cache/chat (scripted demo commands), and voice is off.
Range requests for paged .rad trees work through Starlette's StaticFiles.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
SCENES = [s for s in os.environ.get("MR_HOSTED_SCENES", "blender01,demo").split(",") if s]
CACHE_ONLY = os.environ.get("MR_HOSTED_CACHE_ONLY", "") == "1"


def _skip(path: Path) -> bool:
    """Only what the viewer loads: manifest, PLYs, LOD trees, baked planes, coverage JSON."""
    parts = Path(path).parts
    if not parts or Path(path).suffix == "":
        return False  # directories
    keep = parts[0] in ("objects", "lod", "planes") or parts == ("scene.json",)
    return not (keep or (parts[0] == "coverage" and Path(path).suffix == ".json"))


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "openai>=2.0",
        "meta-sam-parser>=0.1.0",
        "python-dotenv>=1.0",
        "numpy>=2.0",
        "pillow>=11.0",
        "pydantic>=2.8",
        "plyfile>=1.1",
        "fastapi>=0.115",
        "python-multipart>=0.0.12",
        "httpx>=0.27",
    )  # fmt: skip
    .env(
        {
            "MR_SESSIONS": "1",
            "MR_DATA_DIR": "/data",
            "MR_SESSIONS_DIR": "/tmp/sessions",
            "MR_CACHE_ONLY": "1" if CACHE_ONLY else "0",
        }
    )
    .add_local_dir(ROOT / "web" / "dist", "/repo/web/dist")
    .add_local_dir(ROOT / "data" / "cache" / "chat", "/data/cache/chat")
)
for scene in SCENES:
    image = image.add_local_dir(
        ROOT / "data" / "scenes" / scene,
        f"/data/scenes/{scene}",
        ignore=_skip,
    )
image = image.add_local_python_source("movable_reality")

app = modal.App("movable-reality-demo", image=image)
secrets = [] if CACHE_ONLY else [modal.Secret.from_name("movable-reality-keys")]


@app.function(secrets=secrets, max_containers=1, timeout=600)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    # One container: per-visitor state and the daily cap live in its memory.
    import movable_reality.config as cfg  # noqa: PLC0415

    cfg.REPO_ROOT = Path("/repo")
    from movable_reality.server import app as fastapi_app  # noqa: PLC0415

    return fastapi_app
