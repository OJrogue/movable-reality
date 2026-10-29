"""Central configuration. Every model ID and endpoint used in the project lives here."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

META_BASE_URL = "https://api.meta.ai/v1"
BASETEN_BASE_URL = "https://inference.baseten.co/v1"


class Models:
    """Meta model IDs. Standard mode only (no -contributor suffix)."""

    AGENT = "muse-wonder-1.0"  # agent brain, tool calls, video inventory
    FAST = "muse-spark-1.3"  # cheap helper calls
    IMAGE = "muse-image-1.0"  # furniture generation for inserts
    TRANSCRIBE = "muse-voice-transcribe-1.0"  # voice commands
    SAM = "sam-3.1"  # segmentation and tracking
    VERIFIER_OPEN = "meta-models/muse-spark-open-1026"  # open weights on Baseten
    # Open weights run on Modal. MapAnything: Apache-2.0 weights only, never the CC-BY-NC
    # facebook/map-anything (demo_colmap.py --apache selects them).
    MAPANYTHING = "facebook/map-anything-apache"


DATA_DIR = Path(os.environ.get("MR_DATA_DIR", REPO_ROOT / "data"))
SCENES_DIR = DATA_DIR / "scenes"
CACHE_DIR = DATA_DIR / "cache"


def scene_dir(scene_id: str) -> Path:
    return SCENES_DIR / scene_id


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value
