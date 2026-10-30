"""Cached chat calls. Every paid response is stored under data/cache/chat/ keyed by a hash of
the request, so reruns and the hosted no-key demo replay them without a network call."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from ..config import CACHE_DIR

CHAT_CACHE = CACHE_DIR / "chat"


def request_key(provider: str, payload: dict[str, Any]) -> str:
    blob = json.dumps({"provider": provider, **payload}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def cache_only() -> bool:
    """MR_CACHE_ONLY=1: never call a paid API, answer only from the cache (hosted fallback)."""
    return os.environ.get("MR_CACHE_ONLY", "") == "1"


async def cached_chat(client, provider: str, **payload: Any) -> dict[str, Any]:
    """chat.completions.create through the cache. Returns the response as a plain dict."""
    key = request_key(provider, payload)
    path = CHAT_CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    if cache_only():
        raise RuntimeError("not in the cache and MR_CACHE_ONLY=1")
    resp = await client.chat.completions.create(**payload)
    data = resp.model_dump(exclude_none=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data
