"""Muse Voice Transcribe: WAV in, transcript out, cached by the audio's hash.

Endpoint (checked 30 Oct, HANDOFF section 7): POST {META_BASE_URL}/asr/transcribe, multipart
with a JSON `request` part and an `audio` part (WAV, mono, 16-bit PCM, 16 or 24 kHz).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import wave

import httpx

from .config import CACHE_DIR, META_BASE_URL, Models, require_env

VOICE_CACHE = CACHE_DIR / "voice"


def check_wav(data: bytes) -> tuple[int, int, float]:
    """Return (rate, channels, seconds); raise ValueError if the endpoint would reject it."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            rate, ch, width, n = (
                w.getframerate(),
                w.getnchannels(),
                w.getsampwidth(),
                w.getnframes(),
            )
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"not a WAV file: {exc}") from exc
    if ch != 1 or width != 2 or rate not in (16000, 24000):
        raise ValueError(f"need mono 16-bit 16/24 kHz WAV, got {ch} ch, {8 * width} bit, {rate} Hz")
    return rate, ch, n / rate


async def transcribe(data: bytes) -> str:
    check_wav(data)
    key = hashlib.sha256(data).hexdigest()[:32]
    path = VOICE_CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["transcript"]
    if os.environ.get("MR_CACHE_ONLY", "") == "1":
        raise ValueError("voice is off in this demo (no API key); type the command instead")
    request = {"model": Models.TRANSCRIBE, "audioEncoding": "WAV", "mode": "PUSH_TO_TALK"}
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(
            f"{META_BASE_URL}/asr/transcribe",
            headers={"Authorization": f"Bearer {require_env('MODEL_API_KEY')}"},
            files={
                "request": (None, json.dumps(request), "application/json"),
                "audio": ("command.wav", data, "audio/wav"),
            },
        )
    r.raise_for_status()
    out = r.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out.get("transcript", "")
