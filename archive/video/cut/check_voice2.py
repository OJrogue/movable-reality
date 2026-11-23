"""Transcribe the v2 voice track in chunks at scene ends, to check the takes and their order."""
import asyncio
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
from movable_reality.voice import transcribe  # noqa: E402

V = "data/broll/_cut/voice_track2.wav"
CUTS = [0, 5.4, 14.3, 31.7, 38.9, 43.1, 50.9, 59.5, 66.9, 76.2, 82.9, 94.6, 98.6]


async def one(a, b, i):
    p = Path(f"data/broll/_cut/seg/chk2_{i}.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", V, "-ss", str(a), "-to", str(b), "-ac", "1", "-ar", "24000",
                    "-af", "volume=3dB", "-c:a", "pcm_s16le", str(p)], check=True)
    return (await transcribe(p.read_bytes())).strip()


async def main():
    rows = await asyncio.gather(*[one(CUTS[i], CUTS[i + 1], i) for i in range(len(CUTS) - 1)])
    for i, r in enumerate(rows):
        print(f"{CUTS[i]:6.1f}  {r}")


asyncio.run(main())
