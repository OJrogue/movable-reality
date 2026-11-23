"""Transcribe the finished voice track in chunks at its pauses, to check the takes and their order."""
import asyncio
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
from movable_reality.voice import transcribe  # noqa: E402

V = "data/broll/_cut/voice_track.wav"
CUTS = [0, 10.5, 22, 36, 46, 62, 69, 76, 88, 105, 120, 133, 145, 153.5, 168, 175.7]


async def one(a, b, i):
    p = Path(f"data/broll/_cut/seg/chk_{i}.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", V, "-ss", str(a), "-to", str(b), "-ac", "1", "-ar", "24000",
                    "-af", "volume=3dB", "-c:a", "pcm_s16le", str(p)], check=True)
    return (await transcribe(p.read_bytes())).strip()


async def main():
    rows = await asyncio.gather(*[one(CUTS[i], CUTS[i + 1], i) for i in range(len(CUTS) - 1)])
    for i, r in enumerate(rows):
        print(f"{CUTS[i]:6.1f}  {r}")


asyncio.run(main())
