"""Are doubled words in the takes real? Transcribe again with other settings and in halves."""
import asyncio
import json
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
from movable_reality.voice import transcribe  # noqa: E402

SRC = "data/broll/voiceover.m4a"
segs = {r["i"]: r for r in json.loads(Path("data/broll/_cut/segments.json").read_text(encoding="utf8"))}
CHECK = [1, 2, 13, 14, 27, 36, 37, 39, 45, 48, 52]


async def ask(a, b, name, rate, gain):
    p = Path(f"data/broll/_cut/seg/t_{name}.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-ss", str(a), "-to", str(b), "-ac", "1", "-ar", str(rate),
                    "-af", f"highpass=f=80,volume={gain}dB,alimiter=limit=0.9", "-c:a", "pcm_s16le", str(p)], check=True)
    return (await transcribe(p.read_bytes())).strip()


async def main():
    for i in CHECK:
        a, b = segs[i]["a"], segs[i]["b"]
        m = (a + b) / 2
        alt, h1, h2 = await asyncio.gather(
            ask(a - 0.2, b + 0.2, f"{i}a", 24000, 14), ask(a, m + 0.1, f"{i}h1", 24000, 14), ask(m - 0.1, b, f"{i}h2", 24000, 14))
        print(f"{i:3d} first: {segs[i]['text']}\n    alt:   {alt}\n    halves: {h1} | {h2}")


asyncio.run(main())
