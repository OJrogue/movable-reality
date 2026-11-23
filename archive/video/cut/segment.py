"""Split voiceover.m4a at silences, transcribe every piece with Muse Voice Transcribe."""
import asyncio
import json
import re
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
from movable_reality.voice import transcribe  # noqa: E402

SRC = "data/broll/voiceover.m4a"
OUT = Path("data/broll/_cut/seg")
OUT.mkdir(parents=True, exist_ok=True)
log = subprocess.run(
    ["ffmpeg", "-hide_banner", "-i", SRC, "-af", "silencedetect=noise=-38dB:d=0.45", "-f", "null", "-"],
    capture_output=True, encoding="utf8", errors="replace").stderr
starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", log)]
ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
DUR = 165.61
segs = []
for i, e in enumerate(ends):
    s_next = starts[i + 1] if i + 1 < len(starts) else DUR
    if s_next - e > 0.15:
        segs.append((round(max(0, e - 0.08), 3), round(min(DUR, s_next + 0.12), 3)))
if starts and starts[0] > 0.2:
    segs.insert(0, (0.0, starts[0]))


async def one(i, a, b, sem):
    wav = OUT / f"{i:03d}.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-ss", str(a), "-to", str(b), "-ac", "1", "-ar", "16000",
                    "-af", "volume=20dB,alimiter=limit=0.9", "-c:a", "pcm_s16le", str(wav)], check=True)
    async with sem:
        try:
            text = (await transcribe(wav.read_bytes())).strip()
        except Exception as e:  # keep going, a piece that fails is listed
            text = f"[ERROR {type(e).__name__}: {e}]"[:120]
    return {"i": i, "a": a, "b": b, "text": text}


async def main():
    sem = asyncio.Semaphore(6)
    rows = await asyncio.gather(*[one(i, a, b, sem) for i, (a, b) in enumerate(segs)])
    Path("data/broll/_cut/segments.json").write_text(json.dumps(rows, indent=1), encoding="utf8")
    for r in rows:
        print(f"{r['i']:3d} {r['a']:7.2f}-{r['b']:7.2f}  {r['text']}")


asyncio.run(main())
