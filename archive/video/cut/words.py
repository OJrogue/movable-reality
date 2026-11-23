"""Find word gaps inside a take and transcribe growing prefixes, to place a cut."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()
from movable_reality.voice import transcribe  # noqa: E402

SRC = "data/broll/voiceover.m4a"
segs = {r["i"]: r for r in json.loads(Path("data/broll/_cut/segments.json").read_text(encoding="utf8"))}
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", SRC, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True).stdout
x = np.frombuffer(raw, dtype=np.int16).astype(float)


def dips(a, b):
    w = 320  # 20 ms
    s = x[int(a * 16000):int(b * 16000)]
    e = np.sqrt(np.convolve(s**2, np.ones(w) / w, mode="same"))[::160]
    th = np.percentile(e, 45) * 0.5
    out, i = [], 0
    while i < len(e):
        if e[i] < th:
            j = i
            while j < len(e) and e[j] < th:
                j += 1
            if j - i >= 3 and i > 5 and j < len(e) - 5:
                out.append(round(a + (i + j) / 2 * 0.01, 3))
            i = j
        else:
            i += 1
    return out


async def ask(a, b, name):
    p = Path(f"data/broll/_cut/seg/w_{name}.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-ss", str(a), "-to", str(b), "-ac", "1", "-ar", "24000",
                    "-af", "highpass=f=80,volume=14dB,alimiter=limit=0.9", "-c:a", "pcm_s16le", str(p)], check=True)
    return (await transcribe(p.read_bytes())).strip()


async def main():
    for i in [int(v) for v in sys.argv[1:]]:
        a, b = segs[i]["a"], segs[i]["b"]
        d = dips(a, b)
        print(f"== {i} {a}-{b} dips {d}")
        pts = [a] + d + [b]
        rows = await asyncio.gather(*[ask(pts[k], pts[k + 1] + 0.03, f"{i}_{k}") for k in range(len(pts) - 1)])
        for k, r in enumerate(rows):
            print(f"  {pts[k]:7.2f}-{pts[k + 1]:7.2f} {r}")


asyncio.run(main())
