"""Publish kit: final video copy, SRT timed to the voice takes, thumbnail, captioned copy for X."""
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

B = Path("data/broll")
C = B / "_cut"
P = B / "publish"
P.mkdir(exist_ok=True)
(C / "cap").mkdir(exist_ok=True)
SEG = {r["i"]: r for r in json.loads((C / "segments.json").read_text(encoding="utf8"))}
DURS = [5.4, 8.9, 17.4, 4.5, 2.2, 2.4, 2.0, 7.77, 8.67, 7.4, 9.33, 6.7, 11.7, 4.2]
VOICE = [
    [(0.3, 0), (1.95, 1)], [(0.1, 2), (4.1, 9), (7.0, 10)], [(0.25, 11), (3.2, 12), (6.1, 13), (10.6, 14), (13.85, 19)],
    [(0.3, 20), (1.45, 21), (3.05, 22)], [(0.5, 23)], [(0.1, 24)], [(0.1, 25)], [(0.2, 26), (4.7, 27)],
    [(0.15, 28), (1.85, 29), (5.0, 34)], [(0.1, 35), (3.0, 36)], [(0.1, 37), (5.05, 38), (6.2, 39), (8.1, 40)],
    [(0.2, 42), (1.6, 43), (3.5, 45)], [(0.15, 46), (2.05, 47), (3.85, 48), (6.95, 50), (9.92, 51)], [(0.3, 52)],
]
# As spoken in the kept takes. A list splits a long take into cues, timed by length.
TEXT = {
    0: "Movable Reality.", 1: "Record your room and get a customizable 3D room.",
    2: "All it needs is a video walkthrough from the room and the ceiling height.",
    9: "No depth sensor, no floor plan,", 10: "no manual labelling.",
    11: "Muse Wonder watches the video and lists the objects.", 12: "SAM 3.1 tracks each one.",
    13: ["MapAnything finds the cameras,", "and Gaussian splatting builds the room."],
    14: "One command takes it from video to finished room.",
    19: "Every object comes out as its own set of Gaussian splats.",
    20: "Select any,", 21: "move it along an axis,", 22: "or turn it.", 23: "Measure it.",
    24: "Add furniture from a library.", 25: "Or just say what you want.",
    26: ["Ask it to move the sofa half of a metre,", "and it moves it thirty centimetres,"],
    27: "because the side table is in the way.", 28: "It answers questions too.",
    29: "Will the armchair fit between the sofa and the bookshelf?",
    34: ["No.", "The gap is one metre two, and the chair is wider."],
    35: "Ask for a plant, and it picks one and places it.",
    36: ["After every edit,", "Muse Spark Open looks at the room and checks the result."],
    37: ["No walk sees everything.", "The pipeline scores what the video covered and marks the spots"],
    38: "it barely saw.", 39: "The detail you get follows", 40: "the data you give it.",
    42: "This is a working demo,", 43: "on two rooms rendered in Blender.",
    45: "It can be a good starting point for a project or research.",
    46: "Runs on eight Meta models.", 47: "It's MIT licensed,", 48: "each stage is a plain script you can swap,",
    50: "and the test rooms and scoring tools come with it,", 51: "so you can measure your changes.",
    52: "Try the demo at the link below.",
}


def cues():
    out, t0 = [], 0.0
    for dur, takes in zip(DURS, VOICE):
        for off, seg in takes:
            a = t0 + off
            d = SEG[seg]["b"] - SEG[seg]["a"]
            parts = TEXT[seg] if isinstance(TEXT[seg], list) else [TEXT[seg]]
            n = sum(len(x) for x in parts)
            for x in parts:
                s = d * len(x) / n
                out.append([a, a + s, x])
                a += s
        t0 += dur
    for i, c in enumerate(out):
        nxt = out[i + 1][0] if i + 1 < len(out) else sum(DURS)
        c[1] = min(nxt, max(c[1] + 0.15, c[0] + 0.9))
    return out


def stamp(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def thumbnail():
    src = C / "thumb_src.jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "8.6", "-i", str(B / "edit_armchair.mp4"), "-frames:v", "1",
                    "-vf", "crop=1820:1024:50:28,scale=1280:720", "-q:v", "2", str(src)], check=True)
    im = Image.open(src).convert("RGB")
    a = np.array(im).astype(float)
    g = np.clip(1 - np.linspace(-0.25, 1.85, 1280), 0, 1)[None, :, None] * 0.93
    a = a * (1 - g) + np.array([15, 16, 21])[None, None] * g
    im = Image.fromarray(a.astype(np.uint8))
    F = "C:/Windows/Fonts/segoeuib.ttf"
    d = ImageDraw.Draw(im)
    d.text((56, 128), "Film a room.", font=ImageFont.truetype(F, 84), fill=(240, 238, 245))
    f = ImageFont.truetype(F, 84)
    for k, line in enumerate(("Move the", "furniture.")):
        b = d.textbbox((0, 0), line, font=f)
        w, h = b[2] - b[0] + 8, b[3] - b[1] + 8
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).text((4 - b[0], 4 - b[1]), line, font=f, fill=255)
        u = np.linspace(0, 1, w)[None, :, None]
        col = (np.array([196, 202, 255])[None, None] * (1 - u) + np.array([255, 214, 190])[None, None] * u).astype(np.uint8)
        im.paste(Image.fromarray(np.repeat(col, h, 0)), (58, 248 + k * 104), m)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([56, 484, 500, 546], 31, fill=(58, 63, 110))
    d.text((278, 515), "Eight Meta models", font=ImageFont.truetype(F, 34), fill=(240, 238, 245), anchor="mm")
    im.save(P / "youtube_thumbnail.jpg", quality=90)


def burned(cs):
    fs = []
    for i, (a, b, x) in enumerate(cs):
        f = C / "cap" / f"{i:03d}.txt"
        f.write_text(x, encoding="utf8")
        fs.append(f"drawtext=textfile='{f.as_posix()}':fontfile='C\\:/Windows/Fonts/segoeuib.ttf':fontsize=40:fontcolor=white:"
                  f"x=(w-tw)/2:y={840 if i == 2 else 1012}:box=1:boxcolor=black@0.72:boxborderw=12:enable='between(t,{a:.3f},{b:.3f})'")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(P / "movable_reality_final.mp4"), "-vf", ",".join(fs),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "copy",
                    "-movflags", "+faststart", str(P / "movable_reality_x_captions.mp4")], check=True)


cs = cues()
lines = []
for i, (a, b, x) in enumerate(cs, 1):
    wrapped = "\n".join(textwrap.wrap(x, 42))
    lines.append(f"{i}\n{stamp(a)} --> {stamp(b)}\n{wrapped}\n")
(P / "movable_reality.en.srt").write_text("\n".join(lines), encoding="utf8")
print("cues", len(cs), "last ends", round(cs[-1][1], 2), "longest", max(len(c[2]) for c in cs))
assert all(c[1] > c[0] for c in cs) and all(cs[i][1] <= cs[i + 1][0] + 1e-6 for i in range(len(cs) - 1))
shutil.copy2(B / "movable_reality_v2_pulse.mp4", P / "movable_reality_final.mp4")
thumbnail()
burned(cs)
