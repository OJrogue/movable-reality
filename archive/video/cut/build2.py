"""Second cut: fast, timed to the voice. Animated scenes come from anim.py, viewer scenes from the clips."""
import json
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

B = Path("data/broll")
C = B / "_cut"
SC = C / "scenes2"
SC.mkdir(exist_ok=True)
FONT = "C\\:/Windows/Fonts/segoeuib.ttf"
SEG = {r["i"]: r for r in json.loads((C / "segments.json").read_text(encoding="utf8"))}
FULL = "1820:1024:50:28"  # removes the capture's black bars
NEAR = "1520:855:200:197"  # closer on the sofa and the agent's reply

# pieces: (clip, start, source seconds, speed); seq: frames from anim.py; voice: (offset, take)
SCENES = [
    {"pieces": [("16_room_before", 2.5, 2.5, 1), ("17_room_after", 5.0, 2.9, 1)], "crop": FULL, "fadein": 0.4,
     "title": "Movable Reality", "label": "Demo rooms rendered in Blender", "voice": [(0.3, 0), (1.95, 1)]},
    {"seq": "walk", "voice": [(0.1, 2), (4.1, 9), (7.0, 10)]},
    {"seq": "pipeline", "voice": [(0.25, 11), (3.2, 12), (6.1, 13), (10.6, 14), (13.85, 19)]},
    {"pieces": [("edit_armchair", 2.1, 1.5, 1), ("edit_armchair", 4.6, 3.0, 2), ("edit_armchair", 9.1, 2.4, 1.6)], "crop": FULL,
     "voice": [(0.3, 20), (1.45, 21), (3.05, 22)]},
    {"pieces": [("07_measure", 1.5, 8.3, 4.5)], "crop": FULL, "hold": 0.35, "voice": [(0.5, 23)]},
    {"pieces": [("08_library_add", 5.7, 2.4, 1)], "crop": FULL, "voice": [(0.1, 24)]},
    {"pieces": [("13_voice_fast", 5.8, 3.0, 1.5)], "crop": NEAR, "label": "Muse Voice Transcribe · sped up", "voice": [(0.1, 25)]},
    {"pieces": "SOFA", "crop": NEAR, "label": "Muse Wonder · sped up", "voice": [(0.2, 26), (4.7, 27)]},
    {"pieces": [("10_agent_fit", 2.0, 3.6, 1.1), ("10_agent_fit", 5.6, 14.9, 10), ("10_agent_fit", 20.5, 3.9, 1)], "crop": NEAR,
     "label": "Muse Wonder · sped up", "voice": [(0.15, 28), (1.85, 29), (5.0, 34)]},
    {"pieces": [("11_agent_add_fast", 3.6, 3.4, 1.7), ("11_agent_add_fast", 7.0, 6.5, 1.25)], "crop": NEAR, "hold": 0.2,
     "label": "Muse Spark Open checks every edit · sped up", "voice": [(0.1, 35), (3.0, 36)]},
    {"pieces": [("14b_gaps", 0.6, 9.33, 1)], "crop": FULL, "voice": [(0.1, 37), (5.05, 38), (6.2, 39), (8.1, 40)]},
    {"seq": "demo", "voice": [(0.2, 42), (1.6, 43), (3.5, 45)]},
    {"seq": "models", "voice": [(0.15, 46), (2.05, 47), (3.85, 48), (6.95, 50), (9.92, 51)]},
    {"seq": "end", "voice": [(0.3, 52)], "fade": 1.2},
]
SOFA = json.loads((C / "sofa_pieces.json").read_text())
for s in SCENES:
    if s.get("pieces") == "SOFA":
        s["pieces"] = [tuple(p) for p in SOFA]


def text(s, y, size, name):
    f = C / f"lbl2_{name}.txt"
    f.write_text(s, encoding="utf8")
    return (f"drawtext=textfile='{f.as_posix()}':fontfile='{FONT}':fontsize={size}:fontcolor=white:x=56:y={y}:"
            "box=1:boxcolor=black@0.55:boxborderw=16")


def seconds(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)], capture_output=True, encoding="utf8")
    return float(r.stdout)


def render(item):
    i, sc = item
    out = SC / f"{i:02d}.mp4"
    post = []
    if sc.get("seq"):
        n = len(list((C / "frames" / sc["seq"]).glob("*.jpg")))
        inputs = ["-framerate", "30", "-start_number", "0", "-i", str(C / "frames" / sc["seq"] / "%05d.jpg")]
        fc = "[0:v]scale=in_range=full:out_range=tv,format=yuv420p[c]"
        length = n / 30
    else:
        inputs, chains = [], []
        for k, (clip, st, dur, speed) in enumerate(sc["pieces"]):
            inputs += ["-ss", str(st), "-t", str(dur), "-i", str(B / f"{clip}.mp4")]
            chains.append(f"[{k}:v]setpts=(PTS-STARTPTS)/{speed},fps=30,format=yuv420p[v{k}]")
        n = len(sc["pieces"])
        fc = ";".join(chains) + ";" + "".join(f"[v{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0[c]"
        length = sum(p[2] / p[3] for p in sc["pieces"]) + sc.get("hold", 0)
    if sc.get("crop"):
        post.append(f"crop={sc['crop']},scale=1920:1080:flags=lanczos,setsar=1")
    if sc.get("hold"):
        post.append(f"tpad=stop_mode=clone:stop_duration={sc['hold']}")
    if sc.get("title"):
        post.append(text(sc["title"], 860, 64, f"{i}t"))
    if sc.get("label"):
        post.append(text(sc["label"], 960, 34, f"{i}l"))
    if sc.get("fadein"):
        post.append(f"fade=t=in:st=0:d={sc['fadein']}")
    if sc.get("fade"):
        post.append(f"fade=t=out:st={length - sc['fade']}:d={sc['fade']}")
    fc += ";[c]" + (",".join(post) or "null") + "[o]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", fc, "-map", "[o]", "-an",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-r", "30", str(out)], check=True)
    return seconds(out)


def main():
    with ThreadPoolExecutor(4) as ex:
        durs = list(ex.map(render, enumerate(SCENES)))
    total = sum(durs)
    print("scenes", [round(d, 2) for d in durs], "total", round(total, 2))

    with wave.open(str(C / "voice_clean.wav")) as w:
        src = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float)
    SR = 48000
    track = np.zeros(int((total + 1) * SR))
    t0, warn, used, silence = 0.0, [], [], 0.0
    prev_end = 0.0
    for i, sc in enumerate(SCENES):
        last = 0.0
        for off, seg in sc["voice"]:
            a, b = SEG[seg]["a"], SEG[seg]["b"]
            x = src[int(a * SR):int(b * SR)].copy()
            f = int(0.015 * SR)
            x[:f] *= np.linspace(0, 1, f)
            x[-f:] *= np.linspace(1, 0, f)
            if off < last - 0.01:
                warn.append(f"scene {i} take {seg} starts {off} before previous end {last:.2f}")
            last = off + len(x) / SR
            if last > durs[i]:
                warn.append(f"scene {i} take {seg} ends {last:.2f} after scene {durs[i]:.2f}")
            p = int((t0 + off) * SR)
            silence = max(silence, t0 + off - prev_end)
            prev_end = t0 + last
            track[p:p + len(x)] += x
            used.append((round(t0 + off, 1), seg))
        t0 += durs[i]
    print("warnings:", warn or "none")
    print("longest gap between takes:", round(silence, 2))
    print("voice at:", used)
    track = np.clip(track, -32767, 32767).astype(np.int16)
    vt = C / "voice_track2.wav"
    with wave.open(str(vt), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(track.tobytes())

    (C / "list2.txt").write_text("".join(f"file 'scenes2/{i:02d}.mp4'\n" for i in range(len(SCENES))), encoding="utf8")
    pic = C / "picture2.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(C / "list2.txt"), "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-color_range", "tv", "-r", "30", "-an", str(pic)], check=True)

    T = f"{total:.3f}"

    def mux(name, fc, extra):
        out = B / f"movable_reality_v2_{name}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(pic), *extra, "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", T, "-movflags", "+faststart", str(out)], check=True)
        return out

    fade = f"afade=t=out:st={total - 2.0:.2f}:d=2.0"
    outs = [mux("voice_only", f"[1:a]aformat=channel_layouts=stereo,atrim=0:{T},volume=4dB,alimiter=limit=0.89:level=0[a]", ["-i", str(vt)])]
    for name, gain in (("pulse", -17.0), ("pad", -16.0)):
        fc = (f"[1:a]aformat=channel_layouts=stereo,asplit=2[v1][v2];"
              f"[2:a]atrim=0:{T},volume={gain}dB[m];"
              "[m][v1]sidechaincompress=threshold=0.04:ratio=5:attack=60:release=700:makeup=1[d];"
              f"[d]{fade}[d2];[v2][d2]amix=inputs=2:normalize=0,volume=4dB,alimiter=limit=0.89:level=0,atrim=0:{T}[a]")
        outs.append(mux(name, fc, ["-i", str(vt), "-i", str(C / f"music_{name}.wav")]))
    for o in outs:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(o), "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"],
                           capture_output=True, encoding="utf8", errors="replace").stderr
        keep = [ln.strip() for ln in r.splitlines() if ln.strip().startswith(("I:", "LRA:", "Peak:"))]
        print(o.name, round(o.stat().st_size / 1e6, 1), "MB", keep[-3:])


main()
