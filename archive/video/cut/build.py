"""Cut the submission video: scenes, labels, voice takes on a timeline, three audio versions."""
import json
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

B = Path("data/broll")
C = B / "_cut"
SC = C / "scenes"
SC.mkdir(exist_ok=True)
FONT = "C\\:/Windows/Fonts/segoeuib.ttf"
URL = "ojytcanal--movable-reality-demo.us-east.modal.run"
SEG = {r["i"]: r for r in json.loads((C / "segments.json").read_text(encoding="utf8"))}

# pieces: (clip, start, source seconds, speed); hold: seconds of last frame; voice: (offset, [takes])
SCENES = [
    {"pieces": [("16_room_before", 0, 5.0, 1), ("17_room_after", 5.0, 3.9, 1)],
     "title": "Movable Reality", "label": "Demo rooms rendered in Blender",
     "voice": [(1.0, 0), (2.85, 1)]},
    {"pieces": [("landing_card", 0, 13, 1)], "crop": "1536:864:192:108", "label": "Rendered in Blender · 132 s of video",
     "voice": [(2.8, 2), (7.1, 9), (9.85, 10)]},
    {"pieces": [("landing_pipeline", 1, 15, 1)], "crop": "1536:864:192:108", "label": "Muse Wonder · SAM 3.1 · MapAnything",
     "voice": [(0.6, 11), (3.65, 12), (6.65, 13), (11.2, 14)]},
    {"pieces": [("03_drag_armchair", 0, 10, 1)], "voice": [(0.8, 19)]},
    {"pieces": [("04_gizmo_move", 3, 8, 1), ("05_gizmo_rotate", 6, 6, 1)],
     "voice": [(0.4, 20), (3.5, 21), (9.0, 22)]},
    {"pieces": [("07_measure", 1.5, 8.3, 1)], "voice": [(3.0, 23)]},
    {"pieces": [("08_library_add", 0, 8, 1)], "voice": [(1.0, 24)]},
    {"pieces": [("13_voice_fast", 0, 14.8, 1.6)], "label": "Muse Voice Transcribe · sped up", "voice": [(0.5, 25)]},
    {"pieces": [("09_agent_move_fast", 0, 26.5, 1.7)], "hold": 2.5, "label": "Muse Wonder · sped up",
     "voice": [(8.4, 26), (12.85, 27)]},
    {"pieces": [("10_agent_fit", 0, 7, 1), ("10_agent_fit", 7, 12.5, 7), ("10_agent_fit", 19.5, 5, 1)],
     "label": "Muse Wonder · sped up", "voice": [(1.0, 28), (2.62, 29), (8.8, 34)]},
    {"pieces": [("11_agent_add_fast", 0, 14.56, 1)], "hold": 0.8, "label": "Muse Spark Open · sped up",
     "voice": [(3.0, 35), (7.7, 36)]},
    {"pieces": [("14_coverage_gaps", 1, 7, 0.65)], "voice": [(0.5, 37), (5.45, 38), (6.9, 39), (8.84, 40)]},
    {"pieces": [("landing_build", 0, 9, 1)], "crop": "1280:720:330:360", "voice": [(1.0, 42), (2.55, 43), (4.68, 45)]},
    {"pieces": [("landing_models", 0, 13.2, 1)], "crop": "1320:742:300:100", "voice": [(0.5, 46), (2.53, 47), (4.44, 48), (7.67, 50), (10.66, 51)]},
    {"pieces": [("landing_hero", 0, 9, 1)], "label": URL, "sub": "Team Larracat", "voice": [(1.0, 52)], "fade": 2.0},
]


def text(s, y, size, name):
    f = C / f"lbl_{name}.txt"
    f.write_text(s, encoding="utf8")
    rel = f.as_posix()
    return (f"drawtext=textfile='{rel}':fontfile='{FONT}':fontsize={size}:fontcolor=white:x=56:y={y}:"
            "box=1:boxcolor=black@0.55:boxborderw=16")


def render(idx_scene):
    i, sc = idx_scene
    inputs, chains, n = [], [], len(sc["pieces"])
    for k, (clip, st, dur, speed) in enumerate(sc["pieces"]):
        inputs += ["-ss", str(st), "-t", str(dur), "-i", str(B / f"{clip}.mp4")]
        chains.append(f"[{k}:v]setpts=(PTS-STARTPTS)/{speed},fps=30,format=yuv420p[v{k}]")
    tail = "".join(f"[v{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0[c]"
    post = []
    if sc.get("crop"):
        post.append(f"crop={sc['crop']},scale=1920:1080:flags=lanczos,setsar=1")
    if sc.get("hold"):
        post.append(f"tpad=stop_mode=clone:stop_duration={sc['hold']}")
    if sc.get("title"):
        post.append(text(sc["title"], 860, 64, f"{i}t"))
    if sc.get("label"):
        post.append(text(sc["label"], 960, 34, f"{i}l"))
    if sc.get("sub"):
        post.append(text(sc["sub"], 880, 34, f"{i}s"))
    length = scene_len(sc)
    if i == 0:
        post.append("fade=t=in:st=0:d=0.6")
    if sc.get("fade"):
        post.append(f"fade=t=out:st={length - sc['fade']}:d={sc['fade']}")
    fc = ";".join(chains + [tail + (";[c]" + ",".join(post) + "[o]" if post else ";[c]null[o]")])
    out = SC / f"{i:02d}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", fc, "-map", "[o]", "-an",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-r", "30", str(out)], check=True)
    d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                             capture_output=True, encoding="utf8").stdout)
    return d


def scene_len(sc):
    return sum(p[2] / p[3] for p in sc["pieces"]) + sc.get("hold", 0)


def main():
    with ThreadPoolExecutor(4) as ex:
        durs = list(ex.map(render, enumerate(SCENES)))
    total = sum(durs)
    print("scenes", [round(d, 2) for d in durs], "total", round(total, 2))
    assert total < 180

    # voice: clean the whole recording once, then place the takes
    chain = ("highpass=f=75,volume=19dB,afftdn=nr=8:nf=-55:tn=1,deesser=i=0.35:m=0.5:f=0.5,"
             "equalizer=f=180:t=q:w=1:g=-1.5,equalizer=f=3200:t=q:w=1.3:g=2,"
             "acompressor=threshold=-22dB:ratio=2.5:attack=12:release=180:makeup=2,volume=1.8dB,alimiter=limit=0.84:level=0")
    clean = C / "voice_clean.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(B / "voiceover.m4a"), "-af", chain, "-ac", "1", "-ar", "48000",
                    "-c:a", "pcm_s16le", str(clean)], check=True)
    with wave.open(str(clean)) as w:
        src = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float)
    SR = 48000
    track = np.zeros(int((total + 1) * SR))
    t0 = 0.0
    warn = []
    used = []
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
            track[p:p + len(x)] += x
            used.append((round(t0 + off, 1), seg))
        t0 += durs[i]
    print("warnings:", warn or "none")
    print("voice at:", used)
    track = np.clip(track, -32767, 32767).astype(np.int16)
    vt = C / "voice_track.wav"
    with wave.open(str(vt), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(track.tobytes())

    # picture
    (C / "list.txt").write_text("".join(f"file 'scenes/{i:02d}.mp4'\n" for i in range(len(SCENES))), encoding="utf8")
    pic = C / "picture.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(C / "list.txt"), "-c", "copy", str(pic)], check=True)

    # audio versions
    T = f"{total:.3f}"
    def mux(name, fc, extra):
        out = B / f"movable_reality_{name}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(pic), *extra, "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", T, "-movflags", "+faststart", str(out)], check=True)
        return out

    fade = f"afade=t=out:st={total - 2.5:.2f}:d=2.5"
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
        keep = [l.strip() for l in r.splitlines() if l.strip().startswith(("I:", "LRA:", "Peak:"))]
        print(o.name, round(o.stat().st_size / 1e6, 1), "MB", keep[-3:])


main()
