"""Synthetic Blender rooms rendered on Modal GPUs (scripts/blender_room.py, Cycles + OptiX).

    uv run modal run modal_jobs/blender.py --scene blender02                    # both walks
    uv run modal run modal_jobs/blender.py --scene blender03 --walk test --chunk 20 --samples 16

Each container builds the room once (CC0 assets from data/assets, docs/ASSETS.md) and renders
one frame range of one walk with its GT (depth, instance masks); the chunks run in parallel
with `.map`. cameras.json always holds every frame of the walk, so chunks agree. Each chunk
writes its files to the Modal Volume `movable-reality-renders` and marks itself done there, so
a rerun renders only the missing chunks (returning ~300 files per chunk through the client lost
the connection on 2 Nov). The files are then downloaded into data/raw/<scene>/ and
data/raw/<scene>_gt/ exactly as a local render writes them, and the videos are encoded locally
with ffmpeg (blender_room.py --stage encode does the same).
No learned models run here. Blender 4.5 LTS (GPL, a tool, not shipped) is downloaded from
download.blender.org into the image.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
BLENDER = "blender-4.5.10-linux-x64"
URL = f"https://download.blender.org/release/Blender4.5/{BLENDER}.tar.xz"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "xz-utils", "libxi6", "libxxf86vm1", "libxfixes3", "libxrender1",
                 "libgl1", "libxkbcommon0", "libsm6", "libice6", "libegl1", "libglib2.0-0")
    .run_commands(f"wget -q {URL} -O /tmp/b.tar.xz && tar -xJf /tmp/b.tar.xz -C /opt "
                  f"&& rm /tmp/b.tar.xz")
    .add_local_dir(REPO / "data" / "assets", "/work/data/assets")
    .add_local_file(REPO / "scripts" / "blender_room.py", "/work/scripts/blender_room.py")
    .add_local_file(REPO / "scripts" / "blender_bedroom.py", "/work/scripts/blender_bedroom.py")
    .add_local_file(REPO / "scripts" / "blender_living.py", "/work/scripts/blender_living.py")
)  # fmt: skip
app = modal.App("movable-reality-blender", image=image)
VOLUME = "movable-reality-renders"
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)


@app.function(gpu=["L4", "A10G", "L40S"], timeout=3 * 3600, max_containers=40,
              volumes={"/vol": vol})  # fmt: skip
def render_chunk(scene: str, walk: str, frames: str, samples: int) -> int:
    """Build the room, render `frames` (a:b) of `walk`, convert its GT and copy every written
    file under data/raw to /vol/raw (same relative paths). Returns the number of files; a
    chunk already marked done in the volume is skipped (0)."""
    mark = Path("/vol/done") / f"{scene}_{walk}_{frames.replace(':', '-')}_{samples}"
    if mark.exists():
        return 0
    out = Path("/work/data/raw")
    t0 = time.time()
    cmd = [f"/opt/{BLENDER}/blender", "-b", "-P", "/work/scripts/blender_room.py", "--",
           "--scene", scene, "--walk", walk, "--frames", frames, "--stage", "build,render,gt",
           "--samples", str(samples), "--device", "OPTIX"]  # fmt: skip
    res = subprocess.run(cmd, cwd="/work", capture_output=True, text=True)
    if res.returncode != 0:
        tail = f"{res.stdout[-3000:]}\n{res.stderr[-2000:]}"
        raise RuntimeError(f"blender failed ({frames}):\n{tail}")
    n = 0
    for p in out.rglob("*"):
        rel = p.relative_to(out).as_posix()
        if p.is_file() and p.suffix != ".blend" and "/raw/" not in f"/{rel}":
            if rel.endswith("render_log.json"):
                continue  # each chunk's log only holds itself
            dst = Path("/vol/raw") / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
            n += 1
    mark.parent.mkdir(parents=True, exist_ok=True)
    mark.write_text(str(n))
    vol.commit()
    print(f"{scene} {walk} {frames}: {n} files, {time.time() - t0:.0f} s")
    return n


@app.local_entrypoint()
def main(scene: str = "blender02", walk: str = "main,test", chunk: int = 30, samples: int = 16,
         encode: bool = True) -> None:  # fmt: skip
    sys.path.insert(0, str(REPO / "scripts"))
    from blender_room import FPS, SCENES  # noqa: PLC0415 - pure helpers, no bpy at import

    jobs = []
    for w in walk.split(","):
        n = int(round(SCENES[scene]["walks"][w] * FPS))
        jobs += [(scene, w, f"{a}:{min(a + chunk, n)}", samples) for a in range(0, n, chunk)]
    print(f"{scene}: {len(jobs)} chunks of {chunk} frames, {samples} samples")
    raw = REPO / "data" / "raw"
    t0 = time.time()
    done = sum(1 for n in render_chunk.starmap(jobs, order_outputs=False) if n)
    print(f"{done} chunks rendered ({len(jobs) - done} already in the volume), "
          f"{time.time() - t0:.0f} s")  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for sub in (scene, f"{scene}_gt"):
        subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"raw/{sub}", str(raw)],
                       check=True, env=env)  # fmt: skip
    print(f"downloaded -> {raw / scene}, {raw / (scene + '_gt')}")
    if encode:
        for w in walk.split(","):
            mp4 = raw / (scene + ("" if w == "main" else "_test") + ".mp4")
            noise = "noise=c0s=6:c0f=t+u:c1s=3:c1f=t+u:c2s=3:c2f=t+u"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS),
                 "-i", str(raw / scene / w / "rgb" / "%06d.png"),
                 "-vf", f"format=yuv420p,{noise}", "-c:v", "libx264", "-preset", "slow",
                 "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)],
                check=True,
            )  # fmt: skip
            print(f"wrote {mp4}")
