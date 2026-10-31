"""Select sharp frames from a phone video and build the matching SAM input clip.

    uv run python scripts/extract_frames.py data/raw/room01.mp4 --scene room01

Writes data/scenes/<scene>/{input.mp4, frames/000000.jpg..., frames.json, sam_input.mp4}.
Frame i of sam_input.mp4 is frames/{i:06d}.jpg, so SAM and reconstruction share indices.
Needs ffmpeg and ffprobe on PATH.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from movable_reality.config import scene_dir


def probe(video: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate:format=duration",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    info = json.loads(out)
    num, den = info["streams"][0]["avg_frame_rate"].split("/")
    return {"fps": float(num) / float(den or 1), "duration": float(info["format"]["duration"])}


def sharpness(path: Path) -> float:
    """Variance of a 4-neighbour Laplacian on a downscaled grayscale copy."""
    img = Image.open(path).convert("L")
    img.thumbnail((480, 480))
    a = np.asarray(img, dtype=np.float32)
    lap = -4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    return float(lap.var())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--fps", type=float, default=4.0, help="target selected frames per second")
    ap.add_argument("--oversample", type=int, default=4, help="candidates per selected frame")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--long-side", type=int, default=1920)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--focal35",
        type=float,
        help="35 mm-equivalent focal length of the video (e.g. 26 for an iPhone main camera); "
        "writes intrinsics.json so MapAnything gets a known camera instead of self-calibrating",
    )
    args = ap.parse_args()

    out = scene_dir(args.scene)
    if (out / "frames.json").exists() and not args.force:
        print(f"{out / 'frames.json'} exists, skipping (use --force to redo)")
        return 0
    # Clear what an earlier or crashed run left: extra frames would end up in the SAM clip.
    if out.exists():
        for sub in ("frames", "frames.json", "sam_input.mp4"):
            p = out / sub
            shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    src = out / "input.mp4"
    if args.video.resolve() != src.resolve():
        shutil.copyfile(args.video, src)
    info = probe(src)
    cand_fps = min(args.fps * args.oversample, info["fps"])
    print(
        f"source {info['fps']:.2f} fps, {info['duration']:.1f} s; candidates at {cand_fps:.2f} fps"
    )

    L = args.long_side
    scale = f"scale='if(gte(iw,ih),min({L}\\,iw),-2)':'if(gte(iw,ih),-2,min({L}\\,ih))'"
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(src),
                "-vf",
                f"fps={cand_fps},{scale}",
                "-q:v",
                "2",
                str(tmpdir / "%06d.jpg"),
            ],
            check=True,
        )
        cands = sorted(tmpdir.glob("*.jpg"))
        scores = [sharpness(p) for p in cands]
        chosen: list[int] = []
        for start in range(0, len(cands), args.oversample):
            window = range(start, min(start + args.oversample, len(cands)))
            chosen.append(max(window, key=lambda i: scores[i]))
        if len(chosen) > args.max_frames:
            keep = np.linspace(0, len(chosen) - 1, args.max_frames).round().astype(int)
            chosen = [chosen[k] for k in keep]

        meta = []
        for new_idx, ci in enumerate(chosen):
            shutil.copyfile(cands[ci], out / "frames" / f"{new_idx:06d}.jpg")
            meta.append(
                {
                    "index": new_idx,
                    "source_time_s": round(ci / cand_fps, 3),
                    "sharpness": round(scores[ci], 2),
                }
            )

    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(out / "frames" / "%06d.jpg"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(out / "sam_input.mp4"),
        ],
        check=True,
    )
    (out / "frames.json").write_text(json.dumps({"fps": args.fps, "frames": meta}, indent=2))
    print(f"selected {len(meta)} of {len(cands)} candidates -> {out}")
    if args.focal35:
        w, h = Image.open(out / "frames" / "000000.jpg").size
        k = intrinsics_from_focal35(args.focal35, w, h)
        (out / "intrinsics.json").write_text(json.dumps(k, indent=2))
        print(f"intrinsics.json: fx = fy = {k['fx']:.1f} px ({args.focal35} mm equivalent)")
    return 0


def intrinsics_from_focal35(f35: float, width: int, height: int) -> dict[str, float]:
    """Pinhole intrinsics from a phone's 35 mm-equivalent focal length. Phones quote it for
    the 4:3 photo, matched on the diagonal (43.27 mm), so the 4:3 sensor is 34.6 mm wide, and
    16:9 video keeps the full width. Stabilisation crops a few percent more."""
    f = f35 / (43.27 * 0.8) * max(width, height)
    return {"fx": f, "fy": f, "cx": width / 2, "cy": height / 2}


if __name__ == "__main__":
    sys.exit(main())
