"""C2 recapture (full rerun): add a top-up video to a scene, reconstruct again, rescore.

    uv run python scripts/coverage_topup.py --scene room01 --video data/raw/room01_topup.mp4
    uv run python scripts/coverage_topup.py --scene room01 --video t.mp4 --no-recon   # frames only

1. Selects sharp frames from the top-up video (same rule as extract_frames.py) and appends
   them to data/scenes/<scene>/frames with the next free indices; frames.json records
   their source. Rerunning with the same video adds nothing.
2. Rebuilds sam_input.mp4 from all frames, so SAM frame i is still frames/{i:06d}.jpg.
3. Keeps the previous cameras (coverage/topup/prev_sparse) and reruns
   `modal run modal_jobs/reconstruct.py --force` on all frames (MapAnything + gsplat, GPU).
4. Fits the new reconstruction frame to the old one on the shared cameras (Umeyama) and
   updates world_from_recon, so floor, object boxes and old gaps keep their place.
5. Rescores coverage (as score_coverage.py --force).

Objects are not recomputed: the per-object PLYs still come from the old splats until
`scripts/lift_ids.py --scene <scene> --force` runs (paid SAM calls for the new frames).
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

from movable_reality.config import REPO_ROOT, scene_dir
from movable_reality.coverage.colmap import find_sparse_dir, read_model
from movable_reality.coverage.pipeline import recon_dir, score_scene, write_outputs
from movable_reality.coverage.topup import (
    align_world,
    merge_frames,
    sam_clip_cmd,
    select_sharp,
    sharpness,
    video_source,
)
from movable_reality.scene import Scene


def extract(video: Path, fps: float, oversample: int, max_frames: int, long_side: int, tmp: Path):
    L = long_side
    scale = f"scale='if(gte(iw,ih),min({L}\\,iw),-2)':'if(gte(iw,ih),-2,min({L}\\,ih))'"
    cand_fps = fps * oversample
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={cand_fps},{scale}",
         "-q:v", "2", str(tmp / "%06d.jpg")],
        check=True,
    )  # fmt: skip
    cands = sorted(tmp.glob("*.jpg"))
    scores = [sharpness(p) for p in cands]
    chosen = select_sharp(scores, oversample, max_frames)
    meta = [
        {"source_time_s": round(ci / cand_fps, 3), "sharpness": round(scores[ci], 2)}
        for ci in chosen
    ]
    return [cands[ci] for ci in chosen], meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--long-side", type=int, default=1920)
    ap.add_argument("--max-steps", type=int, default=15_000, help="gsplat steps of the rerun")
    ap.add_argument("--no-recon", action="store_true", help="merge frames only, no GPU run")
    args = ap.parse_args()

    folder = scene_dir(args.scene)
    clip_fps = json.loads((folder / "frames.json").read_text()).get("fps", args.fps)
    source = video_source(args.video)
    with tempfile.TemporaryDirectory() as tmp:
        images, meta = extract(
            args.video, args.fps, args.oversample, args.max_frames, args.long_side, Path(tmp)
        )
        new = merge_frames(folder, images, meta, source)
    print(f"{source}: frames {new[0]}..{new[-1]} ({len(new)}) in {folder / 'frames'}")
    subprocess.run(sam_clip_cmd(folder, clip_fps), check=True)
    print(f"rebuilt {folder / 'sam_input.mp4'}")
    if args.no_recon:
        print("--no-recon: stopping before the reconstruction")
        return 0

    prev = folder / "coverage" / "topup" / "prev_sparse"
    if not prev.exists():  # the cameras of the model that scene.json's frame refers to
        shutil.copytree(find_sparse_dir(recon_dir(folder)), prev)
    cmd = [sys.executable, "-m", "modal", "run", "modal_jobs/reconstruct.py",
           "--scene", args.scene, "--force", "--max-steps", str(args.max_steps)]  # fmt: skip
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    scene = Scene.load(folder / "scene.json")
    W, fit = align_world(
        read_model(prev), read_model(recon_dir(folder)), np.asarray(scene.world_from_recon)
    )
    print(f"aligned to the previous frame: {fit}")
    scene.world_from_recon = W.tolist()
    scene.save(folder / "scene.json")
    shutil.rmtree(prev)
    (folder / "coverage" / "topup").mkdir(parents=True, exist_ok=True)
    (folder / "coverage" / "topup" / "align.json").write_text(
        json.dumps({"source": source, "new_frames": len(new), **fit}, indent=2)
    )

    # The residual uncertainty belongs to the old splats; refit it with modal_jobs/coverage.py.
    for name in ("uncertainty.npz", "uncertainty.json"):
        (folder / "coverage" / name).unlink(missing_ok=True)
    res = score_scene(folder)
    write_outputs(folder, res)
    print(json.dumps({k: res.report[k] for k in ("views", "gaps", "gap_share")}, indent=2))
    print(f"next: uv run python scripts/lift_ids.py --scene {args.scene} --force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
