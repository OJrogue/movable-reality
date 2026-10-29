"""Write a synthetic room capture with known hidden regions, in the layout of a real scene.

    uv run python scripts/coverage_synthetic.py                       # -> data/scenes/cov_synth/
    uv run python scripts/coverage_synthetic.py --scene cov_big --spacing 0.004 --views 200 \
        --no-frames

Writes splat/scene.ply (recon frame), recon/sparse/0 (COLMAP), frames/ (CPU renders),
objects/, scene.json (world_from_recon) and synthetic/ground_truth.npz (exact view count and
VR visibility of every true surface sample). Then run scripts/score_coverage.py on it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

from movable_reality.config import scene_dir
from movable_reality.coverage.synthetic import SynthConfig, make_capture, write_scene


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", default="cov_synth")
    ap.add_argument("--spacing", type=float, default=0.02, help="surface sample spacing (m)")
    ap.add_argument("--views", type=int, default=150)
    ap.add_argument("--no-frames", action="store_true", help="skip CPU frame renders")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = scene_dir(args.scene)
    if (out / "scene.json").exists() and not args.force:
        print(f"{out / 'scene.json'} exists, skipping (use --force to redo)")
        return 0
    if out.exists() and args.force:
        shutil.rmtree(out)
    cfg = SynthConfig(
        spacing=args.spacing, n_views=args.views, write_frames=not args.no_frames, seed=args.seed
    )
    t0 = time.perf_counter()
    cap = make_capture(cfg)
    write_scene(cap, out, cfg, args.scene)
    print(
        json.dumps(
            {
                "scene": str(out),
                "gaussians": int(len(cap.means)),
                "surface_samples": int(len(cap.surface.points)),
                "views": len(cap.cameras),
                "samples_never_seen": int((cap.views == 0).sum()),
                "samples_vr_visible": int(cap.vr_visible.sum()),
                "timings_s": cap.timings | {"total_s": round(time.perf_counter() - t0, 1)},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
