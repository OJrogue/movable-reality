"""Score capture coverage of a scene and write the gap map into scene.json.

    uv run python scripts/score_coverage.py --scene cov_synth
    uv run python scripts/score_coverage.py --scene room01 --force --voxel 0.08

Reads scene.json, splat/scene.ply (or the per-object PLYs) and recon/ (COLMAP). Uses
coverage/uncertainty.npz from `modal run modal_jobs/coverage.py` when present. Writes
coverage/{gaussians.npz, gaps.npz, report.json} and the `coverage` section of scene.json.
Skips when coverage/report.json exists unless --force. For a synthetic scene (with
synthetic/ground_truth.npz) it also writes coverage/eval.json with precision and recall.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from movable_reality.config import scene_dir
from movable_reality.coverage.evaluate import evaluate_scene
from movable_reality.coverage.pipeline import ScoreParams, score_scene, write_outputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--voxel", type=float, default=0.1, help="voxel size (m)")
    ap.add_argument("--depth-res", type=int, default=256, help="coarse depth map long side (px)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    folder = scene_dir(args.scene)
    report_path = folder / "coverage" / "report.json"
    if report_path.exists() and not args.force:
        print(f"{report_path} exists, skipping (use --force to redo)")
        return 0
    base = ScoreParams()
    params = replace(base, voxel=args.voxel, depth=replace(base.depth, long_side=args.depth_res))
    res = score_scene(folder, params)
    write_outputs(folder, res)
    summary = {k: res.report[k] for k in ("gaussians", "views", "gaps", "gap_share", "timings_s")}
    if (folder / "synthetic" / "ground_truth.npz").exists():
        ev = evaluate_scene(folder)
        (folder / "coverage" / "eval.json").write_text(json.dumps(ev, indent=2))
        summary["eval"] = ev["overall"] | {"regions": ev["regions"]}
    print(json.dumps(summary, indent=2))
    for g in res.coverage.gaps:
        nbv = g.next_best_views[0] if g.next_best_views else None
        where = f"stand at ({nbv.eye[0]:.2f}, {nbv.eye[2]:.2f}) gain {nbv.gain:.2f}" if nbv else "-"
        print(
            f"{g.id}  r={g.radius:.2f} m  score={g.score:.2f}  near={g.near_object}  "
            f"c=({g.center[0]:.2f}, {g.center[1]:.2f}, {g.center[2]:.2f})  {where}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
