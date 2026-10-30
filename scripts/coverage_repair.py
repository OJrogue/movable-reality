"""Prepare (and optionally run) Muse Image repairs of small coverage gaps (C3).

    uv run python scripts/coverage_repair.py --scene cov_synth                 # dry run, CPU
    uv run python scripts/coverage_repair.py --scene room01 --gpu              # gsplat views
    uv run python scripts/coverage_repair.py --scene room01 --gap gap_003 --gpu --call   # paid
    uv run modal run modal_jobs/coverage.py --scene room01 --job distil        # back into splats

For each selected gap (default: open gaps up to --max-radius that have a next-best view,
at most --limit) and each of its first --views next-best views: renders the gap view with
the gap painted magenta, picks the nearest real frames as references, and writes
coverage/repair/<gap>[_v<k>]/{view.png, mask.png, request.json}. With --gpu the view is an
alpha-blended gsplat render (modal_jobs/coverage.py --job render, about a minute of GPU),
else the CPU point preview. With --call it sends one Muse Image edit per view (Meta Model
API, MODEL_API_KEY, $0.01 per image), cached under data/cache/muse_image/, and writes
repaired.webp next to the view. modal_jobs/coverage.py --job distil then fits new splats
to the repaired views and writes coverage/generated.ply (every splat generated = 1).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np

from movable_reality.config import REPO_ROOT, scene_dir
from movable_reality.coverage.cameras import similarity_parts, transform_points
from movable_reality.coverage.colmap import read_model
from movable_reality.coverage.pipeline import load_splats, recon_dir
from movable_reality.coverage.repair import (
    load_render,
    plan_camera,
    prepare_request,
    repair_view,
    view_dir,
    write_check,
)
from movable_reality.scene import Scene


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--gap", action="append", help="gap id (repeatable); default: small gaps")
    ap.add_argument("--max-radius", type=float, default=0.8, help="small gap: radius <= this (m)")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--views", type=int, default=2, help="repair views per gap (next-best views)")
    ap.add_argument(
        "--refs", type=int, default=1, help="reference frames per view (more move the camera)"
    )
    ap.add_argument("--min-mask", type=float, default=0.01, help="skip views with less gap")
    ap.add_argument("--gpu", action="store_true", help="render gap views with gsplat on Modal")
    ap.add_argument("--call", action="store_true", help="call Muse Image (paid, cached)")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    args = ap.parse_args()

    folder = scene_dir(args.scene)
    scene = Scene.load(folder / "scene.json")
    if not scene.coverage or not scene.coverage.gaps:
        print("no gaps in scene.json; run scripts/score_coverage.py first")
        return 1
    if args.gap:
        gaps = [g for g in scene.coverage.gaps if g.id in set(args.gap)]
    else:
        gaps = [
            g
            for g in scene.coverage.gaps
            if g.status == "open" and g.radius <= args.max_radius and g.next_best_views
        ][: args.limit]
    splats, _ = load_splats(folder, scene)
    W = np.asarray(scene.world_from_recon, dtype=np.float64)
    scale, _ = similarity_parts(W)
    world = (
        transform_points(W, splats.means),
        splats.rgb,
        splats.scales.max(axis=1) * np.float32(scale),
        splats.opacity,
        splats.generated,
    )
    cams = [c.in_frame(W) for c in read_model(recon_dir(folder))]
    jobs = []  # (gap, view index, camera)
    for g in gaps:
        n = max(1, min(args.views, len(g.next_best_views) or 1))
        jobs += [(g, k, plan_camera(g, world, k)) for k in range(n)]

    recon_from_world = np.linalg.inv(W)
    wanted = []
    for g, k, cam in jobs:
        c = cam.in_frame(recon_from_world)
        vm = np.eye(4)
        vm[:3, :3], vm[:3, 3] = c.R, c.t
        name = view_dir(folder, g.id, k).name
        wanted.append({"name": name, "K": c.K.tolist(), "viewmat": vm.tolist(),
                       "width": c.width, "height": c.height})  # fmt: skip
    # Reuse gsplat renders of the same cameras from an earlier --gpu run.
    req_path = folder / "coverage" / "repair" / "render_request.json"
    done = json.loads(req_path.read_text())["cameras"] if req_path.exists() else []
    have = {d["name"] for d in done if d in wanted}
    if args.gpu and any(w["name"] not in have for w in wanted):
        req_path.parent.mkdir(parents=True, exist_ok=True)
        req_path.write_text(json.dumps({"cameras": wanted}, indent=2))
        cmd = [sys.executable, "-m", "modal", "run", "modal_jobs/coverage.py",
               "--scene", args.scene, "--job", "render"]  # fmt: skip
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        have = {w["name"] for w in wanted}
    renders: dict[str, tuple] = {}
    for g, k, _ in jobs:
        npz = view_dir(folder, g.id, k) / "render.npz"
        if npz.parent.name in have and npz.exists():
            renders[f"{g.id}/{k}"] = load_render(npz, scale)

    for g, k, cam in jobs:
        req = prepare_request(
            folder, g, cams, world, k_refs=args.refs, view_index=k, cam=cam,
            render=renders.get(f"{g.id}/{k}"),
        )  # fmt: skip
        line = f"{g.id} view {k}: {req.view.relative_to(folder)}, {len(req.references)} refs"
        share = json.loads((req.view.parent / "request.json").read_text())["mask_pixels"] / (
            cam.width * cam.height
        )
        if share < args.min_mask:  # the gap is hidden from this pose: nothing to repair
            print(f"{line}, gap covers {100 * share:.1f}% of the view, skipped")
            (req.view.parent / "repaired.webp").unlink(missing_ok=True)
            continue
        if args.call:
            res = repair_view(req, req.view.parent / "repaired.webp", force=args.force)
            src = "cache" if res.cached else "API"
            check = write_check(req.view.parent)
            verdict = "accepted" if check["accepted"] else "rejected (viewpoint moved)"
            line += f", repaired -> {res.image.relative_to(folder)} ({src}), "
            line += f"alignment L1 {check['alignment_l1']:.3f} {verdict}"
        print(line)
    if not args.call:
        print("dry run: no Muse Image call (add --call)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
