"""Prepare (and optionally run) Muse Image repairs of small coverage gaps.

    uv run python scripts/coverage_repair.py --scene cov_synth                 # dry run
    uv run python scripts/coverage_repair.py --scene room01 --gap gap_003 --call   # paid

For each selected gap (default: open gaps up to --max-radius, at most --limit): renders
the gap view from its next-best viewpoint with the gap painted magenta, picks the nearest
real frames as references, and writes coverage/repair/<gap>/{view.png, mask.png,
request.json}. With --call it sends one Muse Image edit per gap (Meta Model API,
MODEL_API_KEY, about $0.01 per image), cached under data/cache/muse_image/, and writes
coverage/repair/<gap>/repaired.webp. Distilling repairs back into splats is not done here.
"""

from __future__ import annotations

import argparse

import numpy as np

from movable_reality.config import scene_dir
from movable_reality.coverage.cameras import similarity_parts, transform_points
from movable_reality.coverage.colmap import read_model
from movable_reality.coverage.pipeline import load_splats
from movable_reality.coverage.repair import prepare_request, repair_view
from movable_reality.scene import Scene


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--gap", action="append", help="gap id (repeatable); default: small gaps")
    ap.add_argument("--max-radius", type=float, default=0.8, help="small gap: radius <= this (m)")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--refs", type=int, default=3, help="reference frames per gap")
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
            g for g in scene.coverage.gaps if g.status == "open" and g.radius <= args.max_radius
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
    cams = [c.in_frame(W) for c in read_model(folder / "recon")]
    for g in gaps:
        req = prepare_request(folder, g, cams, world, k_refs=args.refs)
        line = f"{g.id}: view {req.view.relative_to(folder)}, {len(req.references)} references"
        if args.call:
            res = repair_view(req, req.view.parent / "repaired.webp", force=args.force)
            src = "cache" if res.cached else "API"
            line += f", repaired -> {res.image.relative_to(folder)} ({src})"
        print(line)
    if not args.call:
        print("dry run: no Muse Image call (add --call)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
