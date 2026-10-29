"""Bake the floor and ceiling of a scene into opaque textured quads (stage 12, lane bake).

    uv run python scripts/bake_planes.py --scene bakedemo --synthetic   # test room, then bake
    uv run python scripts/bake_planes.py --scene room01                 # skip if baked
    uv run python scripts/bake_planes.py --scene room01 --force         # bake again
    uv run python scripts/bake_planes.py --scene room01 --labels floor  # fallback: floor only
    uv run python scripts/bake_planes.py --scene room01 --revert        # back to splats only

Reads splat/scene.ply, ids/gaussian_ids.npy and ids/labels.json ({"<id>": "floor#1", ...}).
Writes planes/<label>.glb, a reduced background PLY next to the original (which is kept),
and the `planes` and `plane_bake` sections of scene.json; background_ply then points at the
reduced PLY. GPU path: modal_jobs/bake.py (same outputs, gsplat orthographic rasterizer).
"""

from __future__ import annotations

import argparse
import shutil
import sys

from movable_reality.bake.pipeline import BakeOptions, bake_scene, revert_scene
from movable_reality.bake.synthetic import make_room
from movable_reality.config import scene_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--force", action="store_true", help="bake again even if outputs exist")
    ap.add_argument("--labels", default="floor,ceiling", help="comma-separated SAM nouns")
    ap.add_argument("--texel", type=float, default=0.005, help="texel size in metres")
    ap.add_argument("--max-px", type=int, default=4096, help="texture size cap per side")
    ap.add_argument("--threshold", type=float, default=0.03, help="plane inlier distance, m")
    ap.add_argument("--max-tilt", type=float, default=None, help="max plane tilt from up, deg")
    ap.add_argument("--format", choices=["jpeg", "png"], default="jpeg")
    ap.add_argument("--synthetic", action="store_true", help="write the synthetic test room first")
    ap.add_argument("--revert", action="store_true", help="drop planes, restore background")
    args = ap.parse_args()

    out = scene_dir(args.scene)
    if args.revert:
        ok = revert_scene(out)
        print("reverted" if ok else "nothing to revert")
        return 0
    if args.synthetic:
        if (out / "scene.json").exists() and not args.force:
            print(f"{out} exists; keeping it (pass --force to rewrite the synthetic room)")
        else:
            shutil.rmtree(out, ignore_errors=True)
            gt = make_room(out, scene_id=args.scene)
            print(f"wrote synthetic room to {out}: {gt.counts}")
    if not (out / "scene.json").is_file():
        print(f"{out / 'scene.json'} not found", file=sys.stderr)
        return 1

    opts = BakeOptions(
        labels=tuple(s.strip() for s in args.labels.split(",") if s.strip()),
        texel=args.texel,
        max_px=args.max_px,
        threshold=args.threshold,
        max_tilt_deg=args.max_tilt,
        image_format=args.format,
    )
    try:
        report = bake_scene(out, opts, force=args.force)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{args.scene}: {exc}", file=sys.stderr)
        return 1
    if report.status == "skipped":
        print(f"{args.scene}: planes already baked (pass --force to bake again)")
        return 0
    for p in report.planes:
        if p.status != "baked":
            print(f"{p.label}: {p.status}")
    if report.status != "baked":
        print(f"{args.scene}: {report.status}")
        return 1
    print(
        f"{args.scene}: baked {sum(p.status == 'baked' for p in report.planes)} planes,"
        f" removed {report.removed_splats} splats -> {report.background_ply}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
