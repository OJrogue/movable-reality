"""Build one Spark LOD tree (.RAD) per object PLY and one for the background.

    uv run python scripts/build_lod.py demo                  # -> data/scenes/demo/lod/*.rad
    uv run python scripts/build_lod.py demo --force          # rebuild every tree
    uv run python scripts/build_lod.py demo --method quick   # tiny-lod, about 3x faster
    uv run python scripts/build_lod.py --setup               # only fetch and build build-lod

Writes data/scenes/<scene>/lod/<ply stem>.rad and sets SceneObject.lod and
Scene.background_lod in scene.json. Skips trees that exist and are newer than their PLY
unless --force. Uses Spark's build-lod (MIT, Rust): $SPARK_BUILD_LOD, a cached build, or
build-lod on PATH; if none is found it clones Spark at the version pinned in web/ and
builds it with cargo (needs Rust; about 2 min once). See docs/LOD.md.
"""

from __future__ import annotations

import argparse
import sys

from movable_reality.config import scene_dir
from movable_reality.lod.build import build_scene_lod, ensure_build_lod, find_build_lod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("scene", nargs="?")
    ap.add_argument("--force", action="store_true", help="rebuild trees that already exist")
    ap.add_argument("--method", choices=["quality", "quick"], default="quality")
    ap.add_argument("--max-sh", type=int, choices=[0, 1, 2, 3], help="clamp SH degree")
    ap.add_argument("--setup", action="store_true", help="fetch and build build-lod, then exit")
    ap.add_argument("--no-setup", action="store_true", help="fail instead of building build-lod")
    args = ap.parse_args()

    if args.no_setup:
        binary = find_build_lod()
        if binary is None:
            print("build-lod not found (run with --setup, or set SPARK_BUILD_LOD)", file=sys.stderr)
            return 1
    else:
        binary = ensure_build_lod()
    print(f"build-lod: {binary}")
    if args.setup and not args.scene:
        return 0
    if not args.scene:
        ap.error("scene is required")

    path = scene_dir(args.scene) / "scene.json"
    if not path.is_file():
        ap.error(f"{path} not found")
    results = build_scene_lod(
        path, binary, method=args.method, max_sh=args.max_sh, force=args.force
    )
    built = sum(r.status == "built" for r in results)
    with_lod = sum(r.status in ("built", "skipped") for r in results)
    print(f"{args.scene}: {with_lod}/{len(results)} meshes have LOD trees ({built} built now)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
