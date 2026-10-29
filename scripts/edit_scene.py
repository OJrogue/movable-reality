"""Edit a scene manifest by hand (the same operations the agent will call as tools).

    uv run python scripts/edit_scene.py demo list
    uv run python scripts/edit_scene.py demo move "chair#1" 0.5 0 0
    uv run python scripts/edit_scene.py demo rotate "table#1" 90
    uv run python scripts/edit_scene.py demo remove "floor lamp#1"
    uv run python scripts/edit_scene.py demo restore "floor lamp#1"

The viewer polls scene.json, so edits show up within a second.
"""

from __future__ import annotations

import argparse

from movable_reality.config import scene_dir
from movable_reality.scene import Scene


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("op", choices=["list", "move", "rotate", "remove", "restore"])
    ap.add_argument("object_id", nargs="?")
    ap.add_argument("values", nargs="*", type=float)
    args = ap.parse_args()

    path = scene_dir(args.scene) / "scene.json"
    scene = Scene.load(path)
    if args.op == "list":
        for o in scene.objects:
            size = " x ".join(f"{v:.2f}" for v in o.size)
            print(f"{o.id:<20} {o.label:<14} {size} m  visible={o.visible}  {o.transform}")
        return 0
    if not args.object_id:
        ap.error("object_id is required for this operation")
    need = {"move": 3, "rotate": 1}.get(args.op, 0)
    if len(args.values) < need:
        ap.error(f"{args.op} needs {need} value(s), got {len(args.values)}")
    if args.op == "move":
        scene.move(args.object_id, *args.values[:3])
    elif args.op == "rotate":
        scene.rotate(args.object_id, args.values[0])
    else:
        scene.set_visible(args.object_id, args.op == "restore")
    scene.save(path)
    print(f"{args.op} {args.object_id}: saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
