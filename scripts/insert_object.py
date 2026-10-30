"""Insert a generated object by hand (the agent's insert_object tool runs the same code).

    uv run python scripts/insert_object.py --scene m4test "small wooden stool" --x 0.3 --z -0.6

Muse Spark (size) -> Muse Image -> SAM 3.1 -> SAM 3D Objects on Modal -> objects/gen_<n>.ply
and a "gen#<n>" entry in scene.json. Needs `uv run modal deploy modal_jobs/sam3d.py` once.
Every paid step is cached, so the same prompt again costs nothing.
"""

from __future__ import annotations

import argparse
import json

from movable_reality.config import scene_dir
from movable_reality.insert import insert_object
from movable_reality.scene import Scene


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--x", type=float, default=0.0, help="world x of the footprint centre, m")
    ap.add_argument("--z", type=float, default=0.0, help="world z of the footprint centre, m")
    args = ap.parse_args()
    folder = scene_dir(args.scene)
    scene = Scene.load(folder / "scene.json")
    out = insert_object(scene, folder, args.prompt, args.x, args.z, instruction=args.prompt)
    scene.save(folder / "scene.json")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
