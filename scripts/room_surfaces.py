"""Room surfaces replace the scan's wall, floor and ceiling splats (after room_shell.py and
the plane bake).

    uv run python scripts/room_surfaces.py --scene blender01   # --gap 0.06, --revert

The background splats of walls, floor and ceiling are the scan's weakest part: blur, haze,
streaks along the wall tops and smears where furniture stood (blender01, 2 Nov). The room
shell (room_shell.py) has the same surfaces as flat quads textured from the frames with a
depth test, and its hidden texels filled from the surface itself. This stage drops every
background splat that reaches within --gap metres of a shell surface (centre distance less
--reach sigmas of its largest axis) or lies outside the room (behind a wall, under the
floor, above the ceiling), and the haze: splats larger than --haze-scale metres with an
opacity under --haze-opacity (blender01 kept 20,735 splats at a 6 cm centre gap, median
10 cm across at opacity 0.19, a fog over the corners), then the floaters: splats with fewer
than --min-neighbours splats in the 30 cm cube around them (neighbour_counts; coloured blobs
in front of the unfilmed upper walls), and the needles: longest axis over --needle-length
metres and over --needle-ratio times the middle one (dark strokes across the walls), and
the leftovers inside an object's box grown by --object-margin metres (parts of the object
the lift missed: coloured specks inside blender01's side table, a ghost once it moves).
Writes objects/background_room.ply, points
scene.json background_ply at it (background_lod is cleared for build_lod.py), and marks
room/shell.json `replaces_splats`, so the viewer draws the shell in its true planes in every
style and hides the baked floor quad. --revert puts the previous background back.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from movable_reality.bake import gaussians
from movable_reality.config import scene_dir
from movable_reality.roommesh import source_background
from movable_reality.roomshell import neighbour_counts, surface_splats
from movable_reality.scene import Scene

OUT = "objects/background_room.ply"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--gap", type=float, default=0.06, help="metres from a surface that go")
    ap.add_argument("--reach", type=float, default=2.0, help="splat extent in sigmas")
    ap.add_argument("--haze-scale", type=float, default=0.10, help="metres, largest axis")
    ap.add_argument("--haze-opacity", type=float, default=0.3)
    ap.add_argument("--min-neighbours", type=int, default=10, help="0 keeps floaters")
    ap.add_argument("--needle-length", type=float, default=0.05, help="metres")
    ap.add_argument("--needle-ratio", type=float, default=8.0, help="long / middle axis")
    ap.add_argument("--object-margin", type=float, default=0.12, help="metres, <0 keeps them")
    ap.add_argument("--revert", action="store_true", help="restore the previous background")
    args = ap.parse_args()
    root = scene_dir(args.scene)
    shell_path = root / "room" / "shell.json"
    if not shell_path.exists():
        print("no room/shell.json: run room_shell.py first")
        return 1
    shell = json.loads(shell_path.read_text())
    scene = Scene.load(root / "scene.json")
    if args.revert:
        prev = shell.pop("surfaces_previous_background", None)
        shell["replaces_splats"] = False
        if prev:
            scene.background_ply, scene.background_lod = prev["ply"], prev["lod"]
            scene.save(root / "scene.json")
        shell_path.write_text(json.dumps(shell, indent=2))
        print(f"background back to {scene.background_ply}")
        return 0

    src = source_background(scene.model_dump())
    W = np.asarray(scene.world_from_recon, dtype=np.float64)
    ply = gaussians.read_ply(root / src)
    g = gaussians.gaussians_from_vertices(gaussians.vertices(ply))
    m = np.asarray(g.means, dtype=np.float64) @ W[:3, :3].T + W[:3, 3]
    poly = np.asarray(shell["outline_xz"], dtype=np.float64)
    axes = np.sort(np.exp(np.asarray(g.log_scales, dtype=np.float64)), axis=1)
    axes *= np.cbrt(abs(np.linalg.det(W[:3, :3])))
    size = axes[:, 2]
    needles = (size > args.needle_length) & (size > args.needle_ratio * axes[:, 1])
    op = np.asarray(g.opacities)
    haze = (size > args.haze_scale) & (op < args.haze_opacity)
    keep = ~surface_splats(m, size, op, poly, float(shell["height_m"]), args.gap, args.reach,
                           args.haze_scale, args.haze_opacity)  # fmt: skip
    keep &= ~needles
    inside = np.zeros(len(keep), dtype=bool)
    if args.object_margin >= 0:
        for o in scene.objects:  # bbox in the world frame of the unedited object
            lo = np.asarray(o.bbox_min) - args.object_margin
            hi = np.asarray(o.bbox_max) + args.object_margin
            inside |= np.all((m >= lo) & (m <= hi), axis=1)
        keep &= ~inside
    floaters = np.zeros(len(keep), dtype=bool)
    if args.min_neighbours:  # counted among the kept splats, so a fog's last specks go too
        floaters[keep] = neighbour_counts(m[keep]) < args.min_neighbours
        keep &= ~floaters
    n = int(keep.sum())
    if scene.background_ply != OUT:
        shell["surfaces_previous_background"] = {"ply": scene.background_ply,
                                                 "lod": scene.background_lod}  # fmt: skip
    shell.update(replaces_splats=True, surfaces_source_ply=src, surfaces_gap_m=args.gap,
                 surfaces_reach_sigma=args.reach, surfaces_haze_removed=int(haze.sum()),
                 surfaces_floaters_removed=int(floaters.sum()),
                 surfaces_needles_removed=int(needles.sum()),
                 surfaces_in_objects_removed=int(inside.sum()),
                 surfaces_removed=int((~keep).sum()), surfaces_kept=int(n))  # fmt: skip
    shell_path.write_text(json.dumps(shell, indent=2))
    gaussians.write_subset(root / OUT, ply, keep)  # after shell.json: run_pipeline.py mtimes
    scene.background_ply = OUT
    scene.background_lod = None  # its .RAD tree holds the removed splats; build_lod.py rebuilds
    scene.save(root / "scene.json")
    print(f"{src}: {len(keep)} splats, {(~keep).sum()} on or behind the room surfaces or haze "
          f"({haze.sum()}), needles ({needles.sum()}), in objects ({inside.sum()}) or floaters "
          f"({floaters.sum()}) removed, "
          f"{n} kept -> {OUT}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
