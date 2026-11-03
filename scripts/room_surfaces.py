"""Room surfaces replace the scan's wall, floor and ceiling splats (after room_shell.py and
the plane bake).

    uv run python scripts/room_surfaces.py --scene blender01   # --gap 0.06, --revert

The background splats of walls, floor and ceiling are the scan's weakest part: blur, haze,
streaks along the wall tops and smears where furniture stood (blender01, 2 Nov). The room
shell (room_shell.py) has the same surfaces as flat quads textured from the frames with a
depth test, and its hidden texels filled from the surface itself. This stage drops every
background splat that reaches within --gap metres of a shell surface (centre distance less
--reach sigmas of its largest axis) or lies outside the room (behind a wall, under the
floor, above the ceiling), and the haze: splats whose largest axis (sigma) is over
--haze-scale metres with an opacity under --haze-opacity (blender01 kept 20,735 splats at a
6 cm centre gap, median sigma 10 cm at opacity 0.19, a fog over the corners), then the
floaters: splats with fewer than --min-neighbours splats in the 30 cm cube around them
(neighbour_counts; coloured blobs in front of the unfilmed upper walls), and the needles:
longest axis over --needle-length metres and over --needle-ratio times the middle one (dark
strokes across the walls), and the leftovers inside a scanned object's box grown by
--object-margin metres (parts of the object the lift missed: coloured specks inside
blender01's side table, a ghost once it moves), and the surface ghosts: splats within
--ghost-reach metres of a surface whose colour is within --ghost-colour (RGB, 0 to 1) of
the shell texture under them (blender01: 1,509 of the last 1,959, beige fuzz 10 to 60 cm off
the walls and floor in the colour of the wall behind; surface_colour_gap). Writes
objects/background_room.ply, points scene.json background_ply at it (background_lod is
cleared for build_lod.py), and marks room/shell.json `replaces_splats`, so the viewer draws
the shell in its true planes in every style and hides the baked floor quad. --revert puts
the previous background back. When more than --max-kept of the background splats survive,
the room is not a box (sloped ceilings, an attic) and the stage leaves the scan's splats.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from movable_reality.bake import gaussians, glb
from movable_reality.config import scene_dir
from movable_reality.roomshell import (
    SURFACES_PLY,
    Surface,
    background_source,
    neighbour_counts,
    surface_colour_gap,
    surface_splats,
    surfaces,
)
from movable_reality.scene import Scene
from movable_reality.splat_ops import read_vertices, tame_splats, write_vertices

OUT = SURFACES_PLY
SH_C0 = 0.28209479177387814


def shell_textures(root: Path, shell: dict) -> tuple[list[Surface], list[np.ndarray]]:
    """The shell surfaces room_shell.py wrote (room/shell_<kind>_<k>.glb, in the order of
    roomshell.surfaces) with their textures."""
    surfs, texs, counts = [], [], {}
    for s in surfaces(np.asarray(shell["outline_xz"], dtype=np.float64), float(shell["height_m"])):
        k = counts.get(s.kind, 0)
        counts[s.kind] = k + 1
        path = root / "room" / f"shell_{s.kind}_{k}.glb"
        if path.exists():
            doc, blob = glb.read_glb(path.read_bytes())
            texs.append(np.asarray(Image.open(io.BytesIO(glb.image_bytes(doc, blob)))))
            surfs.append(s)
    return surfs, texs


def revert(root: Path, scene: Scene, shell: dict) -> None:
    """Back to the background before this stage (scene.json saved, shell.json updated in
    `shell` only)."""
    prev = shell.pop("surfaces_previous_background", None)
    shell["replaces_splats"] = False
    if prev:
        scene.background_ply, scene.background_lod = prev["ply"], prev["lod"]
        scene.save(root / "scene.json")
    (root / "room" / "shell.json").write_text(json.dumps(shell, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--gap", type=float, default=0.06, help="metres from a surface that go")
    ap.add_argument("--reach", type=float, default=2.0, help="splat extent in sigmas")
    ap.add_argument("--haze-scale", type=float, default=0.10, help="m, largest sigma")
    ap.add_argument("--haze-opacity", type=float, default=0.3)
    ap.add_argument("--min-neighbours", type=int, default=10, help="0 keeps floaters")
    ap.add_argument("--needle-length", type=float, default=0.05, help="metres")
    ap.add_argument("--needle-ratio", type=float, default=8.0, help="long / middle axis")
    ap.add_argument("--object-margin", type=float, default=0.12, help="metres, <0 keeps them")
    ap.add_argument("--ghost-colour", type=float, default=0.10, help="RGB gap, 0 keeps them")
    ap.add_argument("--ghost-reach", type=float, default=1.0, help="metres from a surface")
    ap.add_argument(
        "--max-kept",
        type=float,
        default=0.18,
        help="share of background splats kept above which the room is not a box",
    )
    ap.add_argument("--force", action="store_true", help="apply even when not a box room")
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
        revert(root, scene, shell)
        print(f"background back to {scene.background_ply}")
        return 0

    src = background_source(scene.model_dump(), shell)
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
            if o.source != "scan":
                continue  # generated and inserted objects have no scan leftovers
            lo = np.asarray(o.bbox_min) - args.object_margin
            hi = np.asarray(o.bbox_max) + args.object_margin
            inside |= np.all((m >= lo) & (m <= hi), axis=1)
        keep &= ~inside
    ghosts = np.zeros(len(keep), dtype=bool)
    if args.ghost_colour > 0:
        v = gaussians.vertices(ply)
        dc = np.stack([np.asarray(v[f"f_dc_{i}"], dtype=np.float64) for i in range(3)], 1)
        colour = np.clip(0.5 + SH_C0 * dc[keep], 0.0, 1.0)
        dist, gap = surface_colour_gap(m[keep], colour, *shell_textures(root, shell))
        ghosts[keep] = (dist < args.ghost_reach) & (gap < args.ghost_colour)
        keep &= ~ghosts
    floaters = np.zeros(len(keep), dtype=bool)
    if args.min_neighbours:  # counted among the kept splats, so a fog's last specks go too
        floaters[keep] = neighbour_counts(m[keep]) < args.min_neighbours
        keep &= ~floaters
    n = int(keep.sum())
    if n > args.max_kept * len(keep) and not args.force:
        # The room has structure the shell (vertical walls, flat ceiling) does not model:
        # blender04's attic kept 25 % (its roof slopes), the box rooms 0.1 to 2 % and the
        # cluttered blender06 bedroom 11.5 % (lamp, shelf, curtains, radiator). A share is a
        # heuristic: clutter and slopes are told apart only by how much of them there is.
        # Its splats stay, and with them the scan's own walls.
        revert(root, scene, shell)
        shell["surfaces_skipped"] = (f"{n} of {len(keep)} background splats off the shell "
                                     f"(over --max-kept {args.max_kept})")  # fmt: skip
        shell_path.write_text(json.dumps(shell, indent=2))
        print(f"not a box room: {shell['surfaces_skipped']}; background left at "
              f"{scene.background_ply} (--force applies it)")  # fmt: skip
        return 0
    shell.pop("surfaces_skipped", None)
    if scene.background_ply != OUT:
        shell["surfaces_previous_background"] = {"ply": scene.background_ply,
                                                 "lod": scene.background_lod}  # fmt: skip
    shell.update(replaces_splats=True, surfaces_source_ply=src, surfaces_gap_m=args.gap,
                 surfaces_reach_sigma=args.reach, surfaces_haze_removed=int(haze.sum()),
                 surfaces_floaters_removed=int(floaters.sum()),
                 surfaces_needles_removed=int(needles.sum()),
                 surfaces_in_objects_removed=int(inside.sum()),
                 surfaces_ghosts_removed=int(ghosts.sum()),
                 surfaces_removed=int((~keep).sum()), surfaces_kept=int(n))  # fmt: skip
    shell_path.write_text(json.dumps(shell, indent=2))
    gaussians.write_subset(root / OUT, ply, keep)  # after shell.json: run_pipeline.py mtimes
    # What stays, a pendant lamp or a door frame, gets the objects' limits on smears and hairs.
    world_per_unit = float(np.linalg.norm(np.asarray(scene.world_from_recon)[:3, 0]))
    write_vertices(root / OUT, tame_splats(read_vertices(root / OUT), world_per_unit))
    scene.background_ply = OUT
    scene.background_lod = None  # its .RAD tree holds the removed splats; build_lod.py rebuilds
    scene.save(root / "scene.json")
    print(f"{src}: {len(keep)} splats, {(~keep).sum()} on or behind the room surfaces or haze "
          f"({haze.sum()}), needles ({needles.sum()}), in objects ({inside.sum()}), ghosts "
          f"({ghosts.sum()}) or floaters ({floaters.sum()}) removed, "
          f"{n} kept -> {OUT}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
