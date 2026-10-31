"""Room shell for the mesh-only view: floor, walls and ceiling as flat quads textured from
the video frames (src/movable_reality/roomshell.py).

    uv run python scripts/room_shell.py --scene blender01   # after lift_ids.py (and objtrain)

Reads the background splats (before any plane bake), recon_full/, frames/ and ids/masks.
Writes room/shell_<kind>_<k>.glb (one quad each, recon frame, alpha mask on floor and
ceiling), room/shell.json, and sets room_planes in scene.json. The viewer draws them with
the object meshes in ?objects=meshonly.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time

import numpy as np
from PIL import Image

from movable_reality.bake import gaussians, glb
from movable_reality.config import scene_dir
from movable_reality.coverage.colmap import read_model
from movable_reality.lift import STRUCTURE, ceiling_from_histogram
from movable_reality.roommesh import source_background, unit_m
from movable_reality.roomshell import (
    ON_WALL,
    Projector,
    View,
    finish,
    floor_points,
    room_outline,
    surfaces,
)
from movable_reality.scene import Scene


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--texel", type=float, default=0.006, help="metres per texel")
    ap.add_argument("--height", type=float, help="ceiling height in metres (default: estimate)")
    ap.add_argument("--views", type=int, default=3, help="best views kept per texel")
    ap.add_argument("--force", action="store_true", help="rebuild an existing shell")
    args = ap.parse_args()
    t0 = time.time()
    root = scene_dir(args.scene)
    if (root / "room" / "shell.json").exists() and not args.force:
        print("room/shell.json exists, skipping (--force rebuilds)")
        return 0
    scene = Scene.load(root / "scene.json")
    W = np.asarray(scene.world_from_recon, dtype=np.float64)
    unit = unit_m(W)  # world metres per recon unit; camera depths are in recon units
    lin, trans = W[:3, :3], W[:3, 3]
    bg_rel = source_background(scene.model_dump())
    g = gaussians.gaussians_from_vertices(gaussians.vertices(gaussians.read_ply(root / bg_rel)))
    m = np.asarray(g.means, dtype=np.float64) @ lin.T + trans
    solid = g.opacities >= 0.3
    cams = read_model(root / "recon_full")
    cam_y = np.array([c.center for c in cams]) @ lin.T + trans

    labels = json.loads((root / "ids" / "masks" / "labels.json").read_text())
    noun = {int(k): n.split("#")[0] for k, n in labels.items()}
    blockers = np.array([k for k, n in noun.items() if n not in STRUCTURE + ON_WALL])
    floor_vals = np.array([k for k, n in noun.items() if n == "floor"])
    Winv = np.linalg.inv(W)
    frames = []  # (camera, world R, world t, mask path)
    for c in cams:
        stem = c.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        mpath = root / "ids" / "masks" / f"{stem}.png"
        if mpath.exists():  # no SAM labels: furniture would be painted onto the walls
            # World-frame camera: x_cam = R (W^-1 x) + t, with W a similarity.
            frames.append((c, c.R @ Winv[:3, :3], c.R @ Winv[:3, 3] + c.t, mpath))
    hits = []
    for c, R, t, mpath in frames:
        fm = np.isin(np.asarray(Image.open(mpath)), floor_vals)
        if fm.any():
            hits.append(floor_points(c.K, R, t, fm))
    if not hits:
        print("no frame has floor pixels (the lift needs a 'floor' noun), no shell")
        return 1
    boxes = [(np.asarray(o.bbox_min)[[0, 2]], np.asarray(o.bbox_max)[[0, 2]])
             for o in scene.objects]  # fmt: skip
    poly = room_outline(hits, boxes)
    if args.height:
        height, how = args.height, "given"
    else:
        layer = ceiling_from_histogram(m[solid, 1], cam_y[:, 1])
        if len(layer) >= 100:
            height, how = float(np.median(layer)), "densest layer above the cameras"
        else:
            height, how = float(np.percentile(m[solid, 1], 99)), "99th percentile"
    walls = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    print(f"outline {len(poly)} corners, walls {' '.join(f'{w:.2f}' for w in walls)} m, "
          f"height {height:.2f} m ({how})")  # fmt: skip

    surfs = surfaces(poly, height)
    grids = [s.grid(args.texel) for s in surfs]
    proj = [Projector(p, s.normal, k=args.views) for s, (p, _, _) in zip(surfs, grids, strict=True)]
    used = 0
    for c, R, t, mpath in frames:
        image = np.asarray(Image.open(root / "frames" / c.name).convert("RGB"))
        blocked = np.isin(np.asarray(Image.open(mpath)), blockers)
        view = View(c.K, R, t, c.center @ lin.T + trans, image, blocked, near=0.1 / unit)
        for pr in proj:
            pr.add(view)
        used += 1
    print(f"projected {used} of {len(cams)} frames ({time.time() - t0:.0f} s)")

    out = root / "room"
    out.mkdir(exist_ok=True)
    for old in out.glob("shell_*.glb"):
        old.unlink()
    rels, report = [], {"background_ply": bg_rel, "floor_frames": len(hits),
                        "outline_xz": np.round(poly, 3).tolist(),
                        "wall_lengths_m": np.round(walls, 3).tolist(),
                        "height_m": round(height, 3), "height_from": how, "frames": used,
                        "texel_m": args.texel, "surfaces": []}  # fmt: skip
    counts: dict[str, int] = {}
    for s, (_, w, h), pr in zip(surfs, grids, proj, strict=True):
        rgb, seen = pr.result((h, w))
        rgba = finish(rgb, seen, s.outline_uv, args.texel, (s.size_u, s.size_v))
        masked = s.outline_uv is not None
        buf = io.BytesIO()
        if masked:
            Image.fromarray(rgba).save(buf, format="PNG", optimize=True)
            mime = "image/png"
        else:
            Image.fromarray(rgba[..., :3]).save(buf, format="JPEG", quality=90)
            mime = "image/jpeg"
        k = counts.get(s.kind, 0)
        counts[s.kind] = k + 1
        rel = f"room/shell_{s.kind}_{k}.glb"
        corners_recon = s.corners() @ Winv[:3, :3].T + Winv[:3, 3]
        glb.write_quad_glb(root / rel, corners_recon, buf.getvalue(), mime, name=rel,
                           alpha_mask=masked)  # fmt: skip
        rels.append(rel)
        inside = rgba[..., 3] > 0
        report["surfaces"].append({
            "glb": rel, "kind": s.kind, "size_m": [round(s.size_u, 3), round(s.size_v, 3)],
            "texture_px": [w, h], "seen_share": round(float(seen[inside].mean()), 3),
            "bytes": (root / rel).stat().st_size,
        })  # fmt: skip
        print(f"  {rel}: {s.size_u:.2f} x {s.size_v:.2f} m, {w} x {h} px, "
              f"{seen[inside].mean():.0%} seen")  # fmt: skip
    report["seconds"] = round(time.time() - t0, 1)
    (out / "shell.json").write_text(json.dumps(report, indent=2))
    scene = Scene.load(root / "scene.json")
    scene.room_planes = rels
    scene.save(root / "scene.json")
    print(f"{len(rels)} surfaces -> scene.json room_planes, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
