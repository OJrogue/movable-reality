"""Stage 6: complete each scanned object with SAM 3D Objects (unseen sides included).

    uv run python scripts/complete_objects.py --scene room01
    uv run python scripts/complete_objects.py --scene room01 --only "armchair#2" --force

Needs lift_ids.py output (ids/masks/{i:06d}.png + labels.json, objects/obj_<id>.ply) and the
deployed Modal app (`uv run modal deploy modal_jobs/sam3d.py`). For each object: the frame
with the largest mask, SAM 3D Objects on frame + mask (about 1 min on an L40S, cached under
data/cache/sam3d/), registration to the scanned splats (pose from the frame's camera, scale
from the boxes, ICP on centres), then objects/obj_<id>_completed.ply with every splat flagged
`generated` = 1 and `completed_ply` set in scene.json. Skips objects already completed.

--mesh also asks SAM 3D Objects for the object as a vertex-coloured triangle mesh and writes
objects/obj_<id>_mesh.glb in the recon frame (decimated to --mesh-faces, default 100k),
placed by the same registration as the completed splats, with `mesh` set in scene.json.
The viewer shows either (?objects=, key V).
The mesh run is a separate cache entry, so the first --mesh run calls Modal once per object.

    uv run python scripts/complete_objects.py --scene room01 --mesh
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from PIL import Image

from movable_reality import splat_ops as so
from movable_reality.completion import best_frame, register
from movable_reality.config import scene_dir
from movable_reality.coverage.cameras import similarity_parts
from movable_reality.coverage.colmap import read_model
from movable_reality.meshes import decimate, read_mesh_glb, to_gaussian_frame, write_mesh_glb
from movable_reality.sam3d import mask_png, run_sam3d
from movable_reality.scene import Scene


def completed_name(ply: str) -> str:
    """objects/obj_chair_1.ply -> objects/obj_chair_1_completed.ply"""
    return ply[: -len(".ply")] + "_completed.ply"


def mesh_name(ply: str) -> str:
    """objects/obj_chair_1.ply -> objects/obj_chair_1_mesh.glb"""
    return ply[: -len(".ply")] + "_mesh.glb"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--only", help="one object id, e.g. 'armchair#2'")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mesh", action="store_true", help="also write a vertex-coloured mesh")
    ap.add_argument("--mesh-faces", type=int, default=100_000, help="decimation target")
    args = ap.parse_args()
    root = scene_dir(args.scene)
    scene = Scene.load(root / "scene.json")
    masks_dir = root / "ids" / "masks"
    labels = json.loads((masks_dir / "labels.json").read_text())
    value_of = {v: int(k) for k, v in labels.items()}
    cams = {int(c.name.rsplit("/", 1)[-1].split(".")[0]): c for c in read_model(root / "recon")}
    report_path = root / "completion.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    todo = [o for o in scene.objects if o.source == "scan" and (not args.only or o.id == args.only)]
    if args.only and not todo:
        print(f"no scanned object '{args.only}'")
        return 1
    for obj in todo:
        out_rel = completed_name(obj.ply)
        done = obj.completed_ply and (root / obj.completed_ply).exists()
        if args.mesh:
            done = done and obj.mesh and (root / obj.mesh).exists()
        if done and not args.force:
            print(f"{obj.id:<24} done ({obj.completed_ply}), --force to redo")
            continue
        if obj.id not in value_of:
            print(f"{obj.id:<24} not in ids/masks/labels.json, skipped")
            continue
        t0 = time.time()
        frame, pixels = best_frame(masks_dir, value_of[obj.id])
        if frame not in cams:
            print(f"{obj.id:<24} frame {frame} has no camera, skipped")
            continue
        mask = np.asarray(Image.open(masks_dir / f"{frame:06d}.png"))
        image = (root / "frames" / f"{frame:06d}.jpg").read_bytes()
        res = run_sam3d(image, mask_png(mask == value_of[obj.id]), mesh=args.mesh)
        t_gen = time.time() - t0

        cam = cams[frame]
        A, t, s = res.cam_from_local()
        # SAM 3D's pose is metric; the recon frame is metric / world scale (lift_ids --scale).
        sw, _ = similarity_parts(np.asarray(scene.world_from_recon))
        t, s = np.asarray(t) / sw, s / sw
        recon_from_cam = np.linalg.inv(so.similarity(cam.R, cam.t, 1.0))
        recon_from_local = recon_from_cam @ so.similarity(A, t, s)
        scan_v = so.read_vertices(root / obj.ply)
        splats, info = register(
            res.vertices, recon_from_local, so.xyz(scan_v), scene.world_from_recon,
            scan_alpha=so.opacity(scan_v),
        )  # fmt: skip
        so.write_vertices(root / out_rel, splats)
        obj.completed_ply = out_rel
        M = np.asarray(info.pop("recon_from_local"))
        if args.mesh and res.glb:
            mv, mc, mf = read_mesh_glb(res.glb)
            raw_faces = len(mf)
            mv, mc, mf = decimate(mv, mc, mf, args.mesh_faces)
            R, frame_name, frame_d = to_gaussian_frame(mv, so.xyz(res.vertices))
            mv = (mv @ R.T) @ M[:3, :3].T + M[:3, 3]
            obj.mesh = mesh_name(obj.ply)
            write_mesh_glb(root / obj.mesh, mv, mc, mf, name=obj.id)
            info.update(mesh_vertices=len(mv), mesh_faces=len(mf), mesh_raw_faces=raw_faces,
                        mesh_frame=frame_name, mesh_frame_dist=round(frame_d, 4))  # fmt: skip
        elif args.mesh:
            print(f"{obj.id:<24} SAM 3D returned no mesh")
        scene.save(root / "scene.json")
        report[obj.id] = {
            "frame": frame,
            "mask_px": pixels,
            "splats": len(splats),
            "sam3d_s": round(t_gen, 1),
            "total_s": round(time.time() - t0, 1),
            **info,
            "modal": res.meta.get("timings", {}),
        }
        report_path.write_text(json.dumps(report, indent=2))
        print(f"{obj.id:<24} frame {frame} ({pixels} px) -> {out_rel}, {len(splats)} splats, "
              f"ICP RMS {info['icp_rms_before_m']:.3f} -> {info['icp_rms_after_m']:.3f} m, "
              f"{time.time() - t0:.0f} s")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
