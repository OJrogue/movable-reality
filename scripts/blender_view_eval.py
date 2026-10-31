"""Novel-view check for a synthetic room: render the splat scene from the GT test-walk cameras
and compare with the GT renders (PSNR, SSIM, LPIPS).

    uv run python scripts/blender_view_eval.py --scene blender01            # write the views
    uv run modal run modal_jobs/reconstruct.py --scene blender01 --eval-walk test

The test walk (scripts/blender_room.py) follows a different path from the capture walk, so
none of its frames was trained on. Its GT cameras are moved into the reconstruction frame by
the Sim(3) that maps the GT capture-walk camera centres onto the reconstructed ones (the same
Umeyama fit as blender_gt_eval.py). Pose error in that fit lowers the scores, so the fit
residual is reported next to them. Writes data/scenes/<scene>/eval/views_<walk>.json.
CPU only, no API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_gt_eval import map_frames, read_colmap_images, umeyama  # noqa: E402

from movable_reality.config import DATA_DIR, scene_dir  # noqa: E402


def gt_to_recon(c2w_gt: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """GT cam_to_world -> cam_from_world (4x4) in the reconstruction frame, given
    recon ~ s R gt + t."""
    rot = R @ c2w_gt[:3, :3]
    centre = s * R @ c2w_gt[:3, 3] + t
    w2c = np.eye(4)
    w2c[:3, :3] = rot.T
    w2c[:3, 3] = -rot.T @ centre
    return w2c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", default="blender01")
    ap.add_argument("--walk", default="test")
    ap.add_argument("--stride", type=int, default=10, help="every n-th GT frame (30 fps)")
    args = ap.parse_args()

    scene = scene_dir(args.scene)
    gt = DATA_DIR / "raw" / f"{args.scene}_gt"
    main_cams = json.loads((gt / "main" / "cameras.json").read_text())
    gt_c2w = [np.array(f["cam_to_world"]) for f in main_cams["frames"]]
    frame_map = map_frames(scene, len(gt_c2w), DATA_DIR / "raw" / args.scene / "main" / "rgb")
    poses = read_colmap_images(scene / "recon" / "sparse")
    idx = {n: int(Path(n).stem) for n in sorted(poses) if Path(n).stem.isdigit()}
    pairs = [(poses[n][:3, 3], gt_c2w[frame_map[i]][:3, 3]) for n, i in idx.items()
             if i in frame_map]  # fmt: skip
    rc, gc = (np.array(x) for x in zip(*pairs, strict=True))
    s, R, t = umeyama(gc, rc)
    res = np.linalg.norm(s * gc @ R.T + t - rc, axis=1) / s  # in GT metres

    walk = json.loads((gt / args.walk / "cameras.json").read_text())
    K = np.array(walk["K"])
    K[0, 2] += 0.5  # GT K has pixel centres at integers; gsplat follows COLMAP
    K[1, 2] += 0.5
    rgb = DATA_DIR / "raw" / args.scene / args.walk / "rgb"
    views = [
        {"name": f"{f['index']:06d}", "image": f"{f['index']:06d}.png", "K": K.tolist(),
         "w2c": gt_to_recon(np.array(f["cam_to_world"]), s, R, t).tolist()}
        for f in walk["frames"][:: args.stride]
        if (rgb / f"{f['index']:06d}.png").exists()
    ]  # fmt: skip
    out = {
        "scene": args.scene,
        "walk": args.walk,
        "width": walk["width"],
        "height": walk["height"],
        "rgb_dir": str(rgb.relative_to(DATA_DIR)),  # under data/
        "alignment": {
            "cameras": len(pairs),
            "scale_recon_per_gt": s,
            "residual_rms_m": float(np.sqrt((res**2).mean())),
            "residual_max_m": float(res.max()),
        },  # fmt: skip
        "views": views,
    }
    path = scene / "eval" / f"views_{args.walk}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["alignment"]), f"{len(views)} views -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
