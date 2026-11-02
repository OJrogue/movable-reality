"""Room-surface artifact check for a synthetic room: the viewer at the GT test-walk poses,
scored against the GT renders on wall, floor and ceiling pixels only.

    uv run python scripts/blender_viewer_eval.py views --scene blender01   # poses for the viewer
    # in the viewer (dev server), run web/src/bench/evalViews.ts's driver, see docs/BUILD_LOG.md
    uv run python scripts/blender_viewer_eval.py score --scene blender01 --run <name>
    uv run python scripts/blender_viewer_eval.py sheet --scene blender01 --run <name>

`views` maps every --stride-th GT test-walk camera into the viewer's world frame (the Sim(3)
of blender_view_eval.py, then world_from_recon) and writes eval/viewer_views.json, which the
dev server serves at /scenes/<scene>/eval/viewer_views.json. The viewer saves one PNG per view
through POST /__eval to eval/viewer/<run>/.

`score` compares each PNG with the GT frame on the GT room-surface masks (floor 101, wall
102, ceiling 103), eroded so pose error at object edges does not count. A pixel is an
artifact when the mean absolute sRGB difference of the two 3x3-blurred images is over
--thresh. Writes eval/viewer/<run>/score.json and side-by-side PNGs (viewer, GT, artifacts
in red) of the worst views. `sheet` puts every view in one contact sheet,
eval/viewer/<run>/sheet.png, with the artifact pixels in red and the boxes of
artifact_judge.py (judge_<model>.json, if present) in cyan. CPU only, no API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_gt_eval import map_frames, read_colmap_images, umeyama  # noqa: E402

from movable_reality.config import DATA_DIR, scene_dir  # noqa: E402

SURFACES = {"floor": 101, "wall": 102, "ceiling": 103}
CV_TO_GL = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes to three.js camera axes


def viewer_from_gt(scene: Path, gt: Path) -> tuple[np.ndarray, dict]:
    """4x4 similarity from GT world to the viewer's world (W @ recon_from_gt)."""
    main_cams = json.loads((gt / "main" / "cameras.json").read_text())
    gt_c2w = [np.array(f["cam_to_world"]) for f in main_cams["frames"]]
    frame_map = map_frames(scene, len(gt_c2w), DATA_DIR / "raw" / scene.name / "main" / "rgb")
    poses = read_colmap_images(scene / "recon" / "sparse")
    idx = {n: int(Path(n).stem) for n in sorted(poses) if Path(n).stem.isdigit()}
    pairs = [(poses[n][:3, 3], gt_c2w[frame_map[i]][:3, 3]) for n, i in idx.items()
             if i in frame_map]  # fmt: skip
    rc, gc = (np.array(x) for x in zip(*pairs, strict=True))
    s, R, t = umeyama(gc, rc)
    recon_from_gt = np.eye(4)
    recon_from_gt[:3, :3], recon_from_gt[:3, 3] = s * R, t
    W = np.asarray(json.loads((scene / "scene.json").read_text())["world_from_recon"])
    M = W @ recon_from_gt
    res = np.linalg.norm(s * gc @ R.T + t - rc, axis=1) / s
    return M, {"cameras": len(pairs), "residual_rms_m": float(np.sqrt((res**2).mean())),
               "scale_viewer_per_gt": float(np.cbrt(np.linalg.det(M[:3, :3])))}  # fmt: skip


def cmd_views(args) -> int:
    scene = scene_dir(args.scene)
    gt = DATA_DIR / "raw" / f"{args.scene}_gt"
    M, align = viewer_from_gt(scene, gt)
    sc = align["scale_viewer_per_gt"]
    walk = json.loads((gt / args.walk / "cameras.json").read_text())
    K = np.array(walk["K"])
    fov = float(np.degrees(2 * np.arctan(walk["height"] / 2 / K[1, 1])))
    views = []
    for f in walk["frames"][:: args.stride]:
        c2w = np.array(f["cam_to_world"])
        out = np.eye(4)
        out[:3, :3] = (M[:3, :3] / sc) @ c2w[:3, :3] @ CV_TO_GL
        out[:3, 3] = M[:3, :3] @ c2w[:3, 3] + M[:3, 3]
        views.append({"name": f"{f['index']:06d}", "c2w": out.round(6).tolist()})
    path = scene / "eval" / "viewer_views.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scene": args.scene, "walk": args.walk, "fov_deg": fov,
                                "aspect": walk["width"] / walk["height"], "alignment": align,
                                "views": views}, indent=1))  # fmt: skip
    print(json.dumps(align), f"{len(views)} views, fov {fov:.1f} deg -> {path}")
    return 0


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(img.filter(ImageFilter.MinFilter(2 * px + 1))) > 127


def score_view(viewer: Image.Image, gt_rgb: Image.Image, gt_mask: Image.Image, thresh: float,
               erode_px: int) -> tuple[dict, np.ndarray, np.ndarray]:  # fmt: skip
    """Per-surface stats, the artifact map and the scored mask, at the viewer's size."""
    size = viewer.size
    v = np.asarray(viewer.convert("RGB").filter(ImageFilter.BoxBlur(1)), dtype=np.float32) / 255
    g = gt_rgb.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    g = np.asarray(g.filter(ImageFilter.BoxBlur(1)), dtype=np.float32) / 255
    labels = np.asarray(gt_mask.resize(size, Image.Resampling.NEAREST))
    err = np.abs(v - g).mean(-1)
    bad = err > thresh
    stats, scored = {}, np.zeros(labels.shape, bool)
    for name, value in SURFACES.items():
        m = _erode(labels == value, erode_px)
        scored |= m
        n = int(m.sum())
        stats[name] = {"px": n, "artifact_frac": float(bad[m].mean()) if n else 0.0,
                       "mean_err": float(err[m].mean()) if n else 0.0}  # fmt: skip
    return stats, bad & scored, scored


def cmd_score(args) -> int:
    scene = scene_dir(args.scene)
    run = scene / "eval" / "viewer" / args.run
    meta = json.loads((scene / "eval" / "viewer_views.json").read_text())
    rgb_dir = DATA_DIR / "raw" / args.scene / meta["walk"] / "rgb"
    mask_dir = DATA_DIR / "raw" / f"{args.scene}_gt" / meta["walk"] / "masks"
    per_view, totals = {}, {k: [0, 0.0] for k in SURFACES}
    pics = []
    for view in meta["views"]:
        p = run / f"{view['name']}.png"
        if not p.exists():
            continue
        img = Image.open(p)
        gt_rgb = Image.open(rgb_dir / f"{view['name']}.png")
        stats, bad, scored = score_view(img, gt_rgb, Image.open(mask_dir / f"{view['name']}.png"),
                                        args.thresh, args.erode)  # fmt: skip
        per_view[view["name"]] = stats
        for k, s in stats.items():
            totals[k][0] += s["px"]
            totals[k][1] += s["artifact_frac"] * s["px"]
        frac = bad.sum() / max(1, scored.sum())
        pics.append((frac, view["name"], img, gt_rgb, bad))
    if not per_view:
        print(f"no frames in {run}")
        return 1
    summary = {k: round(v[1] / max(1, v[0]), 5) for k, v in totals.items()}
    allpx = sum(v[0] for v in totals.values())
    summary["all"] = round(sum(v[1] for v in totals.values()) / max(1, allpx), 5)
    pics.sort(key=lambda x: -x[0])
    for _frac, name, img, gt_rgb, bad in pics[: args.worst]:
        w, h = img.size
        over = np.asarray(img.convert("RGB")).copy()
        over[bad] = (0.4 * over[bad] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
        sheet = Image.new("RGB", (w * 3, h))
        sheet.paste(img.convert("RGB"), (0, 0))
        sheet.paste(gt_rgb.convert("RGB").resize((w, h)), (w, 0))
        sheet.paste(Image.fromarray(over), (2 * w, 0))
        sheet.save(run / f"worst_{name}.png")
    out = {"run": args.run, "thresh": args.thresh, "erode_px": args.erode,
           "views": len(per_view), "artifact_frac": summary,
           "worst": [(n, round(float(f), 4)) for f, n, *_ in pics[: args.worst]],
           "per_view": per_view}  # fmt: skip
    (run / "score.json").write_text(json.dumps(out, indent=1))
    print(f"{args.run}: {len(per_view)} views, artifact share {summary}")
    print("worst:", out["worst"])
    return 0


def cmd_sheet(args) -> int:
    d = scene_dir(args.scene) / "eval" / "viewer" / args.run
    meta = json.loads((scene_dir(args.scene) / "eval" / "viewer_views.json").read_text())
    jp = d / f"judge_{args.judge}.json"
    judge = json.loads(jp.read_text())["frames"] if jp.exists() else {}
    gt_dir = DATA_DIR / "raw" / args.scene / meta["walk"] / "rgb"
    mask_dir = DATA_DIR / "raw" / f"{args.scene}_gt" / meta["walk"] / "masks"
    tw, th, cols = 480, 270, 5
    tiles = []
    for v in meta["views"]:
        img = Image.open(d / f"{v['name']}.png").convert("RGB")
        gt, mk = Image.open(gt_dir / f"{v['name']}.png"), Image.open(mask_dir / f"{v['name']}.png")
        _, bad, _ = score_view(img, gt, mk, args.thresh, args.erode)
        a = np.asarray(img).copy()
        a[bad] = (0.4 * a[bad] + [153, 0, 0]).astype(np.uint8)
        t = Image.fromarray(a)
        dr = ImageDraw.Draw(t)
        for art in judge.get(v["name"], {}).get("artifacts") or []:
            b = art.get("box")
            if isinstance(b, list) and len(b) == 4:
                box = [b[0] * t.width, b[1] * t.height, b[2] * t.width, b[3] * t.height]
                dr.rectangle(box, outline=(0, 255, 255), width=3)
        dr.text((8, 8), v["name"], fill=(255, 255, 0))
        tiles.append(t.resize((tw, th)))
    sheet = Image.new("RGB", (cols * tw, -(-len(tiles) // cols) * th))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    sheet.save(d / "sheet.png")
    print(d / "sheet.png")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("views")
    v.add_argument("--scene", default="blender01")
    v.add_argument("--walk", default="test")
    v.add_argument("--stride", type=int, default=20, help="every n-th GT frame (30 fps)")
    s = sub.add_parser("score")
    s.add_argument("--scene", default="blender01")
    s.add_argument("--run", required=True)
    s.add_argument("--thresh", type=float, default=0.12, help="mean abs sRGB error, 0..1")
    s.add_argument("--erode", type=int, default=4, help="mask erosion in viewer pixels")
    s.add_argument("--worst", type=int, default=6)
    h = sub.add_parser("sheet")
    h.add_argument("--scene", default="blender01")
    h.add_argument("--run", required=True)
    h.add_argument("--thresh", type=float, default=0.12)
    h.add_argument("--erode", type=int, default=4)
    h.add_argument("--judge", default="spark", help="judge_<model>.json boxes to draw")
    args = ap.parse_args()
    return {"views": cmd_views, "score": cmd_score, "sheet": cmd_sheet}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
