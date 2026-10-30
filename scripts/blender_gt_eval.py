"""Compare pipeline outputs for a synthetic room (blender01, blender02) against its ground truth.

    uv run python scripts/blender_gt_eval.py --scene blender01
    uv run python scripts/blender_gt_eval.py --scene blender01 --labels data/scenes/blender01/<dir>

Ground truth comes from scripts/blender_room.py (data/raw/blender01_gt/). Checks:
- pose: MapAnything's cameras (the COLMAP model the reconstruction writes, default
  data/scenes/<scene>/recon/sparse) against the GT cameras. ATE = RMSE of camera centres
  after a Sim(3) (Umeyama) alignment; also the median rotation error and the fitted scale,
  which is ~1 when the reconstruction is metric.
- masks: per-frame object label PNGs from the pipeline (value = pipeline object index,
  0 = none, file {i:06d}.png = frames/{i:06d}.jpg, optional labels.json mapping value to
  object id) against the GT masks. Objects are matched one-to-one by greedy max IoU over
  all frames; reports IoU per GT object and the mean over furniture and items.
- completion: chamfer distance of obj_<id>_completed.ply against the GT mesh. The helper
  exists; the CLI hookup waits for M4 (TODO below).
Pipeline frame i maps to GT frame round(source_time_s * 30) from frames.json, refined
within +-2 frames by matching against the noise-free GT renders when they exist.
Writes data/scenes/<scene>/eval/blender_gt.json. CPU only, no API calls.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from movable_reality.config import DATA_DIR, scene_dir

GT_FPS = 30
STRUCTURE_MIN_INDEX = 100  # GT indices >= 100 are floor, wall, ceiling


# ---------------------------------------------------------------- COLMAP cameras


def qvec_to_rotmat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def read_colmap_images(path: Path) -> dict[str, np.ndarray]:
    """images.bin or images.txt (a file or a sparse folder, also sparse/0) -> name ->
    4x4 cam_to_world (OpenCV camera, reconstruction frame)."""
    path = Path(path)
    if path.is_dir():
        for cand in (path, path / "0"):
            for fname in ("images.bin", "images.txt"):
                if (cand / fname).exists():
                    return read_colmap_images(cand / fname)
        raise FileNotFoundError(f"no images.bin or images.txt under {path}")
    poses: dict[str, tuple] = {}
    if path.suffix == ".bin":
        data = path.read_bytes()
        (n,), off = struct.unpack_from("<Q", data, 0), 8
        for _ in range(n):
            vals = struct.unpack_from("<I7dI", data, off)
            off += struct.calcsize("<I7dI")
            end = data.index(b"\x00", off)
            name = data[off:end].decode()
            off = end + 1
            (npts,) = struct.unpack_from("<Q", data, off)
            off += 8 + npts * 24
            poses[name] = (vals[1:5], vals[5:8])
    else:
        lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
        i = 0
        while i < len(lines):
            parts = lines[i].split()
            if not parts:
                i += 1
                continue
            poses[parts[9]] = (tuple(map(float, parts[1:5])), tuple(map(float, parts[5:8])))
            i += 2  # skip the 2D points line, which may be empty
    out = {}
    for name, (q, t) in poses.items():
        r = qvec_to_rotmat(q)
        m = np.eye(4)
        m[:3, :3] = r.T
        m[:3, 3] = -r.T @ np.asarray(t)
        out[name] = m
    return out


# ---------------------------------------------------------------- pose metrics


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares s, R, t with dst ~ s * R @ src + t (Umeyama 1991). src, dst: (N, 3)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rot = u @ sign @ vt
    var_s = (xs**2).sum() / len(src)
    scale = float(np.trace(np.diag(d) @ sign) / var_s) if with_scale else 1.0
    return scale, rot, mu_d - scale * rot @ mu_s


def rotation_angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))))


def pose_errors(pred_c2w: list[np.ndarray], gt_c2w: list[np.ndarray]) -> dict:
    """ATE after Sim(3) alignment of camera centres, rotation errors after the same alignment."""
    if len(pred_c2w) < 3:
        raise ValueError("need at least 3 matched cameras")
    pc = np.array([m[:3, 3] for m in pred_c2w])
    gc = np.array([m[:3, 3] for m in gt_c2w])
    s, rot, t = umeyama(pc, gc)
    res = np.linalg.norm(gc - (s * pc @ rot.T + t), axis=1)
    rot_err = [
        rotation_angle_deg(g[:3, :3].T @ rot @ p[:3, :3])
        for p, g in zip(pred_c2w, gt_c2w, strict=True)
    ]
    return {
        "n_cameras": len(pc),
        "ate_rmse_m": float(np.sqrt((res**2).mean())),
        "ate_mean_m": float(res.mean()),
        "ate_median_m": float(np.median(res)),
        "ate_max_m": float(res.max()),
        "rot_err_median_deg": float(np.median(rot_err)),
        "rot_err_max_deg": float(np.max(rot_err)),
        "sim3_scale": s,
        "gt_path_length_m": float(np.linalg.norm(np.diff(gc, axis=0), axis=1).sum()),
    }


# ---------------------------------------------------------------- frame mapping


def gt_index(source_time_s: float, n_gt: int, fps: float = GT_FPS) -> int:
    return int(min(max(round(source_time_s * fps), 0), n_gt - 1))


@lru_cache(maxsize=64)  # neighbouring pipeline frames share GT candidates
def small_gray(path: Path, size: tuple[int, int] = (240, 135)) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize(size), dtype=np.float32)


def closest_frame(target: np.ndarray, candidates: dict[int, np.ndarray]) -> int:
    """Index of the candidate image with the lowest mean squared difference to target."""
    return min(candidates, key=lambda i: float(((candidates[i] - target) ** 2).mean()))


def map_frames(scene: Path, n_gt: int, rgb_dir: Path | None, window: int = 2) -> dict[int, int]:
    """Pipeline frame index -> GT frame index."""
    meta = json.loads((scene / "frames.json").read_text())["frames"]
    out = {}
    for f in meta:
        guess = gt_index(f["source_time_s"], n_gt)
        if rgb_dir is not None and rgb_dir.is_dir():
            cands = {
                j: small_gray(rgb_dir / f"{j:06d}.png")
                for j in range(max(guess - window, 0), min(guess + window, n_gt - 1) + 1)
                if (rgb_dir / f"{j:06d}.png").exists()
            }
            if cands:
                guess = closest_frame(small_gray(scene / "frames" / f"{f['index']:06d}.jpg"), cands)
        out[f["index"]] = guess
    return out


# ---------------------------------------------------------------- mask metrics


def accumulate_overlap(pairs, counts: dict | None = None) -> dict:
    """Sum pixel counts per (pred label, gt label) over (pred, gt) label image pairs."""
    counts = {} if counts is None else counts
    for pred, gt in pairs:
        pred = np.asarray(pred, dtype=np.int64)
        gt = np.asarray(gt, dtype=np.int64)
        if pred.shape != gt.shape:
            raise ValueError(f"label shapes differ: {pred.shape} vs {gt.shape}")
        codes, n = np.unique(pred * 65536 + gt, return_counts=True)
        for c, k in zip(codes.tolist(), n.tolist(), strict=True):
            key = (c // 65536, c % 65536)
            counts[key] = counts.get(key, 0) + k
    return counts


def match_objects(counts: dict, gt_ids, background: int = 0) -> dict[int, tuple[int | None, float]]:
    """One-to-one greedy match by IoU. Returns gt label -> (pred label or None, IoU)."""
    pred_area: dict[int, int] = {}
    gt_area: dict[int, int] = {}
    for (p, g), k in counts.items():
        pred_area[p] = pred_area.get(p, 0) + k
        gt_area[g] = gt_area.get(g, 0) + k
    ious = []
    for (p, g), k in counts.items():
        if p == background or g == background or g not in gt_ids:
            continue
        ious.append((k / (pred_area[p] + gt_area[g] - k), g, p))
    out: dict[int, tuple[int | None, float]] = {g: (None, 0.0) for g in gt_ids}
    used_p: set[int] = set()
    used_g: set[int] = set()
    for iou, g, p in sorted(ious, reverse=True):
        if g in used_g or p in used_p:
            continue
        out[g] = (p, iou)
        used_g.add(g)
        used_p.add(p)
    return out


def load_label(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path)
    if shape is not None and img.size != (shape[1], shape[0]):
        img = img.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(img).astype(np.int64)


# ---------------------------------------------------------------- completion


def chamfer(a: np.ndarray, b: np.ndarray, max_points: int = 20000, seed: int = 0) -> float:
    """Symmetric chamfer distance (mean of both nearest-neighbour means), metres."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) > max_points:
        a = a[rng.choice(len(a), max_points, replace=False)]
    if len(b) > max_points:
        b = b[rng.choice(len(b), max_points, replace=False)]

    def nn_mean(x: np.ndarray, y: np.ndarray) -> float:
        best = []
        for i in range(0, len(x), 2048):
            d = ((x[i : i + 2048, None, :] - y[None, :, :]) ** 2).sum(-1)
            best.append(np.sqrt(d.min(1)))
        return float(np.concatenate(best).mean())

    return 0.5 * (nn_mean(a, b) + nn_mean(b, a))


# ---------------------------------------------------------------- CLI


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", default="blender01")
    ap.add_argument("--gt", type=Path, help="default data/raw/<scene>_gt")
    ap.add_argument("--walk", default="main")
    ap.add_argument("--colmap", type=Path, help="default data/scenes/<scene>/recon/sparse")
    ap.add_argument("--labels", type=Path, help="folder of pipeline label PNGs (skip if unset)")
    ap.add_argument("--rgb", type=Path, help="GT renders for frame matching (default inferred)")
    args = ap.parse_args()

    scene = scene_dir(args.scene)
    gt = args.gt or DATA_DIR / "raw" / f"{args.scene}_gt"
    cams = json.loads((gt / args.walk / "cameras.json").read_text())
    objects = json.loads((gt / "objects.json").read_text())["objects"]
    gt_c2w = [np.array(f["cam_to_world"]) for f in cams["frames"]]
    rgb = args.rgb or DATA_DIR / "raw" / args.scene / args.walk / "rgb"
    frame_map = map_frames(scene, len(gt_c2w), rgb)
    report: dict = {"scene": args.scene, "walk": args.walk, "synthetic": True}

    colmap = args.colmap or scene / "recon" / "sparse"
    if colmap.exists():
        poses = read_colmap_images(colmap)
        pairs = [(m, gt_c2w[frame_map[int(Path(n).stem)]]) for n, m in sorted(poses.items())
                 if Path(n).stem.isdigit() and int(Path(n).stem) in frame_map]  # fmt: skip
        report["pose"] = pose_errors([p for p, _ in pairs], [g for _, g in pairs])
        report["pose"]["n_frames_selected"] = len(frame_map)
    else:
        print(f"no COLMAP model at {colmap}; skipping the pose check")

    if args.labels:
        names = {}
        if (args.labels / "labels.json").exists():
            names = {
                int(k): v for k, v in json.loads((args.labels / "labels.json").read_text()).items()
            }
        counts: dict = {}
        used = 0
        for i, j in sorted(frame_map.items()):
            p = args.labels / f"{i:06d}.png"
            if not p.exists():
                continue
            g = load_label(gt / args.walk / "masks" / f"{j:06d}.png")
            accumulate_overlap([(load_label(p, g.shape), g)], counts)
            used += 1
        if used == 0:
            raise SystemExit(f"no label PNG in {args.labels} matches a frame in frames.json")
        gt_ids = {o["index"]: o for o in objects}
        match = match_objects(counts, set(gt_ids))
        per = {
            gt_ids[g]["id"]: {"pred": None if p is None else names.get(p, p), "iou": round(iou, 4)}
            for g, (p, iou) in sorted(match.items())
        }
        movable = [match[g][1] for g in gt_ids if g < STRUCTURE_MIN_INDEX]
        report["masks"] = {"n_frames": used, "mean_iou_objects": float(np.mean(movable)),
                           "per_object": per}  # fmt: skip

    # TODO(M4): once obj_<id>_completed.ply exists, map pipeline objects to GT objects through
    # the mask match, bring the splat centres into the GT frame with the pose Sim(3) and
    # world_from_recon, and report chamfer(completed, GT mesh vertices) per object.
    report["completion"] = "not run: needs M4 completions (obj_<id>_completed.ply)"

    out = scene / "eval" / "blender_gt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "masks"}, indent=2))
    if "masks" in report:
        print(f"mask mean IoU (objects): {report['masks']['mean_iou_objects']:.3f}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
