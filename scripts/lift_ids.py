"""Stages 4-5: SAM 3.1 per inventory noun, then lift the masks onto the Gaussians.

    uv run python scripts/lift_ids.py --scene room01          # needs inventory.json, recon/, splat/
    uv run python scripts/lift_ids.py --scene room01 --force

Writes ids/gaussian_ids.npy (one int per Gaussian of splat/scene.ply, -1 = background),
ids/labels.json ({"<int>": "<noun>#<sam id>"}, floor/wall/ceiling included, D9),
ids/masks/{i:06d}.png (per-frame object label images for blender_gt_eval.py --labels),
ids/object_labels.json (scene object id -> mask values, merged children included),
objects/background.ply, objects/obj_<id>.ply and scene.json (objects, world_from_recon from
the floor plane). Items resting on furniture join it (support rule, D12). SAM calls are
cached under data/cache/sam/, so a rerun costs nothing.

Metric scale: MapAnything's scale can be off (1.8x too small on blender01). Pass the room's
floor-to-ceiling height from a tape measure, or a factor:

    uv run python scripts/lift_ids.py --scene room01 --force --reuse-votes --ceiling-height 2.6
    uv run python scripts/lift_ids.py --scene room01 --force --reuse-votes --scale 1.8

--reuse-votes skips SAM and the vote (ids/assigned.npy, written by every full run).

Without either, the scale comes from typical object heights (Muse Spark, metric.py): every
floor-standing object votes typical / measured height. --no-auto-scale keeps the raw scale.

--sam-only runs only the SAM calls (they need the clip, not the reconstruction), so the
cache fills while the GPU job runs (scripts/run_pipeline.py does this).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

from movable_reality.config import scene_dir
from movable_reality.coverage.colmap import read_model
from movable_reality.lift import (
    STRUCTURE,
    LiftParams,
    assign,
    ceiling_from_histogram,
    ceiling_scale,
    floor_alignment,
    label_image,
    largest_cluster,
    scaled,
    support,
    vote,
)
from movable_reality.lod.split import write_split
from movable_reality.metric import height_priors, object_height, scale_from_heights, votes_json
from movable_reality.recon import full_res_camera
from movable_reality.sam import paste_mask, segment_video
from movable_reality.scene import Scene, SceneObject

MIN_SPLATS = 150  # smaller objects are dropped as noise


SPARSE_NOUNS = ("wall", "ceiling")  # large regions: their SAM streams drop on long clips
SPARSE_STEP = 3


def sparse_clip(video: Path, step: int) -> Path:
    """Every `step`-th frame of the scene's frames/ as its own clip (same encoding as
    extract_frames.py). The "wall" stream failed on every 150+ frame clip (31 Oct); a third
    of the frames still gives walls in every direction."""
    root = video.parent
    out = root / f"sam_input_every{step}.mp4"
    if out.exists() and out.stat().st_mtime >= video.stat().st_mtime:
        return out
    frames = sorted((root / "frames").glob("*.jpg"))[::step]
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            shutil.copyfile(f, Path(tmp) / f"{i:06d}.jpg")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-framerate", "4", "-i", str(Path(tmp) / "%06d.jpg"),
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt",
             "yuv420p", "-crf", "18", "-movflags", "+faststart", str(out)],
            check=True,
        )  # fmt: skip
    return out


async def segment_all(video, nouns: list[str]):
    sem = asyncio.Semaphore(3)

    async def one(noun):
        async with sem:
            t0 = time.time()
            sparse = noun.lower() in SPARSE_NOUNS
            clip = sparse_clip(Path(video), SPARSE_STEP) if sparse else video
            for attempt in range(3):  # SAM streams time out now and then (31 Oct, blender01)
                try:
                    inst = await segment_video(clip, noun)
                    if sparse:  # back to indices of the full clip (= frames/{i:06d}.jpg)
                        inst = [replace(i, frame_index=i.frame_index * SPARSE_STEP) for i in inst]
                    break
                except Exception as exc:  # noqa: BLE001 - APITimeoutError, stream errors
                    cause = f"{type(exc).__name__}: {exc.__cause__ or exc}"
                    if attempt == 2 and noun in (
                        "wall",
                        "ceiling",
                    ):  # they only refine yaw and scale
                        print(f"  SAM '{noun}' failed 3 times ({cause}); continuing without it")
                        return noun, []
                    if attempt == 2:
                        raise
                    print(f"  SAM '{noun}' failed ({cause}), retrying")
                    await asyncio.sleep(10 * (attempt + 1))
            print(f"  SAM '{noun}': {len({i.object_id for i in inst})} object(s), "
                  f"{len(inst)} masks, {time.time() - t0:.0f} s")  # fmt: skip
            return noun, inst

    return await asyncio.gather(*(one(n) for n in nouns))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-share", type=float, default=0.6)
    metric = ap.add_mutually_exclusive_group()
    metric.add_argument("--ceiling-height", type=float, help="tape-measured floor to ceiling, m")
    metric.add_argument("--scale", type=float, help="metric correction factor")
    metric.add_argument(
        "--no-auto-scale", action="store_true", help="no scale from typical object heights"
    )
    ap.add_argument("--reuse-votes", action="store_true", help="skip SAM and the vote")
    ap.add_argument("--sam-only", action="store_true", help="only fill the SAM cache, then exit")
    ap.add_argument(
        "--skip",
        default="",
        help="comma-separated nouns not to segment (e.g. wall, whose SAM "
        "stream dropped 6 of 6 times on blender01 and blender02)",
    )
    args = ap.parse_args()
    root = scene_dir(args.scene)
    if args.sam_only:
        inv = json.loads((root / "inventory.json").read_text())
        skip = {n.strip().lower() for n in args.skip.split(",") if n.strip()}
        nouns = [n for n in inv["nouns"] if n.lower() not in skip]
        asyncio.run(segment_all(root / "sam_input.mp4", nouns))
        print(f"SAM cache filled for {len(nouns)} nouns")
        return 0
    if (root / "ids" / "gaussian_ids.npy").exists() and not args.force:
        print("ids/gaussian_ids.npy exists (--force to redo)")
        return 0
    inv = json.loads((root / "inventory.json").read_text())
    p = LiftParams(min_share=args.min_share)

    w, h = Image.open(next(iter(sorted((root / "frames").glob("*.jpg"))))).size
    cams = [full_res_camera(c, w, h) for c in read_model(root / "recon")]
    v = PlyData.read(str(root / "splat" / "scene.ply"), mmap=False)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    if args.reuse_votes:
        ids = np.load(root / "ids" / "assigned.npy")
        names = {
            int(k): n for k, n in json.loads((root / "ids" / "labels.json").read_text()).items()
        }
        if len(ids) != len(xyz):
            raise SystemExit("ids/assigned.npy does not match splat/scene.ply; run without it")
        if args.skip:
            print("--skip has no effect with --reuse-votes (the saved vote already has its nouns)")
    else:
        skip = {n.strip().lower() for n in args.skip.split(",") if n.strip()}
        inv = {**inv, "nouns": [n for n in inv["nouns"] if n.lower() not in skip]}
        names, ids = segment_and_vote(root, inv, p, cams, xyz, v, w, h)
    finish(args, root, p, names, ids, cams, xyz)
    return 0


def segment_and_vote(root, inv, p, cams, xyz, v, w, h):
    # SAM 3.1 per noun on the clip built from the reconstruction frames.
    results = asyncio.run(segment_all(root / "sam_input.mp4", inv["nouns"]))
    names: dict[int, str] = {}
    per_frame: dict[int, list] = {}  # box-local instances; full-frame masks only per frame
    for noun, instances in results:
        for inst in instances:
            key = f"{noun}#{inst.object_id}"
            lab = next((k for k, v in names.items() if v == key), None)
            if lab is None:
                lab = len(names)
                names[lab] = key
            per_frame.setdefault(inst.frame_index, []).append((lab, inst))
    labels_img = {
        i: label_image([(lab, paste_mask(inst, w, h), float(inst.mask.sum())) for lab, inst in v],
                       (h, w))
        for i, v in per_frame.items()
    }  # fmt: skip
    (root / "ids" / "masks").mkdir(parents=True, exist_ok=True)
    for i, img in labels_img.items():  # 0 = none, value = label + 1
        Image.fromarray((img + 1).clip(0, 255).astype(np.uint8)).save(
            root / "ids" / "masks" / f"{i:06d}.png"
        )
    (root / "ids" / "masks" / "labels.json").write_text(
        json.dumps({str(k + 1): v for k, v in names.items()}, indent=2)
    )

    # Vote in the reconstruction frame.
    frame_of = [int(c.name.rsplit("/", 1)[-1].split(".")[0]) for c in cams]
    opacity = 1 / (1 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))
    t0 = time.time()
    # Wall and ceiling were asked of SAM on every SPARSE_STEP-th frame only.
    sparse = np.array([names[k].split("#")[0].lower() in SPARSE_NOUNS for k in range(len(names))])
    queried = [~sparse | (f % SPARSE_STEP == 0) for f in frame_of]
    votes, seen = vote(cams, [labels_img.get(f) for f in frame_of], xyz, opacity, len(names), p,
                       queried=queried)  # fmt: skip
    ids = assign(votes, seen, p)
    shown = (seen.max(axis=1) >= p.min_views).mean()
    print(f"voted over {len(cams)} views, {len(xyz)} Gaussians in {time.time() - t0:.0f} s; "
          f"{shown:.0%} seen in >= {p.min_views} views")  # fmt: skip
    np.save(root / "ids" / "assigned.npy", ids)
    (root / "ids" / "labels.json").write_text(
        json.dumps({str(k): v for k, v in names.items()}, indent=2)
    )
    return names, ids


def auto_scale(world, ids, noun_of, names, p):
    """Scale from typical object heights (metric.py). Measures each object after a provisional
    floater cut, without changing ids. Returns (scale or None, votes)."""
    measured = {}
    for k, noun in noun_of.items():
        if noun in STRUCTURE:
            continue
        pts = world[ids == k]
        if len(pts) < MIN_SPLATS:
            continue
        pts = pts[largest_cluster(pts, p.voxel)]
        if len(pts) >= MIN_SPLATS:
            measured[names[k]] = (noun, *object_height(pts[:, 1]))
    priors = asyncio.run(height_priors([n for n, _, _ in measured.values()]))
    s, votes = scale_from_heights(measured, priors)
    for v in sorted(votes, key=lambda v: -v.weight):
        mark = "x" if v.used else "-"
        print(f"  {mark} {v.object_id:<22} top {v.measured:.3f}  typical {v.prior:.2f} m  "
              f"ratio {v.ratio:.3f}  w {v.weight:.1f}  {v.note}")  # fmt: skip
    if s is None:
        print("too few floor-standing objects for a scale from object heights; raw scale kept")
    return s, votes


def finish(args, root, p, names, ids, cams, xyz) -> None:

    # Floor alignment from the floor Gaussians (fallback: OpenCV to +Y up).
    noun_of = {k: v.split("#")[0] for k, v in names.items()}
    floor = np.isin(ids, [k for k, n in noun_of.items() if n == "floor"])
    wall = np.isin(ids, [k for k, n in noun_of.items() if n == "wall"])
    centres = np.array([c.center for c in cams])
    aligned = floor.sum() > 500
    if aligned:
        M = floor_alignment(xyz[floor].astype(np.float64), centres, xyz[wall].astype(np.float64))
    else:
        print("too few floor Gaussians; keeping the default OpenCV-to-Y-up frame")
        M = np.asarray(Scene(scene_id="x").world_from_recon)
    world = xyz.astype(np.float64) @ M[:3, :3].T + M[:3, 3]
    s = args.scale
    if args.ceiling_height and not aligned:
        raise SystemExit("--ceiling-height needs the floor plane, and floor alignment failed")
    if args.ceiling_height:
        ceiling = np.isin(ids, [k for k, n in noun_of.items() if n == "ceiling"])
        ceiling_y = world[ceiling, 1]
        if len(ceiling_y) < 100:  # SAM found no ceiling: densest layer above the cameras
            cam_y = (centres @ M[:3, :3].T + M[:3, 3])[:, 1]
            ceiling_y = ceiling_from_histogram(world[:, 1], cam_y)
            print(f"no ceiling label; ceiling layer from the height histogram, "
                  f"{len(ceiling_y)} Gaussians")  # fmt: skip
        s = ceiling_scale(ceiling_y, args.ceiling_height)
    method = "scale" if args.scale else "ceiling" if args.ceiling_height else "none"
    votes: list = []
    if not s and aligned and not args.no_auto_scale:
        s, votes = auto_scale(world, ids, noun_of, names, p)
        method = "object heights" if s else "none"
    if s:
        M = scaled(M, s)
        world *= s
        print(f"metric scale x{s:.3f} ({method})")
    (root / "ids" / "scale.json").write_text(
        json.dumps({"scale": s or 1.0, "method": method, "votes": votes_json(votes)}, indent=2)
    )

    # Floaters, tiny objects, then the support rule.
    objects = [k for k, n in noun_of.items() if n not in STRUCTURE]
    for k in objects:
        m = np.nonzero(ids == k)[0]
        keep = largest_cluster(world[m], p.voxel)
        ids[m[~keep]] = -1
        if keep.sum() < MIN_SPLATS:
            ids[m] = -1
    merged: dict[int, int] = {}
    if aligned:  # the support rule assumes the floor at y = 0
        ids, merged = support(world, ids, [k for k in objects if (ids == k).sum()], p)
    for child, parent in merged.items():
        print(f"  {names[child]} rests on {names[parent]}: merged")
    np.save(root / "ids" / "gaussian_ids.npy", ids)
    (root / "ids" / "labels.json").write_text(
        json.dumps({str(k): v for k, v in names.items()}, indent=2)
    )

    # Split and write the manifest. Structure stays in the background (D9).
    object_names = {k: names[k] for k in objects if (ids == k).sum() >= MIN_SPLATS}
    # Mask values (label + 1 in ids/masks) per object, merged children included, for the
    # per-object training (modal_jobs/reconstruct.py --objects).
    (root / "ids" / "object_labels.json").write_text(json.dumps({
        n: [k + 1] + [c + 1 for c, par in merged.items() if par == k]
        for k, n in object_names.items()
    }, indent=2))  # fmt: skip
    split = write_split(root / "splat" / "scene.ply", ids, object_names, root, M.tolist())
    for stale in ("objects_lifted", "objects_trained"):  # per-object training of the old split
        shutil.rmtree(root / stale, ignore_errors=True)
    old = Scene.load(root / "scene.json") if (root / "scene.json").exists() else None
    scene = Scene(scene_id=args.scene, world_from_recon=M.tolist())
    if old and np.allclose(old.world_from_recon, M):  # coverage positions are world frame
        scene.coverage = old.coverage
    elif old and old.coverage:
        print("world_from_recon changed: coverage dropped, rerun score_coverage.py")
    for key, (rel, n, (lo, hi)) in split.items():
        if key == "background":
            continue
        scene.objects.append(
            SceneObject(id=key, label=key.split("#")[0], ply=rel, bbox_min=lo, bbox_max=hi)
        )
        print(f"  {key:<24} {n:>8} splats  size "
              + " x ".join(f"{b - a:.2f}" for a, b in zip(lo, hi, strict=True)) + " m")  # fmt: skip
    scene.save(root / "scene.json")
    print(f"wrote {len(scene.objects)} objects to {root / 'scene.json'}")


if __name__ == "__main__":
    sys.exit(main())
