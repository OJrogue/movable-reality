"""Stages 4-5: SAM 3.1 per inventory noun, then lift the masks onto the Gaussians.

    uv run python scripts/lift_ids.py --scene room01          # needs inventory.json, recon/, splat/
    uv run python scripts/lift_ids.py --scene room01 --force

Writes ids/gaussian_ids.npy (one int per Gaussian of splat/scene.ply, -1 = background),
ids/labels.json ({"<int>": "<noun>#<sam id>"}, floor/wall/ceiling included, D9),
ids/masks/{i:06d}.png (per-frame object label images for blender_gt_eval.py --labels),
objects/background.ply, objects/obj_<id>.ply and scene.json (objects, world_from_recon from
the floor plane). Items resting on furniture join it (support rule, D12). SAM calls are
cached under data/cache/sam/, so a rerun costs nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import numpy as np
from PIL import Image
from plyfile import PlyData

from movable_reality.config import scene_dir
from movable_reality.coverage.colmap import read_model
from movable_reality.lift import (
    STRUCTURE,
    LiftParams,
    assign,
    floor_alignment,
    label_image,
    largest_cluster,
    support,
    vote,
)
from movable_reality.lod.split import write_split
from movable_reality.sam import paste_mask, segment_video
from movable_reality.scene import Scene, SceneObject

MIN_SPLATS = 150  # smaller objects are dropped as noise


async def segment_all(video, nouns: list[str]):
    sem = asyncio.Semaphore(3)

    async def one(noun):
        async with sem:
            t0 = time.time()
            inst = await segment_video(video, noun)
            print(f"  SAM '{noun}': {len({i.object_id for i in inst})} object(s), "
                  f"{len(inst)} masks, {time.time() - t0:.0f} s")  # fmt: skip
            return noun, inst

    return await asyncio.gather(*(one(n) for n in nouns))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-share", type=float, default=0.6)
    args = ap.parse_args()
    root = scene_dir(args.scene)
    if (root / "ids" / "gaussian_ids.npy").exists() and not args.force:
        print("ids/gaussian_ids.npy exists (--force to redo)")
        return 0
    inv = json.loads((root / "inventory.json").read_text())
    p = LiftParams(min_share=args.min_share)

    # SAM 3.1 per noun on the clip built from the reconstruction frames.
    results = asyncio.run(segment_all(root / "sam_input.mp4", inv["nouns"]))
    w, h = Image.open(next(iter(sorted((root / "frames").glob("*.jpg"))))).size
    names: dict[int, str] = {}
    per_frame: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for noun, instances in results:
        for inst in instances:
            key = f"{noun}#{inst.object_id}"
            lab = next((k for k, v in names.items() if v == key), None)
            if lab is None:
                lab = len(names)
                names[lab] = key
            m = paste_mask(inst, w, h)
            per_frame.setdefault(inst.frame_index, []).append((lab, m, float(m.sum())))
    labels_img = {i: label_image(v, (h, w)) for i, v in per_frame.items()}
    (root / "ids" / "masks").mkdir(parents=True, exist_ok=True)
    for i, img in labels_img.items():  # 0 = none, value = label + 1
        Image.fromarray((img + 1).clip(0, 255).astype(np.uint8)).save(
            root / "ids" / "masks" / f"{i:06d}.png"
        )
    (root / "ids" / "masks" / "labels.json").write_text(
        json.dumps({str(k + 1): v for k, v in names.items()}, indent=2)
    )

    # Vote in the reconstruction frame.
    cams = read_model(root / "recon")
    frame_of = [int(c.name.rsplit("/", 1)[-1].split(".")[0]) for c in cams]
    v = PlyData.read(str(root / "splat" / "scene.ply"), mmap=False)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    opacity = 1 / (1 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))
    t0 = time.time()
    votes, seen = vote(cams, [labels_img.get(f) for f in frame_of], xyz, opacity, len(names), p)
    ids = assign(votes, seen, p)
    print(f"voted over {len(cams)} views, {len(xyz)} Gaussians in {time.time() - t0:.0f} s; "
          f"{(seen >= p.min_views).mean():.0%} seen in >= {p.min_views} views")  # fmt: skip

    # Floor alignment from the floor Gaussians (fallback: OpenCV to +Y up).
    noun_of = {k: v.split("#")[0] for k, v in names.items()}
    floor = np.isin(ids, [k for k, n in noun_of.items() if n == "floor"])
    wall = np.isin(ids, [k for k, n in noun_of.items() if n == "wall"])
    centres = np.array([c.center for c in cams])
    if floor.sum() > 500:
        M = floor_alignment(xyz[floor].astype(np.float64), centres, xyz[wall].astype(np.float64))
    else:
        print("too few floor Gaussians; keeping the default OpenCV-to-Y-up frame")
        M = np.asarray(Scene(scene_id="x").world_from_recon)
    world = xyz.astype(np.float64) @ M[:3, :3].T + M[:3, 3]

    # Floaters, tiny objects, then the support rule.
    objects = [k for k, n in noun_of.items() if n not in STRUCTURE]
    for k in objects:
        m = np.nonzero(ids == k)[0]
        keep = largest_cluster(world[m], p.voxel)
        ids[m[~keep]] = -1
        if keep.sum() < MIN_SPLATS:
            ids[m] = -1
    ids, merged = support(world, ids, [k for k in objects if (ids == k).sum()], p)
    for child, parent in merged.items():
        print(f"  {names[child]} rests on {names[parent]}: merged")
    np.save(root / "ids" / "gaussian_ids.npy", ids)
    (root / "ids" / "labels.json").write_text(
        json.dumps({str(k): v for k, v in names.items()}, indent=2)
    )

    # Split and write the manifest. Structure stays in the background (D9).
    object_names = {k: names[k] for k in objects if (ids == k).sum() >= MIN_SPLATS}
    split = write_split(root / "splat" / "scene.ply", ids, object_names, root, M.tolist())
    old = Scene.load(root / "scene.json") if (root / "scene.json").exists() else None
    scene = Scene(scene_id=args.scene, world_from_recon=M.tolist())
    if old:  # keep lane sections that do not depend on object ids
        scene.coverage = old.coverage
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
