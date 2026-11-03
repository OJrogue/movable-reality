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
from pathlib import Path

import numpy as np
from PIL import Image

from movable_reality.bake import gaussians, glb, texture
from movable_reality.config import scene_dir
from movable_reality.coverage.colmap import read_model
from movable_reality.lift import STRUCTURE, ceiling_from_histogram
from movable_reality.roommesh import unit_m
from movable_reality.roomshell import (
    FILL_PROMPT,
    ON_WALL,
    UNSEEN_SHARE,
    Projector,
    View,
    background_source,
    base_mask,
    blend_fill,
    copy_fill,
    finish,
    floor_points,
    footprint_mask,
    hole_mask,
    match_colour,
    occluder_zbuffer,
    outline_keep,
    room_outline,
    shadow_mask,
    size_hint,
    surface_splats,
    surfaces,
    upper_wall_colour,
)
from movable_reality.scene import Scene

TYPICAL_CEILING = 2.5  # metres
SH_C0 = 0.28209479177387814  # degree-0 spherical harmonic: f_dc to colour


def fill_holes(rgb: np.ndarray, holes: np.ndarray, kind: str, work: Path) -> np.ndarray | None:
    """Muse Image edit of a surface texture with its holes painted magenta (coverage.repair,
    cached by request). Returns the edited texture at the input size, None when the call
    fails (the push-pull fill stays)."""
    from movable_reality.coverage.repair import MASK_COLOR, RepairRequest, repair_view

    work.mkdir(parents=True, exist_ok=True)
    view = rgb.copy()
    view[holes] = MASK_COLOR
    Image.fromarray(view).save(work / "view.png")
    h, w = holes.shape
    req = RepairRequest(gap_id=f"shell-{kind}", view=work / "view.png", references=(),
                        prompt=FILL_PROMPT.format(kind=kind), size=size_hint(w, h))  # fmt: skip
    try:
        res = repair_view(req, work / "filled.png")
    except Exception as e:  # noqa: BLE001 - no key, quota or a bad answer: keep push-pull
        print(f"  {kind}: Muse Image fill failed ({e}), push-pull fill kept")
        return None
    out = np.asarray(Image.open(res.image).convert("RGB").resize((w, h), Image.Resampling.LANCZOS))
    print(f"  {kind}: Muse Image fill of {holes.mean():.1%} {'(cached)' if res.cached else ''}")
    return out


def plausible_ceiling(h: float) -> bool:
    """Homes have ceilings of about 2.1 to 4.5 m; anything else was not the ceiling."""
    return 2.1 <= h <= 4.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--texel", type=float, default=0.006, help="metres per texel")
    ap.add_argument("--height", type=float, help="ceiling height in metres (default: estimate)")
    ap.add_argument("--views", type=int, default=3, help="best views kept per texel")
    ap.add_argument("--force", action="store_true", help="rebuild an existing shell")
    ap.add_argument(
        "--fill",
        choices=("copy", "muse", "pushpull"),
        default="copy",
        help="hidden texels: copied from the surface itself, Muse Image edit ($0.01 a "
        "surface, cached) or push-pull",
    )
    ap.add_argument(
        "--reuse-projection",
        action="store_true",
        help="reuse room/fill/projection.npz when its inputs match",
    )
    ap.add_argument(
        "--floor-reach",
        type=float,
        default=1.2,
        help="furniture up to this height (m) is tested for a shadow on the floor",
    )
    ap.add_argument(
        "--floor-contact",
        type=float,
        default=0.12,
        help="furniture this close (m) to the floor stands on it (roomshell.base_mask)",
    )
    ap.add_argument(
        "--wall-reach",
        type=float,
        default=0.4,
        help="furniture this close (m) to a wall shadows it",
    )
    ap.add_argument(
        "--shadow-margin",
        type=float,
        default=0.1,
        help="margin (m) around a footprint that is regenerated too",
    )
    ap.add_argument(
        "--shadow-reach",
        type=float,
        default=0.5,
        help="texels this close (m) to a footprint are tested for a cast shadow",
    )
    ap.add_argument(
        "--occluder-gap",
        type=float,
        default=0.06,
        help="background splats that stay farther than this (m) from a surface occlude it; "
        "the same rule and default as room_surfaces.py --gap, so a splat is either kept as an "
        "occluder or painted onto the shell, never lost",
    )
    ap.add_argument(
        "--z-margin",
        type=float,
        default=0.05,
        help="a texel this far (m) behind an occluder still counts as seen",
    )
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
    shell_path = root / "room" / "shell.json"
    old_shell = json.loads(shell_path.read_text()) if shell_path.exists() else None
    bg_rel = background_source(scene.model_dump(), old_shell)
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
             for o in scene.objects if o.source == "scan"]  # fmt: skip
    poly = room_outline(hits, boxes)
    if args.height:
        height, how = args.height, "given"
    else:
        # Height ranges in metres mean nothing in an unscaled scene: measure only.
        scale_file = root / "ids" / "scale.json"
        metric = not scale_file.exists() or (
            json.loads(scale_file.read_text()).get("method") != "none"
        )
        layer = ceiling_from_histogram(m[solid, 1], cam_y[:, 1])
        p99 = float(np.percentile(m[solid, 1], 99))
        if len(layer) >= 100 and (plausible_ceiling(float(np.median(layer))) or not metric):
            height, how = float(np.median(layer)), "densest layer above the cameras"
        elif plausible_ceiling(p99) or not metric:
            height, how = p99, "99th percentile"
        else:
            # A walk that never looks up (blender01) has no ceiling splats, and its densest
            # layer above the cameras was a shelf at 1.8 m (1 Nov). The scene is metric
            # (lift_ids.py), so a typical height beats a wrong measured one.
            height, how = TYPICAL_CEILING, "typical; ceiling not seen, pass --height"
    walls = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    print(f"outline {len(poly)} corners, walls {' '.join(f'{w:.2f}' for w in walls)} m, "
          f"height {height:.2f} m ({how})")  # fmt: skip

    surfs = surfaces(poly, height)
    grids = [s.grid(args.texel) for s in surfs]
    proj = [Projector(p, s.normal, k=args.views) for s, (p, _, _) in zip(surfs, grids, strict=True)]
    # Occluders for the depth test: every scanned object's splats, and the background splats
    # off the room surfaces (items no noun named). SAM's masks miss object edges, shadows
    # and unnamed items, which then got painted onto the walls and floor (blender01, 2 Nov).
    size = np.exp(np.asarray(g.log_scales, dtype=np.float64)).max(1) * unit
    occ = [m[solid & ~surface_splats(m, size, g.opacities, poly, height, gap=args.occluder_gap)]]
    furniture = []  # (bbox_min, bbox_max, splat centres, colours) per scanned object, world
    for o in scene.objects:
        if o.source == "scan" and (root / o.ply).exists():
            ov = gaussians.vertices(gaussians.read_ply(root / o.ply))
            og = gaussians.gaussians_from_vertices(ov)
            occ.append(np.asarray(og.means, dtype=np.float64)[og.opacities >= 0.3] @ lin.T + trans)
            dc = np.stack([np.asarray(ov[f"f_dc_{i}"], dtype=np.float64) for i in range(3)], 1)
            rgb_o = np.clip(0.5 + SH_C0 * dc[og.opacities >= 0.3], 0, 1) * 255
            furniture.append((np.asarray(o.bbox_min), np.asarray(o.bbox_max), occ[-1], rgb_o))
    furniture_w = np.concatenate(occ[1:]) if len(occ) > 1 else np.zeros((0, 3))
    occ_w = np.concatenate(occ)
    if len(occ_w) > 600_000:
        occ_w = occ_w[np.random.default_rng(0).choice(len(occ_w), 600_000, replace=False)]
    print(f"depth test with {len(occ_w)} occluder splats")
    # The projection is the slow part (2 min on blender01): --reuse-projection keeps its
    # result, keyed by the inputs that change it, while the hole fill is tuned.
    cache = root / "room" / "fill" / "projection.npz"
    key = json.dumps([np.round(poly, 4).tolist(), round(height, 4), args.texel, args.views,
                      args.occluder_gap, args.z_margin, len(frames), len(occ_w),
                      round(float(occ_w.sum()), 2)])  # fmt: skip  # a re-lift changes the occluders
    results, used = None, len(frames)
    if args.reuse_projection and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            results = [(z[f"rgb{i}"], z[f"seen{i}"]) for i in range(len(surfs))]
            print("projection reused")
    if results is None:
        used = 0
        for c, R, t, mpath in frames:
            image = np.asarray(Image.open(root / "frames" / c.name).convert("RGB"))
            blocked = np.isin(np.asarray(Image.open(mpath)), blockers)
            zbuf = occluder_zbuffer(occ_w @ R.T + t, c.K, image.shape[:2])
            view = View(c.K, R, t, c.center @ lin.T + trans, image, blocked, near=0.1 / unit,
                        zbuf=zbuf, z_margin=args.z_margin / unit)  # fmt: skip
            for pr in proj:
                pr.add(view)
            used += 1
        print(f"projected {used} of {len(cams)} frames ({time.time() - t0:.0f} s)")
        results = [pr.result((h, w)) for (_, w, h), pr in zip(grids, proj, strict=True)]
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, key=key, **{f"rgb{i}": r for i, (r, _) in enumerate(results)},
                            **{f"seen{i}": sn for i, (_, sn) in enumerate(results)})  # fmt: skip

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
    upper = upper_wall_colour([r for s, r in zip(surfs, results, strict=True)
                               if s.kind == "wall"])  # fmt: skip
    for s, (_, w, h), (rgb, seen) in zip(surfs, grids, results, strict=True):
        rgba = finish(rgb, seen, s.outline_uv, args.texel, (s.size_u, s.size_v),
                      fallback=upper if s.kind == "ceiling" else None)  # fmt: skip
        keep = outline_keep(s.outline_uv, (s.size_u, s.size_v), seen.shape)
        foot, foreign = None, None
        if s.kind != "ceiling":  # furniture stands on the floor and against the walls
            reach = args.floor_reach if s.kind == "floor" else args.wall_reach
            rgb8 = texture.to_uint8(rgb)
            if s.kind == "floor":
                foot, foreign = base_mask(s, furniture, rgb8, (seen > 0) & keep,
                                          args.floor_contact, args.shadow_margin)  # fmt: skip
            else:
                foot = footprint_mask(s, furniture_w, seen.shape, reach, args.shadow_margin)
            near = footprint_mask(s, furniture_w, seen.shape, reach, args.shadow_reach)
            drop = 0.3 if s.kind == "floor" else 0.18  # plank floors vary more than paint
            foot |= shadow_mask(rgb8, (seen > 0) & keep & ~foot, near, drop=drop)
        holes = hole_mask(seen, keep, foot)
        generated = 0.0
        seen_share = float(seen[keep].mean())
        if args.fill == "copy" and holes.any() and seen_share >= UNSEEN_SHARE:
            rgba[..., :3] = copy_fill(rgba[..., :3], holes, keep, foreign=foreign)
            generated = float(holes[keep].mean())
        elif args.fill == "muse" and holes.any() and seen_share >= UNSEEN_SHARE:
            k_ = counts.get(s.kind, 0)
            filled = fill_holes(rgba[..., :3], holes, s.kind, out / "fill" / f"{s.kind}_{k_}")
            if filled is not None:
                filled = match_colour(filled, rgba[..., :3], holes)
                rgba[..., :3] = blend_fill(rgba[..., :3], filled, holes)
                generated = float(holes[keep].mean())
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
            "generated_share": round(generated, 3),  # texels filled, not photographed
            "normal": np.round(s.normal, 4).tolist(),  # world, into the room
            "bytes": (root / rel).stat().st_size,
        })  # fmt: skip
        print(f"  {rel}: {s.size_u:.2f} x {s.size_v:.2f} m, {w} x {h} px, "
              f"{seen[inside].mean():.0%} seen, {generated:.0%} generated")  # fmt: skip
    report["seconds"] = round(time.time() - t0, 1)
    if old_shell:  # room_surfaces.py's record: the background still points at its output
        report.update({k: v for k, v in old_shell.items()
                       if k.startswith("surfaces_") or k == "replaces_splats"})  # fmt: skip
    (out / "shell.json").write_text(json.dumps(report, indent=2))
    scene = Scene.load(root / "scene.json")
    scene.room_planes = rels
    scene.save(root / "scene.json")
    print(f"{len(rels)} surfaces -> scene.json room_planes, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
