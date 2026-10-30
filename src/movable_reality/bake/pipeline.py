"""Stage 12: bake floor and ceiling of one scene folder into opaque textured quads.

Inputs (data/scenes/<id>/): scene.json, splat/scene.ply, ids/gaussian_ids.npy (one id per
Gaussian of splat/scene.ply), ids/labels.json ({"<id>": "<noun>#<sam id>"}, e.g.
{"4": "floor#1"}), and the background PLY named in scene.json.

Per label: fit a plane to the centres of the labelled Gaussians (world frame), bound it by
its inliers, rasterise the in-plane Gaussians orthographically into a texture, fill holes,
and write planes/<label>.glb (quad in the recon frame, like the PLYs). Then write the
background without the replaced Gaussians next to the original, point scene.json at it,
and record everything in `planes` and `plane_bake`.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from movable_reality.scene import BakedPlane, PlaneBake, Scene

from . import gaussians, glb, mask, planefit, raster, texture

DEFAULT_LABELS = ("floor", "ceiling")
IDS_PATH = "ids/gaussian_ids.npy"
LABELS_PATH = "ids/labels.json"
IDS_PLY = "splat/scene.ply"

# (world means (N, 3), world covariances (N, 3, 3), opacities (N,), rgb (N, 3), view)
#   -> (premultiplied colour (H, W, 3), alpha (H, W))
RasterFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, raster.OrthoView],
    tuple[np.ndarray, np.ndarray],
]


def cpu_raster(means, covs, opacities, rgb, view: raster.OrthoView):
    m2, c2, toward = raster.project(means, covs, view)
    return raster.rasterize(m2, c2, toward, opacities, rgb, view.width, view.height)


@dataclass
class BakeOptions:
    labels: tuple[str, ...] = DEFAULT_LABELS
    texel: float = 0.005  # metres per texel
    max_px: int = 4096  # cap per side; texel grows to fit
    threshold: float = 0.03  # RANSAC inlier distance and bake band, metres
    min_splats: int = 200  # labels with fewer Gaussians are skipped
    max_tilt_deg: float | None = None  # reject plane hypotheses tilted more than this from up
    image_format: str = "jpeg"  # "jpeg" or "png"
    jpeg_quality: int = 90
    seed: int = 0
    # Alpha mask (see mask.py). Texels outside the surface's outline become transparent.
    alpha_mask: bool = True
    mask_cell: float = 0.05  # metres, grid of the mask
    mask_close: float = 0.15  # metres, gaps narrower than about twice this are bridged


@dataclass
class PlaneReport:
    label: str
    status: str  # "baked" or "skipped: <reason>"
    labelled: int = 0
    inliers: int = 0
    baked: int = 0
    behind: int = 0
    removed: int = 0
    normal: tuple[float, float, float] | None = None
    offset: float | None = None
    rms: float | None = None
    size_m: tuple[float, float] | None = None
    texture_px: tuple[int, int] | None = None
    filled: float = 0.0
    transparent: float = 0.0
    seconds: float = 0.0


@dataclass
class BakeReport:
    status: str  # "baked", "skipped" (outputs exist) or "nothing to bake"
    planes: list[PlaneReport] = field(default_factory=list)
    background_ply: str | None = None
    removed_splats: int = 0
    notes: list[str] = field(default_factory=list)


def reduced_background_name(source: str) -> str:
    p = Path(source)
    return p.with_name(f"{p.stem}_baked{p.suffix}").as_posix()


def load_label_ids(scene_dir: Path, labels: Iterable[str]) -> dict[str, list[int]]:
    """Integer ids per label from ids/labels.json; the label is the noun before '#'."""
    path = scene_dir / LABELS_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing: it maps integer ids in {IDS_PATH} to object ids such as 'floor#1'"
        )
    mapping = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[int]] = {label: [] for label in labels}
    for key, obj_id in mapping.items():
        noun = str(obj_id).split("#", 1)[0].strip().lower()
        if noun in out:
            out[noun].append(int(key))
    return out


def outputs_exist(scene_dir: Path, scene: Scene) -> bool:
    if scene.plane_bake is None or not scene.planes:
        return False
    paths = [scene.plane_bake.background_ply] + [p.mesh for p in scene.planes]
    return all((scene_dir / p).is_file() for p in paths)


def _encode(img: np.ndarray, opts: BakeOptions) -> tuple[bytes, str]:
    """JPEG or PNG as asked. An RGBA image (alpha mask) is always PNG, as JPEG has no alpha."""
    buf = io.BytesIO()
    if opts.image_format == "png" or img.shape[-1] == 4:
        Image.fromarray(img).save(buf, format="PNG", optimize=True)
        return buf.getvalue(), "image/png"
    Image.fromarray(img).save(buf, format="JPEG", quality=opts.jpeg_quality, subsampling=0)
    return buf.getvalue(), "image/jpeg"


def bake_scene(
    scene_dir: Path,
    opts: BakeOptions | None = None,
    *,
    force: bool = False,
    label_ids: dict[str, list[int]] | None = None,
    raster_fn: RasterFn = cpu_raster,
    log: Callable[[str], None] = print,
) -> BakeReport:
    opts = opts or BakeOptions()
    scene_dir = Path(scene_dir)
    manifest = scene_dir / "scene.json"
    scene = Scene.load(manifest)
    if not force and outputs_exist(scene_dir, scene):
        return BakeReport(status="skipped", background_ply=scene.background_ply)

    source_bg = scene.plane_bake.source_background_ply if scene.plane_bake else scene.background_ply
    for rel in (IDS_PATH, IDS_PLY, source_bg):
        if not (scene_dir / rel).is_file():
            raise FileNotFoundError(
                f"{scene_dir / rel} missing: the bake needs the trained scene, per-Gaussian ids"
                " and the background split (stage 5, lift_ids)"
            )
    ids = np.load(scene_dir / IDS_PATH)
    full = gaussians.read_ply(scene_dir / IDS_PLY)
    verts = gaussians.vertices(full)
    if ids.shape != (len(verts),):
        raise ValueError(f"{IDS_PATH} has shape {ids.shape}, {IDS_PLY} has {len(verts)} splats")
    if label_ids is None:
        label_ids = load_label_ids(scene_dir, opts.labels)

    world_from_recon = np.asarray(scene.world_from_recon, dtype=np.float64)
    lin, trans = world_from_recon[:3, :3], world_from_recon[:3, 3]
    recon_from_world = np.linalg.inv(world_from_recon)
    means_recon = np.stack([np.asarray(verts[c], dtype=np.float64) for c in "xyz"], axis=1)
    means_world = means_recon @ lin.T + trans
    step = max(1, len(means_world) // 200_000)
    interior = np.median(means_world[::step], axis=0)  # a point inside the room
    up = lin @ np.array([0.0, -1.0, 0.0])  # OpenCV -Y in world; +Y after alignment
    up_world = np.array([0.0, 1.0, 0.0])

    report = BakeReport(status="baked")
    planes: list[BakedPlane] = []
    removed_idx: dict[str, np.ndarray] = {}
    for label in opts.labels:
        t0 = time.time()
        sel_idx = np.flatnonzero(np.isin(ids, label_ids.get(label, [])))
        pr = PlaneReport(label=label, status="baked", labelled=len(sel_idx))
        if len(sel_idx) < opts.min_splats:
            pr.status = f"skipped: {len(sel_idx)} labelled splats (< {opts.min_splats})"
            report.planes.append(pr)
            continue
        pts = means_world[sel_idx]
        plane = planefit.fit_plane(
            pts,
            opts.threshold,
            interior=interior,
            up=up_world if opts.max_tilt_deg is not None else None,
            max_tilt_deg=opts.max_tilt_deg,
            seed=opts.seed,
        )
        rect = planefit.bounding_rect(pts[plane.inliers], plane)
        rect, texel, width, height = planefit.snap_to_texels(rect, opts.texel, opts.max_px)
        inside = rect.contains(pts, margin=texel)
        dist = plane.signed_distance(pts)
        bake_local = plane.inliers & inside
        behind_local = (dist <= -opts.threshold) & inside  # hidden behind the opaque quad

        g = gaussians.gaussians_from_vertices(verts[sel_idx[bake_local]])
        covs = np.einsum("ij,njk,lk->nil", lin, g.covariances(), lin)
        view = raster.OrthoView(rect.origin, rect.u, rect.v, rect.normal, texel, width, height)
        premult, alpha = raster_fn(pts[bake_local], covs, g.opacities, g.rgb, view)
        rgb = texture.push_pull(premult, alpha)
        keep = np.ones(alpha.shape, dtype=bool)
        if opts.alpha_mask:
            # Objects stand on floors, so the floor under them exists. Ceilings have no such rule.
            on_floor = float(np.dot(plane.normal, up_world)) > 0.9
            supports = (
                [(np.asarray(o.bbox_min), np.asarray(o.bbox_max)) for o in scene.objects]
                if on_floor
                else None
            )
            keep = mask.plane_mask(
                alpha,
                texel,
                origin=rect.origin,
                u=rect.u,
                v=rect.v,
                supports=supports,
                cell=opts.mask_cell,
                close_m=opts.mask_close,
            )
        img = texture.to_uint8(rgb)
        masked = not keep.all()
        if masked:
            img = np.dstack([img, np.where(keep, 255, 0).astype(np.uint8)])
        img_bytes, mime = _encode(img, opts)

        corners_world = rect.corners()
        corners_recon = corners_world @ recon_from_world[:3, :3].T + recon_from_world[:3, 3]
        mesh_rel = f"planes/{label}.glb"
        glb.write_quad_glb(
            scene_dir / mesh_rel,
            corners_recon,
            img_bytes,
            mime,
            name=label,
            alpha_mask=masked,
            extras={"label": label, "texel_m": texel, "frame": "recon"},
        )
        removed_idx[label] = sel_idx[bake_local | behind_local]

        pr.inliers = int(plane.inliers.sum())
        pr.baked = int(bake_local.sum())
        pr.behind = int(behind_local.sum())
        pr.normal = tuple(float(x) for x in plane.normal)  # type: ignore[assignment]
        pr.offset = float(plane.offset)
        pr.rms = plane.rms
        pr.size_m = (rect.size_u, rect.size_v)
        pr.texture_px = (width, height)
        pr.filled = float(np.mean((alpha < 0.5) & keep))
        pr.transparent = float(np.mean(~keep))
        pr.seconds = time.time() - t0
        report.planes.append(pr)
        planes.append(
            BakedPlane(
                id=label,
                label=label,
                mesh=mesh_rel,
                normal=pr.normal,
                offset=pr.offset,
                corners=[tuple(float(x) for x in c) for c in corners_world],  # type: ignore[misc]
                texel_m=texel,
                texture_px=(width, height),
                fit_rms_m=plane.rms,
                filled_fraction=pr.filled,
                transparent_fraction=pr.transparent,
            )
        )
        tilt = planefit.angle_deg(plane.normal, up)
        log(
            f"{label}: {pr.labelled} labelled, {pr.inliers} inliers"
            f" (rms {plane.rms * 1000:.1f} mm),"
            f" {tilt:.1f} deg from recon up, {rect.size_u:.2f} x {rect.size_v:.2f} m,"
            f" {width} x {height} px, {pr.filled:.1%} filled, {pr.transparent:.1%} transparent,"
            f" {pr.seconds:.1f} s"
        )

    if not planes:
        report.status = "nothing to bake"
        return report

    # Background without the replaced Gaussians. Same source PLY, so the float32 centres
    # match exactly; the original stays untouched.
    bg = gaussians.read_ply(scene_dir / source_bg)
    bg_verts = gaussians.vertices(bg)
    hit_all = np.zeros(len(bg_verts), dtype=bool)
    for p in planes:
        hit = gaussians.match_positions(bg_verts, verts[removed_idx[p.label]])
        p.removed_splats = int((hit & ~hit_all).sum())
        hit_all |= hit
        missing = len(removed_idx[p.label]) - int(hit.sum())
        if missing > 0:
            report.notes.append(
                f"{p.label}: {missing} replaced splats are not in {source_bg} (in an object PLY?)"
            )
    for pr in report.planes:
        match = next((p for p in planes if p.label == pr.label), None)
        pr.removed = match.removed_splats if match else 0
    reduced = reduced_background_name(source_bg)
    gaussians.write_subset(scene_dir / reduced, bg, ~hit_all)

    scene = Scene.load(manifest)  # re-read: the agent may have edited it meanwhile
    scene.planes = planes
    scene.plane_bake = PlaneBake(
        source_background_ply=source_bg,
        background_ply=reduced,
        ids=IDS_PATH,
        ids_ply=IDS_PLY,
        removed_splats=int(hit_all.sum()),
    )
    scene.background_ply = reduced
    if scene.background_lod and Path(scene.background_lod).stem != Path(reduced).stem:
        # background_lod belongs to the lod lane, so it is left as is; the viewer loads it
        # instead of the reduced PLY until build_lod.py builds a tree for the new background.
        report.notes.append(
            f"background_lod ({scene.background_lod}) was built from the full background;"
            " run scripts/build_lod.py so it is rebuilt from the reduced PLY"
        )
    scene.save(manifest)
    report.background_ply = reduced
    report.removed_splats = int(hit_all.sum())
    for note in report.notes:
        log(f"note: {note}")
    return report


def revert_scene(scene_dir: Path) -> bool:
    """Point scene.json back at the original background and drop the planes (files stay).
    Used for the "without baked planes" benchmark row. Returns False if nothing was baked.
    A background_lod built from the reduced PLY needs build_lod.py again afterwards."""
    manifest = Path(scene_dir) / "scene.json"
    scene = Scene.load(manifest)
    if scene.plane_bake is None:
        return False
    scene.background_ply = scene.plane_bake.source_background_ply
    scene.planes = []
    scene.plane_bake = None
    scene.save(manifest)
    return True
