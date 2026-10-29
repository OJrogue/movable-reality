"""Synthetic room capture with exact ground truth for coverage.

A 5 m x 4 m x 2.6 m room with a sofa pulled out from the wall, a coffee table, a desk with
side panels and a chair, and a tall bookshelf. A phone "orbit" walks a loop around the
coffee table at hand height and points across the room. Every surface sample's exact view
count comes from analytic ray-box tests, so hidden regions (behind the sofa, under the
desk, under the table, the ceiling above the loop, the top of the shelf) are known.

The splat set imitates a gsplat result: samples seen by many views become tight,
opaque Gaussians; samples seen once or twice are noisier; most never-seen samples are
missing (holes) and the rest are faint floaters.

Output uses the layout of a real scene (docs/ARCHITECTURE.md), in a reconstruction frame
equal to the first camera's OpenCV frame, with `world_from_recon` in scene.json:

  splat/scene.ply              all Gaussians, recon frame
  recon/sparse/0/*.bin         COLMAP model (PINHOLE), cam_from_recon
  frames/000000.jpg ...        CPU renders of the true surfaces (optional)
  frames.json
  objects/background.ply, objects/obj_<id>.ply, scene.json
  synthetic/ground_truth.npz   true surface samples, exact view counts, VR visibility
  synthetic/regions.json       named hidden regions (world frame boxes)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..scene import Scene, SceneObject
from .cameras import Camera, camera_from_yaw_pitch, transform_points
from .colmap import write_model
from .ply import write_splats

ROOM_LO = np.array([-2.5, 0.0, -2.0])
ROOM_HI = np.array([2.5, 2.6, 2.0])


@dataclass(frozen=True)
class Box:
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]


@dataclass(frozen=True)
class SynthObject:
    id: str
    label: str
    color: tuple[float, float, float]
    boxes: tuple[Box, ...]


def _legs(x0, x1, z0, z1, h, t=0.04) -> tuple[Box, ...]:
    return tuple(Box((x, 0.0, z), (x + t, h, z + t)) for x in (x0, x1 - t) for z in (z0, z1 - t))


OBJECTS: tuple[SynthObject, ...] = (
    SynthObject(
        "sofa#1",
        "sofa",
        (0.30, 0.38, 0.55),
        (
            Box((-1.1, 0.0, -1.6), (1.1, 0.45, -0.75)),  # seat
            Box((-1.1, 0.0, -1.6), (1.1, 0.85, -1.4)),  # back
            Box((-1.1, 0.0, -1.6), (-0.92, 0.62, -0.75)),  # arms
            Box((0.92, 0.0, -1.6), (1.1, 0.62, -0.75)),
        ),
    ),
    SynthObject(
        "coffee table#1",
        "coffee table",
        (0.55, 0.40, 0.25),
        (Box((-0.4, 0.33, 0.1), (0.4, 0.37, 0.7)),) + _legs(-0.4, 0.4, 0.1, 0.7, 0.33),
    ),
    SynthObject(
        "desk#1",
        "desk",
        (0.80, 0.78, 0.72),
        (
            Box((1.7, 0.71, 0.3), (2.45, 0.75, 1.5)),  # top
            Box((1.7, 0.0, 0.3), (2.45, 0.71, 0.33)),  # side panels
            Box((1.7, 0.0, 1.47), (2.45, 0.71, 1.5)),
            Box((2.40, 0.0, 0.33), (2.45, 0.71, 1.47)),  # back panel
        ),
    ),
    SynthObject(
        "chair#1",
        "chair",
        (0.20, 0.20, 0.22),
        (
            Box((1.15, 0.42, 0.7), (1.6, 0.47, 1.15)),  # seat
            Box((1.15, 0.47, 0.7), (1.2, 0.95, 1.15)),  # back
        )
        + _legs(1.15, 1.6, 0.7, 1.15, 0.42, 0.03),
    ),
    SynthObject(
        "bookshelf#1",
        "bookshelf",
        (0.45, 0.30, 0.20),
        (Box((-2.45, 0.0, 0.5), (-2.1, 1.45, 1.4)),),  # above phone height, below VR eyes
    ),
)

# Named hidden regions for per-region reporting (world frame). Ground truth itself comes
# from exact visibility of every sample, not from these boxes.
REGIONS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "behind_sofa": ((-1.1, 0.0, -2.0), (1.1, 0.9, -1.6)),
    "under_desk": ((1.7, 0.0, 0.33), (2.5, 0.71, 1.47)),
    "under_coffee_table": ((-0.4, 0.0, 0.1), (0.4, 0.33, 0.7)),
    "ceiling": ((-2.5, 2.5, -2.0), (2.5, 2.6, 2.0)),
    "top_of_bookshelf": ((-2.5, 1.4, 0.5), (-2.1, 1.5, 1.4)),
}

WALL_COLOR = (0.86, 0.84, 0.78)
FLOOR_COLOR = (0.58, 0.44, 0.30)
CEILING_COLOR = (0.93, 0.93, 0.92)


@dataclass(frozen=True)
class SynthConfig:
    spacing: float = 0.02  # surface sample spacing, metres (about 255k Gaussians)
    n_views: int = 150
    width: int = 640
    height: int = 360
    hfov_deg: float = 65.0
    loop_center: tuple[float, float] = (0.0, 0.35)  # x, z
    loop_radii: tuple[float, float] = (1.35, 0.85)
    hold_height: float = 1.3
    vr_eye_heights: tuple[float, ...] = (1.6, 1.0)
    vr_spacing: float = 0.25
    wall_clearance: float = 0.3
    object_clearance: float = 0.25
    hole_prob: float = 0.8  # never-seen samples dropped from the splat set
    write_frames: bool = True
    seed: int = 7


@dataclass
class Surface:
    points: np.ndarray  # (S, 3) world
    normals: np.ndarray  # (S, 3) unit, axis-aligned
    colors: np.ndarray  # (S, 3)
    object_index: np.ndarray  # (S,) int, -1 = room shell


# ---------------------------------------------------------------- geometry


def all_boxes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Object boxes as arrays lo (B, 3), hi (B, 3) and the object index of each."""
    lo, hi, idx = [], [], []
    for i, obj in enumerate(OBJECTS):
        for b in obj.boxes:
            lo.append(b.lo)
            hi.append(b.hi)
            idx.append(i)
    return np.array(lo, float), np.array(hi, float), np.array(idx)


def object_bounds(obj: SynthObject) -> tuple[np.ndarray, np.ndarray]:
    lo = np.min([b.lo for b in obj.boxes], axis=0)
    hi = np.max([b.hi for b in obj.boxes], axis=0)
    return lo, hi


def _face_samples(rng, lo, hi, axis: int, side: int, spacing: float):
    """Jittered grid on one axis-aligned face. Returns points (n, 3) and normal (3,)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    a, b = [k for k in range(3) if k != axis]
    na = max(1, int(round((hi[a] - lo[a]) / spacing)))
    nb = max(1, int(round((hi[b] - lo[b]) / spacing)))
    ga, gb = np.meshgrid(np.arange(na), np.arange(nb), indexing="ij")
    ua = (ga.ravel() + rng.uniform(0.15, 0.85, ga.size)) / na
    ub = (gb.ravel() + rng.uniform(0.15, 0.85, gb.size)) / nb
    pts = np.empty((ua.size, 3))
    pts[:, a] = lo[a] + ua * (hi[a] - lo[a])
    pts[:, b] = lo[b] + ub * (hi[b] - lo[b])
    pts[:, axis] = hi[axis] if side == 1 else lo[axis]
    normal = np.zeros(3)
    normal[axis] = 1.0 if side == 1 else -1.0
    return pts, normal


def _inside_any(points: np.ndarray, lo: np.ndarray, hi: np.ndarray, eps: float = 1e-3):
    out = np.zeros(len(points), dtype=bool)
    for bl, bh in zip(lo, hi, strict=True):
        out |= np.all((points > bl + eps) & (points < bh - eps), axis=1)
    return out


def sample_surfaces(spacing: float, seed: int = 7) -> Surface:
    """Surface samples of the room shell (inward normals) and every object box (outward),
    without samples hidden inside other boxes or under boxes that stand on the floor."""
    rng = np.random.default_rng(seed)
    pts, nrm, col, oid = [], [], [], []

    def add(p, n, c, o, texture=0.04):
        pts.append(p)
        nrm.append(np.repeat(n[None], len(p), axis=0))
        base = np.asarray(c, float)
        # Low-frequency texture so rendered frames are not flat (helps the residual job).
        tex = 0.5 + 0.5 * np.sin(p @ np.array([7.3, 5.1, 6.7]) + rng.uniform(0, 6.3))
        shade = base[None] * (1.0 - texture + 2 * texture * tex[:, None])
        col.append(np.clip(shade + rng.normal(0, 0.015, (len(p), 3)), 0, 1))
        oid.append(np.full(len(p), o))

    for axis in range(3):
        for side in (0, 1):
            p, n = _face_samples(rng, ROOM_LO, ROOM_HI, axis, side, spacing)
            n = -n  # shell faces point into the room
            c = FLOOR_COLOR if (axis == 1 and side == 0) else WALL_COLOR
            c = CEILING_COLOR if (axis == 1 and side == 1) else c
            add(p, n, c, -1, texture=0.06)
    for i, obj in enumerate(OBJECTS):
        for b in obj.boxes:
            for axis in range(3):
                for side in (0, 1):
                    if axis == 1 and side == 0 and b.lo[1] <= 0.0:
                        continue  # bottom face on the floor
                    p, n = _face_samples(rng, b.lo, b.hi, axis, side, spacing)
                    add(p, n, obj.color, i)
    points = np.concatenate(pts)
    normals = np.concatenate(nrm)
    lo, hi, _ = all_boxes()
    keep = ~_inside_any(points + 2e-3 * normals, lo, hi)
    return Surface(
        points=points[keep],
        normals=normals[keep],
        colors=np.concatenate(col)[keep],
        object_index=np.concatenate(oid)[keep],
    )


# ---------------------------------------------------------------- exact visibility


def segment_hits_box(o: np.ndarray, t: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Slab test: does the segment o -> t (P, 3) pass through the box [lo, hi]?"""
    d = t - o
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(np.abs(d) > 1e-12, 1.0 / d, np.inf)
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
    t1 = np.where(np.isnan(t1), -np.inf, t1)
    t2 = np.where(np.isnan(t2), np.inf, t2)
    # Axis with zero direction: inside the slab -> (-inf, inf), outside -> empty.
    zero = np.abs(d) <= 1e-12
    inside = (o >= lo) & (o <= hi)
    tmin_ax = np.where(zero, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
    tmax_ax = np.where(zero, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
    tmin = tmin_ax.max(axis=1)
    tmax = tmax_ax.min(axis=1)
    return (tmax >= np.maximum(tmin, 0.0)) & (tmin <= 1.0)


def visible_mask(
    eye: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """Exact visibility of surface samples from one eye: front-facing and not blocked by
    any box (boxes shrunk by eps, samples lifted 2 eps along their normal)."""
    eye = np.asarray(eye, float)
    front = np.einsum("ij,ij->i", normals, eye - points) > 0
    out = front.copy()
    idx = np.nonzero(front)[0]
    if idx.size == 0:
        return out
    tgt = points[idx] + 2 * eps * normals[idx]
    o = np.broadcast_to(eye, tgt.shape)
    blocked = np.zeros(idx.size, dtype=bool)
    for bl, bh in zip(lo + eps, hi - eps, strict=True):
        live = ~blocked
        blocked[live] = segment_hits_box(o[live], tgt[live], bl, bh)
    out[idx] = ~blocked
    return out


def exact_view_counts(surf: Surface, cameras: list[Camera]) -> np.ndarray:
    """Number of cameras that see each sample (inside the image, front-facing, unblocked)."""
    lo, hi, _ = all_boxes()
    counts = np.zeros(len(surf.points), dtype=np.int32)
    for cam in cameras:
        inf = cam.in_frustum(surf.points)
        idx = np.nonzero(inf)[0]
        vis = visible_mask(cam.center, surf.points[idx], surf.normals[idx], lo, hi)
        counts[idx[vis]] += 1
    return counts


def walkable_eyes(cfg: SynthConfig) -> np.ndarray:
    """Analytic walkable standing positions (x, z) on a grid, away from walls and objects."""
    lo = ROOM_LO[[0, 2]] + cfg.wall_clearance
    hi = ROOM_HI[[0, 2]] - cfg.wall_clearance
    xs = np.arange(lo[0], hi[0] + 1e-9, cfg.vr_spacing)
    zs = np.arange(lo[1], hi[1] + 1e-9, cfg.vr_spacing)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    xz = np.stack([gx.ravel(), gz.ravel()], axis=1)
    ok = np.ones(len(xz), dtype=bool)
    for obj in OBJECTS:
        blo, bhi = object_bounds(obj)
        c = cfg.object_clearance
        ok &= ~(
            (xz[:, 0] > blo[0] - c)
            & (xz[:, 0] < bhi[0] + c)
            & (xz[:, 1] > blo[2] - c)
            & (xz[:, 1] < bhi[2] + c)
        )
    return xz[ok]


def vr_visibility(surf: Surface, cfg: SynthConfig, max_dist: float = 8.0) -> np.ndarray:
    """Samples visible from at least one walkable eye (any direction: the head turns)."""
    lo, hi, _ = all_boxes()
    xz = walkable_eyes(cfg)
    seen = np.zeros(len(surf.points), dtype=bool)
    for h in cfg.vr_eye_heights:
        for x, z in xz:
            todo = np.nonzero(~seen)[0]
            if todo.size == 0:
                return seen
            eye = np.array([x, h, z])
            near = np.linalg.norm(surf.points[todo] - eye, axis=1) <= max_dist
            todo = todo[near]
            seen[todo[visible_mask(eye, surf.points[todo], surf.normals[todo], lo, hi)]] = True
    return seen


# ---------------------------------------------------------------- capture


def orbit_cameras(cfg: SynthConfig) -> list[Camera]:
    """Phone loop around the coffee table at hand height, pointing across the room with a
    slow up/down and left/right sweep."""
    cams = []
    cx, cz = cfg.loop_center
    rx, rz = cfg.loop_radii
    for i in range(cfg.n_views):
        th = 2 * np.pi * i / cfg.n_views
        pos = np.array(
            [cx + rx * np.cos(th), cfg.hold_height + 0.08 * np.sin(5 * th), cz + rz * np.sin(th)]
        )
        # Inward direction (towards the far side of the room) as yaw; yaw 0 looks at -Z.
        inward = np.array([cx, cz]) - pos[[0, 2]]
        yaw = np.degrees(np.arctan2(inward[0], -inward[1])) + 35.0 * np.sin(3 * th)
        pitch = -12.0 + 20.0 * np.sin(4 * th + 0.5)
        cam = camera_from_yaw_pitch(pos, yaw, pitch, cfg.width, cfg.height, cfg.hfov_deg)
        cams.append(Camera(f"{i:06d}.jpg", cam.width, cam.height, cam.K, cam.R, cam.t))
    return cams


def splats_from_samples(surf: Surface, views: np.ndarray, cfg: SynthConfig):
    """Imitate a gsplat result: returns (keep index into samples, means, scales, opacity)."""
    rng = np.random.default_rng(cfg.seed + 1)
    n = len(surf.points)
    never = views == 0
    keep = ~(never & (rng.random(n) < cfg.hole_prob))
    idx = np.nonzero(keep)[0]
    noise = np.where(views[idx] == 0, 0.04, np.where(views[idx] <= 2, 0.01, 0.002))
    means = surf.points[idx] + rng.normal(size=(idx.size, 3)) * noise[:, None]
    sigma = 0.6 * cfg.spacing
    scales = np.full((idx.size, 3), sigma)
    flat_axis = np.argmax(np.abs(surf.normals[idx]), axis=1)
    scales[np.arange(idx.size), flat_axis] = 0.2 * sigma
    scales[views[idx] == 0] *= 1.5
    opacity = np.where(views[idx] == 0, rng.uniform(0.3, 0.6, idx.size), 0.95)
    return idx, means, scales, opacity


@dataclass
class SyntheticCapture:
    cameras: list[Camera]  # world frame
    surface: Surface
    views: np.ndarray  # exact view count per sample
    vr_visible: np.ndarray  # per sample
    region: np.ndarray  # per sample, index into REGIONS or -1
    splat_sample: np.ndarray  # per Gaussian, index of its source sample
    means: np.ndarray  # per Gaussian, world frame
    scales: np.ndarray
    opacity: np.ndarray
    world_from_recon: np.ndarray
    timings: dict[str, float]


def sample_regions(points: np.ndarray) -> np.ndarray:
    out = np.full(len(points), -1)
    for i, (lo, hi) in enumerate(REGIONS.values()):
        inside = np.all((points >= np.asarray(lo) - 1e-6) & (points <= np.asarray(hi) + 1e-6), 1)
        out[inside & (out < 0)] = i
    return out


def make_capture(cfg: SynthConfig) -> SyntheticCapture:
    t0 = time.perf_counter()
    surf = sample_surfaces(cfg.spacing, cfg.seed)
    cams = orbit_cameras(cfg)
    t1 = time.perf_counter()
    views = exact_view_counts(surf, cams)
    t2 = time.perf_counter()
    vr = vr_visibility(surf, cfg)
    t3 = time.perf_counter()
    idx, means, scales, opacity = splats_from_samples(surf, views, cfg)
    # Recon frame = first camera's OpenCV frame (what MapAnything returns before alignment).
    c0 = cams[0]
    world_from_recon = np.eye(4)
    world_from_recon[:3, :3] = c0.R.T
    world_from_recon[:3, 3] = c0.center
    timings = {
        "sample_s": round(t1 - t0, 2),
        "exact_views_s": round(t2 - t1, 2),
        "vr_visibility_s": round(t3 - t2, 2),
    }
    return SyntheticCapture(
        cameras=cams,
        surface=surf,
        views=views,
        vr_visible=vr,
        region=sample_regions(surf.points),
        splat_sample=idx,
        means=means,
        scales=scales,
        opacity=opacity,
        world_from_recon=world_from_recon,
        timings=timings,
    )


# ---------------------------------------------------------------- writing the scene folder


def _obj_ply_name(obj_id: str) -> str:
    return "objects/obj_" + obj_id.replace(" ", "_").replace("#", "_") + ".ply"


def write_scene(cap: SyntheticCapture, out: Path, cfg: SynthConfig, scene_id: str) -> None:
    out = Path(out)
    recon_from_world = np.linalg.inv(cap.world_from_recon)
    rgb = cap.surface.colors[cap.splat_sample]
    means_recon = transform_points(recon_from_world, cap.means)
    write_splats(out / "splat" / "scene.ply", means_recon, rgb, cap.scales, cap.opacity)

    cams_recon = [c.in_frame(recon_from_world) for c in cap.cameras]
    seen = cap.views[cap.splat_sample] > 0
    write_model(
        out / "recon" / "sparse" / "0", cams_recon, means_recon[seen][::50], rgb[seen][::50]
    )

    obj_of = cap.surface.object_index[cap.splat_sample]
    objects = []
    for i, obj in enumerate(OBJECTS):
        sel = obj_of == i
        ply = _obj_ply_name(obj.id)
        write_splats(out / ply, means_recon[sel], rgb[sel], cap.scales[sel], cap.opacity[sel])
        lo, hi = object_bounds(obj)
        objects.append(
            SceneObject(
                id=obj.id,
                label=obj.label,
                ply=ply,
                bbox_min=tuple(float(v) for v in lo),
                bbox_max=tuple(float(v) for v in hi),
            )
        )
    bg = obj_of < 0
    write_splats(
        out / "objects" / "background.ply",
        means_recon[bg],
        rgb[bg],
        cap.scales[bg],
        cap.opacity[bg],
    )
    scene = Scene(
        scene_id=scene_id,
        world_from_recon=[[float(v) for v in row] for row in cap.world_from_recon],
        objects=objects,
    )
    scene.save(out / "scene.json")

    if cfg.write_frames:
        write_frames(cap, out)

    gt = out / "synthetic"
    gt.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        gt / "ground_truth.npz",
        points=cap.surface.points.astype(np.float32),
        normals=cap.surface.normals.astype(np.int8),
        object_index=cap.surface.object_index.astype(np.int16),
        views=cap.views.astype(np.int32),
        vr_visible=cap.vr_visible,
        region=cap.region.astype(np.int16),
        splat_sample=cap.splat_sample.astype(np.int32),
    )
    (gt / "regions.json").write_text(
        json.dumps(
            {
                "regions": {k: {"lo": v[0], "hi": v[1]} for k, v in REGIONS.items()},
                "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                "timings": cap.timings,
                "frame": "world (scene.json), metres",
            },
            indent=2,
        )
    )


def write_frames(cap: SyntheticCapture, out: Path) -> None:
    from PIL import Image

    from .render import render_points, to_uint8

    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    s = cap.surface
    meta = []
    for i, cam in enumerate(cap.cameras):
        img, _ = render_points(
            cam, s.points, s.colors, scales=np.full(len(s.points), 0.012), background=(0.1,) * 3
        )
        Image.fromarray(to_uint8(img)).save(frames / cam.name, quality=90)
        meta.append({"index": i, "source_time_s": round(i / 4.0, 3), "sharpness": 0.0})
    (out / "frames.json").write_text(json.dumps({"fps": 4.0, "frames": meta}, indent=2))
