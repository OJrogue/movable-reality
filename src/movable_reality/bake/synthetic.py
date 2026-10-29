"""Synthetic test room for the bake lane, laid out like a real scene folder.

A 4 x 5 m room, 2.5 m high, with flat, semi-transparent floor and ceiling Gaussians that
carry a visible pattern (floor: 0.5 m checker of two wood tones and a red rug; ceiling:
0.6 m panel grid and a round lamp), noise normal to the plane, mislabelled outliers, a
floor island seen through a doorway, walls, and box objects standing on the floor. No floor
Gaussians lie under the sofa (the phone never saw there), so the bake has a hole to fill.

Everything is written in the reconstruction frame (OpenCV) with a known world_from_recon
(tilted, rotated and offset like a real capture), so the bake must convert both ways.

    uv run python scripts/bake_planes.py --scene bakedemo --synthetic
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from movable_reality.scene import Scene, SceneObject

from .gaussians import SH_C0, Gaussians, rotmat_to_quat, write_gaussians

ROOM_X = (-2.0, 2.0)
ROOM_Z = (-2.5, 2.5)
ROOM_H = 2.5
CHECKER = 0.5
FLOOR_A = np.array([0.62, 0.45, 0.28])  # light wood
FLOOR_B = np.array([0.38, 0.25, 0.14])  # dark wood
RUG = np.array([0.70, 0.12, 0.10])
RUG_X, RUG_Z = (-1.8, -0.4), (-1.3, -0.2)
CEIL = np.array([0.90, 0.90, 0.88])
CEIL_LINE = np.array([0.55, 0.55, 0.55])
LAMP = np.array([1.0, 0.95, 0.75])
LAMP_C, LAMP_R = (0.0, 0.0), 0.35
FLOOR_ID, CEILING_ID = 0, 1


def floor_colour(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Ground-truth floor colour at world (x, z)."""
    cell = (np.floor(x / CHECKER) + np.floor(z / CHECKER)).astype(int) % 2
    rgb = np.where(cell[:, None] == 0, FLOOR_A, FLOOR_B)
    rug = (x > RUG_X[0]) & (x < RUG_X[1]) & (z > RUG_Z[0]) & (z < RUG_Z[1])
    rgb[rug] = RUG
    return rgb


def ceiling_colour(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    rgb = np.tile(CEIL, (len(x), 1))
    fx, fz = np.mod(x, 0.6), np.mod(z, 0.6)
    line = (fx < 0.03) | (fz < 0.03)
    rgb[line] = CEIL_LINE
    lamp = (x - LAMP_C[0]) ** 2 + (z - LAMP_C[1]) ** 2 < LAMP_R**2
    rgb[lamp] = LAMP
    return rgb


def _rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def default_world_from_recon() -> np.ndarray:
    """First camera 1.45 m above the floor, pitched 8 degrees down, yawed 35 degrees."""
    flip = np.diag([1.0, -1.0, -1.0])  # OpenCV (+Y down, +Z forward) to +Y up
    m = np.eye(4)
    m[:3, :3] = _rot_y(35.0) @ _rot_x(-8.0) @ flip
    m[:3, 3] = [0.4, 1.45, -0.7]
    return m


@dataclass
class Box:
    id: str
    label: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    colour: tuple[float, float, float]


# The sofa stands against the far wall, in the viewer's default view (camera at the origin
# looking down -Z), so moving it shows the unseen floor it stood on.
BOXES = [
    Box("sofa#1", "sofa", (-0.3, 0.0, -2.4), (1.5, 0.85, -1.5), (0.25, 0.35, 0.55)),
    Box("table#1", "table", (-0.6, 0.0, 0.6), (0.6, 0.75, 1.4), (0.45, 0.3, 0.15)),
    Box("chair#1", "chair", (-1.5, 0.0, -1.0), (-1.0, 0.9, -0.5), (0.2, 0.55, 0.3)),
]


@dataclass
class GroundTruth:
    world_from_recon: np.ndarray
    floor_normal: np.ndarray  # into the room
    floor_offset: float
    ceiling_normal: np.ndarray
    ceiling_offset: float
    floor_ids: list[int]
    ceiling_ids: list[int]
    counts: dict[str, int]


def _flat_layer(rng, xs, zs, y, spacing, normal_sigma, colour_fn, opacity):
    gx, gz = np.meshgrid(np.arange(*xs, spacing), np.arange(*zs, spacing), indexing="ij")
    x = gx.ravel() + rng.uniform(-0.4, 0.4, gx.size) * spacing
    z = gz.ravel() + rng.uniform(-0.4, 0.4, gz.size) * spacing
    yy = y + rng.normal(0.0, normal_sigma, x.size)
    pts = np.stack([x, yy, z], 1)
    rgb = np.clip(colour_fn(x, z) + rng.normal(0, 0.02, (x.size, 3)), 0, 1)
    n = len(pts)
    scales = np.stack(
        [
            spacing * rng.uniform(0.5, 0.9, n),
            np.full(n, 0.0015),
            spacing * rng.uniform(0.5, 0.9, n),
        ],
        1,
    )
    rots = np.stack([_rot_y(a) for a in rng.uniform(0, 360, n)])
    op = rng.uniform(*opacity, n)
    return pts, rots, scales, op, rgb


def _box_surface(rng, box: Box, n: int):
    lo, hi = np.asarray(box.lo), np.asarray(box.hi)
    pts = rng.uniform(lo, hi, size=(n, 3))
    axis = rng.choice([0, 1, 2], n, p=[0.35, 0.3, 0.35])
    side = rng.integers(0, 2, n)
    side[axis == 1] = 1  # no bottom face: boxes stand on the floor
    pts[np.arange(n), axis] = np.where(side == 1, hi[axis], lo[axis])
    rgb = np.clip(np.asarray(box.colour) + rng.normal(0, 0.03, (n, 3)), 0, 1)
    rots = np.tile(np.eye(3), (n, 1, 1))
    scales = np.full((n, 3), 0.012)
    return pts, rots, scales, np.full(n, 0.9), rgb


def _wall(rng, a, b, spacing):
    """Vertical wall from a to b (x, z), floor to ceiling."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    t, h = np.meshgrid(np.arange(0, length, spacing), np.arange(0, ROOM_H, spacing))
    t, h = t.ravel(), h.ravel()
    d = (b - a) / length
    x, z = a[0] + d[0] * t, a[1] + d[1] * t
    pts = np.stack([x, h, z], 1) + rng.normal(0, 0.002, (t.size, 3))
    rgb = np.clip(np.array([0.82, 0.8, 0.74]) + rng.normal(0, 0.02, (t.size, 3)), 0, 1)
    # Flat axis along the wall normal.
    yaw = math.degrees(math.atan2(d[0], d[1]))
    r = _rot_y(yaw) @ np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    rots = np.tile(r, (t.size, 1, 1))
    scales = np.tile([0.0015, spacing * 0.7, spacing * 0.7], (t.size, 1))
    return pts, rots, scales, np.full(t.size, 0.85), rgb


def make_room(
    scene_dir: Path,
    *,
    seed: int = 0,
    floor_spacing: float = 0.025,
    ceiling_spacing: float = 0.04,
    wall_spacing: float = 0.05,
    outlier_fraction: float = 0.03,
    world_from_recon: np.ndarray | None = None,
    scene_id: str | None = None,
) -> GroundTruth:
    rng = np.random.default_rng(seed)
    w = default_world_from_recon() if world_from_recon is None else np.asarray(world_from_recon)
    recon_from_world = np.linalg.inv(w)
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []

    # Floor, without the area under the sofa (never observed).
    f = _flat_layer(rng, ROOM_X, ROOM_Z, 0.0, floor_spacing, 0.003, floor_colour, (0.4, 0.95))
    sofa = BOXES[0]
    under = (
        (f[0][:, 0] > sofa.lo[0] + 0.05)
        & (f[0][:, 0] < sofa.hi[0] - 0.05)
        & (f[0][:, 2] > sofa.lo[2] + 0.05)
        & (f[0][:, 2] < sofa.hi[2] - 0.05)
    )
    groups.append((*(a[~under] for a in f), FLOOR_ID))  # type: ignore[arg-type]

    # Floor island seen through a doorway, beyond the +x wall (same plane, labelled floor).
    isl = _flat_layer(
        rng, (2.4, 3.0), (-0.5, 0.4), 0.0, floor_spacing, 0.003, floor_colour, (0.4, 0.95)
    )
    groups.append((*isl, FLOOR_ID))

    # Ceiling.
    c = _flat_layer(rng, ROOM_X, ROOM_Z, ROOM_H, ceiling_spacing, 0.004, ceiling_colour, (0.3, 0.9))
    groups.append((*c, CEILING_ID))

    # Mislabelled outliers: random points in the room volume carrying the floor or ceiling id.
    for gid, n_lab in ((FLOOR_ID, len(groups[0][0])), (CEILING_ID, len(c[0]))):
        n = int(outlier_fraction * n_lab)
        pts = rng.uniform([ROOM_X[0], 0.1, ROOM_Z[0]], [ROOM_X[1], ROOM_H - 0.1, ROOM_Z[1]], (n, 3))
        rots = np.tile(np.eye(3), (n, 1, 1))
        groups.append(
            (pts, rots, np.full((n, 3), 0.01), np.full(n, 0.6), rng.uniform(0, 1, (n, 3)), gid)
        )

    # Walls (background).
    corners = [
        (ROOM_X[0], ROOM_Z[0]),
        (ROOM_X[1], ROOM_Z[0]),
        (ROOM_X[1], ROOM_Z[1]),
        (ROOM_X[0], ROOM_Z[1]),
    ]
    for i in range(4):
        groups.append((*_wall(rng, corners[i], corners[(i + 1) % 4], wall_spacing), -1))

    # Objects standing on the floor.
    names = {FLOOR_ID: "floor#1", CEILING_ID: "ceiling#1"}
    for k, box in enumerate(BOXES, start=2):
        groups.append((*_box_surface(rng, box, 3000), k))
        names[k] = box.id

    pts = np.concatenate([g[0] for g in groups])
    rots = np.concatenate([g[1] for g in groups])
    scales = np.concatenate([g[2] for g in groups])
    op = np.concatenate([g[3] for g in groups])
    rgb = np.concatenate([g[4] for g in groups])
    ids = np.concatenate([np.full(len(g[0]), g[5], dtype=np.int32) for g in groups])

    # To the reconstruction frame.
    r_rw = recon_from_world[:3, :3]
    pts_r = pts @ r_rw.T + recon_from_world[:3, 3]
    quats = rotmat_to_quat(np.einsum("ij,njk->nik", r_rw, rots))
    g = Gaussians(
        means=pts_r,
        quats=quats,
        log_scales=np.log(scales),
        opacity_logits=np.log(op / (1 - op)),
        f_dc=(rgb - 0.5) / SH_C0,
    )

    scene_dir = Path(scene_dir)
    write_gaussians(scene_dir / "splat" / "scene.ply", g)
    (scene_dir / "ids").mkdir(parents=True, exist_ok=True)
    np.save(scene_dir / "ids" / "gaussian_ids.npy", ids)
    (scene_dir / "ids" / "labels.json").write_text(
        json.dumps({str(k): v for k, v in names.items()}, indent=2), encoding="utf-8"
    )
    is_bg = ids < 2  # floor, ceiling and walls stay in the background
    write_gaussians(scene_dir / "objects" / "background.ply", g.subset(is_bg))
    objects = []
    for k, box in enumerate(BOXES, start=2):
        sel = ids == k
        ply = f"objects/obj_{box.id.replace('#', '_')}.ply"
        write_gaussians(scene_dir / ply, g.subset(sel))
        objects.append(
            SceneObject(
                id=box.id,
                label=box.label,
                ply=ply,
                bbox_min=tuple(float(v) for v in pts[sel].min(0)),  # type: ignore[arg-type]
                bbox_max=tuple(float(v) for v in pts[sel].max(0)),  # type: ignore[arg-type]
            )
        )
    Scene(
        scene_id=scene_id or scene_dir.name,
        world_from_recon=w.tolist(),
        objects=objects,
    ).save(scene_dir / "scene.json")
    return GroundTruth(
        world_from_recon=w,
        floor_normal=np.array([0.0, 1.0, 0.0]),
        floor_offset=0.0,
        ceiling_normal=np.array([0.0, -1.0, 0.0]),
        ceiling_offset=-ROOM_H,
        floor_ids=[FLOOR_ID],
        ceiling_ids=[CEILING_ID],
        counts={
            "total": int(len(ids)),
            "floor": int((ids == FLOOR_ID).sum()),
            "ceiling": int((ids == CEILING_ID).sum()),
            "background": int(is_bg.sum()),
        },
    )
