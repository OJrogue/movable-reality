"""Write room-shaped synthetic splat scenes for renderer benchmarks (lane renderer).

    uv run python scripts/bench_make_splats.py                     # 50k ... 2M
    uv run python scripts/bench_make_splats.py --counts 300000     # one scene
    -> data/scenes/bench_<N>/objects/background.ply + scene.json

Each scene is a 6 x 5 x 2.7 m room in the world frame (+Y up, floor at y = 0): floor,
ceiling and four walls, plus clutter (furniture-sized boxes and small objects). Splats are
standard 3DGS (degree-0 SH, log scales, logit opacity, wxyz quaternions), with scales and
opacities drawn from log-normal and beta-like distributions so the scene has thin
surface splats, large soft splats and a few floaters, like a trained capture. The same
seed gives the same file; a larger N is a denser sampling of the same room, so frame-time
differences between scenes come from the splat count alone.

All splats go in one background PLY with no objects, so the viewer renders one SplatMesh
and the numbers do not depend on the lod lane's per-object trees.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from movable_reality.config import scene_dir
from movable_reality.scene import Scene

SH_C0 = 0.28209479177387814
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
DEFAULT_COUNTS = (50_000, 150_000, 300_000, 600_000, 1_000_000, 2_000_000)

# Room extent in metres: x in [-3, 3], z in [-2.5, 2.5], y in [0, 2.7].
ROOM_X, ROOM_Z, ROOM_H = 3.0, 2.5, 2.7

# Share of splats per part. Floor and walls dominate in a real room capture.
SHARES = {
    "floor": 0.22,
    "ceiling": 0.10,
    "walls": 0.33,
    "furniture": 0.27,
    "small": 0.06,
    "air": 0.02,
}

FIELDS = [
    "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
]  # fmt: skip


def quat_from_z_to(normals: np.ndarray) -> np.ndarray:
    """wxyz quaternions rotating +Z onto each unit normal (so scale_2 is the thin axis)."""
    z = np.array([0.0, 0.0, 1.0])
    n = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    dot = n @ z
    axis = np.cross(np.broadcast_to(z, n.shape), n)
    w = 1.0 + dot
    q = np.concatenate([w[:, None], axis], axis=1)
    flip = w < 1e-8  # normal = -Z: rotate 180 degrees about X
    q[flip] = [0.0, 1.0, 0.0, 0.0]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def random_quats(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def surface_scales(rng: np.random.Generator, n: int, spacing: float, thin: float) -> np.ndarray:
    """In-plane scales around the sampling spacing (log-normal), thin out of plane."""
    s = spacing * np.exp(rng.normal(0.0, 0.45, size=(n, 2)))
    t = thin * np.exp(rng.normal(0.0, 0.3, size=(n, 1)))
    return np.concatenate([s, t], axis=1)


def opacities(rng: np.random.Generator, n: int) -> np.ndarray:
    """Mostly opaque, with a long tail of faint splats (beta(5, 1.5)-like), in (0, 1)."""
    return np.clip(rng.beta(5.0, 1.5, size=n), 0.02, 0.995)


def plane_samples(
    rng: np.random.Generator, n: int, origin, u, v, normal
) -> tuple[np.ndarray, np.ndarray]:
    origin, u, v = (np.asarray(a, float) for a in (origin, u, v))
    a = rng.uniform(0.0, 1.0, size=(n, 1))
    b = rng.uniform(0.0, 1.0, size=(n, 1))
    pts = origin + a * u + b * v
    normals = np.broadcast_to(np.asarray(normal, float), pts.shape).copy()
    return pts, normals


def checker(pts: np.ndarray, size: float, axes: tuple[int, int]) -> np.ndarray:
    i = np.floor(pts[:, axes[0]] / size).astype(int) + np.floor(pts[:, axes[1]] / size).astype(int)
    return (i % 2).astype(float)


FURNITURE = [
    # (min corner, max corner, rgb) in metres: sofa, table, shelf, bed, desk, cabinet, chair.
    ((-2.6, 0.0, -2.3), (-0.6, 0.85, -1.4), (0.35, 0.40, 0.55)),
    ((-0.4, 0.0, -0.6), (0.8, 0.75, 0.4), (0.55, 0.38, 0.22)),
    ((2.5, 0.0, -2.2), (2.95, 2.0, -0.6), (0.62, 0.52, 0.40)),
    ((1.0, 0.0, 1.0), (2.9, 0.55, 2.4), (0.80, 0.78, 0.74)),
    ((-2.9, 0.0, 0.8), (-2.2, 0.78, 2.3), (0.30, 0.28, 0.27)),
    ((1.4, 0.0, -2.45), (2.3, 1.1, -2.0), (0.70, 0.66, 0.60)),
    ((-1.6, 0.0, 0.7), (-1.1, 0.95, 1.2), (0.25, 0.45, 0.35)),
]


def box_surface(rng: np.random.Generator, lo, hi, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Points on the 5 visible faces of a box (no bottom), area-weighted, with normals."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    d = hi - lo
    # faces: +x, -x, +y, +z, -z
    areas = np.array([d[1] * d[2], d[1] * d[2], d[0] * d[2], d[0] * d[1], d[0] * d[1]])
    face = rng.choice(5, size=n, p=areas / areas.sum())
    pts = rng.uniform(lo, hi, size=(n, 3))
    normals = np.zeros((n, 3))
    for f, (axis, side, sign) in enumerate(
        [(0, hi, 1), (0, lo, -1), (1, hi, 1), (2, hi, 1), (2, lo, -1)]
    ):
        m = face == f
        pts[m, axis] = side[axis]
        normals[m, axis] = sign
    return pts, normals


def build_room(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Arrays for one bench scene: xyz, rgb, opacity, scales (linear), quats (wxyz)."""
    rng = np.random.default_rng(seed)
    counts = {k: int(round(v * n)) for k, v in SHARES.items()}
    counts["floor"] += n - sum(counts.values())  # exact total

    parts: list[tuple[np.ndarray, ...]] = []
    floor_area = 4 * ROOM_X * ROOM_Z
    wall_area = 2 * ROOM_H * (2 * ROOM_X + 2 * ROOM_Z)

    def surface(pts, normals, rgb, area):
        k = len(pts)
        spacing = np.sqrt(area / max(k, 1)) * 0.6
        scales = surface_scales(rng, k, spacing, thin=0.002)
        quats = quat_from_z_to(normals)
        rgb = np.clip(rgb + rng.normal(0, 0.03, size=rgb.shape), 0, 1)
        parts.append((pts, rgb, opacities(rng, k), scales, quats))

    # floor: wooden planks, alternating tone every 0.2 m along x
    k = counts["floor"]
    pts, nrm = plane_samples(
        rng, k, (-ROOM_X, 0, -ROOM_Z), (2 * ROOM_X, 0, 0), (0, 0, 2 * ROOM_Z), (0, 1, 0)
    )
    tone = np.floor(pts[:, 0] / 0.2) % 2
    rgb = np.where(tone[:, None] == 1, [0.55, 0.42, 0.30], [0.48, 0.36, 0.25])
    surface(pts, nrm, rgb.astype(float), floor_area)

    # ceiling: plain white with a slight checker so motion is visible
    k = counts["ceiling"]
    pts, nrm = plane_samples(
        rng, k, (-ROOM_X, ROOM_H, -ROOM_Z), (2 * ROOM_X, 0, 0), (0, 0, 2 * ROOM_Z), (0, -1, 0)
    )
    c = checker(pts, 0.6, (0, 2))[:, None]
    surface(pts, nrm, 0.88 + 0.04 * c * np.ones((k, 3)), floor_area)

    # walls: four colours, checker texture
    walls = [
        ((-ROOM_X, 0, -ROOM_Z), (2 * ROOM_X, 0, 0), (0, ROOM_H, 0), (0, 0, 1), (0.80, 0.76, 0.68)),
        ((-ROOM_X, 0, ROOM_Z), (2 * ROOM_X, 0, 0), (0, ROOM_H, 0), (0, 0, -1), (0.66, 0.74, 0.80)),
        ((-ROOM_X, 0, -ROOM_Z), (0, 0, 2 * ROOM_Z), (0, ROOM_H, 0), (1, 0, 0), (0.78, 0.70, 0.70)),
        ((ROOM_X, 0, -ROOM_Z), (0, 0, 2 * ROOM_Z), (0, ROOM_H, 0), (-1, 0, 0), (0.70, 0.78, 0.70)),
    ]
    lengths = np.array([2 * ROOM_X, 2 * ROOM_X, 2 * ROOM_Z, 2 * ROOM_Z])
    split = np.floor(counts["walls"] * lengths / lengths.sum()).astype(int)
    split[0] += counts["walls"] - split.sum()
    for (o, u, v, nrm_w, col), k in zip(walls, split, strict=True):
        pts, nrm = plane_samples(rng, int(k), o, u, v, nrm_w)
        axes = (0, 1) if nrm_w[2] != 0 else (2, 1)
        c = checker(pts, 0.5, axes)[:, None]
        rgb = np.asarray(col) * (0.92 + 0.08 * c)
        surface(pts, nrm, rgb, wall_area / 4)

    # furniture: boxes, area-weighted share of the furniture budget
    def box_area(lo, hi):
        d = np.subtract(hi, lo)
        return 2 * d[1] * d[2] + d[0] * d[2] + 2 * d[0] * d[1]

    areas = np.array([box_area(lo, hi) for lo, hi, _ in FURNITURE])
    split = np.floor(counts["furniture"] * areas / areas.sum()).astype(int)
    split[0] += counts["furniture"] - split.sum()
    for (lo, hi, col), k, area in zip(FURNITURE, split, areas, strict=True):
        pts, nrm = box_surface(rng, lo, hi, int(k))
        rgb = np.tile(col, (int(k), 1)).astype(float)
        surface(pts, nrm, rgb, area)

    # small clutter: many small blobs on the furniture tops and the floor, random colours,
    # random orientation, scales 1-4 cm
    k = counts["small"]
    if k:
        on_floor = rng.uniform(
            [-ROOM_X + 0.3, 0.02, -ROOM_Z + 0.3], [ROOM_X - 0.3, 0.1, ROOM_Z - 0.3], (40, 3)
        )
        on_tops = np.array(
            [[(lo[0] + hi[0]) / 2, hi[1] + 0.05, (lo[2] + hi[2]) / 2] for lo, hi, _ in FURNITURE]
        )
        centres = np.concatenate([on_floor, on_tops])
        which = rng.integers(0, len(centres), k)
        pts = centres[which] + rng.normal(0, 0.06, size=(k, 3)) * [1, 0.6, 1]
        pts[:, 1] = np.maximum(pts[:, 1], 0.005)
        palette = rng.uniform(0.15, 0.95, size=(len(centres), 3))
        rgb = np.clip(palette[which] + rng.normal(0, 0.05, size=(k, 3)), 0, 1)
        scales = 0.012 * np.exp(rng.normal(0, 0.5, size=(k, 3)))
        parts.append((pts, rgb, opacities(rng, k), scales, random_quats(rng, k)))

    # air: floaters and large soft splats, the kind training leaves behind
    k = counts["air"]
    if k:
        lo = [-ROOM_X + 0.1, 0.1, -ROOM_Z + 0.1]
        hi = [ROOM_X - 0.1, ROOM_H - 0.1, ROOM_Z - 0.1]
        pts = rng.uniform(lo, hi, size=(k, 3))
        rgb = rng.uniform(0.3, 0.8, size=(k, 3))
        scales = 0.05 * np.exp(rng.normal(0, 0.6, size=(k, 3)))
        alpha = np.clip(rng.beta(1.2, 8.0, size=k), 0.02, 0.6)
        parts.append((pts, rgb, alpha, scales, random_quats(rng, k)))

    xyz, rgb, alpha, scales, quats = (np.concatenate(p) for p in zip(*parts, strict=True))
    order = rng.permutation(len(xyz))  # no spatial order in the file, as after training
    return {
        "xyz": xyz[order].astype(np.float32),
        "rgb": rgb[order].astype(np.float32),
        "opacity": alpha[order].astype(np.float32),
        "scales": scales[order].astype(np.float32),
        "quats": quats[order].astype(np.float32),
    }


def write_ply(path: Path, room: dict[str, np.ndarray]) -> None:
    """Standard binary little-endian 3DGS PLY, degree-0 SH."""
    n = len(room["xyz"])
    data = np.zeros(n, dtype=[(f, "f4") for f in FIELDS])
    data["x"], data["y"], data["z"] = room["xyz"].T
    dc = (room["rgb"] - 0.5) / SH_C0
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = dc.T
    a = np.clip(room["opacity"], 1e-4, 1 - 1e-4)
    data["opacity"] = np.log(a / (1 - a))
    logs = np.log(np.maximum(room["scales"], 1e-7))
    data["scale_0"], data["scale_1"], data["scale_2"] = logs.T
    data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = room["quats"].T
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".ply.tmp")
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(tmp))
    tmp.replace(path)


def make_scene(n: int, out_dir: Path, seed: int = 0, force: bool = False) -> bool:
    """Write bench_<n>; returns False when it already exists and force is off."""
    ply = out_dir / "objects" / "background.ply"
    manifest = out_dir / "scene.json"
    if ply.exists() and manifest.exists() and not force:
        return False
    write_ply(ply, build_room(n, seed))
    Scene(scene_id=out_dir.name, world_from_recon=IDENTITY, objects=[]).save(manifest)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--counts", type=int, nargs="+", default=list(DEFAULT_COUNTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="rewrite scenes that exist")
    args = ap.parse_args()
    for n in args.counts:
        out = scene_dir(f"bench_{n}")
        if make_scene(n, out, seed=args.seed, force=args.force):
            size = (out / "objects" / "background.ply").stat().st_size / 1e6
            print(f"wrote {out} ({n:,} splats, {size:.0f} MB)")
        else:
            print(f"skip {out} (exists; --force to rewrite)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
