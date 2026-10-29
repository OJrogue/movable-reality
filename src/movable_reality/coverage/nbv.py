"""Walkable standing positions, VR exposure of voxels, and next-best-view ranking.

World frame of scene.json: floor at y = 0, +Y up. The user stands on the floor; the phone
is held at eye height (1.6 m) for a top-up video.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..scene import NextBestView
from .gaps import FREE, OCCUPIED, UNKNOWN
from .voxels import VoxelGrid, fibonacci_sphere, ray_voxels


@dataclass(frozen=True)
class WalkParams:
    spacing: float = 0.25  # standing grid, metres
    clearance: float = 0.3  # from any obstacle voxel, metres
    min_height: float = 0.1  # obstacles count between these heights
    max_height: float = 1.9
    eye_height: float = 1.6
    exposure_heights: tuple[float, ...] = (1.6, 1.0)  # standing and crouching
    exposure_spacing: float = 0.5  # coarser grid for the exposure pass


@dataclass(frozen=True)
class NbvParams:
    max_targets: int = 400  # per gap
    min_dist: float = 0.4
    max_dist: float = 4.0
    cone_half_deg: float = 22.0  # phone field of view around the aim point (landscape)
    top_k: int = 3
    min_separation: float = 0.75  # metres between returned standing positions
    seed: int = 0


def walkable_positions(grid: VoxelGrid, state: np.ndarray, p: WalkParams) -> np.ndarray:
    """Standing positions (M, 2) as (x, z) on a regular grid where no occupied voxel
    between min_height and max_height lies within `clearance`."""
    st = state.reshape(grid.dims)
    ys = grid.centres(np.arange(grid.dims[1]) * grid.dims[2])[:, 1]  # y of each layer
    band = (ys >= p.min_height) & (ys <= p.max_height)
    obstacle = (st[:, band, :] == OCCUPIED).any(axis=1)  # (nx, nz)
    r = int(np.ceil(p.clearance / grid.size))
    blocked = obstacle.copy()
    nx, nz = obstacle.shape
    for dx in range(-r, r + 1):
        for dz in range(-r, r + 1):
            if dx * dx + dz * dz > r * r:
                continue
            src = (slice(max(0, -dx), nx - max(0, dx)), slice(max(0, -dz), nz - max(0, dz)))
            dst = (slice(max(0, dx), nx - max(0, -dx)), slice(max(0, dz), nz - max(0, -dz)))
            blocked[dst] |= obstacle[src]
    # The room interior: columns between the outermost obstacles in x and z.
    lo = grid.origin[[0, 2]] + (r + 0.5) * grid.size
    hi = grid.origin[[0, 2]] + (np.asarray(grid.dims)[[0, 2]] - r - 0.5) * grid.size
    xs = np.arange(lo[0], hi[0] + 1e-9, p.spacing)
    zs = np.arange(lo[1], hi[1] + 1e-9, p.spacing)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    xz = np.stack([gx.ravel(), gz.ravel()], axis=1)
    pts = np.stack([xz[:, 0], np.full(len(xz), p.eye_height), xz[:, 1]], axis=1)
    ijk, inside = grid.ijk(pts)
    ok = inside.copy()
    ok[inside] = ~blocked[ijk[inside, 0], ijk[inside, 2]]
    # Do not stand inside never-observed space at eye height (outside the captured room).
    flat = grid.flat(np.clip(ijk, 0, np.asarray(grid.dims) - 1))
    ok &= state[flat] != UNKNOWN
    return xz[ok]


def eyes_at(xz: np.ndarray, heights: tuple[float, ...]) -> np.ndarray:
    return np.concatenate([np.stack([xz[:, 0], np.full(len(xz), h), xz[:, 1]], 1) for h in heights])


def exposure(
    grid: VoxelGrid,
    solid: np.ndarray,
    mark: np.ndarray,
    eyes: np.ndarray,
    n_dirs: int = 6000,
    max_dist: float = 8.0,
    skip: float = 0.15,
    chunk_rays: int = 2000,
) -> np.ndarray:
    """Number of eyes (n,) that see each voxel. A ray from the eye passes through
    non-solid voxels and stops at the first `solid` one (included). Only voxels in `mark`
    are counted (for example: everything not seen as free space).

    Typical use: solid = occupied voxels, mark = not free. Rays then pass through
    never-observed empty space (a hole in the capture is see-through in VR) and count it
    as exposed, until they reach a surface."""
    hits = np.zeros(grid.n, dtype=np.uint16)
    dirs = fibonacci_sphere(n_dirs)
    step = 0.5 * grid.size
    for eye in np.asarray(eyes, dtype=np.float32):
        seen = np.zeros(grid.n, dtype=bool)
        for s in range(0, len(dirs), chunk_rays):
            flat, ok = ray_voxels(grid, eye, dirs[s : s + chunk_rays], max_dist, step, skip)
            ok &= np.cumsum(~ok, axis=1) == 0  # leaving the grid ends a ray
            blocked = (ok & solid[flat]).astype(np.int8)
            upto = (np.cumsum(blocked, axis=1) - blocked) == 0  # samples up to the first hit
            seen[flat[ok & upto & mark[flat]]] = True
        hits += seen
    return hits


def gap_targets(
    grid: VoxelGrid,
    labels: np.ndarray,
    gap: int,
    means: np.ndarray,
    gauss_voxel: np.ndarray,
    gauss_cov: np.ndarray,
    state: np.ndarray,
    cov_threshold: float,
    max_targets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Points to observe for one gap: its low-coverage Gaussians plus the centres of its
    unknown voxels, subsampled to `max_targets`."""
    in_gap = labels == gap
    g = (gauss_voxel >= 0) & (gauss_cov < cov_threshold)
    g[g] = in_gap[gauss_voxel[g]]
    pts = [means[g]]
    unk = np.nonzero(in_gap & (state == UNKNOWN))[0]
    pts.append(grid.centres(unk))
    t = np.concatenate(pts).astype(np.float32)
    if len(t) > max_targets:
        t = t[rng.choice(len(t), max_targets, replace=False)]
    return t


def line_of_sight(
    grid: VoxelGrid,
    blocking: np.ndarray,
    eye: np.ndarray,
    targets: np.ndarray,
    stop_short: float,
) -> np.ndarray:
    """Bool (T,): no blocking voxel between `eye` and each target (the last `stop_short`
    metres before the target are not tested, so its own voxel does not block)."""
    d = targets - eye
    dist = np.linalg.norm(d, axis=1)
    dirs = d / np.maximum(dist[:, None], 1e-6)
    flat, ok = ray_voxels(grid, eye, dirs, dist - stop_short, 0.5 * grid.size, 0.1)
    return ~(ok & blocking[flat]).any(axis=1)


def rank_views(
    grid: VoxelGrid,
    state: np.ndarray,
    labels: np.ndarray,
    gap: int,
    targets: np.ndarray,
    candidates_xz: np.ndarray,
    walk: WalkParams,
    p: NbvParams,
) -> list[NextBestView]:
    """Top standing viewpoints for one gap, by how many of its targets a phone held at eye
    height and aimed at the gap's target centroid would see."""
    if len(targets) == 0 or len(candidates_xz) == 0:
        return []
    aim = targets.mean(axis=0)
    # Occupied voxels block; unknown voxels block unless they belong to this gap (the
    # unseen inside of the gap is probably air).
    blocking = (state == OCCUPIED) | ((state == UNKNOWN) & (labels != gap))
    eyes = eyes_at(candidates_xz, (walk.eye_height,)).astype(np.float32)
    cos_cone = np.cos(np.radians(p.cone_half_deg))
    scored = []
    for eye in eyes:
        d = targets - eye
        dist = np.linalg.norm(d, axis=1)
        axis = aim - eye
        axis /= max(float(np.linalg.norm(axis)), 1e-6)
        in_cone = (d @ axis) / np.maximum(dist, 1e-6) >= cos_cone
        ok = in_cone & (dist >= p.min_dist) & (dist <= p.max_dist)
        if not ok.any():
            continue
        vis = np.zeros(len(targets), dtype=bool)
        vis[ok] = line_of_sight(grid, blocking, eye, targets[ok], 1.0 * grid.size)
        n = int(vis.sum())
        if n:
            # Prefer closer views at equal count (more pixels on the gap).
            scored.append((n, -float(np.linalg.norm(aim - eye)), eye))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    out: list[NextBestView] = []
    for n, _, eye in scored:
        if any(np.linalg.norm(np.subtract(eye, v.eye)[[0, 2]]) < p.min_separation for v in out):
            continue
        out.append(
            NextBestView(
                eye=tuple(round(float(v), 3) for v in eye),  # type: ignore[arg-type]
                target=tuple(round(float(v), 3) for v in aim),  # type: ignore[arg-type]
                visible=n,
                gain=round(n / len(targets), 3),
            )
        )
        if len(out) >= p.top_k:
            break
    return out


__all__ = [
    "FREE",
    "NbvParams",
    "WalkParams",
    "exposure",
    "eyes_at",
    "gap_targets",
    "line_of_sight",
    "rank_views",
    "walkable_positions",
]
