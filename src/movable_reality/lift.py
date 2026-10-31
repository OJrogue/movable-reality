"""Stage 5: lift SAM 3.1 video masks onto the trained Gaussians (ARCHITECTURE stage 5).

For every training view, project the Gaussian centres, drop the ones hidden behind nearer
splats (a coarse depth buffer of the opaque centres stands in for gsplat's expected depth),
and vote for the label under each visible centre. A Gaussian takes a label when it wins
`min_share` of its votes over at least `min_views` views. Then floor alignment (RANSAC on
"floor" Gaussians) sets world_from_recon, floaters are cut per object, and the support rule
(D12) gives each piece of furniture the splats and small items resting on it.

Pure numpy, CPU only, so it runs and tests anywhere.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .bake.planefit import fit_plane
from .coverage.cameras import Camera

STRUCTURE = ("floor", "wall", "ceiling")  # stay in the background (D9)


@dataclass
class LiftParams:
    depth_grid: int = 240  # long side of the occlusion depth buffer, pixels
    depth_tol: float = 0.05  # relative depth tolerance (5 % of depth), covers grazing floors
    depth_abs: float = 0.03  # plus an absolute slack, metres
    depth_erode: int = 0  # min-filter radius of the depth buffer, cells
    min_share: float = 0.6
    min_views: int = 3
    opaque: float = 0.3  # opacity above which a centre may occlude others
    voxel: float = 0.04  # floater filter voxel size, metres
    support_height: float = 0.6  # how far above a top surface the support rule reaches, metres


def label_image(masks: list[tuple[int, np.ndarray, float]], shape: tuple[int, int]) -> np.ndarray:
    """Per-pixel label (int32, -1 = none) from (label, full-frame bool mask, area) triples.
    Smaller masks are painted last, so a vase on a table keeps its own pixels."""
    out = np.full(shape, -1, dtype=np.int32)
    for lab, m, _ in sorted(masks, key=lambda t: -t[2]):
        out[m] = lab
    return out


def _min_filter(img: np.ndarray, r: int) -> np.ndarray:
    """Min over a (2r+1)^2 window; the border is padded with inf, so nothing wraps around."""
    if r <= 0:
        return img
    h, w = img.shape
    pad = np.pad(img, r, constant_values=np.inf)
    out = img.copy()
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out = np.minimum(out, pad[dy : dy + h, dx : dx + w])
    return out


def visible_mask(
    cam: Camera, xyz: np.ndarray, opacity: np.ndarray, p: LiftParams
) -> tuple[np.ndarray, np.ndarray]:
    """(visible bool (N,), pixel uv (N, 2)) for one view. Occlusion: a coarse z-buffer of the
    opaque centres (optionally min-filtered, p.depth_erode)."""
    uv, z = cam.project(xyz)
    inside = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width)
    inside &= (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
    s = p.depth_grid / max(cam.width, cam.height)
    gw, gh = max(1, int(cam.width * s) + 1), max(1, int(cam.height * s) + 1)
    gx = np.clip((uv[:, 0] * s).astype(np.int64), 0, gw - 1)
    gy = np.clip((uv[:, 1] * s).astype(np.int64), 0, gh - 1)
    zbuf = np.full(gh * gw, np.inf, dtype=np.float32)
    occ = inside & (opacity > p.opaque)
    np.minimum.at(zbuf, gy[occ] * gw + gx[occ], z[occ])
    zbuf = _min_filter(zbuf.reshape(gh, gw), p.depth_erode).ravel()
    front = z <= zbuf[gy * gw + gx] * (1 + p.depth_tol) + p.depth_abs
    return inside & front, uv


def vote(
    cams: list[Camera],
    labels: list[np.ndarray | None],
    xyz: np.ndarray,
    opacity: np.ndarray,
    n_labels: int,
    p: LiftParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Votes (N, n_labels + 1; last column = unlabelled) and views seen (N,)."""
    n = len(xyz)
    votes = np.zeros((n, n_labels + 1), dtype=np.uint16)
    seen = np.zeros(n, dtype=np.uint16)
    for cam, lab in zip(cams, labels, strict=True):
        if lab is None:
            continue
        vis, uv = visible_mask(cam, xyz, opacity, p)
        idx = np.nonzero(vis)[0]
        h, w = lab.shape
        px = np.clip((uv[idx, 0] * w / cam.width).astype(np.int64), 0, w - 1)
        py = np.clip((uv[idx, 1] * h / cam.height).astype(np.int64), 0, h - 1)
        col = lab[py, px]
        col = np.where(col < 0, n_labels, col)
        np.add.at(votes, (idx, col), 1)
        seen[idx] += 1
    return votes, seen


def assign(votes: np.ndarray, seen: np.ndarray, p: LiftParams) -> np.ndarray:
    """Label per Gaussian, -1 when no label wins min_share of min_views views or more."""
    n_labels = votes.shape[1] - 1
    best = votes[:, :n_labels].argmax(axis=1) if n_labels else np.zeros(len(votes), int)
    best_votes = votes[np.arange(len(votes)), best] if n_labels else np.zeros(len(votes))
    share = best_votes / np.maximum(seen, 1)
    ok = (seen >= p.min_views) & (share >= p.min_share) & (best_votes > 0)
    return np.where(ok, best, -1).astype(np.int32)


def largest_cluster(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """Bool mask of the points in the largest 26-connected voxel cluster (floater filter)."""
    if len(xyz) == 0:
        return np.zeros(0, dtype=bool)
    ijk = np.floor((xyz - xyz.min(axis=0)) / voxel).astype(np.int64)
    keys, inv, counts = np.unique(ijk, axis=0, return_inverse=True, return_counts=True)
    index = {tuple(k): i for i, k in enumerate(keys)}
    comp = np.full(len(keys), -1, dtype=np.int64)
    sizes: list[int] = []
    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    for s in range(len(keys)):
        if comp[s] >= 0:
            continue
        lab, size = len(sizes), 0
        comp[s] = lab
        q = deque([s])
        while q:
            i = q.popleft()
            size += int(counts[i])
            k = keys[i]
            for o in offsets:
                j = index.get((k[0] + o[0], k[1] + o[1], k[2] + o[2]))
                if j is not None and comp[j] < 0:
                    comp[j] = lab
                    q.append(j)
        sizes.append(size)
    return comp[inv.ravel()] == int(np.argmax(sizes))


def trim_outliers(xyz: np.ndarray, pct: float = 2.0, margin: float = 0.1) -> np.ndarray:
    """Bool mask of the points inside the object's pct..100-pct box, widened by `margin` of its
    size per side. Stray splats that touch the object's cluster stretched blender01's boxes
    (coffee table 1.10 m tall against 0.39 m) although the 2D masks matched (IoU 0.91)."""
    if len(xyz) < 20:
        return np.ones(len(xyz), dtype=bool)
    lo, hi = np.percentile(xyz, pct, axis=0), np.percentile(xyz, 100 - pct, axis=0)
    pad = (hi - lo) * margin
    return np.all((xyz >= lo - pad) & (xyz <= hi + pad), axis=1)


def floor_alignment(
    floor_xyz: np.ndarray, cam_centres: np.ndarray, wall_xyz: np.ndarray | None = None
) -> np.ndarray:
    """4x4 world_from_recon: floor normal to +Y (towards the cameras), floor at y = 0, and a
    yaw that lines the walls (or the floor outline) up with x and z. Scale stays 1; see
    `ceiling_scale` for the metric correction."""
    interior = cam_centres.mean(axis=0)
    plane = fit_plane(floor_xyz, threshold=0.02, interior=interior)
    up = plane.normal / np.linalg.norm(plane.normal)
    # Rotation taking `up` to +Y (Rodrigues), then a yaw about +Y.
    y = np.array([0.0, 1.0, 0.0])
    v, c = np.cross(up, y), float(up @ y)
    if np.linalg.norm(v) < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    ref = wall_xyz if wall_xyz is not None and len(wall_xyz) > 100 else floor_xyz
    xz = (ref @ R.T)[:, [0, 2]]
    yaw = dominant_yaw(xz)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R = Ry @ R
    M = np.eye(4)
    M[:3, :3] = R
    M[1, 3] = -plane.offset  # n . p = offset becomes y = offset, shift to 0
    return M


def ceiling_scale(ceiling_y: np.ndarray, height: float) -> float:
    """Metric correction from one tape measurement: the factor that puts the ceiling (world
    y of the "ceiling" Gaussians, floor at y = 0) at `height` metres. MapAnything's scale
    came out 1.4x to 1.8x too small on blender01 and smoke01 (31 Oct)."""
    if len(ceiling_y) < 100:
        raise ValueError(
            f"only {len(ceiling_y)} ceiling Gaussians (ceiling not reconstructed); pass --scale"
        )
    measured = float(np.median(ceiling_y))
    if measured <= 0:
        raise ValueError(f"ceiling at y = {measured:.3f}, below the floor")
    return height / measured


def ceiling_from_histogram(
    world_y: np.ndarray, cam_y: np.ndarray, bin_m: float = 0.01
) -> np.ndarray:
    """Heights of the Gaussians in the ceiling layer when SAM labelled no ceiling (blender01):
    the densest 1 cm height bin above every camera, plus its neighbours. A ceiling is the one
    large horizontal surface up there, so it dominates the histogram."""
    above = world_y[world_y > float(np.max(cam_y)) + 0.1]
    if len(above) < 100:
        return above
    edges = np.arange(above.min(), above.max() + bin_m, bin_m)
    counts, edges = np.histogram(above, bins=edges)
    peak = int(np.argmax(counts))
    # A real ceiling is a spike. On blender01 (camera tilted 28 deg down) the ceiling was never
    # reconstructed and the heights above the cameras only tail off: refuse that case.
    away = int(round(0.1 / bin_m))
    near = [counts[i] for i in (peak - away, peak + away) if 0 <= i < len(counts)]
    if not near or counts[peak] < 2 * max(near):
        return above[:0]
    lo, hi = edges[max(peak - 2, 0)], edges[min(peak + 3, len(edges) - 1)]
    return above[(above >= lo) & (above <= hi)]


def scaled(M: np.ndarray, s: float) -> np.ndarray:
    """world_from_recon followed by a uniform scale about the world origin (floor stays y = 0)."""
    S = np.diag([s, s, s, 1.0])
    return S @ M


def dominant_yaw(xz: np.ndarray, bins: int = 90) -> float:
    """Angle (radians, modulo 90 degrees) of the main straight edges in a 2D point set,
    from the orientation that minimises the area of the points' bounding box."""
    pts = xz - xz.mean(axis=0)
    if len(pts) > 20000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 20000, replace=False)]

    def area(a: float) -> float:
        c, s = np.cos(a), np.sin(a)
        r = pts @ np.array([[c, -s], [s, c]])
        # 0.5/99.5 percentiles: tight enough that a square box wins, loose enough for floaters
        lo, hi = np.percentile(r, 0.5, axis=0), np.percentile(r, 99.5, axis=0)
        return float(np.prod(hi - lo))

    step = np.pi / 2 / bins
    best = min(np.linspace(0, np.pi / 2, bins, endpoint=False), key=area)
    best = min(np.linspace(best - step, best + step, 41), key=area)  # 0.05 deg refinement
    best = best % (np.pi / 2)
    return float(best)


def support(
    world: np.ndarray,
    ids: np.ndarray,
    objects: list[int],
    p: LiftParams,
) -> tuple[np.ndarray, dict[int, int]]:
    """Support rule (D12). Unlabelled splats, and whole small objects, that rest on top of
    a piece of furniture inside its footprint join it. Returns new ids and {child: parent}."""
    ids = ids.copy()
    tops = {}
    for o in objects:
        m = ids == o
        if m.sum() < 20:
            continue
        pts = world[m]
        lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
        tops[o] = (lo, hi)
    merged: dict[int, int] = {}
    # Whole objects first: an object whose bottom sits on another's top, inside its footprint.
    order = sorted(tops, key=lambda o: tops[o][0][1])  # lowest first
    for child in order[::-1]:
        clo, chi = tops[child]
        if clo[1] < 0.08:
            continue  # stands on the floor
        ccx, ccz = (clo[0] + chi[0]) / 2, (clo[2] + chi[2]) / 2
        for parent in order:
            if parent == child or parent in merged:
                continue
            plo, phi = tops[parent]
            on_top = abs(clo[1] - phi[1]) < 0.12
            inside = plo[0] <= ccx <= phi[0] and plo[2] <= ccz <= phi[2]
            if (
                on_top
                and inside
                and (phi[0] - plo[0]) * (phi[2] - plo[2]) > (chi[0] - clo[0]) * (chi[2] - clo[2])
            ):
                merged[child] = parent
                ids[ids == child] = parent
                break
    # Then loose splats above each remaining top surface.
    free = ids < 0
    for o, (lo, hi) in tops.items():
        if o in merged:
            continue
        if hi[1] < 0.15:
            continue  # rugs and other flat things carry nothing
        m = (
            free
            & (world[:, 0] > lo[0] + 0.02)
            & (world[:, 0] < hi[0] - 0.02)
            & (world[:, 2] > lo[2] + 0.02)
            & (world[:, 2] < hi[2] - 0.02)
            & (world[:, 1] > hi[1] - 0.03)
            & (world[:, 1] < hi[1] + p.support_height)
        )
        ids[m] = o
        free &= ~m
    return ids, merged
