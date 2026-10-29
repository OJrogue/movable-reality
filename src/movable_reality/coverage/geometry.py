"""Geometric coverage: for every Gaussian, how many training views saw it and from how wide a
range of directions; for every voxel, whether some view saw it as free space.

Per view (vectorised numpy, chunked over Gaussians):
1. Project every Gaussian centre into a coarse copy of the camera (long side 256 px).
2. Build a coarse depth map from opaque Gaussians (a random subset when there are many):
   each writes its depth into the pixels its footprint covers (radius ~1.5 sigma of its
   largest scale, at most 3 px), keeping the minimum.
3. A Gaussian is seen by the view when it is inside the frustum and not behind the depth
   map by more than a tolerance: share of depth + pixel footprints + local depth slope.
   The slope term keeps surfaces seen at grazing angles from occluding themselves.
4. A voxel centre is seen free when it lies in front of the depth map by more than a
   tighter carving tolerance. A voxel that still holds part of the surface is occupied
   anyway, so a tight tolerance only trims the shell of unknown voxels in front of
   surfaces.

Angle spread uses the mean resultant length R of the unit vectors from the Gaussian to the
cameras that saw it: spread = 2 * acos(R). For two views this is exactly the angle between
them; for one view it is 0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .cameras import Camera


@dataclass(frozen=True)
class DepthParams:
    long_side: int = 256  # coarse depth map resolution (pixels, long side)
    near: float = 0.05  # metres
    far: float = 12.0  # Gaussians further than this do not count as seen
    occluder_min_opacity: float = 0.5  # only these write into the depth map
    max_occluders: int = 600_000  # random subset above this
    footprint_sigma: float = 1.5  # footprint radius in units of the largest scale
    max_footprint_px: int = 3
    rel_tol: float = 0.02  # visibility tolerance: share of depth
    px_tol: float = 1.5  # + this many coarse pixel footprints
    slope_k: float = 2.0  # + this many times the local depth change per pixel
    slope_cap: float = 0.1  # local slope is capped at this share of depth per pixel
    carve_rel_tol: float = 0.01  # free-space carving tolerance: share of depth
    carve_px_tol: float = 1.0  # + this many coarse pixel footprints
    carve_abs_tol: float = 0.02  # + this many metres
    chunk: int = 1_000_000  # Gaussians per projection chunk
    seed: int = 0


@dataclass
class GaussianCoverage:
    view_count: np.ndarray  # (N,) uint16
    dir_sum: np.ndarray  # (N, 3) float32, sum of unit vectors Gaussian -> camera
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def resultant(self) -> np.ndarray:
        """Mean resultant length in [0, 1] (1 for a single view, 0 where never seen)."""
        n = np.maximum(self.view_count, 1).astype(np.float32)
        return np.linalg.norm(self.dir_sum, axis=1) / n

    @property
    def spread_deg(self) -> np.ndarray:
        return spread_from_sums(self.view_count, self.dir_sum)


def spread_from_sums(view_count: np.ndarray, dir_sum: np.ndarray) -> np.ndarray:
    n = np.maximum(view_count, 1).astype(np.float32)
    r = np.clip(np.linalg.norm(dir_sum, axis=1) / n, 0.0, 1.0)
    spread = np.degrees(2.0 * np.arccos(r)).astype(np.float32)
    return np.where(view_count >= 2, spread, np.float32(0.0))


def geometric_score(
    view_count: np.ndarray, spread_deg: np.ndarray, views_ok: float = 3.0, spread_ok: float = 10.0
) -> np.ndarray:
    """Per-Gaussian geometric coverage in [0, 1]: saturates at `views_ok` views and
    `spread_ok` degrees of angle spread. A Gaussian seen once scores 0."""
    cv = np.clip(view_count.astype(np.float32) / views_ok, 0.0, 1.0)
    ca = np.clip(spread_deg.astype(np.float32) / spread_ok, 0.0, 1.0)
    return (cv * ca).astype(np.float32)


# ---------------------------------------------------------------- per-view primitives


def _project_chunk(cam: Camera, pts: np.ndarray, p: DepthParams):
    """Coarse pixel coordinates, depth and in-frustum mask for a chunk of points."""
    uv, z = cam.project(pts, p.near)
    u = np.floor(uv[:, 0]).astype(np.int32)
    v = np.floor(uv[:, 1]).astype(np.int32)
    ok = (z > p.near) & (z < p.far) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    return u, v, z, ok


def splat_depth(
    width: int, height: int, u: np.ndarray, v: np.ndarray, z: np.ndarray, radius_px: np.ndarray
) -> np.ndarray:
    """Min-depth map (H*W,) with each point covering a square of `radius_px` around its
    pixel. Empty pixels stay +inf."""
    depth = np.full(width * height, np.inf, dtype=np.float32)
    for r in np.unique(radius_px):
        sel = radius_px == r
        us, vs, zs = u[sel], v[sel], z[sel]
        for dy in range(-int(r), int(r) + 1):
            for dx in range(-int(r), int(r) + 1):
                uu, vv = us + dx, vs + dy
                ok = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
                np.minimum.at(depth, vv[ok] * width + uu[ok], zs[ok])
    return depth


def depth_slope(depth: np.ndarray, width: int, height: int, cap: float) -> np.ndarray:
    """Local depth change per pixel (H*W,): per axis the smaller of the two one-sided
    differences (so an occlusion edge on one side does not count), max over the two axes,
    capped at `cap` * depth. Pixels next to empty pixels get the cap."""
    d = depth.reshape(height, width).astype(np.float32)
    pad = np.pad(d, 1, mode="edge")
    c = pad[1:-1, 1:-1]
    with np.errstate(invalid="ignore"):
        dx = np.minimum(np.abs(pad[1:-1, 2:] - c), np.abs(c - pad[1:-1, :-2]))
        dy = np.minimum(np.abs(pad[2:, 1:-1] - c), np.abs(c - pad[:-2, 1:-1]))
        s = np.maximum(dx, dy)
    lim = cap * c
    s = np.where(np.isfinite(s), np.minimum(s, lim), lim)
    return np.where(np.isfinite(c), s, 0.0).ravel().astype(np.float32)


def view_depth_map(
    cam: Camera,
    means: np.ndarray,
    max_scale: np.ndarray,
    occluders: np.ndarray,
    p: DepthParams,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Coarse depth map of one (already coarse) camera plus the per-chunk projections
    (indices of in-frustum Gaussians, their flat pixel, their depth)."""
    f = float(cam.K[0, 0])
    chunks = []
    for s in range(0, len(means), p.chunk):
        u, v, z, ok = _project_chunk(cam, means[s : s + p.chunk], p)
        idx = np.nonzero(ok)[0]
        chunks.append((idx + s, v[idx] * cam.width + u[idx], z[idx]))
    u, v, z, ok = _project_chunk(cam, means[occluders], p)
    u, v, z = u[ok], v[ok], z[ok]
    r = np.rint(p.footprint_sigma * max_scale[occluders][ok] * f / z)
    r = np.clip(r, 0, p.max_footprint_px).astype(np.int32)
    return splat_depth(cam.width, cam.height, u, v, z, r), chunks


def visibility_tolerance(z: np.ndarray, slope: np.ndarray, focal: float, p: DepthParams):
    return p.rel_tol * z + p.px_tol * z / focal + p.slope_k * slope


def carve_voxels(cam: Camera, depth: np.ndarray, centres: np.ndarray, p: DepthParams) -> np.ndarray:
    """Bool mask of voxel centres this view saw as free space. Pixels without a surface
    carve nothing: an empty pixel means a missing reconstruction, not free space."""
    u, v, z, ok = _project_chunk(cam, centres, p)
    out = np.zeros(len(centres), dtype=bool)
    idx = np.nonzero(ok)[0]
    d = depth[v[idx] * cam.width + u[idx]]
    fin = np.isfinite(d)
    idx, d = idx[fin], d[fin]
    tol = p.carve_rel_tol * d + p.carve_px_tol * d / float(cam.K[0, 0]) + p.carve_abs_tol
    out[idx] = z[idx] < d - tol
    return out


def select_occluders(opacity: np.ndarray, p: DepthParams) -> np.ndarray:
    idx = np.nonzero(opacity >= p.occluder_min_opacity)[0]
    if len(idx) > p.max_occluders:
        rng = np.random.default_rng(p.seed)
        idx = np.sort(rng.choice(idx, p.max_occluders, replace=False))
    return idx


# ---------------------------------------------------------------- main entry


def compute_coverage(
    means: np.ndarray,
    opacity: np.ndarray,
    scales: np.ndarray,
    cameras: list[Camera],
    params: DepthParams | None = None,
    voxel_centres: np.ndarray | None = None,
) -> tuple[GaussianCoverage, np.ndarray | None]:
    """View counts and direction sums per Gaussian; optionally, per voxel, the number of
    views that saw it as free space. All inputs in the world frame."""
    p = params or DepthParams()
    means = np.ascontiguousarray(means, dtype=np.float32)
    opacity = np.asarray(opacity, dtype=np.float32)
    max_scale = np.asarray(scales, dtype=np.float32).max(axis=1)
    occluders = select_occluders(opacity, p)
    n = len(means)
    count = np.zeros(n, dtype=np.uint16)
    dir_sum = np.zeros((n, 3), dtype=np.float32)
    free = None if voxel_centres is None else np.zeros(len(voxel_centres), dtype=np.uint16)
    t_depth = t_test = t_carve = 0.0
    for cam in cameras:
        small = cam.scaled(p.long_side)
        t0 = time.perf_counter()
        depth, chunks = view_depth_map(small, means, max_scale, occluders, p)
        focal = float(small.K[0, 0])
        slope = depth_slope(depth, small.width, small.height, p.slope_cap)
        t1 = time.perf_counter()
        c = cam.center.astype(np.float32)
        for idx, pix, z in chunks:
            d = depth[pix]
            seen = ~np.isfinite(d) | (z <= d + visibility_tolerance(d, slope[pix], focal, p))
            sidx = idx[seen]
            count[sidx] += 1
            vec = c - means[sidx]
            vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-8)
            dir_sum[sidx] += vec
        t2 = time.perf_counter()
        if free is not None:
            free += carve_voxels(small, depth, voxel_centres, p)
        t3 = time.perf_counter()
        t_depth += t1 - t0
        t_test += t2 - t1
        t_carve += t3 - t2
    timings = {
        "project_and_depth_s": round(t_depth, 3),
        "visibility_test_s": round(t_test, 3),
        "voxel_carve_s": round(t_carve, 3),
    }
    return GaussianCoverage(view_count=count, dir_sum=dir_sum, timings=timings), free
