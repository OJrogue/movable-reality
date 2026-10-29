"""Plane fit for flat structure (floor, ceiling): RANSAC on Gaussian centres, least-squares
refinement, and an oriented bounding rectangle of the inliers.

All inputs are world-frame points in metres. Pure numpy, deterministic for a given seed.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Plane:
    normal: np.ndarray  # (3,) unit; points into the room when an interior point was given
    offset: float  # plane is {p : normal . p = offset}
    inliers: np.ndarray  # bool mask over the fitted points
    rms: float  # RMS distance of the inliers to the refined plane, metres

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, dtype=np.float64) @ self.normal - self.offset


@dataclass(frozen=True)
class Rect:
    """Oriented rectangle in a plane. Texel (row i, col j) centre is
    origin + (j + 0.5) * texel * u + (i + 0.5) * texel * v.

    u x v = -normal, so (u, v, -normal) is a right-handed camera frame (OpenCV: x right,
    y down, z forward) that looks at the plane from the room side.
    """

    origin: np.ndarray  # (3,) corner, world
    u: np.ndarray  # (3,) unit, along the long side
    v: np.ndarray  # (3,) unit
    normal: np.ndarray  # (3,) unit, room side
    size_u: float  # metres
    size_v: float

    def corners(self) -> np.ndarray:
        """(4, 3): origin, +u, +u+v, +v."""
        o, du, dv = self.origin, self.u * self.size_u, self.v * self.size_v
        return np.stack([o, o + du, o + du + dv, o + dv])

    def to_plane_coords(self, pts: np.ndarray) -> np.ndarray:
        """(N, 2) metres from the origin along u and v."""
        d = np.asarray(pts, dtype=np.float64) - self.origin
        return np.stack([d @ self.u, d @ self.v], axis=1)

    def contains(self, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        uv = self.to_plane_coords(pts)
        return (
            (uv[:, 0] >= -margin)
            & (uv[:, 0] <= self.size_u + margin)
            & (uv[:, 1] >= -margin)
            & (uv[:, 1] <= self.size_v + margin)
        )


def _lsq_plane(pts: np.ndarray) -> tuple[np.ndarray, float]:
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    return n, float(n @ c)


def fit_plane(
    pts: np.ndarray,
    threshold: float = 0.02,
    *,
    interior: np.ndarray | None = None,
    up: np.ndarray | None = None,
    max_tilt_deg: float | None = None,
    iterations: int = 1000,
    max_sample: int = 50_000,
    refine_steps: int = 3,
    seed: int = 0,
) -> Plane:
    """RANSAC plane through `pts` (N, 3), then least-squares refinement on the inliers.

    threshold: inlier distance in metres. up and max_tilt_deg: optional prior that rejects
    hypotheses whose normal is more than max_tilt_deg from +-up; if no hypothesis passes,
    the fit runs again without the prior. interior: a point on the room side, used to
    orient the normal.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        raise ValueError(f"need at least 3 points for a plane, got {len(pts)}")
    rng = np.random.default_rng(seed)
    sample = pts if len(pts) <= max_sample else pts[rng.choice(len(pts), max_sample, False)]

    best_n, best_d, best_count = None, 0.0, -1
    min_cos = None
    if up is not None and max_tilt_deg is not None:
        up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
        min_cos = math.cos(math.radians(max_tilt_deg))
    batch = 128
    for start in range(0, iterations, batch):
        k = min(batch, iterations - start)
        tri = sample[rng.integers(0, len(sample), size=(k, 3))]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm = np.linalg.norm(n, axis=1)
        ok = norm > 1e-12
        n = n[ok] / norm[ok, None]
        p0 = tri[ok, 0]
        if min_cos is not None:
            keep = np.abs(n @ up) >= min_cos
            n, p0 = n[keep], p0[keep]
        if len(n) == 0:
            continue
        d = np.einsum("ij,ij->i", n, p0)
        counts = (np.abs(sample @ n.T - d) < threshold).sum(axis=0)
        i = int(np.argmax(counts))
        if counts[i] > best_count:
            best_n, best_d, best_count = n[i], float(d[i]), int(counts[i])

    if best_n is None:
        if min_cos is None:
            raise ValueError("RANSAC found no plane (degenerate points)")
        return fit_plane(
            pts,
            threshold,
            interior=interior,
            iterations=iterations,
            max_sample=max_sample,
            refine_steps=refine_steps,
            seed=seed,
        )

    n, d = best_n, best_d
    inl = np.abs(pts @ n - d) < threshold
    for _ in range(refine_steps):
        if inl.sum() < 3:
            break
        n, d = _lsq_plane(pts[inl])
        inl = np.abs(pts @ n - d) < threshold
    if interior is not None and float(np.asarray(interior) @ n - d) < 0:
        n, d = -n, -d
    rms = float(np.sqrt(np.mean((pts[inl] @ n - d) ** 2))) if inl.any() else float("nan")
    return Plane(normal=n, offset=d, inliers=inl, rms=rms)


# ---------------------------------------------------------------- rectangle


def _basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Any orthonormal in-plane pair (a, b)."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    a = np.cross(normal, helper)
    a /= np.linalg.norm(a)
    return a, np.cross(normal, a)


def largest_component(xy: np.ndarray, cell: float, min_count: int = 1) -> np.ndarray:
    """Bool mask of the points in the largest 8-connected cluster of occupied grid cells.

    Removes islands such as floor seen through a doorway, or stray mislabelled clusters.
    """
    lo = xy.min(axis=0)
    ij = np.floor((xy - lo) / cell).astype(np.int64)
    shape = ij.max(axis=0) + 1
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    counts = np.bincount(flat, minlength=int(shape[0] * shape[1])).reshape(shape)
    occ = counts >= min_count
    label = np.full(occ.shape, -1, dtype=np.int64)
    sizes: list[int] = []
    for start in zip(*np.nonzero(occ), strict=True):
        if label[start] >= 0:
            continue
        lab = len(sizes)
        label[start] = lab
        queue, size = deque([start]), 0
        while queue:
            i, j = queue.popleft()
            size += int(counts[i, j])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if 0 <= a < shape[0] and 0 <= b < shape[1] and occ[a, b] and label[a, b] < 0:
                        label[a, b] = lab
                        queue.append((a, b))
        sizes.append(size)
    if not sizes:
        return np.zeros(len(xy), dtype=bool)
    return label[ij[:, 0], ij[:, 1]] == int(np.argmax(sizes))


def _robust_extent(
    xy: np.ndarray, angle: float, trim: float, grow: float = 0.0
) -> tuple[float, float, float, float]:
    """Trimmed-percentile extent along the rotated axes, then grown to the extreme points
    that lie within `grow` of it (so the quad reaches the walls without chasing strays)."""
    c, s = math.cos(angle), math.sin(angle)
    a = xy @ np.array([c, s])
    b = xy @ np.array([-s, c])
    a0, a1 = np.percentile(a, [trim, 100 - trim])
    b0, b1 = np.percentile(b, [trim, 100 - trim])
    if grow > 0:
        a0, a1 = max(a.min(), a0 - grow), min(a.max(), a1 + grow)
        b0, b1 = max(b.min(), b0 - grow), min(b.max(), b1 + grow)
    return float(a0), float(a1), float(b0), float(b1)


def bounding_rect(
    pts: np.ndarray,
    plane: Plane,
    *,
    trim_percent: float = 0.5,
    cell: float = 0.1,
    min_cell_count: int = 2,
) -> Rect:
    """Oriented bounding rectangle of in-plane points (PCA axes), robust to outliers.

    1. Keep the largest connected cluster on a coarse occupancy grid (drops islands).
    2. PCA of the remaining points gives the axes; extents are trimmed percentiles, grown
       by at most one grid cell to the extreme points.
    3. PCA axes are arbitrary for near-square footprints, so a sweep over angles looks for
       a smaller robust rectangle and is used when it is at least 3% smaller.
    The long side becomes u.
    """
    n = plane.normal
    a, b = _basis(n)
    proj = np.asarray(pts, dtype=np.float64) - np.outer(plane.signed_distance(pts), n)
    xy = np.stack([proj @ a, proj @ b], axis=1)
    keep = largest_component(xy, cell, min_cell_count)
    if keep.sum() < 3:
        keep = np.ones(len(xy), dtype=bool)
    xy = xy[keep]
    full_xy = xy
    if len(xy) > 100_000:  # the angle search needs only the distribution
        xy = xy[np.random.default_rng(0).choice(len(xy), 100_000, replace=False)]

    centred = xy - xy.mean(axis=0)
    evals, evecs = np.linalg.eigh(centred.T @ centred)
    major = evecs[:, int(np.argmax(evals))]
    best_angle = math.atan2(major[1], major[0])
    ext = _robust_extent(xy, best_angle, trim_percent)
    best_area = (ext[1] - ext[0]) * (ext[3] - ext[2])

    def area(t: float) -> float:
        e = _robust_extent(xy, t, trim_percent)
        return (e[1] - e[0]) * (e[3] - e[2])

    coarse = np.radians(np.arange(0.0, 90.0, 1.0))
    areas = [area(t) for t in coarse]
    t0 = float(coarse[int(np.argmin(areas))])
    fine = t0 + np.radians(np.arange(-1.0, 1.0001, 0.1))
    t1 = float(fine[int(np.argmin([area(t) for t in fine]))])
    if area(t1) < 0.97 * best_area:
        best_angle = t1
    ext = _robust_extent(full_xy, best_angle, trim_percent, grow=cell)

    c, s = math.cos(best_angle), math.sin(best_angle)
    du = c * a + s * b
    dv_dir = -s * a + c * b
    a0, a1, b0, b1 = ext
    size_u, size_v = a1 - a0, b1 - b0
    corner = a0 * du + b0 * dv_dir
    if size_v > size_u:  # long side along u
        corner = corner + size_u * du
        du, dv_dir = dv_dir, -du
        size_u, size_v = size_v, size_u
    # Enforce u x v = -n by flipping v if needed (moves the origin to the other u edge).
    if np.dot(np.cross(du, dv_dir), n) > 0:
        corner = corner + size_v * dv_dir
        dv_dir = -dv_dir
    origin = corner + plane.offset * n
    return Rect(origin=origin, u=du, v=dv_dir, normal=n, size_u=size_u, size_v=size_v)


def snap_to_texels(rect: Rect, texel: float, max_px: int) -> tuple[Rect, float, int, int]:
    """Grow the rectangle (about its centre) to whole texels. Texel size grows if a side
    would exceed max_px. Returns (rect, texel, width_px, height_px)."""
    longest = max(rect.size_u, rect.size_v)
    if longest / texel > max_px:
        texel = longest / max_px
    w = max(1, min(max_px, math.ceil(rect.size_u / texel - 1e-9)))
    h = max(1, min(max_px, math.ceil(rect.size_v / texel - 1e-9)))
    su, sv = w * texel, h * texel
    origin = rect.origin - 0.5 * (su - rect.size_u) * rect.u - 0.5 * (sv - rect.size_v) * rect.v
    out = Rect(origin=origin, u=rect.u, v=rect.v, normal=rect.normal, size_u=su, size_v=sv)
    return out, texel, w, h


def angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    """Angle between two directions, ignoring sign."""
    c = abs(float(np.dot(n1, n2)) / (np.linalg.norm(n1) * np.linalg.norm(n2)))
    return math.degrees(math.acos(min(1.0, c)))
