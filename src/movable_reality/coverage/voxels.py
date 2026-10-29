"""Voxel grid over the room, occupancy from Gaussians, ray sampling and connected components.

Voxel states (see gaps.py): occupied (holds opaque Gaussians), free (a training ray passed
through it before reaching a surface), unknown (the capture has no evidence about it).
The grid is aligned so the room bounds pass through voxel centres: walls, floor and
ceiling fall inside the outermost voxel layers. Flat index order: (ix * ny + iy) * nz + iz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The 13 neighbour offsets that are lexicographically positive; with their negatives they
# form the 26-neighbourhood.
HALF_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
]


@dataclass(frozen=True)
class VoxelGrid:
    origin: np.ndarray  # (3,) world position of the min corner
    size: float  # edge length, metres
    dims: tuple[int, int, int]  # nx, ny, nz

    @staticmethod
    def from_bounds(lo: np.ndarray, hi: np.ndarray, size: float) -> VoxelGrid:
        """Grid whose outermost voxel centres lie on `lo` and `hi` (rounded up)."""
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        dims = tuple(int(v) for v in np.ceil((hi - lo) / size - 1e-6).astype(int) + 1)
        return VoxelGrid(origin=lo - 0.5 * size, size=float(size), dims=dims)  # type: ignore[arg-type]

    @property
    def n(self) -> int:
        nx, ny, nz = self.dims
        return nx * ny * nz

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.dims

    def ijk(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Integer voxel coordinates (..., 3) and an inside mask (...)."""
        rel = (np.asarray(points, dtype=np.float32) - self.origin.astype(np.float32)) / np.float32(
            self.size
        )
        ijk = np.floor(rel).astype(np.int32)
        inside = np.all((ijk >= 0) & (ijk < np.asarray(self.dims, dtype=np.int32)), axis=-1)
        return ijk, inside

    def flat(self, ijk: np.ndarray) -> np.ndarray:
        _, ny, nz = self.dims
        return (ijk[..., 0].astype(np.int64) * ny + ijk[..., 1]) * nz + ijk[..., 2]

    def index(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flat voxel index (clipped into the grid) and inside mask for (..., 3) points."""
        ijk, inside = self.ijk(points)
        ijk = np.clip(ijk, 0, np.asarray(self.dims) - 1)
        return self.flat(ijk), inside

    def unflat(self, flat: np.ndarray) -> np.ndarray:
        _, ny, nz = self.dims
        flat = np.asarray(flat, dtype=np.int64)
        return np.stack([flat // (ny * nz), (flat // nz) % ny, flat % nz], axis=-1)

    def centres(self, flat: np.ndarray | None = None) -> np.ndarray:
        """World centres (M, 3) float32 of the given voxels (all voxels when None)."""
        if flat is None:
            flat = np.arange(self.n)
        return (self.origin + (self.unflat(flat) + 0.5) * self.size).astype(np.float32)

    def to_dict(self) -> dict:
        return {"origin": self.origin.tolist(), "size": self.size, "dims": list(self.dims)}

    @staticmethod
    def from_dict(d: dict) -> VoxelGrid:
        return VoxelGrid(np.asarray(d["origin"], float), float(d["size"]), tuple(d["dims"]))


def room_bounds(
    means: np.ndarray, opacity: np.ndarray, pct: float = 0.5, min_opacity: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Robust room box from opaque Gaussian centres (world frame): percentiles per axis,
    floor at y = 0 or lower."""
    keep = opacity >= min_opacity
    pts = means[keep] if keep.any() else means
    lo = np.percentile(pts, pct, axis=0)
    hi = np.percentile(pts, 100 - pct, axis=0)
    lo[1] = min(0.0, float(lo[1]))
    return lo.astype(np.float64), hi.astype(np.float64)


def occupancy(
    grid: VoxelGrid, means: np.ndarray, opacity: np.ndarray, min_opacity: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Opaque Gaussians per voxel (n,) and each Gaussian's voxel (N,), -1 outside the grid."""
    flat, inside = grid.index(means)
    vox = np.where(inside, flat, -1)
    keep = inside & (opacity >= min_opacity)
    counts = np.bincount(flat[keep], minlength=grid.n).astype(np.int32)
    return counts, vox


def ray_voxels(
    grid: VoxelGrid,
    origin: np.ndarray,
    dirs: np.ndarray,
    t_end: np.ndarray | float,
    step: float,
    t_start: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample rays origin + t * dir for t = t_start, t_start + step, ... up to t_end (per
    ray or scalar). Returns flat voxel indices (R, S) and a mask (R, S) of samples that are
    within t_end and inside the grid."""
    dirs = np.asarray(dirs, dtype=np.float32)
    t_end = np.broadcast_to(np.asarray(t_end, dtype=np.float32), (len(dirs),))
    n_s = int(np.ceil((float(t_end.max(initial=t_start)) - t_start) / step)) + 1
    t = t_start + np.arange(max(n_s, 1), dtype=np.float32) * np.float32(step)
    pts = np.asarray(origin, dtype=np.float32) + t[None, :, None] * dirs[:, None, :]
    flat, inside = grid.index(pts)
    return flat, inside & (t[None, :] <= t_end[:, None])


def fibonacci_sphere(n: int) -> np.ndarray:
    """n roughly uniform unit directions (n, 3) float32."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], axis=1
    ).astype(np.float32)


# ---------------------------------------------------------------- neighbourhoods


def _shift_slices(d: tuple[int, int, int], shape: tuple[int, int, int]):
    src = tuple(slice(max(0, -k), n - max(0, k)) for k, n in zip(d, shape, strict=True))
    dst = tuple(slice(max(0, k), n - max(0, -k)) for k, n in zip(d, shape, strict=True))
    return src, dst


def label_components(mask3d: np.ndarray) -> np.ndarray:
    """26-connected components of a 3D bool mask. Returns (n,) int32 labels, -1 outside the
    mask, components numbered 0..k-1 in order of their smallest flat index."""
    mask3d = np.asarray(mask3d, dtype=bool)
    labels = np.full(mask3d.size, -1, dtype=np.int32)
    idx = np.flatnonzero(mask3d)
    if idx.size == 0:
        return labels
    flat = np.arange(mask3d.size).reshape(mask3d.shape)
    compact = np.full(mask3d.size, -1, dtype=np.int64)
    compact[idx] = np.arange(idx.size)
    pa, pb = [], []
    for d in HALF_26:
        src, dst = _shift_slices(d, mask3d.shape)
        both = mask3d[src] & mask3d[dst]
        pa.append(compact[flat[src][both]])
        pb.append(compact[flat[dst][both]])
    a, b = np.concatenate(pa), np.concatenate(pb)
    lab = np.arange(idx.size, dtype=np.int64)
    while True:
        prev = lab.copy()
        m = np.minimum(lab[a], lab[b])
        np.minimum.at(lab, a, m)
        np.minimum.at(lab, b, m)
        while True:  # pointer jumping; lab[i] always names a member of i's component
            jumped = lab[lab]
            if np.array_equal(jumped, lab):
                break
            lab = jumped
        if np.array_equal(lab, prev):
            break
    _, dense = np.unique(lab, return_inverse=True)
    labels[idx] = dense.astype(np.int32)
    return labels


def dilate(mask3d: np.ndarray, steps: int = 1) -> np.ndarray:
    """26-neighbourhood dilation."""
    out = np.asarray(mask3d, dtype=bool).copy()
    offsets = HALF_26 + [(-a, -b, -c) for a, b, c in HALF_26]
    for _ in range(steps):
        grown = out.copy()
        for d in offsets:
            src, dst = _shift_slices(d, out.shape)
            grown[dst] |= out[src]
        out = grown
    return out
