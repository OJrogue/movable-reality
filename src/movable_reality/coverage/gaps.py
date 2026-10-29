"""From per-Gaussian scores to gap entries.

1. Voxel states: occupied (opaque Gaussians inside), free (a training ray passed through it
   before reaching a surface), unknown (neither).
2. Voxel gap score: unknown = 1; occupied = 1 - opacity-weighted mean coverage of its
   Gaussians; free = 0.
3. Candidates: unknown voxels and occupied voxels with coverage below a threshold.
4. Exposure: a VR user standing anywhere in the walkable area (several eye heights,
   looking in every direction) would see the voxel as the first solid voxel along a ray.
   Unknown voxels count as solid: the user would look at whatever is (or is not) there.
5. Gap voxels: exposed candidates grown by a few voxels through connected candidates, so a
   gap includes the hidden region behind its visible face (behind the sofa, under the
   desk) but not the deep interior of objects.
6. Gaps: 26-connected components with centre, radius, mean score and nearest object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..scene import Gap, Scene
from .voxels import VoxelGrid, dilate, label_components

FREE, OCCUPIED, UNKNOWN = 0, 1, 2


@dataclass(frozen=True)
class GapParams:
    cov_threshold: float = 0.35  # occupied voxels below this coverage are candidates
    grow: int = 2  # voxels to grow from the exposed face into hidden candidates
    min_voxels: int = 4  # smaller components are dropped
    near_object_margin: float = 0.4  # metres beyond the gap radius


def voxel_states(occ_count: np.ndarray, free_count: np.ndarray, min_occ: int = 1) -> np.ndarray:
    state = np.full(occ_count.shape, UNKNOWN, dtype=np.uint8)
    state[free_count > 0] = FREE
    state[occ_count >= min_occ] = OCCUPIED
    return state


def voxel_coverage(
    grid: VoxelGrid, gauss_voxel: np.ndarray, coverage: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """Weighted mean of per-Gaussian coverage per voxel (n,); NaN where no weight."""
    ok = gauss_voxel >= 0
    v, w = gauss_voxel[ok], weight[ok].astype(np.float64)
    num = np.bincount(v, weights=w * coverage[ok], minlength=grid.n)
    den = np.bincount(v, weights=w, minlength=grid.n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan).astype(np.float32)


def voxel_gap_score(state: np.ndarray, vox_cov: np.ndarray) -> np.ndarray:
    score = np.zeros(state.shape, dtype=np.float32)
    score[state == UNKNOWN] = 1.0
    occ = state == OCCUPIED
    score[occ] = 1.0 - np.nan_to_num(vox_cov[occ], nan=0.0)
    return np.clip(score, 0.0, 1.0)


def candidates(state: np.ndarray, vox_cov: np.ndarray, p: GapParams) -> np.ndarray:
    low = (state == OCCUPIED) & (np.nan_to_num(vox_cov, nan=0.0) < p.cov_threshold)
    return (state == UNKNOWN) | low


def grow_within(seed: np.ndarray, allowed: np.ndarray, shape, steps: int) -> np.ndarray:
    """Geodesic dilation of `seed` inside `allowed` (both flat bool arrays)."""
    cur = (seed & allowed).reshape(shape)
    allowed3 = allowed.reshape(shape)
    for _ in range(steps):
        cur = dilate(cur, 1) & allowed3
    return cur.ravel()


def gap_voxels(cand: np.ndarray, exposed: np.ndarray, grid: VoxelGrid, p: GapParams) -> np.ndarray:
    """Component label per voxel (n,), -1 where not a gap; small components removed and
    the rest renumbered by size (largest first)."""
    mask = grow_within(cand & exposed, cand, grid.dims, p.grow)
    labels = label_components(mask.reshape(grid.dims))
    if labels.max() < 0:
        return labels
    sizes = np.bincount(labels[labels >= 0])
    keep = np.nonzero(sizes >= p.min_voxels)[0]
    order = keep[np.argsort(-sizes[keep], kind="stable")]
    remap = np.full(len(sizes), -1, dtype=np.int32)
    remap[order] = np.arange(len(order), dtype=np.int32)
    return np.where(labels >= 0, remap[np.maximum(labels, 0)], -1).astype(np.int32)


def _aabb_distance(p: np.ndarray, lo, hi) -> float:
    d = np.maximum(np.maximum(np.asarray(lo) - p, 0.0), p - np.asarray(hi))
    return float(np.linalg.norm(np.maximum(d, 0.0)))


def nearest_object(scene: Scene | None, center: np.ndarray, radius: float, margin: float):
    if scene is None:
        return None
    best, best_d = None, np.inf
    for obj in scene.objects:
        if not obj.visible:
            continue
        lo, hi = obj.world_aabb()
        d = _aabb_distance(center, lo, hi)
        if d < best_d:
            best, best_d = obj.id, d
    return best if best_d <= radius + margin else None


def build_gaps(
    grid: VoxelGrid,
    labels: np.ndarray,
    score: np.ndarray,
    scene: Scene | None,
    p: GapParams,
) -> list[Gap]:
    gaps = []
    for g in range(int(labels.max()) + 1):
        idx = np.nonzero(labels == g)[0]
        pts = grid.centres(idx).astype(np.float64)
        c = pts.mean(axis=0)
        r = float(np.linalg.norm(pts - c, axis=1).max() + 0.5 * grid.size)
        gaps.append(
            Gap(
                id=f"gap_{g:03d}",
                near_object=nearest_object(scene, c, r, p.near_object_margin),
                center=tuple(round(float(v), 3) for v in c),  # type: ignore[arg-type]
                radius=round(r, 3),
                score=round(float(score[idx].mean()), 3),
            )
        )
    return gaps


def carry_over(new: list[Gap], old: list[Gap]) -> list[Gap]:
    """Keep the label and status of an old gap whose centre lies inside a new gap's
    half radius, so Muse Wonder's names and recapture requests survive a re-score."""
    out = []
    for g in new:
        match = None
        for o in old:
            if np.linalg.norm(np.subtract(g.center, o.center)) <= 0.5 * g.radius:
                match = o
                break
        if match is not None:
            g = g.model_copy(update={"label": match.label, "status": match.status})
        out.append(g)
    return out
