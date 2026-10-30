"""Alpha mask for a baked plane, marking the texels outside the surface's real outline.

The quad is the bounding rectangle of the plane's Gaussians, so for a room that is not a
rectangle (an L-shaped floor) part of the quad lies behind the walls. Push-pull fills
every texel, so without a mask that part shows as floor sticking out past the walls.

The mask works on a coarse grid (`cell` metres, default 5 cm) in four steps.

1. Covered cells: any texel has baked alpha >= `threshold`, or the cell lies under an
   object's footprint (`supports`, floors only). The floor under a sofa exists even though
   no splat shows it.
2. Closing: a disc of radius `close_m` bridges gaps narrower than about 2 * close_m
   (sparse splats along a wall, a doorway threshold).
3. Holes: uncovered regions that do not touch the quad's border lie inside the surface
   (under furniture, glare) and stay opaque. Push-pull fills their colour.
4. Outside: what remains uncovered touches the border. A region that never reaches more
   than `close_m` from the surface (the strip between a sofa and the wall) stays opaque.
   A deeper region becomes transparent when walls run along its edge against the surface
   (`walls`, background splats above the plane), so the surface really ends there. A
   region without walls along that edge (an unobserved corner of a rectangular room)
   stays opaque and keeps the push-pull fill.

The result is upsampled to texels by nearest neighbour. Covered cells count when any
texel in them is covered, so the opaque region extends at most one cell past the splats,
which hides behind the wall instead of leaving a gap in front of it.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a disc of `radius` cells (outside the grid counts as False)."""
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    p = np.pad(mask, radius)
    out = np.zeros_like(mask)
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di * di + dj * dj <= radius * radius:
                out |= p[radius + di : radius + di + h, radius + dj : radius + dj + w]
    return out


def close(mask: np.ndarray, radius: int) -> np.ndarray:
    """Morphological closing (dilate, then erode) with a disc. Padding keeps regions that
    touch the border from being eaten by the erosion."""
    if radius <= 0:
        return mask.copy()
    p = np.pad(mask, radius, mode="edge")
    grown = _dilate(p, radius)
    closed = ~_dilate(~grown, radius)
    return closed[radius:-radius, radius:-radius]


def outside(uncovered: np.ndarray) -> np.ndarray:
    """Cells of `uncovered` connected (4-neighbourhood) to the grid border."""
    reach = np.zeros_like(uncovered)
    reach[0, :], reach[-1, :] = uncovered[0, :], uncovered[-1, :]
    reach[:, 0] |= uncovered[:, 0]
    reach[:, -1] |= uncovered[:, -1]
    while True:
        grown = reach.copy()
        grown[1:] |= reach[:-1]
        grown[:-1] |= reach[1:]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown &= uncovered
        if np.array_equal(grown, reach):
            return reach
        reach = grown


def cell_centres(
    origin: np.ndarray, u: np.ndarray, v: np.ndarray, cell_m: float, shape: tuple[int, int]
) -> np.ndarray:
    """(rows, cols, 3) world positions of the coarse cell centres."""
    i = (np.arange(shape[0]) + 0.5) * cell_m
    j = (np.arange(shape[1]) + 0.5) * cell_m
    return origin + i[:, None, None] * v + j[None, :, None] * u


def plane_mask(
    alpha: np.ndarray,
    texel: float,
    *,
    origin: np.ndarray | None = None,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
    supports: list[tuple[np.ndarray, np.ndarray]] | None = None,
    walls: np.ndarray | None = None,
    cell: float = 0.05,
    close_m: float = 0.15,
    threshold: float = 0.5,
    wall_min: int = 5,
    wall_near_m: float = 0.1,
    wall_share: float = 0.5,
) -> np.ndarray:
    """(H, W) bool, True where the texel is part of the surface (opaque).

    `supports` are world-frame axis-aligned boxes (min, max) of objects standing on the
    plane. Given `origin`, `u` and `v` (the quad's frame, see planefit.Rect), the cells under
    them count as covered (only their x and z extent is used).

    `walls` (world points, for example background splats 0.2 to 1.5 m off the plane) make
    the cut conservative. A border region is then cut only when at least `wall_share` of
    its edge against the surface lies within `wall_near_m` of a wall cell (`wall_min` or
    more points). An unobserved corner of a rectangular room has no wall along that edge
    and stays opaque. Without `walls` every border region is cut.
    """
    a = np.asarray(alpha, dtype=np.float64)
    h, w = a.shape
    k = max(1, round(cell / texel))
    ch, cw = math.ceil(h / k), math.ceil(w / k)
    fine = np.zeros((ch * k, cw * k), dtype=bool)
    fine[:h, :w] = a >= threshold
    covered = fine.reshape(ch, k, cw, k).any(axis=(1, 3))
    if supports and origin is not None and u is not None and v is not None:
        c = cell_centres(origin, u, v, k * texel, (ch, cw))
        for lo, hi in supports:
            covered |= (
                (c[..., 0] >= lo[0])
                & (c[..., 0] <= hi[0])
                & (c[..., 2] >= lo[2])
                & (c[..., 2] <= hi[2])
            )
    radius = round(close_m / (k * texel))
    closed = close(covered, radius)
    # A border region thinner than close_m everywhere (the strip between a sofa and the
    # wall) stays opaque. Only regions reaching deeper than that are cut.
    deep = ~_dilate(closed, radius)
    near = None
    if walls is not None and origin is not None and u is not None and v is not None:
        cells = wall_cells(walls, origin, u, v, k * texel, (ch, cw), wall_min)
        near = _dilate(cells, max(1, round(wall_near_m / (k * texel))))
    opaque = ~_cut_regions(outside(~closed), closed, deep, near, wall_share)
    return np.repeat(np.repeat(opaque, k, axis=0), k, axis=1)[:h, :w]


def wall_cells(
    points: np.ndarray,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    cell_m: float,
    shape: tuple[int, int],
    min_count: int,
) -> np.ndarray:
    """(rows, cols) bool, cells holding at least `min_count` of the points (projected)."""
    d = np.asarray(points, dtype=np.float64).reshape(-1, 3) - origin
    i = np.floor(d @ v / cell_m).astype(np.int64)
    j = np.floor(d @ u / cell_m).astype(np.int64)
    ok = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
    counts = np.bincount(i[ok] * shape[1] + j[ok], minlength=shape[0] * shape[1])
    return counts.reshape(shape) >= min_count


def _cut_regions(
    out: np.ndarray,
    surface: np.ndarray,
    deep: np.ndarray,
    near_wall: np.ndarray | None,
    share: float,
) -> np.ndarray:
    """The components of `out` to cut: those with a `deep` cell and, given `near_wall`,
    whose edge against `surface` runs along walls."""
    keep = np.zeros_like(out)
    seen = np.zeros_like(out)
    h, w = out.shape
    for start in zip(*np.nonzero(out), strict=True):
        if seen[start]:
            continue
        comp = []
        queue = deque([start])
        seen[start] = True
        while queue:
            i, j = queue.popleft()
            comp.append((i, j))
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < h and 0 <= b < w and out[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    queue.append((a, b))
        ci, cj = np.array(comp).T
        # Edge cells: component cells with a surface cell among their 4 neighbours.
        edge = np.zeros(len(ci), dtype=bool)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = ci + di, cj + dj
            ok = (a >= 0) & (a < h) & (b >= 0) & (b < w)
            edge[ok] |= surface[a[ok], b[ok]]
        if not deep[ci, cj].any():
            continue
        if near_wall is None or not edge.any() or near_wall[ci[edge], cj[edge]].mean() >= share:
            keep[ci, cj] = True
    return keep
