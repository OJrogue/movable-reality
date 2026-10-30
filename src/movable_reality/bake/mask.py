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
4. Outside: what remains uncovered touches the border, lies outside the outline and
   becomes transparent.

The result is upsampled to texels by nearest neighbour. Covered cells count when any
texel in them is covered, so the opaque region extends at most one cell past the splats,
which hides behind the wall instead of leaving a gap in front of it.
"""

from __future__ import annotations

import math

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
    cell: float = 0.05,
    close_m: float = 0.15,
    threshold: float = 0.5,
) -> np.ndarray:
    """(H, W) bool, True where the texel is part of the surface (opaque).

    `supports` are world-frame axis-aligned boxes (min, max) of objects standing on the
    plane. Given `origin`, `u` and `v` (the quad's frame, see planefit.Rect), the cells under
    them count as covered (only their x and z extent is used).
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
    closed = close(covered, round(close_m / (k * texel)))
    opaque = ~outside(~closed)
    return np.repeat(np.repeat(opaque, k, axis=0), k, axis=1)[:h, :w]
