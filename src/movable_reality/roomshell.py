"""The room shell for the mesh-only view: floor, walls and ceiling as flat textured
surfaces, textured from the video frames themselves.

    uv run python scripts/room_shell.py --scene blender01

The background splats are no use for this. On blender01 (31 Oct) the floor splats fit a
plane to 9 mm, but the wall splats form no plane at all, and RANSAC on the fused room mesh
gave walls tilted 17 to 40 deg. A textureless wall gives 3DGS nothing to fix depth with.
So the shell is built from what is reliable.

- Outline. SAM's floor pixels in every frame are cast onto the plane y = 0 (the floor of
  the world frame, set by lift_ids.py). Cells that at least 3 frames hit, plus the
  footprints of the scene objects, fill a grid turned to the room's main wall direction.
  A closing bridges gaps, holes are filled, the largest region is kept and made
  orthogonally convex (furniture against a wall leaves notches), and its border
  becomes a polygon (Douglas-Peucker), whose near-axis edges are snapped to the main
  wall directions and whose small steps are merged (rectify). Floor splats were tried
  first and spilled 0.3 m past the walls on blender01.
- Height. The densest layer of background splats above every camera
  (lift.ceiling_from_histogram), else the 99th percentile of their heights.
- Surfaces. One vertical quad per outline edge from the floor to the ceiling, and one
  quad each for the floor and the ceiling over the outline's bounding box, cut to the
  outline by an alpha mask.
- Texture. Each texel's world point is projected into every frame that faces it. Pixels
  SAM marked as a non-structure item (furniture, which hides the wall behind it) do not
  count. The texel keeps the colours of its `k` best views (nearest and most frontal)
  and takes their median. Texels no frame saw are filled by push-pull (bake.texture).

No learned weights. Assumes a room whose walls are vertical and whose outline is convex
or close to it: a wall that hides another from a camera is not tested for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bake import texture

# Flat things in the plane of a wall: their pixels texture the wall instead of being
# skipped as furniture, so a window or a door is drawn on the shell where it is.
ON_WALL = ("window", "door", "blinds", "curtain", "picture", "painting", "poster", "mirror")


@dataclass
class Surface:
    kind: str  # floor, ceiling or wall
    origin: np.ndarray  # world corner of texel (0, 0)
    u: np.ndarray  # unit, along the columns
    v: np.ndarray  # unit, along the rows
    normal: np.ndarray  # unit, into the room
    size_u: float
    size_v: float
    outline_uv: np.ndarray | None = None  # polygon in (u, v) metres; None = whole quad
    stats: dict = field(default_factory=dict)

    def corners(self) -> np.ndarray:
        o, du, dv = self.origin, self.u * self.size_u, self.v * self.size_v
        return np.stack([o, o + du, o + du + dv, o + dv])

    def grid(self, texel: float) -> tuple[np.ndarray, int, int]:
        w, h = max(1, round(self.size_u / texel)), max(1, round(self.size_v / texel))
        j, i = np.meshgrid((np.arange(w) + 0.5) * self.size_u / w,
                           (np.arange(h) + 0.5) * self.size_v / h)  # fmt: skip
        pts = self.origin + j[..., None] * self.u + i[..., None] * self.v
        return pts, w, h


# ---- outline


def _close(m: np.ndarray, r: int) -> np.ndarray:
    from .bake import mask  # noqa: PLC0415

    return mask.close(m, r)


def _fill_holes(m: np.ndarray) -> np.ndarray:
    from .bake import mask  # noqa: PLC0415

    return ~mask.outside(~m)


def _largest(m: np.ndarray) -> np.ndarray:
    """Largest 4-connected region of a boolean grid."""
    lab = np.zeros(m.shape, dtype=np.int32)
    best, best_n, n = 0, 0, 0
    for start in zip(*np.nonzero(m), strict=True):
        if lab[start]:
            continue
        n += 1
        stack, count = [start], 0
        lab[start] = n
        while stack:
            i, j = stack.pop()
            count += 1
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < m.shape[0] and 0 <= b < m.shape[1] and m[a, b] and not lab[a, b]:
                    lab[a, b] = n
                    stack.append((a, b))
        if count > best_n:
            best, best_n = n, count
    return lab == best


def ortho_convex(m: np.ndarray) -> np.ndarray:
    """Fill every row and every column between its first and last set cell, until nothing
    changes. Notches of any width along a wall close (furniture hid the floor there), and
    an L-shaped or bay-windowed outline is left as it is (it is already orthogonally
    convex). Only a U-shaped room would lose its inner notch."""
    m = m.copy()
    while True:
        before = m.copy()
        for grid in (m, m.T):
            any_ = grid.any(1)
            first = np.argmax(grid, 1)
            last = grid.shape[1] - 1 - np.argmax(grid[:, ::-1], 1)
            cols = np.arange(grid.shape[1])
            grid |= any_[:, None] & (cols >= first[:, None]) & (cols <= last[:, None])
        if np.array_equal(m, before):
            return m


def _trace(m: np.ndarray) -> np.ndarray:
    """Outer border of the region (grid corners, counter-clockwise in (col, row)), as the
    chain of cell edges with the region on the left."""
    nxt: dict[tuple[int, int], tuple[int, int]] = {}
    h, w = m.shape
    pad = np.pad(m, 1)
    for i, j in zip(*np.nonzero(m), strict=True):
        if not pad[i, j + 1]:  # row above (i - 1) empty: edge along the top, right to left
            nxt[(j + 1, i)] = (j, i)
        if not pad[i + 2, j + 1]:  # row below empty: left to right along the bottom
            nxt[(j, i + 1)] = (j + 1, i + 1)
        if not pad[i + 1, j]:  # left empty: top to bottom along the left side
            nxt[(j, i)] = (j, i + 1)
        if not pad[i + 1, j + 2]:  # right empty: bottom to top along the right side
            nxt[(j + 1, i + 1)] = (j + 1, i)
    start = min(nxt)  # leftmost corner lies on the outer border
    loop, p = [start], nxt[start]
    while p != start and len(loop) <= len(nxt):
        loop.append(p)
        p = nxt[p]
    return np.array(loop, dtype=np.float64)


def _dp(pts: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker on an open polyline."""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-12:
        d = np.linalg.norm(pts - a, axis=1)
    else:
        r = pts - a
        d = np.abs(ab[0] * r[:, 1] - ab[1] * r[:, 0]) / n
    k = int(np.argmax(d))
    if d[k] <= tol:
        return np.stack([a, b])
    return np.concatenate([_dp(pts[: k + 1], tol)[:-1], _dp(pts[k:], tol)])


def simplify_loop(loop: np.ndarray, tol: float) -> np.ndarray:
    """Closed polygon simplified by Douglas-Peucker, split at the two farthest points."""
    i = 0
    k = int(np.argmax(np.linalg.norm(loop - loop[i], axis=1)))
    a = _dp(np.concatenate([loop[i : k + 1]]), tol)
    b = _dp(np.concatenate([loop[k:], loop[: i + 1]]), tol)
    return np.concatenate([a[:-1], b[:-1]])


def main_angle(xz: np.ndarray) -> float:
    """Rotation (radians, 0 to 90 deg) whose axis-aligned box around the points is smallest:
    the room's main wall direction."""

    def area(th: float) -> float:
        c, s = np.cos(th), np.sin(th)
        r = xz @ np.array([[c, -s], [s, c]])
        return float(np.prod(np.percentile(r, 99, 0) - np.percentile(r, 1, 0)))

    coarse = np.radians(np.arange(0.0, 90.0, 1.0))
    best = coarse[int(np.argmin([area(t) for t in coarse]))]
    fine = best + np.radians(np.arange(-1.0, 1.0, 0.05))
    return float(fine[int(np.argmin([area(t) for t in fine]))])


def rectify(poly: np.ndarray, snap_deg: float = 10.0, step_m: float = 0.4) -> np.ndarray:
    """Polygon (in the grid frame, axes along the main walls) with edges within `snap_deg`
    of an axis made axis-parallel, neighbouring parallel edges less than `step_m` apart
    merged into one wall (length-weighted position, after edges shorter than `step_m` are
    dropped, and edges off the axes shorter than twice that, which are corners the grid
    trace cut), and corners at the intersections of
    consecutive walls. Edges at other angles (a bay, an angled wall) are kept as lines."""
    lines = []  # (unit direction, unit normal, offset along the normal, length)
    for a, b in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            continue
        d = d / length
        ang = np.degrees(np.arctan2(d[1], d[0]))
        snapped = round(ang / 90.0) * 90.0
        axis = abs(ang - snapped) <= snap_deg
        if axis:
            r = np.radians(snapped)
            d = np.array([np.cos(r), np.sin(r)]).round(12)
        n = np.array([-d[1], d[0]])
        lines.append([d, n, float(np.dot((a + b) / 2, n)), length, axis])
    # Edges shorter than a step (the riser of a 15 cm notch) go first, so the walls on
    # either side become neighbours and merge. A cut corner (a short diagonal between two
    # axis walls) goes too, while a bay or an angled wall is longer than 2 steps.
    while len(lines) > 4:
        short = [ln[3] / (1.0 if ln[4] else 2.0) for ln in lines]
        k = int(np.argmin(short))
        if short[k] >= step_m:
            break
        del lines[k]
    merged = True
    while merged and len(lines) > 3:
        merged = False
        for i in range(len(lines)):
            j = (i + 1) % len(lines)
            li, lj = lines[i], lines[j]
            if np.dot(li[0], lj[0]) > 0.9999 and abs(li[2] - lj[2]) < step_m:
                w = li[3] + lj[3]
                lines[i] = [li[0], li[1], (li[2] * li[3] + lj[2] * lj[3]) / w, w, li[4]]
                del lines[j]
                merged = True
                break
    out = []
    for i in range(len(lines)):
        (_, n0, o0, _, _), (_, n1, o1, _, _) = lines[i - 1], lines[i]
        A = np.stack([n0, n1])
        if abs(np.linalg.det(A)) < 1e-6:  # parallel neighbours further apart: keep both
            continue
        out.append(np.linalg.solve(A, [o0, o1]))
    return np.array(out)


def room_outline(
    floor_hits: list[np.ndarray],
    boxes_xz: list[tuple[np.ndarray, np.ndarray]],
    cell: float = 0.05,
    min_frames: int = 3,
    close_m: float = 0.3,
    tol_m: float = 0.08,
) -> np.ndarray:
    """Room outline (N, 2) in world x, z, with positive signed area in those axes.

    floor_hits: per frame, the x, z points where its floor pixels meet the floor plane. A
    cell is floor when at least `min_frames` frames hit it, so a few pixels of wall that
    SAM put in the floor mask (which land behind the wall) do not count. The grid is
    turned to the room's main wall direction, so walls run along cell edges."""
    allp = np.concatenate(floor_hits)
    th = main_angle(allp[:: max(1, len(allp) // 50_000)])
    c, s = np.cos(th), np.sin(th)
    rot = np.array([[c, -s], [s, c]])  # world -> grid: p @ rot
    lo = np.percentile(allp @ rot, 0.1, 0) - 3 * close_m
    hi = np.percentile(allp @ rot, 99.9, 0) + 3 * close_m
    shape = tuple(np.ceil((hi - lo) / cell).astype(int)[::-1])  # rows = second axis
    hits = np.zeros(shape, dtype=np.int32)
    for p in floor_hits:
        ij = ((p @ rot - lo) / cell).astype(int)
        ok = (ij >= 0).all(1) & (ij[:, 0] < shape[1]) & (ij[:, 1] < shape[0])
        once = np.zeros(shape, dtype=bool)
        once[ij[ok, 1], ij[ok, 0]] = True
        hits += once
    occ = hits >= min_frames
    for blo, bhi in boxes_xz:  # objects stand on floor no frame saw
        corners = np.array([[blo[0], blo[1]], [bhi[0], blo[1]], [bhi[0], bhi[1]], [blo[0], bhi[1]]])
        g = (corners @ rot - lo) / cell
        a = np.clip(np.floor(g.min(0)).astype(int), 0, np.array(shape[::-1]) - 1)
        b = np.clip(np.floor(g.max(0)).astype(int), 0, np.array(shape[::-1]) - 1)
        occ[a[1] : b[1] + 1, a[0] : b[0] + 1] = True
    occ = ortho_convex(_largest(_fill_holes(_close(occ, max(1, round(close_m / cell))))))
    loop = _trace(occ) * cell + lo
    poly = rectify(simplify_loop(loop, tol_m)) @ rot.T  # back to world x, z
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    return poly if area > 0 else poly[::-1]


def floor_points(K, R, t, floor_mask: np.ndarray, stride: int = 6) -> np.ndarray:
    """x, z where the floor pixels of one frame meet the plane y = 0. R, t map world to
    camera (x_cam = R x + t), R may carry a scale."""
    v, u = np.nonzero(floor_mask[::stride, ::stride])
    pix = np.stack([u * stride + 0.5 * stride, v * stride + 0.5 * stride, np.ones(len(u))], 1)
    d_cam = pix @ np.linalg.inv(K).T
    Rinv = np.linalg.inv(R)
    centre = -Rinv @ t
    d = d_cam @ Rinv.T
    lam = -centre[1] / np.where(np.abs(d[:, 1]) > 1e-9, d[:, 1], np.nan)
    ok = lam > 0
    p = centre + lam[ok, None] * d[ok]
    return p[:, [0, 2]]


def inside_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd test of (N, 2) points against a polygon (M, 2)."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        cross = (y0 > y) != (y1 > y)
        xi = x0 + (y - y0) * (x1 - x0) / np.where(y1 != y0, y1 - y0, 1e-12)
        inside ^= cross & (x < xi)
    return inside


def surfaces(poly: np.ndarray, height: float) -> list[Surface]:
    """Walls along the outline (counter-clockwise in x, z), floor and ceiling quads."""
    up = np.array([0.0, 1.0, 0.0])
    out = []
    lo, hi = poly.min(0), poly.max(0)
    size = hi - lo
    # Floor: u = +x, v = +z, seen from above. Ceiling: u = +x, v = -z, seen from below.
    out.append(Surface("floor", np.array([lo[0], 0.0, lo[1]]), np.array([1.0, 0, 0]),
                       np.array([0, 0, 1.0]), up.copy(), size[0], size[1],
                       outline_uv=poly - lo))  # fmt: skip
    ceil_uv = np.stack([poly[:, 0] - lo[0], hi[1] - poly[:, 1]], 1)
    out.append(Surface("ceiling", np.array([lo[0], height, hi[1]]), np.array([1.0, 0, 0]),
                       np.array([0, 0, -1.0]), -up, size[0], size[1],
                       outline_uv=ceil_uv))  # fmt: skip
    # The outline has positive signed area in (x, z), so the room lies to the left of each
    # edge d in those axes and the inward normal is (-d_z, d_x). This holds for an L-shaped
    # outline too, where a test against the centroid would flip walls.
    for a, b in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 0.2:
            continue
        d /= length
        n = np.array([-d[1], d[0]])
        # u along the wall, v downwards; the texture is seen from inside.
        origin = np.array([a[0], height, a[1]])
        u = np.array([d[0], 0.0, d[1]])
        out.append(Surface("wall", origin, u, -up, np.array([n[0], 0.0, n[1]]), length, height))
    return out


# ---- texture


@dataclass
class View:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    centre: np.ndarray
    image: np.ndarray  # (H, W, 3) uint8
    blocked: np.ndarray | None  # (H, W) bool, pixels that hide the room (furniture)


class Projector:
    """Keeps the `k` best samples per texel across views."""

    def __init__(self, pts: np.ndarray, normal: np.ndarray, k: int = 3, min_cos: float = 0.2):
        self.pts = pts.reshape(-1, 3)
        self.normal = normal
        self.k, self.min_cos = k, min_cos
        n = len(self.pts)
        self.score = np.zeros((n, k))
        self.col = np.zeros((n, k, 3))

    def add(self, view: View) -> int:
        to_cam = view.centre - self.pts
        dist = np.linalg.norm(to_cam, axis=1)
        cos = (to_cam @ self.normal) / np.maximum(dist, 1e-9)
        pc = self.pts @ view.R.T + view.t
        z = pc[:, 2]
        ok = (cos > self.min_cos) & (z > 0.1)
        uv = pc[:, :2] / np.maximum(z, 1e-9)[:, None] @ view.K[:2, :2].T + view.K[:2, 2]
        h, w = view.image.shape[:2]
        u, v = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
        ok &= (u >= 2) & (u < w - 2) & (v >= 2) & (v < h - 2)
        idx = np.flatnonzero(ok)
        if view.blocked is not None:
            idx = idx[~view.blocked[v[idx], u[idx]]]
        s = cos[idx] / np.maximum(dist[idx], 1e-6)
        slot = np.argmin(self.score[idx], axis=1)
        better = s > self.score[idx, slot]
        idx, slot, s = idx[better], slot[better], s[better]
        self.score[idx, slot] = s
        self.col[idx, slot] = view.image[v[idx], u[idx]] / 255.0
        return len(idx)

    def result(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """(H, W, 3) median colour and (H, W) coverage (1 where any view saw the texel)."""
        have = self.score > 0
        col = np.where(have[..., None], self.col, np.nan)
        with np.errstate(all="ignore"):
            med = np.nanmedian(col, axis=1)
        seen = have.any(1)
        med[~seen] = 0
        return med.reshape(*shape, 3), seen.reshape(shape).astype(np.float64)


def finish(rgb: np.ndarray, seen: np.ndarray, outline_uv, texel: float, size) -> np.ndarray:
    """RGBA uint8: unseen texels filled by push-pull, alpha 0 outside the outline."""
    filled = texture.push_pull(rgb * seen[..., None], seen)
    img = texture.to_uint8(filled)
    h, w = seen.shape
    if outline_uv is None:
        return np.dstack([img, np.full((h, w), 255, np.uint8)])
    j, i = np.meshgrid((np.arange(w) + 0.5) * size[0] / w, (np.arange(h) + 0.5) * size[1] / h)
    keep = inside_polygon(np.stack([j.ravel(), i.ravel()], 1), outline_uv).reshape(h, w)
    return np.dstack([img, np.where(keep, 255, 0).astype(np.uint8)])
