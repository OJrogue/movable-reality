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

import warnings
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


def main_angle(xz: np.ndarray, cell: float = 0.03, blur: int = 4) -> float:
    """Rotation (radians, 0 to 90 deg) of the room's main wall direction: the points are
    rasterised, and the edge directions of that coverage image are averaged with 4-fold
    symmetry (angle x 4 on the unit circle), so every wall along either axis votes for the
    same angle. The smallest bounding box, used before, turned blender02's L-shaped floor
    6 deg off, its corner half seen; this is 0.2 deg off on blender01 and blender02 (3 cm
    cells, 12 cm blur)."""
    p = np.asarray(xz, dtype=np.float64)
    lo = np.percentile(p, 0.5, 0) - 4 * cell
    ij = np.floor((p - lo) / cell).astype(int)
    ok = (ij >= 0).all(1)
    ij = ij[ok]
    shape = tuple(ij.max(0)[::-1] + 5)
    img = np.zeros(shape)
    np.add.at(img, (ij[:, 1], ij[:, 0]), 1.0)
    occ = _box_blur((img > 0)[..., None].astype(np.float64), blur)[..., 0]
    gy, gx = np.gradient(occ)
    w = gx**2 + gy**2
    return float(np.angle(np.sum(w * np.exp(4j * np.arctan2(gy, gx)))) / 4 % (np.pi / 2))


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
    box_reach_m: float = 0.1,
) -> np.ndarray:
    """Room outline (N, 2) in world x, z, with positive signed area in those axes.

    floor_hits: per frame, the x, z points where its floor pixels meet the floor plane. A
    cell is floor when at least `min_frames` frames hit it, so a few pixels of wall that
    SAM put in the floor mask (which land behind the wall) do not count. The grid is
    turned to the room's main wall direction, so walls run along cell edges. Object boxes
    add the floor under them (a bed no frame saw under) only within `box_reach_m` of the
    floor the frames saw, closed and filled: blender02's lift gave a desk a box 33 cm
    through the wall, and the outline followed it."""
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
    r_close = max(1, round(close_m / cell))
    seen = _grow(ortho_convex(_largest(_fill_holes(_close(occ, r_close)))),
                 round(box_reach_m / cell))  # fmt: skip
    for blo, bhi in boxes_xz:  # objects stand on floor no frame saw
        corners = np.array([[blo[0], blo[1]], [bhi[0], blo[1]], [bhi[0], bhi[1]], [blo[0], bhi[1]]])
        g = (corners @ rot - lo) / cell
        a = np.clip(np.floor(g.min(0)).astype(int), 0, np.array(shape[::-1]) - 1)
        b = np.clip(np.floor(g.max(0)).astype(int), 0, np.array(shape[::-1]) - 1)
        occ[a[1] : b[1] + 1, a[0] : b[0] + 1] |= seen[a[1] : b[1] + 1, a[0] : b[0] + 1]
    occ = ortho_convex(_largest(_fill_holes(_close(occ, r_close))))
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


def surface_distance(pts: np.ndarray, poly: np.ndarray, height: float) -> np.ndarray:
    """Distance (world units) from each world point to the nearest room surface: the floor
    y = 0, the ceiling y = height and the walls along the outline. Points outside the
    outline, below the floor or above the ceiling get a negative distance."""
    p = np.asarray(pts, dtype=np.float64)
    d = np.minimum(p[:, 1], height - p[:, 1])
    xz = p[:, [0, 2]]
    wall = np.full(len(p), np.inf)
    for a, b in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        e = b - a
        t = np.clip(((xz - a) @ e) / max(float(e @ e), 1e-12), 0.0, 1.0)
        wall = np.minimum(wall, np.linalg.norm(xz - (a + t[:, None] * e), axis=1))
    inside = inside_polygon(xz, poly)
    return np.where(inside, np.minimum(d, wall), -np.minimum(np.abs(d), wall))


def surface_splats(pts: np.ndarray, size: np.ndarray, opacity: np.ndarray, poly: np.ndarray,
                   height: float, gap: float = 0.06, reach: float = 2.0,
                   haze_scale: float = 0.10, haze_opacity: float = 0.3) -> np.ndarray:  # fmt: skip
    """Bool per splat (world centres, largest axis `size` in metres): True for the splats the
    room shell replaces. A splat goes when its extent (`reach` sigmas) comes within `gap` of a
    surface, when it lies outside the room, or when it is haze (larger than `haze_scale` and
    fainter than `haze_opacity`)."""
    d = surface_distance(pts, poly, height)
    haze = (size > haze_scale) & (opacity < haze_opacity)
    return (d - reach * size <= gap) | haze


SURFACES_PLY = "objects/background_room.ply"  # room_surfaces.py's reduced background


def background_source(scene: dict, shell: dict | None) -> str:
    """The background PLY the room stages read: the scan's, never room_surfaces.py's own
    reduced output (a rerun would filter its own result), then the pre-bake PLY."""
    if scene.get("background_ply") == SURFACES_PLY and shell and shell.get("surfaces_source_ply"):
        return shell["surfaces_source_ply"]
    bake = scene.get("plane_bake") or {}
    return bake.get("source_background_ply") or scene.get("background_ply")


def neighbour_counts(pts: np.ndarray, cell: float = 0.1) -> np.ndarray:
    """Per point, the number of points (itself included) in the 3 x 3 x 3 grid cells of size
    `cell` around its own. A splat with only a few is a floater (blender01: coloured 7 cm
    blobs 25 cm in front of the unfilmed upper walls, 2 or 3 neighbours)."""
    key = np.floor(np.asarray(pts, dtype=np.float64) / cell).astype(np.int64)
    uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    lookup = {tuple(k): c for k, c in zip(uniq.tolist(), cnt.tolist(), strict=True)}
    out = np.zeros(len(key), dtype=np.int64)
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    per_cell = np.array([sum(lookup.get((x + a, y + b, z + c), 0) for a, b, c in offs)
                         for x, y, z in uniq.tolist()])  # fmt: skip
    out[:] = per_cell[inv]
    return out


def occluder_zbuffer(pts_cam: np.ndarray, K: np.ndarray, shape: tuple[int, int],
                     cell: int = 4, grow: int = 2) -> np.ndarray:  # fmt: skip
    """Nearest depth of the occluder points (camera frame, +Z forward) on a grid of
    `cell` x `cell` pixel cells, min-filtered by `grow` cells so a cloud of splat centres
    closes into a surface. Upsampled to `shape` (H, W); inf where nothing is in front."""
    h, w = shape
    gh, gw = -(-h // cell), -(-w // cell)
    z = pts_cam[:, 2]
    ok = z > 1e-6
    uv = pts_cam[ok, :2] / z[ok, None] @ K[:2, :2].T + K[:2, 2]
    u, v = np.floor(uv[:, 0] / cell).astype(int), np.floor(uv[:, 1] / cell).astype(int)
    inb = (u >= 0) & (u < gw) & (v >= 0) & (v < gh)
    buf = np.full(gh * gw, np.inf)
    np.minimum.at(buf, v[inb] * gw + u[inb], z[ok][inb])
    buf = buf.reshape(gh, gw)
    if grow:
        k = 2 * grow + 1
        pad = np.pad(buf, grow, mode="edge")
        buf = np.lib.stride_tricks.sliding_window_view(pad, (k, k)).min(axis=(2, 3))
    return np.repeat(np.repeat(buf, cell, 0), cell, 1)[:h, :w]


@dataclass
class View:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    centre: np.ndarray
    image: np.ndarray  # (H, W, 3) uint8
    blocked: np.ndarray | None  # (H, W) bool, pixels that hide the room (furniture)
    near: float = 0.1  # nearest camera depth used, in the units of R, t
    zbuf: np.ndarray | None = None  # (H, W) nearest occluder depth (occluder_zbuffer)
    z_margin: float = 0.0  # a texel this far behind the occluder depth counts as hidden


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
        ok = (cos > self.min_cos) & (z > view.near)
        uv = pc[:, :2] / np.maximum(z, 1e-9)[:, None] @ view.K[:2, :2].T + view.K[:2, 2]
        h, w = view.image.shape[:2]
        u, v = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
        ok &= (u >= 2) & (u < w - 2) & (v >= 2) & (v < h - 2)
        idx = np.flatnonzero(ok)
        if view.blocked is not None:
            idx = idx[~view.blocked[v[idx], u[idx]]]
        if view.zbuf is not None:  # something not on a room surface stands in front
            idx = idx[z[idx] < view.zbuf[v[idx], u[idx]] + view.z_margin]
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are unseen texels
            med = np.nanmedian(col, axis=1)
        seen = have.any(1)
        med[~seen] = 0
        return med.reshape(*shape, 3), seen.reshape(shape).astype(np.float64)


UNSEEN_SHARE = 0.02  # a surface seen less than this gets the fallback colour

FILL_PROMPT = (
    "This is a flat, straight-on texture of the {kind} of a real room, made from photos. The "
    "areas painted magenta were hidden behind furniture and never photographed. Fill only "
    "the magenta areas with the same {kind} surface, continuing its material, pattern "
    "(planks, tiles, grain or paint) with the same alignment, colour and soft lighting as the "
    "surrounding pixels. No furniture, no objects, no shadows of objects, no text. Keep every "
    "non-magenta pixel and the framing exactly as they are."
)


def hole_mask(seen: np.ndarray, keep: np.ndarray, footprint: np.ndarray | None = None,
              grow: int = 2, island: int = 4) -> np.ndarray:  # fmt: skip
    """Texels to generate inside the outline: unseen ones, the `footprint` (furniture and its
    shadow, footprint_mask), and seen texels whose (2 * island + 1)^2 neighbourhood is mostly
    unseen (speckles of stray samples along a hole's edge). Grown by `grow` texels so the
    seam against a hole's blurred rim is replaced too."""
    bad = seen <= 0
    if footprint is not None:
        bad |= footprint
    if island:
        k = 2 * island + 1
        frac = np.lib.stride_tricks.sliding_window_view(
            np.pad(bad.astype(np.float32), island, mode="edge"), (k, k)
        ).mean(axis=(2, 3))
        bad |= frac > 0.5
    return _grow(bad & keep, grow) & keep


def _grow(m: np.ndarray, r: int) -> np.ndarray:
    if r <= 0 or not m.any():
        return m
    k = 2 * r + 1
    return np.lib.stride_tricks.sliding_window_view(np.pad(m, r), (k, k)).any(axis=(2, 3))


def footprint_mask(s: Surface, pts: np.ndarray, shape: tuple[int, int], reach: float,
                   grow_m: float) -> np.ndarray:  # fmt: skip
    """(H, W) texels of surface `s` that have an occluder point (furniture, world frame) in
    front of them within `reach` metres along the normal, grown by `grow_m`. The floor under
    and around furniture, and the wall behind it, hold its shadow: those texels are
    regenerated, or the shadow stays behind as a stain when the furniture moves."""
    h, w = shape
    rel = np.asarray(pts, dtype=np.float64) - s.origin
    d = rel @ s.normal
    sel = (d > 0) & (d < reach)
    u = np.floor(rel[sel] @ s.u / s.size_u * w).astype(int)
    v = np.floor(rel[sel] @ s.v / s.size_v * h).astype(int)
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    m = np.zeros((h, w), dtype=bool)
    m[v[inb], u[inb]] = True
    m = _grow(m, 1)  # close the gaps between splat centres
    return _grow(m, max(0, round(grow_m / (s.size_u / w))))


def shadow_mask(rgb: np.ndarray, good: np.ndarray, near: np.ndarray, cell: int = 24,
                drop: float = 0.18) -> np.ndarray:  # fmt: skip
    """(H, W) real texels in a furniture shadow: within `near` (texels close to a footprint)
    and more than `drop` darker than the surface's local bright level, the 80th percentile
    luminance of the real texels in `cell` x `cell` blocks, taken as the max over each
    block's 3 x 3 neighbourhood and blended back to texels. Plain paint and plank floors
    vary far less than a cast shadow."""
    h, w = good.shape
    lum = rgb.astype(np.float64) @ [0.299, 0.587, 0.114]
    gh, gw = -(-h // cell), -(-w // cell)
    ref = np.full((gh, gw), np.nan)
    for i in range(gh):
        for j in range(gw):
            blk = (slice(i * cell, (i + 1) * cell), slice(j * cell, (j + 1) * cell))
            v = lum[blk][good[blk]]
            if len(v) >= cell:
                ref[i, j] = np.percentile(v, 80)
    if np.isnan(ref).all():
        return np.zeros((h, w), dtype=bool)
    ref = np.where(np.isnan(ref), np.nanmedian(ref), ref)
    pad = np.pad(ref, 1, mode="edge")
    ref = np.lib.stride_tricks.sliding_window_view(pad, (3, 3)).max(axis=(2, 3))
    full = np.repeat(np.repeat(ref, cell, 0), cell, 1)[:h, :w]
    full = _box_blur(full[..., None], cell // 2)[..., 0]
    return near & good & (lum < (1 - drop) * full)


def match_colour(
    filled: np.ndarray, base: np.ndarray, holes: np.ndarray, ring: int = 8
) -> np.ndarray:
    """`filled` with its hole texels shifted and scaled per channel to the mean and spread
    of the real texels in a `ring` around the holes (Muse Image drifts in brightness)."""
    band = _grow(holes, ring) & ~holes
    if band.sum() < 50:
        return filled
    f = filled.astype(np.float64)
    src, ref = f[holes], base[band].astype(np.float64)
    gain = np.clip(ref.std(0) / np.maximum(src.std(0), 1e-6), 0.5, 2.0)
    f[holes] = (src - src.mean(0)) * gain + ref.mean(0)
    return np.clip(np.round(f), 0, 255).astype(np.uint8)


def _box_blur(img: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r + 1)^2 window (edges clamped), by cumulative sums."""
    k = 2 * r + 1
    p = np.pad(img.astype(np.float64), ((r + 1, r), (r + 1, r), (0, 0)), mode="edge")
    c = p.cumsum(0).cumsum(1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def surface_colour_gap(pts: np.ndarray, colours: np.ndarray, surfs: list[Surface],
                       textures: list[np.ndarray], blur: float = 0.05,
                       ) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Per world point, the distance to the nearest surface in front of it (the one it lies
    over, along that surface's normal) and the largest RGB difference (0 to 1) between its
    colour and that surface's texture under it, blurred over `blur` metres. Textures are
    uint8 (rows along v, columns along u) as room_shell.py writes them. Points over no
    surface get an infinite distance."""
    p = np.asarray(pts, dtype=np.float64)
    dist = np.full(len(p), np.inf)
    gap = np.full(len(p), np.inf)
    for s, tex in zip(surfs, textures, strict=True):
        h, w = tex.shape[:2]
        r = max(1, round(blur * w / s.size_u))
        img = _box_blur(tex[..., :3], r) / 255.0
        q = p - s.origin
        dn, u, v = q @ s.normal, q @ s.u, q @ s.v
        over = (dn >= 0) & (dn < dist) & (u >= 0) & (u < s.size_u) & (v >= 0) & (v < s.size_v)
        if tex.shape[-1] == 4:  # outside the floor or ceiling outline
            j = np.clip((u / s.size_u * w).astype(int), 0, w - 1)
            i = np.clip((v / s.size_v * h).astype(int), 0, h - 1)
            over &= tex[i, j, 3] > 0
        j = np.clip((u[over] / s.size_u * w).astype(int), 0, w - 1)
        i = np.clip((v[over] / s.size_v * h).astype(int), 0, h - 1)
        dist[over] = dn[over]
        gap[over] = np.abs(img[i, j] - colours[over]).max(1)
    return dist, gap


def components(m: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a bool mask, each as a bool mask (largest first)."""
    lab = np.zeros(m.shape, dtype=np.int32)
    out = []
    h, w = m.shape
    for seed in zip(*np.nonzero(m), strict=True):
        if lab[seed]:
            continue
        n = len(out) + 1
        stack, cells = [seed], []
        lab[seed] = n
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and m[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    stack.append((yy, xx))
        out.append(lab == n)
    return sorted(out, key=lambda c: -int(c.sum()))


def copy_fill(rgb: np.ndarray, holes: np.ndarray, keep: np.ndarray, ring: int = 6,
              step: int = 3, detail_r: int = 10, max_shift: float = 0.06,
              fade: int = 30, edge_shift: float = 0.2, seed: int = 0) -> np.ndarray:  # fmt: skip
    """Fill `holes` of a room-surface texture (H, W, 3) uint8 from the surface itself.

    Per hole, the offset whose shifted copy lands on real texels (inside `keep`, not a
    hole) and best matches the real texels in a `ring` around the hole (sum of squared
    differences on a sample of the ring) is chosen, so planks and tiles continue in line.
    The copy keeps its own colour and lighting; a smooth correction interpolated from the
    rim mismatch (its `detail_r` box blur, clamped to +-`max_shift`) removes the seam
    without letting a dark rim (a contact shadow) spread into the hole. Within `fade` texels of
    the rim a larger correction (up to +-`edge_shift`) fades in on top, so a soft shadow that
    runs on past the hole's edge meets the copy without a step (blender01: the armchair's
    shadow beyond the 1.2 m footprint reach). No learned weights.
    A hole with no valid offset takes push-pull from its rim."""
    h, w = holes.shape
    good = keep & ~holes
    f = rgb.astype(np.float64) / 255.0
    base = texture.push_pull(f * good[..., None], good.astype(np.float64))
    out = np.where(holes[..., None], base, f)
    rng = np.random.default_rng(seed)
    for comp in components(holes):
        band = _grow(comp, ring) & good
        ys, xs = np.nonzero(comp)
        by, bx = np.nonzero(band)
        if len(by) == 0:
            continue
        pick = rng.choice(len(by), min(300, len(by)), replace=False)
        by, bx = by[pick], bx[pick]
        target = f[by, bx]
        hy, hx = ys[:: max(1, len(ys) // 400)], xs[:: max(1, len(xs) // 400)]
        best, best_off = np.inf, None
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        for dy in range(-y0, h - y1, step):
            for dx in range(-x0, w - x1, step):
                if abs(dy) < y1 - y0 + ring and abs(dx) < x1 - x0 + ring:
                    continue  # the copy would overlap the hole itself
                if not good[hy + dy, hx + dx].mean() > 0.97:
                    continue
                yy, xx = by + dy, bx + dx
                inb = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
                if inb.mean() < 0.9:
                    continue
                e = float(((f[yy[inb], xx[inb]] - target[inb]) ** 2).mean())
                if e < best:
                    best, best_off = e, (dy, dx)
        if best_off is None:
            continue
        dy, dx = best_off
        ry0, ry1 = max(0, y0 - ring), min(h, y1 + ring + 1)
        rx0, rx1 = max(0, x0 - ring), min(w, x1 + ring + 1)
        sl = (slice(ry0, ry1), slice(rx0, rx1))
        cy, cx = np.mgrid[ry0:ry1, rx0:rx1]
        copy = f[np.clip(cy + dy, 0, h - 1), np.clip(cx + dx, 0, w - 1)]
        # The mismatch is taken between the two images' local means over `detail_r` texels
        # (real texels only), so the planks' own detail does not leak into the correction.
        rim = _grow(comp, ring)[sl] & good[sl]
        gw = _box_blur(good[sl][..., None].astype(np.float64), detail_r)
        mean_f = _box_blur(f[sl] * good[sl][..., None], detail_r) / np.maximum(gw, 1e-6)
        cgood = good[np.clip(cy + dy, 0, h - 1), np.clip(cx + dx, 0, w - 1)][..., None]
        cw = _box_blur(cgood.astype(np.float64), detail_r)
        mean_c = _box_blur(copy * cgood, detail_r) / np.maximum(cw, 1e-6)
        rim &= (gw[..., 0] > 0.5) & (cw[..., 0] > 0.5)  # both means stand on real texels
        if not rim.any():
            rim = _grow(comp, ring)[sl] & good[sl]
            mean_f, mean_c = f[sl], copy
        diff = np.where(rim[..., None], mean_f - mean_c, 0.0)
        corr = texture.push_pull(diff, rim.astype(np.float64))
        corr = _box_blur(corr, detail_r)
        near = np.zeros(comp[sl].shape)  # 1 at the rim, 0 from `fade` texels in
        ring_m = ~comp[sl]
        for k in range(fade, 0, -max(1, fade // 10)):
            near = np.maximum(near, np.where(_grow(ring_m, k), 1.0 - k / (fade + 1), 0.0))
        edge = np.clip(corr, -edge_shift, edge_shift) - np.clip(corr, -max_shift, max_shift)
        corr = np.clip(corr, -max_shift, max_shift) + near[..., None] * edge
        m = comp[sl]
        out[sl][m] = (copy + corr)[m]
    return texture.to_uint8(out)


def size_hint(w: int, h: int) -> str:
    """Muse Image's aspect hint closest to a w x h texture."""
    r = w / h
    return "1536x1024" if r > 1.25 else "1024x1536" if r < 0.8 else "1024x1024"


def blend_fill(base: np.ndarray, filled: np.ndarray, mask: np.ndarray,
               feather: int = 3) -> np.ndarray:  # fmt: skip
    """`base` (H, W, 3) uint8 with the masked texels taken from `filled` (same size), the
    weight ramping over `feather` texels outside the mask."""
    w = mask.astype(np.float64)
    for _ in range(feather):  # each pass grows the soft edge by one texel
        pad = np.pad(w, 1, mode="edge")
        grown = np.lib.stride_tricks.sliding_window_view(pad, (3, 3)).max(axis=(2, 3))
        w = np.maximum(w, 0.7 * grown)
    out = base.astype(np.float64) * (1 - w[..., None]) + filled.astype(np.float64) * w[..., None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def upper_wall_colour(walls: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    """Median colour of the seen texels in the upper half of the walls (row 0 is the top).
    A ceiling the walk never looked at takes it instead of push-pull's black."""
    px = [rgb[: len(rgb) // 2][seen[: len(seen) // 2] > 0] for rgb, seen in walls]
    px = [p for p in px if len(p)]
    return np.median(np.concatenate(px), axis=0) if px else None


def outline_keep(outline_uv, size, shape: tuple[int, int]) -> np.ndarray:
    """(H, W) bool, texels inside the outline (all of them without one)."""
    h, w = shape
    if outline_uv is None:
        return np.ones((h, w), dtype=bool)
    j, i = np.meshgrid((np.arange(w) + 0.5) * size[0] / w, (np.arange(h) + 0.5) * size[1] / h)
    return inside_polygon(np.stack([j.ravel(), i.ravel()], 1), outline_uv).reshape(h, w)


def finish(rgb: np.ndarray, seen: np.ndarray, outline_uv, texel: float, size,
           fallback: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    """RGBA uint8: unseen texels filled by push-pull, alpha 0 outside the outline. A surface
    seen less than UNSEEN_SHARE has nothing to push-pull from: its unseen texels take
    `fallback` (the upper wall colour) when given."""
    keep = outline_keep(outline_uv, size, seen.shape)
    # Seen share inside the outline, as shell.json reports it and the viewer reads it.
    if fallback is not None and float(np.mean(seen[keep] > 0)) < UNSEEN_SHARE:
        rgb = np.where(seen[..., None] > 0, rgb, np.asarray(fallback, dtype=rgb.dtype))
        seen = np.ones_like(seen)
    img = texture.to_uint8(texture.push_pull(rgb * seen[..., None], seen))
    # Opaque 2 texels past the outline: texels the edge cuts were clear, and filtered they
    # left a dotted crack along the wall base (blender01, 2 Nov). The walls hide the rest.
    alpha = np.where(_grow(keep, 2), 255, 0).astype(np.uint8)  # all True without an outline
    return np.dstack([img, alpha])
