"""Background behind and under objects, so that moving an object leaves no mark.

Nothing saw those surfaces, so their splats keep what training left there: the television's
view-dependent colour on the wall behind it, darker blotchy floor under the armchair (blender07).
A splat counts as unseen when hardly any frame shows it outside every object's mask, so floor
under a table, seen from the side, stays as it is. Per surface (floor, each wall of the room
outline), unseen splats get the colour and opacity of a seen splat, found by mirroring across the
edge of the covered area, so wood grain carries on without a seam; their view-dependent colour is
cleared. Empty covered cells get new splats and what lies behind a covered surface goes, so a
moved object leaves no hole. Wall splats that show a hung object join it, so nothing changes until
it moves. Background splats off every surface that lie on an object (books no noun named) move
into that object.

No learned model: geometry on the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CELL = 0.03  # m, grid of the covered area
GROW = 0.06  # m, a covered area's rim (contact shadows, mask edges)
FLOOR_COVER_TOP = 0.7  # m, object splats above this do not count as covering the floor
WALL_REACH = 0.5  # m, object splats further from a wall do not count as covering it
MIN_SEEN = 3  # views: a splat fewer frames show outside every object's mask is unseen...
MIN_SEEN_SHARE = 0.08  # ...and so is one this share of the frames that have it in view show
FLAT_HEIGHT = 0.06  # m, an object whose median splat is lower lies flat on the floor (a rug)
PLAIN_NEIGHBOURS = 128  # seen wall splats whose median colour a covered wall splat takes
HUNG_DEPTH = 0.2  # m, an object whose median splat is this close to a wall hangs on it...
HUNG_HEIGHT = 1.5  # m, ...if it is no taller than this
HUNG_THIN = 0.1  # m, a hung object nearly all this close (a picture) owns all the wall it covers
HUNG_COLOUR = 0.12 / 0.28209  # f_dc, about 0.12 in rgb: a wall splat this far from the wall's
# median colour, on or next to a hung object, shows that object (a picture's canvas, the
# television's dark edge), so it moves with it; repainting it would wipe the picture off the wall
SH_C0 = 0.28209  # f_dc * this + 0.5 is rgb
WALL_BEHIND = 0.02  # m, new wall splats sit this far back, so what hangs there draws over them
ON_SURFACE = 0.08  # m, a background splat this close to a surface is that surface
MAX_SIGMA = 0.03  # m, largest size of a repainted splat; a new splat on an empty cell has this
JUNK_AGREE = 0.1  # a splat near an object matching fewer of the frames that show it is junk
AGREE_COLOUR = 0.12  # rgb, largest difference from a frame's pixel that still counts as a match
AGREE_STEP = 3  # use every this many frames for that
ON_OBJECT = 0.5  # m, an off-surface background splat this close to an object may join it...
WALL_TINT = 0.1  # rgb, a splat of an object by a wall this close to the wall's colour is wall
WALL_NEAR = 0.5  # m, an object's median splat this close to a wall stands by it
SHOWN_AGREE = 0.5  # what lies past a covered surface is real if it matches this share of frames
BOX_MARGIN = (0.05, 0.12)  # m, ...if it is inside that object's box, grown by this, a quarter of
# the box's smallest side within these limits (blender07: at 4 cm the armchair's back stayed behind
# as a floating ghost and a dark patch; at 12 cm the soundbar took a slab of the wall's fog along;
# the nearest object is not always the right one, so the smallest box holding the splat decides)


def covered_grid(
    uv: np.ndarray, cover: np.ndarray, cell: float = CELL, grow: float = GROW
) -> tuple[np.ndarray, np.ndarray] | None:
    """Grid over a surface's splats uv [n, 2] of what cover [m, 2] lies over: (origin, bool grid),
    or None when nothing does."""
    from scipy.ndimage import binary_closing, binary_dilation  # noqa: PLC0415

    if not len(uv) or not len(cover):
        return None
    lo = uv.min(0)
    shape = np.ceil((uv.max(0) - lo) / cell).astype(int) + 1
    ij = np.floor((cover - lo) / cell).astype(int)
    ij = ij[np.all((ij >= 0) & (ij < shape), 1)]
    if not len(ij):
        return None
    grid = np.zeros(shape, bool)
    grid[ij[:, 0], ij[:, 1]] = True
    grid = binary_closing(grid, iterations=2)  # sparse splats still cover what lies between
    return lo, binary_dilation(grid, iterations=max(1, round(grow / cell)))


def in_grid(p: np.ndarray, lo: np.ndarray, grid: np.ndarray, cell: float = CELL) -> np.ndarray:
    """Per point p [n, 2]: inside the grid and covered."""
    at = np.floor((p - lo) / cell).astype(int)
    ok = np.all((at >= 0) & (at < grid.shape), 1)
    out = np.zeros(len(p), bool)
    out[ok] = grid[at[ok, 0], at[ok, 1]]
    return out


def mirror(
    p: np.ndarray,
    uv: np.ndarray,
    seen: np.ndarray,
    lo: np.ndarray,
    grid: np.ndarray,
    cell: float = CELL,
    colours: np.ndarray | None = None,
    k: int = PLAIN_NEIGHBOURS,
) -> np.ndarray:
    """For points p [k, 2] on covered cells: the index of the seen splat nearest each one's mirror
    image across the edge of the covered area. With colours [n, c] (a plain surface: a painted
    wall), the one of its k nearest closest to their median colour instead, so splats dirtied by
    the object's own edge are not copied."""
    from scipy.ndimage import distance_transform_edt  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    idx = distance_transform_edt(grid, return_distances=False, return_indices=True)
    at = np.floor((p - lo) / cell).astype(int)
    edge = lo + (np.stack([idx[0][at[:, 0], at[:, 1]], idx[1][at[:, 0], at[:, 1]]], 1) + 0.5) * cell
    tree = cKDTree(uv[seen])
    if colours is None or len(seen) < k:
        _, near = tree.query(2 * edge - p)
        return seen[near]
    _, near = tree.query(2 * edge - p, k=k)
    c = colours[seen[near]]
    best = np.linalg.norm(c - np.median(c, axis=1, keepdims=True), axis=2).argmin(axis=1)
    return seen[near[np.arange(len(p)), best]]


def mirror_donors(
    uv: np.ndarray, cover: np.ndarray, cell: float = CELL, grow: float = GROW
) -> tuple[np.ndarray, np.ndarray]:
    """uv [n, 2]: a surface's splats. cover [m, 2]: what lies over it. Returns the indices of the
    covered splats and, for each, the seen splat nearest its mirror image across the edge."""
    none = np.zeros(0, dtype=np.int64)
    got = covered_grid(uv, cover, cell, grow)
    if got is None:
        return none, none
    lo, grid = got
    hidden = in_grid(uv, lo, grid, cell)
    if not hidden.any() or hidden.all():
        return none, none
    h = np.flatnonzero(hidden)
    return h, mirror(uv[h], uv, np.flatnonzero(~hidden), lo, grid, cell)


def empty_cells(uv: np.ndarray, lo: np.ndarray, grid: np.ndarray, cell: float = CELL) -> np.ndarray:
    """Centres [k, 2] of the covered cells that hold no splat: holes a moved object would show."""
    full = np.zeros(grid.shape, bool)
    at = np.floor((uv - lo) / cell).astype(int)
    ok = np.all((at >= 0) & (at < grid.shape), 1)
    full[at[ok, 0], at[ok, 1]] = True
    return lo + (np.argwhere(grid & ~full) + 0.5) * cell


def repaint(
    v: np.ndarray, hidden: np.ndarray, donor: np.ndarray, max_log: float | None = None
) -> None:
    """In place: covered splats take a seen splat's base colour and opacity, no view dependence.
    max_log caps their log scale: a long one would poke out past the object's edge once bright."""
    for name in v.dtype.names:
        if name.startswith("f_dc_") or name == "opacity":
            v[name][hidden] = v[name][donor]
        elif name.startswith("f_rest_"):
            v[name][hidden] = 0
        elif name.startswith("scale_") and max_log is not None:
            v[name][hidden] = np.minimum(v[name][hidden], max_log)


def wall_coords(xyz: np.ndarray, a, b) -> tuple[np.ndarray, np.ndarray]:
    """World points against the wall over outline points a, b (x, z): (u along it, y) and the
    distance from its plane."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = (b - a) / np.linalg.norm(b - a)
    p = xyz[:, [0, 2]] - a
    return np.stack([p @ t, xyz[:, 1]], 1), np.abs(p[:, 0] * t[1] - p[:, 1] * t[0])


def unseen_splats(
    points: np.ndarray,
    cams: list,
    masks: Path,
    object_values: set[int],
    min_seen: int = MIN_SEEN,
    min_share: float = MIN_SEEN_SHARE,
) -> np.ndarray:
    """Per splat, points [n, 3] in the cameras' frame: True when hardly any frame shows it, i.e.
    nearly every frame that has it in view has an object's mask there. A floor splat under a
    table is seen from the side and stays; one under the sofa is not. Frames with no mask count
    for nothing."""
    from PIL import Image  # noqa: PLC0415

    seen = np.zeros(len(points), np.int32)
    in_view = np.zeros(len(points), np.int32)
    lut = np.zeros(256, bool)
    lut[sorted(object_values)] = True
    for c in cams:
        path = masks / f"{Path(c.name).stem}.png"
        if not path.exists():
            continue
        mask = np.asarray(Image.open(path))
        pc = c.to_camera(points)
        front = np.flatnonzero(pc[:, 2] > 0.01)
        K = np.asarray(c.K, dtype=np.float64)
        u = (K[0, 0] * pc[front, 0] / pc[front, 2] + K[0, 2]) * mask.shape[1] / c.width
        v = (K[1, 1] * pc[front, 1] / pc[front, 2] + K[1, 2]) * mask.shape[0] / c.height
        ok = (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
        idx = front[ok]
        in_view[idx] += 1
        seen[idx] += ~lut[mask[v[ok].astype(int), u[ok].astype(int)]]
    return (seen < min_seen) | (seen < min_share * in_view)


def frame_agreement(
    points: np.ndarray,
    rgb: np.ndarray,
    cams: list,
    masks: Path,
    frames: Path,
    object_values: set[int],
    step: int = AGREE_STEP,
) -> np.ndarray:
    """Per splat, points [n, 3] in the cameras' frame with rgb [n, 3] in 0..1: the share of frames
    showing it outside every object's mask whose pixel has its colour; 1 when none shows it. A real
    surface matches. A clump left behind an object does not, however often it is in view: an
    occluded lamp leg and the leather-coloured fog around it are seen equally often (blender07)."""
    from PIL import Image  # noqa: PLC0415

    seen = np.zeros(len(points), np.int32)
    agree = np.zeros(len(points), np.int32)
    lut = np.zeros(256, bool)
    lut[sorted(object_values)] = True
    for c in cams[::step]:
        mask_path = masks / f"{Path(c.name).stem}.png"
        frame_path = frames / Path(c.name).name
        if not mask_path.exists() or not frame_path.exists():
            continue
        mask = np.asarray(Image.open(mask_path))
        image = np.asarray(Image.open(frame_path).convert("RGB"), np.float32) / 255
        pc = c.to_camera(points)
        front = np.flatnonzero(pc[:, 2] > 0.01)
        K = np.asarray(c.K, dtype=np.float64)
        u = (K[0, 0] * pc[front, 0] / pc[front, 2] + K[0, 2]) / c.width
        v = (K[1, 1] * pc[front, 1] / pc[front, 2] + K[1, 2]) / c.height
        ok = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        front, u, v = front[ok], u[ok], v[ok]
        shown = ~lut[mask[(v * mask.shape[0]).astype(int), (u * mask.shape[1]).astype(int)]]
        front, u, v = front[shown], u[shown], v[shown]
        px = image[(v * image.shape[0]).astype(int), (u * image.shape[1]).astype(int)]
        seen[front] += 1
        agree[front] += np.abs(px - rgb[front]).max(1) < AGREE_COLOUR
    return np.where(seen > 0, agree / np.maximum(seen, 1), 1.0)


def wall_tinted(points: np.ndarray, rgb: np.ndarray, walls: list) -> np.ndarray:
    """Per splat of an object, points [n, 3] with rgb [n, 3]: True when it shows the wall behind it
    rather than the object, a mask having bled over the edge: a slab of wall would move along with
    a soundbar. walls: (a, b, rgb) per wall. Nothing for an object flat on its wall (a picture's
    canvas may match), away from every wall or itself of the wall's colour."""
    none = np.zeros(len(points), bool)
    if not len(points) or not walls:
        return none
    dists = [wall_coords(points, a, b)[1] for a, b, _ in walls]
    best = int(np.argmin([np.median(d) for d in dists]))
    wall = np.asarray(walls[best][2])
    depth = dists[best]
    if np.median(depth) > WALL_NEAR or np.percentile(depth, 95) < HUNG_THIN:
        return none
    if np.linalg.norm(np.median(rgb, axis=0) - wall) < 2 * WALL_TINT:
        return none
    return np.linalg.norm(rgb - wall, axis=1) < WALL_TINT


@dataclass
class Cleaned:
    joins: dict[str, np.ndarray]  # object id: background splats that join it
    drop: np.ndarray  # background splats behind a covered surface
    new_xyz: np.ndarray  # world points [k, 3] of new splats on empty covered cells...
    new_donor: np.ndarray  # ...and the background splat each copies
    stats: dict


def clean(
    bg: np.ndarray,
    bg_xyz: np.ndarray,
    objects: dict[str, np.ndarray],
    outline_xz: list,
    height: float,
    unit: float = 1.0,
    unseen: np.ndarray | None = None,
    flat: set[str] = frozenset(),
    agreement: np.ndarray | None = None,
) -> Cleaned:
    """bg: background vertices, changed in place where covered. bg_xyz [n, 3] in world, and
    objects {id: world points [m, 3]}; unit: metres per bg scale unit. unseen [n] (unseen_splats)
    picks the splats to repaint; without it, all that an object lies over. flat objects (a rug)
    cover nothing: the floor splats among them show the rug. agreement [n] (frame_agreement)
    finds junk near objects. The caller applies the rest: joins and drop leave the background,
    the new splats join it."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    max_log = float(np.log(MAX_SIGMA / unit))
    solid = [v for o, v in objects.items() if o not in flat]
    covers = np.concatenate(solid) if solid else np.zeros((0, 3))
    on_surface = np.abs(bg_xyz[:, 1]) < ON_SURFACE
    on_surface |= np.abs(bg_xyz[:, 1] - height) < ON_SURFACE
    stats = {"floor": 0, "walls": 0}
    drops, new_xyz, new_donor = [], [], []
    taken = np.zeros(len(bg), bool)  # wall splats that joined a hung object
    wall_joins: dict[str, list[np.ndarray]] = {}
    outside = np.zeros(len(bg), bool)  # past a wall: a garden through a window

    dc = [n for n in bg.dtype.names if n.startswith("f_dc_")]
    colours = np.stack([bg[n] for n in dc], 1) if dc else None

    def surface(idx, uv_all, cover, behind, place, key, plain=False, hung=()):
        """idx: this surface's splats; behind: splats past it. place(cells) -> world points. hung:
        (object id, its uv [m, 2], thin) lying on this surface; the splats showing it join it."""
        got = covered_grid(uv_all[idx], cover)
        if got is None:
            return
        lo, grid = got
        uv = uv_all[idx]
        joined = np.zeros(len(idx), bool)
        rim = np.zeros(len(idx), bool)
        tint = colours[idx] if plain and colours is not None else None
        for oid, huv, thin in hung:
            near = covered_grid(uv, huv)
            on = covered_grid(uv, huv, grow=CELL)
            if near is None or on is None:
                continue
            by_it = in_grid(uv, *near)
            rim |= by_it
            mine = in_grid(uv, *on) if thin else np.zeros(len(idx), bool)
            if tint is not None:
                far = np.linalg.norm(tint - np.median(tint, axis=0), axis=1) > HUNG_COLOUR
                mine |= far & by_it
            mine &= ~joined & ~taken[idx]
            if mine.any():
                joined |= mine
                taken[idx[mine]] = True
                wall_joins.setdefault(oid, []).append(idx[mine])
        hidden = ((in_grid(uv, lo, grid) if unseen is None else unseen[idx]) | rim) & ~joined
        seen = np.flatnonzero(~hidden & ~joined)
        if tint is not None and len(seen):  # no donors dirtied by what stands there
            mid = np.median(tint[seen], axis=0)
            clean_seen = np.linalg.norm(tint[seen] - mid, axis=1) < HUNG_COLOUR
            seen = seen[clean_seen] if clean_seen.any() else seen
        if not len(seen) or not (hidden.any() or joined.any()):
            return
        h = np.flatnonzero(hidden)
        if len(h):
            repaint(bg, idx[h], idx[mirror(uv[h], uv, seen, lo, grid, colours=tint)], max_log)
        stats[key] += len(h)
        past = behind[in_grid(uv_all[behind], lo, grid)]
        # A garden through a window stays: frames show it and it matches them. Junk past the wall
        # goes, though frames that look past the television's edge have it in view.
        real = past[:0] if unseen is None else past[~unseen[past]]
        if agreement is not None:
            real = real[agreement[real] >= SHOWN_AGREE]
        shown = real
        drops.append(np.setdiff1d(past, real))
        cells = empty_cells(uv[~joined], lo, grid)
        if len(cells) and len(shown):  # no wall to fill in where frames look past it
            gap, _ = cKDTree(uv_all[shown]).query(cells)
            cells = cells[gap > 2 * CELL]
        if len(cells):
            new_xyz.append(place(cells))
            new_donor.append(idx[mirror(cells, uv, seen, lo, grid, colours=tint)])

    floor = np.flatnonzero(np.abs(bg_xyz[:, 1]) < ON_SURFACE)
    if len(floor):
        level = float(np.median(bg_xyz[floor, 1]))
        covering = covers[covers[:, 1] < FLOOR_COVER_TOP]
        surface(
            floor,
            bg_xyz[:, [0, 2]],
            covering[:, [0, 2]],
            np.flatnonzero(bg_xyz[:, 1] <= -ON_SURFACE),
            lambda c: np.stack([c[:, 0], np.full(len(c), level), c[:, 1]], 1),
            "floor",
        )

    centre = np.mean(np.asarray(outline_xz, float), axis=0)
    for a, b in zip(outline_xz, [*outline_xz[1:], outline_xz[0]], strict=True):
        a_, b_ = np.asarray(a, float), np.asarray(b, float)
        t = (b_ - a_) / np.linalg.norm(b_ - a_)
        n = np.array([t[1], -t[0]])
        n = n if (centre - a_) @ n > 0 else -n  # towards the room
        uv, _ = wall_coords(bg_xyz, a, b)
        signed = (bg_xyz[:, [0, 2]] - a_) @ n
        near = np.abs(signed) < ON_SURFACE
        on_surface |= near
        wall = np.flatnonzero(near & (bg_xyz[:, 1] > ON_SURFACE))
        if not len(wall):
            continue
        outside |= signed <= -ON_SURFACE
        off = float(np.median(signed[wall]))
        cuv, cdist = wall_coords(covers, a, b)
        hung = []  # a television, a picture: flat on this wall and not floor to ceiling (a curtain)
        for o, pts in objects.items():
            if o in flat or not len(pts):
                continue
            y = np.percentile(pts[:, 1], [2, 98])
            huv, hdist = wall_coords(pts, a, b)
            if y[1] - y[0] < HUNG_HEIGHT and np.median(hdist) < HUNG_DEPTH:
                hung.append((o, huv, bool(np.percentile(hdist, 95) < HUNG_THIN)))

        def place(c, a_=a_, t=t, n=n, off=off):
            xz = a_ + c[:, 0, None] * t + (off - WALL_BEHIND) * n
            return np.stack([xz[:, 0], c[:, 1], xz[:, 1]], 1)

        surface(
            wall,
            uv,
            cuv[cdist < WALL_REACH],
            np.flatnonzero(signed <= -ON_SURFACE),
            place,
            "walls",
            plain=True,
            hung=hung,
        )

    drop = np.unique(np.concatenate(drops)) if drops else np.zeros(0, dtype=np.int64)
    joins: dict[str, np.ndarray] = {o: np.concatenate(g) for o, g in wall_joins.items()}
    stats["wall_joined"] = int(taken.sum())
    left = np.flatnonzero(~on_surface)
    left = left[~np.isin(left, drop)]
    # Twice: what joined widens a box, and the rest of a chair's back then falls inside it.
    for _ in range(2):
        grown = {
            o: np.concatenate([p, bg_xyz[joins[o]]]) if o in joins else p
            for o, p in objects.items()
            if len(p)
        }
        boxes = {o: np.percentile(p, [1, 99], axis=0) for o, p in grown.items()}
        # The smallest box first: an item on a shelf takes what lies on it before the shelf does.
        for oid in sorted(boxes, key=lambda o: np.prod(boxes[o][1] - boxes[o][0])):
            lo, hi = boxes[oid]
            pts = bg_xyz[left]
            margin = np.clip(0.25 * np.min(hi - lo), *BOX_MARGIN)
            inside = np.all((pts >= lo - margin) & (pts <= hi + margin), axis=1)
            inside = np.flatnonzero(inside)
            if not len(inside):
                continue
            dist, _ = cKDTree(grown[oid]).query(pts[inside], distance_upper_bound=ON_OBJECT)
            mine = inside[np.isfinite(dist)]
            if len(mine):
                joins[oid] = np.concatenate(
                    [joins.get(oid, np.zeros(0, dtype=np.int64)), left[mine]]
                )
                left = np.delete(left, mine)
    junk = np.zeros(0, dtype=np.int64)
    if agreement is not None and len(solid):
        rest = left[(agreement[left] < JUNK_AGREE) & ~outside[left] & (bg_xyz[left, 1] > 0)]
        if len(rest):
            dist, _ = cKDTree(covers).query(bg_xyz[rest], distance_upper_bound=ON_OBJECT)
            junk = rest[np.isfinite(dist)]
            drop = np.union1d(drop, junk)
    stats["junk"] = len(junk)
    stats["joined"] = int(sum(len(j) for j in joins.values())) - stats["wall_joined"]
    stats["behind_removed"] = len(drop) - len(junk)
    xyz = np.concatenate(new_xyz) if new_xyz else np.zeros((0, 3))
    donor = np.concatenate(new_donor) if new_donor else np.zeros(0, dtype=np.int64)
    stats["new"] = len(xyz)
    return Cleaned(joins, drop, xyz, donor, stats)


def run(root: Path) -> dict:
    """install() afresh (so this can run again), then clean: objects/background_clean.ply, the
    joined splats in their objects, boxes refitted, shell off. build_lod.py must follow."""
    import json  # noqa: PLC0415

    from .coverage.colmap import read_model  # noqa: PLC0415
    from .objtrain import install  # noqa: PLC0415
    from .scene import Scene  # noqa: PLC0415
    from .splat_ops import read_vertices, tame_splats, write_vertices  # noqa: PLC0415

    install(root)
    shell = json.loads((root / "room" / "shell.json").read_text())
    scene = Scene.load(root / "scene.json")
    m = np.asarray(scene.world_from_recon, dtype=np.float64)
    unit = float(np.linalg.norm(m[:3, 0]))

    def world(v: np.ndarray) -> np.ndarray:
        return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64) @ m[:3, :3].T + m[:3, 3]

    bg = read_vertices(root / "objects" / "background.ply")
    shown = [o for o in scene.objects if o.source == "scan" and o.visible and o.ply]
    verts = {o.id: read_vertices(root / o.ply) for o in shown}
    points = {i: world(v) for i, v in verts.items()}
    flat = {i for i, p in points.items() if np.median(p[:, 1]) < FLAT_HEIGHT}
    labels = json.loads((root / "ids" / "masks" / "labels.json").read_text())
    values = {
        int(k)
        for k, name in labels.items()
        if name.split("#")[0] not in ("floor", "wall", "ceiling") and name not in flat
    }
    xyz_recon = np.stack([bg["x"], bg["y"], bg["z"]], 1)
    cams = read_model(root / "recon_full")
    unseen = unseen_splats(xyz_recon, cams, root / "ids" / "masks", values)
    print(f"{int(unseen.sum())} of {len(bg)} background splats unseen, flat objects {sorted(flat)}")
    rgb = np.clip(np.stack([bg[f"f_dc_{i}"] for i in range(3)], 1) * SH_C0 + 0.5, 0, 1)
    agreement = frame_agreement(
        xyz_recon, rgb, cams, root / "ids" / "masks", root / "frames", values
    )
    done = clean(
        bg,
        world(bg),
        points,
        shell["outline_xz"],
        float(shell["height_m"]),
        unit,
        unseen,
        flat,
        agreement,
    )
    outline = shell["outline_xz"]
    bg_world = world(bg)
    walls = []
    for a, b in zip(outline, [*outline[1:], outline[0]], strict=True):
        near = wall_coords(bg_world, a, b)[1] < ON_SURFACE
        if near.any():
            walls.append((a, b, np.median(rgb[near], axis=0)))
    stayed = []
    for obj in shown:
        got = done.joins.get(obj.id)
        v = verts[obj.id]
        if got is not None and len(got):
            add = np.zeros(len(got), dtype=v.dtype)
            for name in add.dtype.names:
                if name in bg.dtype.names:
                    add[name] = bg[name][got]
            v = tame_splats(np.concatenate([v, add]), unit)
        own = np.clip(np.stack([v[f"f_dc_{i}"] for i in range(3)], 1) * SH_C0 + 0.5, 0, 1)
        wall_part = wall_tinted(world(v), own, walls)
        if wall_part.any():
            back = np.zeros(int(wall_part.sum()), dtype=bg.dtype)
            for name in back.dtype.names:
                if name in v.dtype.names:
                    back[name] = v[name][wall_part]
            stayed.append(back)
            v = v[~wall_part]
        elif got is None or not len(got):
            continue
        write_vertices(root / obj.ply, v)
        xyz = world(v)
        lo, hi = np.percentile(xyz, 0.5, axis=0), np.percentile(xyz, 99.5, axis=0)
        obj.bbox_min, obj.bbox_max = tuple(map(float, lo)), tuple(map(float, hi))
    done.stats["wall_tinted"] = int(sum(len(x) for x in stayed))
    # New splats: a copy of the donor at the cell, round and without view dependence.
    new = bg[done.new_donor].copy()
    recon = (done.new_xyz - m[:3, 3]) @ np.linalg.inv(m[:3, :3]).T
    for i, axis in enumerate("xyz"):
        new[axis] = recon[:, i]
    for name in new.dtype.names:
        if name == "opacity":
            new[name] = 4.0  # opaque: nothing behind the wall shows through
        elif name.startswith("scale_"):
            new[name] = np.log(MAX_SIGMA / unit)
        elif name.startswith("f_rest_"):
            new[name] = 0
    keep = np.ones(len(bg), bool)
    keep[done.drop] = False
    for got in done.joins.values():
        keep[got] = False
    write_vertices(
        root / "objects" / "background_clean.ply", np.concatenate([bg[keep], new, *stayed])
    )
    scene.background_ply, scene.background_lod = "objects/background_clean.ply", None
    scene.save(root / "scene.json")
    return done.stats
