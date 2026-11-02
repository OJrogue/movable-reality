"""blender07: a realistic living room, the second main synthetic test room (2 Nov).

    "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b -P scripts/blender_living.py \
        -- --out data/raw/blender07                   # build -> blender07.blend + GT objects
    ... -- --stage build,preview --samples 64         # + preview stills in <out>/preview/
    ... -- --stage preview --views tv_wall,sofa_wall   # stills from the saved .blend

A 5.6 x 4.8 m living room, 2.7 m high, in one modern Scandinavian style: light oak, warm
grey fabric, matte black metal, white ceramics, abstract art. Two casement windows in the
north wall look out onto a park (an HDRI) and let the daylight in, each with a radiator
under it and floor-length linen curtains; a tall plant in a white planter stands between
them. Two closed flush doors in the south wall, white (hall) and oak, with black levers.
The west wall is a deep teal accent with a wall-mounted 55" TV (on, a mountain lake at
dusk), a soundbar, a slatted oak media console (games console, succulent, vase) and a
cable raceway; a white floor vase beside it. Facing it on the east wall: a low grey
three-seat sofa (built here) with linen cushions and a cloth-draped throw, Kandinsky's
Yellow-Red-Blue on canvas above it and an oak side table with a lamp and books. In
between: a round marble coffee table (books, a tray with a candle and the remote, a mug, a
bowl) on a bordered wool rug, a knitted pouf, an oak and leather lounge chair, a tripod
floor lamp and an opal globe pendant. An oak and black-steel bookshelf between the doors
holds books, vases, a statue, a photo frame, a candle, a succulent and a basket. Framed
Mondrian, Kandinsky, Klee and af Klint prints, switches, sockets, a thermostat and a smoke
detector complete it.

Assets: CC0 Poly Haven models, textures and the HDRI, public-domain paintings from Wikimedia
Commons (scripts/blender_assets.py, docs/ASSETS.md). Everything else is built here, mostly
with the blender06 helpers in scripts/blender_bedroom.py.

Light: the HDRI through both windows (a light portal in each) plus a soft ceiling-sized
fill, so no lamp leaves a hotspot that would stay behind when furniture moves. --warm-lamps
also lights the pendant, the floor lamp and the table lamp.

Outputs (out = data/raw/blender07):
  <out>/blender07.blend           the scene, every object named as in objects.json
  <out>/tv_screen.png             the picture on the TV (made here)
  <out>/preview/<view>.png        preview stills (phone camera, 26 mm)
  <out>_gt/objects.json           index -> id, noun, support, world bbox (blender_room format)
  <out>_gt/meshes/<name>.ply      each object's final mesh in world coordinates

Same conventions as scripts/blender_room.py: Blender +Z up, world frame +Y up with the
floor at y = 0, objects' pass_index = their index in objects.json (structure 101-106; both
windows are one "window" object, both doors one "door" object). Walks are not defined yet.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_bedroom import (  # noqa: E402
    Parts,
    _ring,
    apply_modifiers,
    bake_transform,
    book_mats,
    build_book_stack,
    build_mug,
    build_poster,
    check_overlaps,
    cloth_sheet,
    curtain_panel,
    drop_material_faces,
    flat_mat,
    image_mat,
    join,
    pbr_mat,
    pillow,
    place,
    quads_object,
    rect_x,
    rect_y,
    top_z,
)
from blender_bedroom import materials as bedroom_materials  # noqa: E402
from blender_room import (  # noqa: E402
    FOCAL_MM,
    SENSOR_MM,
    STRUCTURE,
    WORLD_FROM_BLENDER,
    _bpy,
    _export_mesh,
    _world_bbox_project,
    import_asset,
    look_at,
    setup_render,
    world_bbox,
)

SCENE = "blender07"
RX, RY, RH = 5.6, 4.8, 2.7  # Blender x (width), y (length), z (height)
HX, HY = RX / 2, RY / 2
REPO = Path(__file__).resolve().parents[1]

# Two windows in the north wall (y = HY): x ranges, shared sill and head height, reveal depth.
WINDOWS = [(-2.15, -0.85), (0.55, 1.85)]
WIN_Z0, WIN_Z1, WIN_DEPTH = 0.80, 2.30, 0.30
# Two doors in the south wall (y = -HY): x range, style, hinge side. Both hinge west.
DOOR_H = 2.05
DOORS = [(-2.45, -1.60, "white"), (1.72, 2.57, "oak")]
WALL_TINT = (0.93, 0.91, 0.87)  # warm white
ACCENT_TINT = (0.30, 0.40, 0.43)  # deep teal, the TV wall (west)
RUG_T = 0.012  # rug thickness; what stands on the rug stands this much higher

# Preview views: name -> (eye, target), Blender metres, phone at chest or eye height.
VIEWS = {
    "from_hall_door": ((-1.95, -1.75, 1.55), (1.4, 1.0, 0.9)),
    "tv_wall": ((1.35, 0.15, 1.35), (-2.8, 0.15, 1.0)),
    "sofa_wall": ((-1.6, 0.0, 1.55), (2.6, 0.1, 1.0)),
    "windows": ((0.6, -1.7, 1.55), (-0.3, 2.4, 1.25)),
    "shelf_doors": ((0.2, 1.4, 1.6), (1.0, -2.4, 1.0)),
    "coffee_close": ((0.05, -0.55, 1.15), (0.8, 0.25, 0.35)),
    "shelf_close": ((0.9, -0.9, 1.45), (1.0, -2.4, 1.15)),
    "tv_close": ((-1.2, -0.6, 1.3), (-2.7, 0.3, 0.75)),
}


# ---------------------------------------------------------------- helpers


def surface_z(obj, x: float, y: float, z0: float = 3.0) -> float:
    """Height of the first upward surface of obj below (x, y, z0), world coordinates."""
    from mathutils import Vector  # noqa: PLC0415

    _bpy().context.view_layer.update()
    mw = obj.matrix_world
    inv = mw.inverted()
    ok, loc, _, _ = obj.ray_cast(inv @ Vector((x, y, z0)), inv.to_3x3() @ Vector((0, 0, -1)))
    if not ok:
        raise RuntimeError(f"no surface of {obj.name} under ({x}, {y})")
    return float((mw @ loc).z)


def against(obj, wall: str, gap: float = 0.01) -> None:
    """Slide obj along the wall's normal until it is gap from that wall."""
    lo, hi = world_bbox(obj)
    if wall == "west":
        obj.location.x += -HX + gap - lo[0]
    elif wall == "east":
        obj.location.x += HX - gap - hi[0]
    elif wall == "south":
        obj.location.y += -HY + gap - lo[1]
    else:
        obj.location.y += HY - gap - hi[1]
    _bpy().context.view_layer.update()


def _bvh(obj, offset=(0.0, 0.0, 0.0)):
    """BVH of obj's mesh in world coordinates (shifted by offset) and its world bbox."""
    from mathutils.bvhtree import BVHTree  # noqa: PLC0415

    me = obj.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    mw = np.array(obj.matrix_world)
    w = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3] + np.asarray(offset)
    tree = BVHTree.FromPolygons([tuple(v) for v in w], [tuple(f.vertices) for f in me.polygons])
    return tree, (w.min(0), w.max(0))


def settle(obj, other, step, max_steps: int = 60) -> None:
    """Move obj (its mesh) by step until it no longer intersects other."""
    from mathutils import Matrix  # noqa: PLC0415

    t_other = _bvh(other)[0]
    for _ in range(max_steps):
        if not _bvh(obj)[0].overlap(t_other):
            return
        obj.data.transform(Matrix.Translation(step))
    raise RuntimeError(f"{obj.name} still intersects {other.name}")


def check_clipping(objs: list, skip: set, tol: float = 0.003) -> None:
    """Fail the build when two meshes interpenetrate. Touching (an item on its support, a
    frame on the wall) is fine: a pair whose triangles intersect counts as clipping only
    when moving the smaller one by tol along no axis separates them. Pairs with both names
    in skip are not checked: the structure (window frames fit their openings exactly) and
    the electrics, which sit on all four walls and the ceiling at once."""
    data = {o.name: (o, *_bvh(o)) for o in objs if o.type == "MESH"}
    names = sorted(data)
    nudges = [(0, 0, tol), (0, 0, -tol), (tol, 0, 0), (-tol, 0, 0), (0, tol, 0), (0, -tol, 0)]
    bad = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if a in skip and b in skip:
                continue
            (oa, ta, (la, ha)), (ob, tb, (lb, hb)) = data[a], data[b]
            if (np.minimum(ha, hb) - np.maximum(la, lb) < 0).any() or not ta.overlap(tb):
                continue
            small, big = (oa, tb) if len(oa.data.vertices) <= len(ob.data.vertices) else (ob, ta)
            if not any(not _bvh(small, d)[0].overlap(big) for d in nudges):
                bad.append(f"{a} / {b}")
    if bad:
        raise RuntimeError("meshes clip: " + "; ".join(bad))


# ---------------------------------------------------------------- materials


def materials(assets: Path) -> dict:
    """The blender06 set with this room's wall, floor, fabrics, plus its own pieces."""
    a = assets
    m = bedroom_materials(assets)
    m.update(
        floor=pbr_mat("floor", a / "wood_floor", 1.7, tint=(0.97, 0.88, 0.76)),
        wall=pbr_mat("wall", a / "white_stucco", 2.0, tint=WALL_TINT, rough_add=0.15, bump=0.4),
        accent=pbr_mat(
            "accent_wall", a / "white_stucco", 2.0, tint=ACCENT_TINT, rough_add=0.15, bump=0.4
        ),
        curtain=pbr_mat(
            "curtain", a / "rough_linen", 0.27, uv=True, sat=0.2, tint=(0.80, 0.74, 0.62), sheen=0.3
        ),
        rug=pbr_mat(
            "rug", a / "poly_wool_herringbone", 0.45, sat=0.4, tint=(0.93, 0.88, 0.80), sheen=0.5
        ),
        rug_border=pbr_mat(
            "rug_border",
            a / "poly_wool_herringbone",
            0.45,
            sat=0.0,
            tint=(0.25, 0.27, 0.30),
            sheen=0.5,
        ),  # fmt: skip
        throw=pbr_mat(
            "throw",
            a / "waffle_pique_cotton",
            0.28,
            uv=True,
            sat=0.6,
            tint=(0.75, 0.42, 0.25),
            sheen=0.4,
        ),  # fmt: skip
        door_oak=pbr_mat("door_oak", a / "white_oak_veneer", 0.8, tint=(0.95, 0.85, 0.70)),
        shade=pbr_mat(
            "lamp_shade", a / "rough_linen", 0.2, sat=0.0, tint=(0.93, 0.90, 0.84), sheen=0.2
        ),
        tv_black=flat_mat("tv_black", (0.012, 0.012, 0.013), rough=0.35),
        tv_back=flat_mat("tv_back", (0.04, 0.04, 0.045), rough=0.6),
        soundbar=flat_mat("soundbar_fabric", (0.025, 0.025, 0.028), rough=0.9, noise=0.1),
        canvas_edge=flat_mat("canvas_edge", (0.80, 0.77, 0.70), rough=0.8, noise=0.05),
        brass=flat_mat("brass", (0.80, 0.62, 0.32), rough=0.3, metal=1.0),
        lamp_base=flat_mat("lamp_base_glaze", (0.42, 0.52, 0.44), rough=0.15, coat=0.5),
        wax=flat_mat("candle_wax", (0.93, 0.90, 0.83), rough=0.5),
        jar=flat_mat("candle_jar", (0.12, 0.13, 0.14), rough=0.25, coat=0.4),
        wick=flat_mat("wick", (0.05, 0.04, 0.03), rough=0.8),
        remote=flat_mat("remote", (0.05, 0.05, 0.055), rough=0.35),
        button=flat_mat("remote_buttons", (0.25, 0.25, 0.27), rough=0.5),
        red_button=flat_mat("power_button", (0.55, 0.06, 0.05), rough=0.4),
        tray=pbr_mat("tray_wood", a / "white_oak_veneer", 0.4, tint=(0.55, 0.40, 0.28)),
        ceramic=flat_mat("mug_glaze", (0.85, 0.82, 0.76), rough=0.2, coat=0.4, noise=0.05),
        cable=flat_mat("cable", (0.02, 0.02, 0.02), rough=0.5),
        thermostat=flat_mat("thermostat_display", (0.05, 0.08, 0.10), rough=0.1),
        sofa=pbr_mat(
            "sofa_fabric", a / "rough_linen", 0.25, sat=0.0, tint=(0.60, 0.58, 0.55), sheen=0.5
        ),
        knit=pbr_mat(
            "pouf_knit",
            a / "waffle_pique_cotton",
            0.12,
            sat=0.0,
            tint=(0.86, 0.82, 0.74),
            sheen=0.4,
        ),  # fmt: skip
        cushion_mustard=pbr_mat(
            "cushion_mustard", a / "rough_linen", 0.25, sat=0.0, tint=(0.78, 0.56, 0.18), sheen=0.4
        ),
        cushion_rust=pbr_mat(
            "cushion_rust", a / "rough_linen", 0.25, sat=0.0, tint=(0.62, 0.30, 0.18), sheen=0.4
        ),
        cushion_sage=pbr_mat(
            "cushion_sage", a / "rough_linen", 0.25, sat=0.0, tint=(0.50, 0.56, 0.46), sheen=0.4
        ),
        planter=flat_mat("planter", (0.86, 0.85, 0.82), rough=0.6, noise=0.04),
        soil=flat_mat("soil", (0.07, 0.05, 0.035), rough=0.95, noise=0.2),
        handle=flat_mat("black_handle", (0.02, 0.02, 0.02), rough=0.35, metal=0.8),
    )
    return m


# ---------------------------------------------------------------- room shell


def build_shell(m: dict) -> dict:
    """Floor, walls (two window holes and reveals in the north wall), ceiling, skirting."""
    h = RH
    floor = quads_object("floor", [[(-HX, -HY, 0), (HX, -HY, 0), (HX, HY, 0), (-HX, HY, 0)]],
                         [m["floor"]])  # fmt: skip
    ceiling = quads_object(
        "ceiling", [[(-HX, -HY, h), (-HX, HY, h), (HX, HY, h), (HX, -HY, h)]], [m["ceiling"]]
    )
    yw, yo = HY, HY + WIN_DEPTH
    quads = [rect_y(-HX, -HY, HY, 0, h, +1)]  # west (accent, material 1)
    quads += [rect_y(HX, -HY, HY, 0, h, -1), rect_x(-HY, -HX, HX, 0, h, +1)]  # east, south
    edges = [-HX] + [x for w in WINDOWS for x in w] + [HX]
    for a, b in zip(edges[::2], edges[1::2], strict=True):  # full-height piers
        quads.append(rect_x(yw, a, b, 0, h, -1))
    for x0, x1 in WINDOWS:
        quads += [
            rect_x(yw, x0, x1, 0, WIN_Z0, -1),
            rect_x(yw, x0, x1, WIN_Z1, h, -1),
            rect_y(x0, yw, yo, WIN_Z0, WIN_Z1, +1),
            rect_y(x1, yw, yo, WIN_Z0, WIN_Z1, -1),
            [(x0, yw, WIN_Z1), (x0, yo, WIN_Z1), (x1, yo, WIN_Z1), (x1, yw, WIN_Z1)],
            [(x0, yw, WIN_Z0), (x1, yw, WIN_Z0), (x1, yo, WIN_Z0), (x0, yo, WIN_Z0)],
        ]
    walls = quads_object("wall", quads, [m["wall"], m["accent"]], [1] + [0] * (len(quads) - 1))

    # Skirting boards, 10 cm high and 1.8 cm deep, except across the doors.
    sk = Parts("skirting")
    t, hs, w = 0.018, 0.10, m["white_paint"]
    sk.box((-HX, -HY, 0), (-HX + t, HY, hs), w)
    sk.box((HX - t, -HY, 0), (HX, HY, hs), w)
    sk.box((-HX, HY - t, 0), (HX, HY, hs), w)
    xs = [-HX] + [x for x0, x1, _ in DOORS for x in (x0 - 0.075, x1 + 0.075)] + [HX]
    for a, b in zip(xs[::2], xs[1::2], strict=True):
        sk.box((a, -HY, 0), (b, -HY + t, hs), w)
    trim = sk.build(bevel=0.004)
    walls = join("wall", [walls, trim])
    for obj, key in ((floor, "floor"), (walls, "wall"), (ceiling, "ceiling")):
        obj.pass_index = STRUCTURE[key]
    return {"floor": floor, "wall": walls, "ceiling": ceiling}


def _window(p: Parts, m: dict, x0: float, x1: float) -> None:
    """Two-casement white window in the reveal x0..x1, glass, handle, interior sill."""
    w = m["white_plastic"]
    yf0, yf1 = HY + 0.12, HY + 0.19  # frame depth inside the reveal
    z0, z1 = WIN_Z0, WIN_Z1
    f = 0.06
    p.box((x0, yf0, z0), (x1, yf1, z0 + f), w)
    p.box((x0, yf0, z1 - f), (x1, yf1, z1), w)
    p.box((x0, yf0, z0), (x0 + f, yf1, z1), w)
    p.box((x1 - f, yf0, z0), (x1, yf1, z1), w)
    xm = (x0 + x1) / 2
    s = 0.055
    for a, b in ((x0 + f, xm), (xm, x1 - f)):
        ys0, ys1 = yf0 - 0.012, yf1 - 0.02
        p.box((a, ys0, z0 + f), (b, ys1, z0 + f + s), w)
        p.box((a, ys0, z1 - f - s), (b, ys1, z1 - f), w)
        p.box((a, ys0, z0 + f), (a + s, ys1, z1 - f), w)
        p.box((b - s, ys0, z0 + f), (b, ys1, z1 - f), w)
        p.box((a + s, yf0 + 0.02, z0 + f + s), (b - s, yf0 + 0.026, z1 - f - s), m["glass"])
        # Transom bar: the upper third of each casement is a separate light.
        zt = z1 - f - s - 0.42
        p.box((a + s, ys0 + 0.01, zt - 0.02), (b - s, ys1 - 0.005, zt + 0.02), w)
    hx = xm + 0.028
    p.box((hx - 0.014, yf0 - 0.024, 1.47), (hx + 0.014, yf0 - 0.012, 1.55), w)
    p.cyl((hx, yf0 - 0.03, 1.51), 0.007, 0.007, 0.02, m["chrome"], rot=(math.pi / 2, 0, 0))
    p.box((hx - 0.009, yf0 - 0.045, 1.39), (hx + 0.009, yf0 - 0.032, 1.52), w)
    # Interior sill: a pale stone board 4 cm proud of the wall.
    p.box((x0 - 0.05, HY - 0.04, WIN_Z0 - 0.03), (x1 + 0.05, yf0, WIN_Z0), m["sill"])


def build_windows(m: dict):
    p = Parts("window")
    for x0, x1 in WINDOWS:
        _window(p, m, x0, x1)
    win = p.build(bevel=0.002)
    win.pass_index = STRUCTURE["window"]
    return win


def build_curtains(m: dict, x0: float, x1: float, name: str):
    """Rod 25 cm wider than the window on each side, two gathered floor-length panels."""
    rod = Parts(name + "_rod")
    yr, zr = HY - 0.10, 2.50
    a, b = x0 - 0.28, x1 + 0.28
    rod.cyl(((a + b) / 2, yr, zr), 0.012, 0.012, b - a, m["black_metal"],
            rot=(0, math.pi / 2, 0))  # fmt: skip
    for xe in (a, b):
        rod.sphere((xe, yr, zr), (0.024, 0.024, 0.024), m["black_metal"], segs=16)
        xb = xe + (0.06 if xe == a else -0.06)
        rod.cyl((xb, HY - 0.05, zr), 0.008, 0.008, 0.1, m["black_metal"], rot=(math.pi / 2, 0, 0))
        rod.box((xb - 0.025, HY - 0.006, zr - 0.035), (xb + 0.025, HY, zr + 0.035),
                m["black_metal"])  # fmt: skip
    parts = [rod.build()]
    for cx0, cx1 in ((a + 0.04, x0 + 0.12), (x1 - 0.12, b - 0.04)):
        parts.append(curtain_panel(cx0, cx1, yr, zr - 0.03, m))
        for i in range(9):
            parts.append(_ring(cx0 + (cx1 - cx0) * i / 8, yr, zr, m))
    return join(name, parts)


def build_doors(m: dict):
    """Two closed flush doors in the south wall, one object: white casings, a white door
    (hall) and an oak door, black lever handles on the east (latch) side, hinges west."""
    p = Parts("door")
    y = -HY
    cw, cd, d = 0.075, 0.02, 0.04
    w = m["white_paint"]
    for x0, x1, style in DOORS:
        p.box((x0 - cw, y, 0), (x0, y + cd, DOOR_H + cw), w)
        p.box((x1, y, 0), (x1 + cw, y + cd, DOOR_H + cw), w)
        p.box((x0, y, DOOR_H), (x1, y + cd, DOOR_H + cw), w)
        slab = m["door"] if style == "white" else m["door_oak"]
        yf = y + d - 0.012
        p.box((x0 + 0.003, y, 0.008), (x1 - 0.003, yf, DOOR_H - 0.003), slab)
        hx, hz = x1 - 0.065, 1.02  # latch on the east side, lever towards the hinge
        p.box((hx - 0.022, yf, hz - 0.085), (hx + 0.022, yf + 0.008, hz + 0.03), m["handle"])
        p.cyl((hx, yf + 0.03, hz), 0.009, 0.009, 0.05, m["handle"], rot=(math.pi / 2, 0, 0))
        p.cyl((hx - 0.06, yf + 0.055, hz), 0.0085, 0.0085, 0.13, m["handle"],
              rot=(0, math.pi / 2, 0))  # fmt: skip
        p.cyl((hx, yf + 0.01, hz - 0.06), 0.006, 0.006, 0.006, m["black_metal"],
              rot=(math.pi / 2, 0, 0))  # keyhole  # fmt: skip
        for zh in (0.22, 1.05, 1.82):
            p.cyl((x0 + 0.004, y + 0.03, zh), 0.007, 0.007, 0.1, m["handle"])
    door = p.build(bevel=0.002)
    door.pass_index = STRUCTURE["door"]
    return door


def build_radiator(m: dict, x0: float, x1: float, name: str):
    """Panel radiator under a window: ribbed front, top grille, valve and pipes."""
    p = Parts(name)
    z0, z1 = 0.15, 0.65
    y1 = HY - 0.05
    y0 = y1 - 0.10
    p.box((x0, y0 + 0.01, z0), (x1, y1, z1), m["radiator"])
    n = int((x1 - x0) / 0.033)
    for i in range(n):
        xc = x0 + (i + 0.5) * (x1 - x0) / n
        p.box((xc - 0.009, y0, z0 + 0.01), (xc + 0.009, y0 + 0.012, z1 - 0.01), m["radiator"])
    p.box((x0 + 0.01, y0 + 0.015, z1), (x1 - 0.01, y1 - 0.005, z1 + 0.004), m["grille"])
    p.box((x0 - 0.01, y0 + 0.02, z0 + 0.03), (x0, y1 - 0.02, z1 - 0.03), m["radiator"])
    p.box((x1, y0 + 0.02, z0 + 0.03), (x1 + 0.01, y1 - 0.02, z1 - 0.03), m["radiator"])
    p.cyl((x1 + 0.05, y0 + 0.05, z0 + 0.06), 0.009, 0.009, 0.1, m["radiator"],
          rot=(0, math.pi / 2, 0))  # fmt: skip
    p.cyl((x1 + 0.1, y0 + 0.05, z0 + 0.1), 0.026, 0.024, 0.09, m["white_plastic"], segs=20)
    p.cyl((x1 + 0.1, y0 + 0.05, z0 + 0.15), 0.024, 0.02, 0.012, m["white_plastic"], segs=20)
    p.cyl((x1 + 0.1, y0 + 0.05, z0 / 2 + 0.02), 0.009, 0.009, z0 + 0.04, m["radiator"])
    p.cyl((x0 - 0.04, y0 + 0.05, z0 / 2 + 0.05), 0.009, 0.009, z0 + 0.1, m["radiator"])
    return p.build(bevel=0.0025)


def build_electrics(m: dict, tv_y: float, tv_bottom: float, console_top: float):
    """Switches by both doors, sockets, a thermostat, the TV cable raceway on the accent
    wall and a smoke detector on the ceiling: one object."""
    p = Parts("electrics")
    w = m["white_plastic"]

    def box(wall, a, z, da0, dz0, d0, da1, dz1, d1, mat):
        if wall == "south":
            p.box((a + da0, -HY + d0, z + dz0), (a + da1, -HY + d1, z + dz1), mat)
        elif wall == "east":
            p.box((HX - d1, a + da0, z + dz0), (HX - d0, a + da1, z + dz1), mat)
        elif wall == "north":
            p.box((a + da0, HY - d1, z + dz0), (a + da1, HY - d0, z + dz1), mat)
        else:
            p.box((-HX + d0, a + da0, z + dz0), (-HX + d1, a + da1, z + dz1), mat)

    def plate(wall, a, z, sockets):
        wdt = 0.086 if sockets <= 1 else 0.158
        box(wall, a, z, -wdt / 2, -0.043, 0, wdt / 2, 0.043, 0.009, w)
        if sockets == 0:  # double rocker switch
            for c in (-0.014, 0.014):
                box(wall, a, z, c - 0.013, -0.027, 0.009, c + 0.013, 0.027, 0.014, w)
        for i in range(sockets):
            c = (i - (sockets - 1) / 2) * 0.072
            box(wall, a, z, c - 0.024, -0.024, 0.009, c + 0.024, 0.024, 0.011, m["grey_plastic"])

    plate("south", DOORS[0][1] + 0.19, 1.05, 0)  # by the hall door, latch side
    plate("south", DOORS[1][1] + 0.15, 1.05, 0)  # by the oak door
    plate("west", 1.25, 0.30, 2)  # media console end
    plate("west", tv_y + 0.25, tv_bottom + 0.12, 2)  # behind the TV
    plate("east", 1.95, 0.30, 2)  # side table
    plate("north", -0.55, 0.30, 2)  # beside the plant
    plate("south", -0.85, 0.30, 1)  # floor lamp
    # Thermostat above the hall switch: a white rounded box with a dark display.
    a = DOORS[0][1] + 0.19
    box("south", a, 1.45, -0.045, -0.06, 0, 0.045, 0.06, 0.022, w)
    box("south", a, 1.47, -0.03, -0.018, 0.022, 0.03, 0.022, 0.0235, m["thermostat"])
    # Cable raceway from the TV down behind the console, white, 4 x 1.5 cm.
    yr = tv_y + 0.18
    p.box((-HX, yr - 0.02, console_top - 0.02), (-HX + 0.015, yr + 0.02, tv_bottom + 0.05), w)
    # Smoke detector.
    p.cyl((0.2, -1.0, RH - 0.022), 0.055, 0.06, 0.044, w, segs=32)
    p.cyl((0.2, -1.0, RH - 0.046), 0.03, 0.03, 0.004, m["grey_plastic"], segs=24)
    return p.build(bevel=0.0015)


# ---------------------------------------------------------------- furniture built here


def screen_image(path: Path) -> None:
    """A mountain lake at dusk (a nature documentary paused on the TV), 1920 x 1080 PNG."""
    w, h = 1920, 1080
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = yy / h  # 0 top .. 1 bottom
    horizon = 0.58
    sky_top, sky_low = np.array((0.10, 0.16, 0.38)), np.array((0.98, 0.62, 0.36))
    t = np.clip(v / horizon, 0, 1)[..., None] ** 1.6
    img = sky_top * (1 - t) + sky_low * t
    sx, sy = 0.62 * w, 0.50 * h  # sun just above the far ridge
    d = np.hypot(xx - sx, yy - sy) / h
    img += (np.exp(-((d / 0.05) ** 2)) * 0.9 + np.exp(-d / 0.18) * 0.25)[..., None] * np.array(
        (1.0, 0.8, 0.55)
    )
    rng = np.random.default_rng(7)

    def ridge(base, amp, freqs):
        x = np.arange(w) / w
        r = np.full(w, base, np.float32)
        for f, a in zip(freqs, amp, strict=True):
            r += a * np.sin(2 * np.pi * f * x + rng.uniform(0, 6.28))
        return r * h

    layers = [  # far to near: base height, amplitudes, frequencies, colour
        (0.47, (0.05, 0.02, 0.008), (1.3, 3.7, 11), (0.42, 0.36, 0.52)),
        (0.51, (0.06, 0.025, 0.01), (0.9, 4.3, 13), (0.24, 0.22, 0.36)),
        (0.55, (0.04, 0.03, 0.012), (1.7, 5.1, 17), (0.10, 0.11, 0.18)),
    ]
    land = np.zeros((h, w), bool)
    for base, amp, fr, col in layers:
        r = ridge(base, amp, fr)
        mask = (yy >= r[None, :]) & (yy < horizon * h)
        img[mask] = np.array(col)
        land |= mask
    # The lake mirrors the sky and ridges, darker and rippled.
    lake = yy >= horizon * h
    src = np.clip((2 * horizon * h - yy[lake]).astype(int), 0, h - 1)
    ripple = (np.sin(yy[lake] * 0.9 + 3 * np.sin(xx[lake] * 0.01)) * 2).astype(int)
    img[lake] = img[src, np.clip(xx[lake].astype(int) + ripple, 0, w - 1)] * 0.7
    img[(yy > horizon * h - 2) & (yy < horizon * h + 1)] *= 1.15
    # Pine silhouettes on the near shore, left.
    for _ in range(26):
        cx = rng.uniform(0, 0.38) * w
        base = horizon * h + rng.uniform(-4, 6)
        th = rng.uniform(0.06, 0.16) * h
        half = (yy - (base - th)) / th * rng.uniform(0.18, 0.26) * th * 0.5  # wide at the foot
        tree = (yy <= base) & (yy >= base - th) & (np.abs(xx - cx) < np.maximum(half, 1.5))
        img[tree] = (0.04, 0.05, 0.06)
    img = np.clip(img, 0, 1).astype(np.float32)
    bpy = _bpy()
    im = bpy.data.images.new("tv_screen_src", w, h, alpha=True)
    rgba = np.concatenate([img, np.ones((h, w, 1), np.float32)], axis=2)
    im.pixels.foreach_set(rgba[::-1].ravel())
    im.filepath_raw, im.file_format = str(path), "PNG"
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save()
    bpy.data.images.remove(im)


def build_tv(m: dict, screen: Path, y: float, zc: float):
    """55" flat TV on a slim wall mount on the west wall: thin bezel, deeper back box,
    picture on (emissive), a small status LED. Built facing -y, turned to face +x."""
    W, H = 1.232, 0.712
    p = Parts("tv")
    p.box((-W / 2, -0.026, -H / 2), (W / 2, 0.0, H / 2), m["tv_black"])  # panel
    p.box((-0.40, 0.0, -0.25), (0.40, 0.035, 0.22), m["tv_back"])  # electronics box
    p.box((-0.20, 0.035, -0.15), (0.20, 0.055, 0.15), m["black_metal"])  # wall plate
    p.box((-0.012, -0.0275, -H / 2 + 0.004), (0.012, -0.026, -H / 2 + 0.008), m["red_button"])
    body = p.build(bevel=0.003)
    sm = 0.007
    disp = quads_object("tv_screen", [[(-W / 2 + sm, -0.0265, -H / 2 + sm),
                                       (W / 2 - sm, -0.0265, -H / 2 + sm),
                                       (W / 2 - sm, -0.0265, H / 2 - sm),
                                       (-W / 2 + sm, -0.0265, H / 2 - sm)]],
                        [image_mat("tv_picture", screen, emit=2.2)])  # fmt: skip
    obj = join("tv", [body, disp])
    place(obj, -HX + 0.056, y, zc, yaw=90)
    bake_transform(obj)
    return obj


def build_soundbar(m: dict, x: float, y: float, z: float):
    p = Parts("soundbar")
    L, D, T = 0.80, 0.095, 0.062
    p.box((-L / 2, -D / 2, 0.006), (L / 2, D / 2, T), m["soundbar"])
    for xf in (-0.33, 0.33):
        p.box((xf - 0.03, -0.03, 0), (xf + 0.03, 0.03, 0.006), m["grey_plastic"])
    p.box((-L / 2 - 0.002, -D / 2 + 0.01, T - 0.004), (L / 2 + 0.002, D / 2, T + 0.002),
          m["tv_black"])  # top cap  # fmt: skip
    for i in range(3):
        p.cyl((-0.05 + i * 0.05, -0.02, T + 0.003), 0.006, 0.006, 0.002, m["grey_plastic"], segs=12)
    obj = p.build(bevel=0.012, segments=3)
    place(obj, x, y, z, yaw=90)
    bake_transform(obj)
    return obj


def build_canvas(name: str, image: Path, centre, wall: str, height: float, m: dict, img_wh):
    """Gallery-wrapped canvas (3.5 cm deep, linen edges) in a thin oak floater frame."""
    aspect = img_wh[0] / img_wh[1]
    ih, iw = height, height * aspect
    dep, gap, fw = 0.035, 0.012, 0.012
    p = Parts(name)
    p.box((-iw / 2, -dep, -ih / 2), (iw / 2, -0.004, ih / 2), m["canvas_edge"])
    o = gap + fw  # floater frame: L profile, a back plate and a lip around the canvas
    yf = -dep - 0.006
    for a, b in (((-iw / 2 - o, yf, -ih / 2 - o), (iw / 2 + o, 0.0, -ih / 2 - gap)),
                 ((-iw / 2 - o, yf, ih / 2 + gap), (iw / 2 + o, 0.0, ih / 2 + o)),
                 ((-iw / 2 - o, yf, -ih / 2 - gap), (-iw / 2 - gap, 0.0, ih / 2 + gap)),
                 ((iw / 2 + gap, yf, -ih / 2 - gap), (iw / 2 + o, 0.0, ih / 2 + gap))):  # fmt: skip
        p.box(a, b, m["oak"])
    p.box((-iw / 2 - gap, -0.006, -ih / 2 - gap), (iw / 2 + gap, 0.0, ih / 2 + gap), m["tv_black"])
    frame = p.build(bevel=0.0015)
    art = quads_object(name + "_art", [[(-iw / 2, -dep - 0.0004, -ih / 2),
                                        (iw / 2, -dep - 0.0004, -ih / 2),
                                        (iw / 2, -dep - 0.0004, ih / 2),
                                        (-iw / 2, -dep - 0.0004, ih / 2)]],
                       [image_mat(name + "_paint", image, rough=0.7)])  # fmt: skip
    obj = join(name, [frame, art])
    yaw = {"north": 0.0, "south": 180.0, "west": 90.0, "east": -90.0}[wall]
    place(obj, *centre, yaw=yaw)
    bake_transform(obj)
    return obj


def build_rug(m: dict, x0, x1, y0, y1):
    """Wool rug with a dark 12 cm border."""
    p = Parts("rug")
    b = 0.12
    p.box((x0 + b, y0 + b, 0), (x1 - b, y1 - b, RUG_T), m["rug"])
    for lo, hi in (((x0, y0, 0), (x1, y0 + b, RUG_T)), ((x0, y1 - b, 0), (x1, y1, RUG_T)),
                   ((x0, y0 + b, 0), (x0 + b, y1 - b, RUG_T)),
                   ((x1 - b, y0 + b, 0), (x1, y1 - b, RUG_T))):  # fmt: skip
        p.box(lo, hi, m["rug_border"])
    return p.build(bevel=0.004)


def build_table_lamp(m: dict, x, y, z):
    """Glazed ceramic gourd base, brass stem, linen drum shade (open, double-walled)."""
    p = Parts("table_lamp")
    p.cyl((0, 0, 0.01), 0.07, 0.07, 0.02, m["lamp_base"], segs=32)
    p.sphere((0, 0, 0.12), (0.10, 0.10, 0.11), m["lamp_base"], segs=32)
    p.sphere((0, 0, 0.245), (0.05, 0.05, 0.05), m["lamp_base"], segs=24)
    p.cyl((0, 0, 0.33), 0.008, 0.008, 0.12, m["brass"], segs=12)
    p.cyl((0, 0, 0.395), 0.02, 0.02, 0.03, m["brass"], segs=16)  # bulb holder
    zs0, hs = 0.33, 0.24
    p.cyl((0, 0, zs0 + hs / 2), 0.17, 0.15, hs, m["shade"], segs=48, caps=False)
    p.cyl((0, 0, zs0 + hs / 2), 0.167, 0.147, hs - 0.002, m["shade"], segs=48, caps=False)
    for zr, r in ((zs0, 0.169), (zs0 + hs, 0.149)):
        p.torus((0, 0, zr), r, 0.002, m["brass"], rot=(math.pi / 2, 0, 0), n1=48, n2=6)
    obj = p.build()
    place(obj, x, y, z)
    bake_transform(obj)
    return obj


def build_floor_lamp(m: dict, x, y):
    """Tripod floor lamp: three splayed oak legs, a brass collar, a drum shade at 1.3 m."""
    p = Parts("floor_lamp")
    top, foot = 1.30, 0.26
    for k in range(3):
        a = 2 * math.pi * k / 3 + 0.3
        fx, fy = foot * math.cos(a), foot * math.sin(a)
        length = math.sqrt(foot**2 + top**2)
        t = math.atan2(foot, top)
        # z axis turned to (-fx, -fy, top) / length: Euler XYZ, x then y
        rx = math.asin(math.sin(t) * math.sin(a))
        ry = math.atan2(-math.sin(t) * math.cos(a), math.cos(t))
        p.cyl((fx / 2, fy / 2, top / 2), 0.014, 0.011, length, m["oak"], rot=(rx, ry, 0),
              segs=12)  # fmt: skip
    p.cyl((0, 0, top + 0.02), 0.022, 0.022, 0.06, m["brass"], segs=16)
    p.cyl((0, 0, top + 0.09), 0.008, 0.008, 0.10, m["brass"], segs=12)
    zs0, hs = top + 0.08, 0.30
    p.cyl((0, 0, zs0 + hs / 2), 0.23, 0.23, hs, m["shade"], segs=56, caps=False)
    p.cyl((0, 0, zs0 + hs / 2), 0.227, 0.227, hs - 0.002, m["shade"], segs=56, caps=False)
    for zr in (zs0, zs0 + hs):
        p.torus((0, 0, zr), 0.229, 0.0022, m["brass"], rot=(math.pi / 2, 0, 0), n1=56, n2=6)
    obj = p.build()
    place(obj, x, y, 0.0)
    bake_transform(obj)
    return obj


def build_tray(m: dict, x, y, z, yaw):
    """Rectangular oak serving tray with a low rim."""
    p = Parts("tray")
    L, W, h, t = 0.42, 0.28, 0.035, 0.008
    p.box((-L / 2, -W / 2, 0), (L / 2, W / 2, 0.008), m["tray"])
    for lo, hi in (((-L / 2, -W / 2, 0), (L / 2, -W / 2 + t, h)),
                   ((-L / 2, W / 2 - t, 0), (L / 2, W / 2, h)),
                   ((-L / 2, -W / 2, 0), (-L / 2 + t, W / 2, h)),
                   ((L / 2 - t, -W / 2, 0), (L / 2, W / 2, h))):  # fmt: skip
        p.box(lo, hi, m["tray"])
    obj = p.build(bevel=0.002)
    place(obj, x, y, z, yaw)
    bake_transform(obj)
    return obj


def build_candle(m: dict, x, y, z, name="candle"):
    """Candle in a dark glazed ceramic jar, wax a little below the rim, a burnt wick."""
    p = Parts(name)
    r, h = 0.045, 0.09
    p.cyl((0, 0, h / 2), r, r, h, m["jar"], segs=40, caps=False)
    p.cyl((0, 0, h / 2), r - 0.004, r - 0.004, h, m["jar"], segs=40, caps=False)
    p.torus((0, 0, h), r - 0.002, 0.002, m["jar"], rot=(math.pi / 2, 0, 0), n1=40, n2=6)
    p.cyl((0, 0, 0.004), r, r, 0.008, m["jar"], segs=40)
    p.cyl((0, 0, h - 0.016), r - 0.004, r - 0.004, 0.004, m["wax"], segs=40)
    p.cyl((0, 0, h - 0.008), 0.0012, 0.0008, 0.012, m["wick"], segs=6)
    obj = p.build()
    place(obj, x, y, z)
    bake_transform(obj)
    return obj


def build_remote(m: dict, x, y, z, yaw):
    p = Parts("remote")
    L, W, T = 0.17, 0.045, 0.017
    p.box((-W / 2, -L / 2, 0), (W / 2, L / 2, T), m["remote"])
    obj = p.build(bevel=0.006, segments=3)
    q = Parts("remote_buttons")
    q.cyl((0, L / 2 - 0.02, T + 0.0005), 0.005, 0.005, 0.002, m["red_button"], segs=12)
    q.cyl((0, 0.02, T + 0.0005), 0.013, 0.013, 0.002, m["button"], segs=24)  # d-pad ring
    for r in range(4):
        for c in (-1, 0, 1):
            yb = -0.015 - r * 0.012
            q.box((c * 0.012 - 0.004, yb - 0.003, T - 0.0005),
                  (c * 0.012 + 0.004, yb + 0.003, T + 0.0012), m["button"])  # fmt: skip
    obj = join("remote", [obj, q.build()])
    place(obj, x, y, z, yaw)
    bake_transform(obj)
    return obj


def build_book_row(name: str, x: float, y: float, z: float, yaw: float, length: float,
                   hmax: float, covers: list, m: dict, seed: int,
                   lean_last: bool = True):  # fmt: skip
    """Books standing upright side by side, spines to -y before yaw, the last one leaning."""
    rng = np.random.default_rng(seed)
    p = Parts(name)
    xc = -length / 2
    h_last = 0.0
    while True:
        t = rng.uniform(0.016, 0.045)
        hb = rng.uniform(0.62, 1.0) * hmax
        d = rng.uniform(0.13, 0.20)
        if xc + t > length / 2:
            break
        cover = covers[rng.integers(len(covers))]
        lo = np.array((xc, -0.10, 0.0))
        p.box(lo, lo + (t, d, hb), cover)
        p.box(lo + (0.002, 0.004, 0.003), lo + (t - 0.002, d + 0.002, hb - 0.003), m["pages"])
        if rng.random() < 0.6:  # a printed band on the spine
            zb = rng.uniform(0.6, 0.85) * hb
            p.box(lo + (0.002, -0.0008, zb), lo + (t - 0.002, 0.0, zb + 0.012), m["gold"])
        xc += t + 0.001
        h_last = hb
    if lean_last and xc > -length / 2:
        # Tipped back onto the row: it stands on its near bottom edge (the pivot) and its
        # face rests on the last upright book's top corner (or its top on that book's side).
        t, hb = 0.03, 0.85 * hmax
        a = math.radians(16)
        xe = xc - 0.001
        if hb * math.cos(a) >= h_last:
            xp = xe + h_last * math.tan(a)
        else:
            xp = xe + hb * math.sin(a)
        lo = (xp, -0.10, 0.0)
        p.box(lo, (xp + t, -0.10 + 0.17, hb), covers[rng.integers(len(covers))],
              rot=(0, -a, 0), pivot=(xp, -0.02, 0.0))  # fmt: skip
    obj = p.build(bevel=0.0012)
    place(obj, x, y, z, yaw)
    bake_transform(obj)
    return obj


def build_sofa(m: dict):
    """Low modern three-seat sofa, 2.25 x 0.92 m: upholstered base and slim track arms,
    three seat and three back cushions (soft, the back ones leaning), tapered oak legs.
    Built facing -y with the back at +y."""
    W, D = 2.25, 0.92
    a, hb = 0.15, 0.17  # arm width, back frame depth
    fab = m["sofa"]
    p = Parts("sofa_frame")
    p.box((-W / 2, -D / 2, 0.13), (W / 2, D / 2, 0.31), fab)  # base
    for sx in (-1, 1):
        x0, x1 = sorted((sx * W / 2, sx * (W / 2 - a)))
        p.box((x0, -D / 2, 0.13), (x1, D / 2, 0.62), fab)
    p.box((-W / 2 + a, D / 2 - hb, 0.31), (W / 2 - a, D / 2, 0.70), fab)  # back frame
    frame = p.build(bevel=0.025, segments=4)
    c = Parts("sofa_cushions")
    inner = W - 2 * a
    n, g = 3, 0.008
    cw = (inner - (n - 1) * g) / n
    for i in range(n):
        x0 = -inner / 2 + i * (cw + g)
        c.box((x0, -D / 2 + 0.01, 0.31), (x0 + cw, D / 2 - hb - 0.01, 0.46), fab)
        c.box((x0, D / 2 - hb - 0.19, 0.46), (x0 + cw, D / 2 - hb - 0.01, 0.88), fab,
              rot=(math.radians(-11), 0, 0), pivot=(0, D / 2 - hb - 0.01, 0.46))  # fmt: skip
    cushions = c.build(bevel=0.045, segments=6)
    leg = Parts("sofa_legs")
    for sx in (-1, 1):
        for sy in (-1, 1):
            leg.cyl((sx * (W / 2 - 0.09), sy * (D / 2 - 0.09), 0.065), 0.016, 0.022, 0.13,
                    m["oak"], segs=16)  # fmt: skip
    return join("sofa", [frame, cushions, leg.build()])


def build_pouf(m: dict, x, y, z):
    """Round knitted pouf, 56 cm across, 40 cm high, very soft edges."""
    p = Parts("pouf")
    p.cyl((0, 0, 0.20), 0.28, 0.28, 0.40, m["knit"], segs=64)
    obj = p.build(bevel=0.12, segments=8)
    place(obj, x, y, z)
    bake_transform(obj)
    return obj


SHELF_X0, SHELF_X1, SHELF_D = 0.35, 1.55, 0.32
SHELF_GAP = 0.022  # off the wall, clear of the 18 mm skirting
SHELF_BOARDS = (0.06, 0.42, 0.78, 1.14, 1.50, 1.86)  # board bottoms, 2.5 cm oak boards


def build_bookshelf(m: dict):
    """Open bookshelf on the south wall: six oak boards on a black steel frame (corner posts,
    side rails under every board, an X brace at the back). Returns it and the board tops."""
    p = Parts("bookshelf")
    x0, x1, y0 = SHELF_X0, SHELF_X1, -HY + SHELF_GAP
    y1, t, s, top = y0 + SHELF_D, 0.025, 0.02, SHELF_BOARDS[-1] + 0.025
    blk = m["black_metal"]
    for xa in (x0, x1 - s):
        for ya in (y0, y1 - s):
            p.box((xa, ya, 0), (xa + s, ya + s, top), blk)
    for zb in SHELF_BOARDS:
        p.box((x0 + s, y0, zb), (x1 - s, y1, zb + t), m["oak"])
        for xa in (x0, x1 - s):
            p.box((xa, y0, zb - 0.02), (xa + s, y1, zb), blk)
    zl, zh = SHELF_BOARDS[0], SHELF_BOARDS[-1]
    ang = math.atan2(zh - zl, x1 - x0 - 2 * s)
    length = math.hypot(zh - zl, x1 - x0 - 2 * s)
    xc, zc = (x0 + x1) / 2, (zl + zh) / 2
    for sign in (1, -1):
        p.box((xc - length / 2, y0 + 0.004, zc - 0.006), (xc + length / 2, y0 + 0.01, zc + 0.006),
              blk, rot=(0, -sign * ang, 0))  # fmt: skip
    obj = p.build(bevel=0.002)
    return obj, [zb + t for zb in SHELF_BOARDS]


def modern_planter(obj, soil_z: float, m: dict, taper: float = 1.0, scale: float = 1.0):
    """Replace a plant asset's terracotta pot with a matte white planter: delete the pot
    faces below the soil line (the pot material also holds the stems), add a cylinder (or
    a tapered cone) with a rim and a soil disc around their footprint. Call before place()."""
    import bmesh  # noqa: PLC0415

    idx = {i for i, mt in enumerate(obj.data.materials) if mt and "pot" in mt.name}
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    faces = [f for f in bm.faces if f.material_index in idx
             and all(v.co.z < soil_z + 0.01 for v in f.verts)]  # fmt: skip
    co = np.array([v.co[:] for f in faces for v in f.verts])
    bmesh.ops.delete(bm, geom=faces, context="FACES")
    bm.to_mesh(obj.data)
    bm.free()
    cx, cy = (co[:, :2].min(0) + co[:, :2].max(0)) / 2
    r = float(np.hypot(co[:, 0] - cx, co[:, 1] - cy).max()) * scale
    h = soil_z + 0.02
    p = Parts("planter")
    p.cyl((cx, cy, h / 2), r * taper, r, h, m["planter"], segs=48, caps=False)
    p.cyl((cx, cy, h / 2 + 0.006), r * taper - 0.01, r - 0.01, h - 0.012, m["planter"], segs=48,
          caps=False)  # fmt: skip
    p.torus((cx, cy, h), r - 0.005, 0.005, m["planter"], rot=(math.pi / 2, 0, 0), n1=48, n2=6)
    p.cyl((cx, cy, 0.005), r * taper, r * taper, 0.01, m["planter"], segs=48)
    p.cyl((cx, cy, soil_z - 0.01), r - 0.01, r - 0.01, 0.01, m["soil"], segs=48)
    return join(obj.name, [obj, p.build()])


# ---------------------------------------------------------------- scene


def build_living(assets: Path, out: Path, warm_lamps: bool = False) -> tuple[dict, list]:
    """Build the whole room in the open (empty) Blender file. Returns the structure objects
    (floor, wall, ceiling, window, door) and the object records (name, noun, asset,
    support, obj) in GT index order."""
    from mathutils import Matrix  # noqa: PLC0415

    bpy = _bpy()
    m = materials(assets)
    covers = book_mats(m)
    structure = build_shell(m)
    structure["window"] = build_windows(m)
    structure["door"] = build_doors(m)

    def asset(name: str, folder: str, x, y, z=0.0, yaw=0.0, scale=1.0):
        obj = import_asset(assets / folder / f"{folder}.gltf", name)
        # Some glTFs carry keyframes; the cloth solver's frame steps would replay them and
        # snap the object back to its imported pose.
        obj.animation_data_clear()
        place(obj, x, y, z, yaw, scale)
        return obj

    rec: list = []

    def add(obj, noun: str, src: str | None, support: str | None = None):
        rec.append((obj.name, noun, src, support, obj))
        return obj

    # Media wall (west): console, TV, soundbar and what stands on the console.
    console = asset("media_console", "modern_wooden_cabinet", -2.5, 0.10, yaw=90, scale=0.75)
    against(console, "west", 0.022)
    add(console, "tv stand", "modern_wooden_cabinet")
    cz = top_z(console)
    clo, chi = world_bbox(console)
    cx = (clo[0] + chi[0]) / 2
    screen = out / "tv_screen.png"
    screen_image(screen)
    tv_y, tv_zc = 0.10, 1.18
    add(build_tv(m, screen, tv_y, tv_zc), "tv", None)
    add(build_soundbar(m, cx + 0.04, tv_y, cz), "soundbar", None, "tv stand")
    add(asset("games_console", "gaming_console", cx + 0.02, -0.55, cz, yaw=90), "game console",
        "gaming_console", "tv stand")  # fmt: skip
    add(asset("console_succulent", "potted_plant_04", cx, 0.62, cz, yaw=30), "succulent",
        "potted_plant_04", "tv stand")  # fmt: skip
    add(asset("console_vase", "ceramic_vase_04", cx - 0.03, 0.86, cz, yaw=10), "vase",
        "ceramic_vase_04", "tv stand")  # fmt: skip
    add(asset("floor_vase", "ceramic_vase_03", -2.56, -1.12, yaw=0, scale=1.5), "vase",
        "ceramic_vase_03")  # fmt: skip

    # Seating group: sofa (east wall), cushions, throw, side table and lamp, coffee table.
    sofa = build_sofa(m)
    place(sofa, 2.3, 0.10, 0.0, yaw=-90)
    against(sofa, "east", 0.03)
    bake_transform(sofa)
    add(sofa, "sofa", None)
    slo, shi = world_bbox(sofa)
    side = asset("side_table", "side_table_01", 2.50, 1.80)
    against(side, "east", 0.04)
    add(side, "side table", "side_table_01")
    # Cushions lean on the back cushions (whose front is ~0.4 m in from the sofa front).
    pillows = []
    for i, (yc, mat, lean, yaw) in enumerate(
        ((-0.72, "cushion_mustard", 72, -6), (-0.22, "cushion_rust", 70, 8),
         (0.52, "cushion_sage", 74, -4))
    ):  # fmt: skip
        pc = pillow(f"cushion_{i}", (slo[0] + 0.47, 0.10 + yc, 0.66), (0.46, 0.46, 0.15), m[mat],
                    rot=(0, math.radians(lean), math.radians(yaw)), seed=10 + i)  # fmt: skip
        apply_modifiers(pc)
        clo, chi = world_bbox(pc)
        seat = surface_z(sofa, (clo[0] + chi[0]) / 2, (clo[1] + chi[1]) / 2, z0=0.6)
        pc.data.transform(Matrix.Translation((0, 0, seat + 0.004 - clo[2])))
        settle(pc, sofa, (-0.005, 0.0, 0.0))
        pillows.append(add(pc, "pillow", None, "sofa"))
    print("cloth: throw")
    throw = cloth_sheet("throw", slo[0] + 0.08, slo[0] + 0.68, 0.62, 1.40, 0.95, 32, 42, m["throw"],
                        [sofa, side, *pillows], frames=90, stiffness=1.2, mass=0.35,
                        thickness=0.008, seed=8, noise=0.005, half=(HX, HY))  # fmt: skip
    add(throw, "blanket", "waffle_pique_cotton (texture)", "sofa")
    tz = top_z(side)
    tlo, thi = world_bbox(side)
    tcy = (tlo[1] + thi[1]) / 2
    add(build_table_lamp(m, thi[0] - 0.19, tcy + 0.07, tz), "table lamp", None, "side table")
    add(build_book_stack("side_books", tlo[0] + 0.12, tcy - 0.12, tz, 70, covers, m, 2,
                         seed=21), "books", None, "side table")  # fmt: skip

    ct_x, ct_y = 0.70, 0.15
    table = asset("coffee_table", "coffee_table_round_01", ct_x, ct_y, RUG_T, scale=0.8)
    add(table, "coffee table", "coffee_table_round_01")
    kz = top_z(table)
    add(build_tray(m, ct_x + 0.16, ct_y - 0.16, kz, -18), "tray", None, "coffee table")
    yz = kz + 0.008
    add(build_candle(m, ct_x + 0.08, ct_y - 0.14, yz), "candle", None, "tray")
    add(build_remote(m, ct_x + 0.26, ct_y - 0.22, yz, -18 + 90 - 10), "remote", None, "tray")
    add(build_book_stack("coffee_books", ct_x - 0.18, ct_y + 0.20, kz, 25, covers, m, 3,
                         seed=31), "books", None, "coffee table")  # fmt: skip
    add(asset("bowl", "wooden_bowl_02", ct_x + 0.22, ct_y + 0.24, kz), "bowl", "wooden_bowl_02",
        "coffee table")  # fmt: skip
    add(build_mug(m, ct_x - 0.24, ct_y - 0.12, kz, yaw=200), "mug", None, "coffee table")

    add(build_rug(m, -0.75, 1.72, -0.78, 1.28), "rug", "poly_wool_herringbone (texture)")
    add(build_pouf(m, -0.60, 1.00, RUG_T), "pouf", "waffle_pique_cotton (texture)")
    add(asset("armchair", "modern_arm_chair_01", -0.30, -1.45, yaw=150), "armchair",
        "modern_arm_chair_01")  # fmt: skip
    add(build_floor_lamp(m, -1.15, -1.98), "floor lamp", None)

    # Bookshelf between the doors, facing the room (+y), and what is on it.
    shelf, z = build_bookshelf(m)
    add(shelf, "bookshelf", None)
    by = -HY + SHELF_GAP + SHELF_D / 2
    board: dict = {}  # name -> shelf board, so the overlap check compares one board only

    def on_shelf(name, folder, x, k, yaw=180.0, scale=1.0, noun=None):
        board[name] = k
        return add(asset(name, folder, x, by, z[k], yaw=yaw, scale=scale), noun or name,
                   folder, "bookshelf")  # fmt: skip

    def books(k, name, x, length=None, seed=0, lean=True):
        """An upright row (length given) or a flat stack of books on board k."""
        if length:
            obj = build_book_row(name, x, by + 0.05, z[k], 180, length, 0.27, covers, m, seed,
                                 lean_last=lean)  # fmt: skip
        else:
            obj = build_book_stack(name, x, by, z[k], 5, covers, m, 4, seed=seed)
        board[name] = k
        return add(obj, "books", None, "bookshelf")

    on_shelf("basket", "wicker_basket_02", 0.62, 0, yaw=185, noun="basket")
    books(0, "shelf_books_0", 1.20, 0.46, seed=40)
    books(1, "shelf_books_1", 0.72, 0.44, seed=41)
    on_shelf("shelf_vase", "ceramic_vase_02", 1.30, 1, scale=0.9, noun="vase")
    on_shelf("horse_statue", "horse_statue_01", 0.60, 2, yaw=200, noun="statue")
    books(2, "shelf_books_2", 1.14, 0.42, seed=42)
    # This model's front is its +x side (not -y like the others): yaw 100 faces the room.
    frame = on_shelf("photo_frame", "standing_picture_frame_01", 0.66, 3, yaw=100,
                     noun="picture frame")  # fmt: skip
    drop_material_faces(frame, "glass")  # opaque here (no alpha in the glTF), hid the photo
    books(3, "shelf_stack_3", 1.02, seed=45)
    board["shelf_candle"] = 3
    add(build_candle(m, 1.36, by, z[3], name="shelf_candle"), "candle", None, "bookshelf")
    on_shelf("shelf_plant", "potted_plant_04", 0.62, 4, noun="succulent")
    books(4, "shelf_books_4", 1.14, 0.50, seed=44, lean=False)
    on_shelf("top_vase", "ceramic_vase_01", 0.66, 5, scale=0.8, noun="vase")
    on_shelf("shelf_bowl", "wooden_bowl_02", 1.24, 5, noun="bowl")

    # Plants in white planters, pendant, radiators, curtains, electrics.
    tall = import_asset(assets / "potted_plant_01" / "potted_plant_01.gltf", "tall_plant")
    tall = modern_planter(tall, 0.58, m, taper=0.85, scale=0.85)
    place(tall, -0.15, 1.96, yaw=25)
    add(tall, "potted plant", "potted_plant_01 (planter built here)")
    corner = import_asset(assets / "potted_plant_02" / "potted_plant_02.gltf", "corner_plant")
    corner = modern_planter(corner, 0.32, m, taper=0.8, scale=0.9)
    place(corner, -2.40, 1.78, yaw=-35)
    add(corner, "potted plant", "potted_plant_02 (planter built here)")
    lamp = asset("pendant", "modern_ceiling_lamp_01", ct_x, ct_y, 0.0, scale=0.75)
    lamp.location.z = RH - top_z(lamp)
    add(lamp, "ceiling lamp", "modern_ceiling_lamp_01")
    for i, (x0, x1) in enumerate(WINDOWS):
        add(build_radiator(m, x0 + 0.24, x1 - 0.32, f"radiator_{i}"), "radiator", None)
    for i, (x0, x1) in enumerate(WINDOWS):
        add(build_curtains(m, x0, x1, f"curtains_{i}"), "curtains", "rough_linen (texture)")
    add(build_electrics(m, tv_y, tv_zc - 0.356, cz), "socket", None)

    # Art, all modernist: a canvas over the sofa, framed prints on the other walls.
    art = assets / "posters"

    def img(name):
        path = art / f"{name}.jpg"
        return path, bpy.data.images.load(str(path)).size

    p, wh = img("painting_yellow_red_blue")
    add(build_canvas("canvas_yellow_red_blue", p, (HX, 0.10, 1.66), "east", 0.72, m, wh),
        "painting", "Yellow-Red-Blue (Kandinsky, 1925, public domain)")  # fmt: skip
    oak_frame = {**m, "frame": m["oak"]}
    for name, centre, wall, hgt, mm, src in (
        ("painting_mondrian", (-0.30, -HY, 1.60), "south", 0.52, m,
         "Composition II in Red, Blue and Yellow (Mondrian, 1930, public domain)"),
        ("painting_kandinsky", (-HX, 1.55, 1.52), "west", 0.40, oak_frame,
         "Color Study, Squares with Concentric Circles (Kandinsky, 1913, public domain)"),
        ("painting_klee", (-HX, -1.55, 1.50), "west", 0.42, m,
         "Ad Parnassum (Klee, 1932, public domain)"),
        ("painting_af_klint", (HX, -1.85, 1.52), "east", 0.62, oak_frame,
         "The Ten Largest, No. 3, Youth (af Klint, 1907, public domain)"),
    ):  # fmt: skip
        p, wh = img(name)
        add(build_poster(name, p, centre, wall, hgt, mm, wh), "painting", src)

    for idx, (_, _, _, _, obj) in enumerate(rec, start=1):
        if obj.modifiers:
            apply_modifiers(obj)
        obj.pass_index = idx
    check_overlaps([(n, nn, a, f"{s}@{board[n]}" if n in board else s, o)
                    for n, nn, a, s, o in rec])  # fmt: skip
    for obj in structure.values():
        apply_modifiers(obj)
    check_clipping([o for *_, o in rec] + list(structure.values()),
                   {o.name for o in structure.values()} | {"electrics"})  # fmt: skip
    build_lights(assets, warm_lamps, rec)
    return structure, rec


def _point(name: str, loc, power: float, colour=(1.0, 0.78, 0.55), size: float = 0.03):
    bpy = _bpy()
    light = bpy.data.lights.new(name, "POINT")
    light.energy, light.shadow_soft_size, light.color = power, size, colour
    o = bpy.data.objects.new(name, light)
    o.location = loc
    bpy.context.collection.objects.link(o)


def build_lights(assets: Path, warm_lamps: bool, rec) -> None:
    """HDRI sky through the windows (a light portal in each opening) and a soft ceiling fill.
    With warm_lamps, bulbs in the pendant, the floor lamp and the table lamp too."""
    bpy = _bpy()
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    nt = world.node_tree
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(next((assets / "eilenriede_park").glob("*.hdr"))))
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (0, 0, math.radians(160))
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    bg = nt.nodes["Background"]
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 3.0
    bpy.context.scene.world = world

    for i, (x0, x1) in enumerate(WINDOWS):
        portal = bpy.data.lights.new(f"window_portal_{i}", "AREA")
        portal.shape = "RECTANGLE"
        portal.size, portal.size_y = x1 - x0, WIN_Z1 - WIN_Z0
        portal.cycles.is_portal = True
        obj = bpy.data.objects.new(f"window_portal_{i}", portal)
        obj.location = ((x0 + x1) / 2, HY + WIN_DEPTH - 0.01, (WIN_Z0 + WIN_Z1) / 2)
        obj.rotation_euler = (math.radians(-90), 0, 0)  # faces -y, into the room
        bpy.context.collection.objects.link(obj)

    fill = bpy.data.lights.new("ceiling_fill", "AREA")
    fill.shape, fill.size, fill.size_y = "RECTANGLE", RX - 0.4, RY - 0.4
    fill.energy, fill.color = 280.0, (1.0, 0.96, 0.9)
    obj = bpy.data.objects.new("ceiling_fill", fill)
    obj.visible_glossy = False  # the glass and the TV showed it as a pale patch
    obj.location = (0.0, 0.0, RH - 0.01)
    bpy.context.collection.objects.link(obj)

    if warm_lamps:
        objs = {n: o for n, *_, o in rec}
        for name, dz, power in (("pendant", 0.08, 35.0), ("table_lamp", 0.42, 18.0),
                                ("floor_lamp", 1.48, 30.0)):  # fmt: skip
            lo, hi = world_bbox(objs[name])
            z = lo[2] + dz if name == "pendant" else dz + lo[2]
            _point(f"{name}_bulb", ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, z), power)


def write_gt(gt: Path, structure: dict, rec: list) -> None:
    """objects.json and per-object meshes, same format as blender_room.py."""
    meshes = gt / "meshes"
    objects = []
    for idx, (name, noun, src, support, obj) in enumerate(rec, start=1):
        _export_mesh(obj, meshes / f"{name}.ply")
        lo, hi = _world_bbox_project(obj)
        objects.append(
            {"index": idx, "id": f"{noun.replace(' ', '_')}#{idx}", "name": name, "noun": noun,
             "kind": "item" if support else "furniture", "asset": src or "built in "
             "blender_living.py", "support": support, "mesh": f"meshes/{name}.ply",
             "bbox_min": [round(float(v), 4) for v in lo],
             "bbox_max": [round(float(v), 4) for v in hi]}
        )  # fmt: skip
    for key, obj in structure.items():
        _export_mesh(obj, meshes / f"{key}.ply")
        objects.append(
            {"index": STRUCTURE[key], "id": key, "name": key, "noun": key, "kind": "structure",
             "asset": "built in blender_living.py", "support": None, "mesh": f"meshes/{key}.ply"}
        )  # fmt: skip
    # blender_room.world_openings format: world x, z (Blender x, -y), heights, normal
    openings = [
        {"kind": "window", "index": STRUCTURE["window"], "wall": "north",
         "outline_xz": [[x0, -HY], [x1, -HY]], "bottom": WIN_Z0, "top": WIN_Z1,
         "normal_xz": [0.0, 1.0]}
        for x0, x1 in WINDOWS
    ] + [
        {"kind": "door", "index": STRUCTURE["door"], "wall": "south",
         "outline_xz": [[x0, HY], [x1, HY]], "bottom": 0.0, "top": DOOR_H,
         "normal_xz": [0.0, -1.0]}
        for x0, x1, _ in DOORS
    ]  # fmt: skip
    (gt / "objects.json").write_text(
        json.dumps(
            {
                "scene": SCENE,
                "synthetic": True,
                "units": "m",
                "world": "+Y up, floor at y = 0 (scene contract)",
                "world_from_blender": WORLD_FROM_BLENDER.tolist(),
                "background_index": 0,
                "room_size_m": [RX, RH, RY],
                "room_outline_xz": [[-HX, HY], [HX, HY], [HX, -HY], [-HX, -HY]],
                "openings": openings,
                "objects": objects,
            },
            indent=2,
        )
    )


def render_previews(out: Path, samples: int, views: list[str], scale: int) -> None:
    bpy = _bpy()
    from mathutils import Matrix  # noqa: PLC0415

    scene = bpy.context.scene
    setup_render(samples, "AUTO")
    scene.render.use_motion_blur = False
    scene.render.resolution_percentage = scale
    scene.cycles.use_animated_seed = False
    cam_data = bpy.data.cameras.new("phone")
    cam_data.lens, cam_data.sensor_width, cam_data.sensor_fit = FOCAL_MM, SENSOR_MM, "HORIZONTAL"
    cam_data.clip_start = 0.05
    cam = bpy.data.objects.new("phone", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    for name in views:
        eye, target = VIEWS[name]
        rot = look_at(np.array(eye, float), np.array(target, float))
        mw = np.eye(4)
        mw[:3, :3], mw[:3, 3] = rot, eye
        cam.matrix_world = Matrix(mw.tolist())
        scene.render.filepath = str(out / "preview" / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"preview {name}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "raw" / SCENE)
    ap.add_argument("--assets", type=Path, default=REPO / "data" / "assets")
    ap.add_argument("--stage", default="build", help="comma list of build, preview")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--scale", type=int, default=50, help="preview resolution percent of 1080p")
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--warm-lamps", action="store_true", help="light the pendant and the lamps")
    args = ap.parse_args(argv)
    bpy = _bpy()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    gt = out.parent / f"{out.name}_gt"
    blend = out / f"{SCENE}.blend"
    stages = set(args.stage.split(","))
    if "build" in stages:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        structure, rec = build_living(args.assets.resolve(), out, args.warm_lamps)
        write_gt(gt, structure, rec)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend))
        print(f"{len(rec)} objects -> {blend}")
    elif "preview" in stages:
        bpy.ops.wm.open_mainfile(filepath=str(blend))
    if "preview" in stages:
        render_previews(out, args.samples, args.views.split(","), args.scale)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []))
