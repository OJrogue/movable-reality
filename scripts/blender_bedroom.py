"""blender06: a realistic single bedroom, the main synthetic test room (2 Nov).

    "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b -P scripts/blender_bedroom.py \
        -- --out data/raw/blender06                   # build -> blender06.blend + GT objects
    ... -- --stage build,preview --samples 64         # + preview stills in <out>/preview/
    ... -- --stage preview                            # stills from the saved .blend

A 3.6 x 4.2 m bedroom, 2.6 m high. A casement window on the north wall shows a park (an
HDRI) and lets the daylight in; a closed flush door on the south wall. A 120 x 200 cm bed
with an oak frame and an upholstered headboard stands against the sage-green west wall,
its duvet and the throw at its foot draped by a cloth simulation. Next to it: a
nightstand (alarm clock, books) and a potted plant in the corner. On the east
wall a desk with a laptop (screen on), mouse, mug, notebook, succulent and desk lamp, a
wall shelf of books and a framed Earthrise print; a chair at the desk. On the south wall a
chest of drawers with toys on top (tin robot, astronaut and hero figures, a wooden
elephant), a wall clock above it and a laundry basket beside it. Also a wool rug,
floor-length linen curtains on a rod, a panel radiator under the window, skirting boards,
a light switch, sockets, a pendant lamp and a ukulele leaning
on the wall. The Great Wave print hangs above the bed.

Assets: CC0 Poly Haven models, textures and the HDRI, public-domain posters from Wikimedia
Commons (scripts/blender_assets.py, docs/ASSETS.md). Everything else is built here.

Light: the HDRI through the window (with a light portal) plus a soft ceiling-sized fill,
so no lamp leaves a hotspot that would stay behind when furniture moves. --warm-lamps also
lights the pendant and the desk lamp, for a cosier look that bakes highlights in.

Outputs (out = data/raw/blender06):
  <out>/blender06.blend           the scene, every object named as in objects.json
  <out>/laptop_screen.png         the texture on the laptop screen (made here)
  <out>/preview/<view>.png        preview stills (phone camera, 26 mm)
  <out>_gt/objects.json           index -> id, noun, support, world bbox (blender_room format)
  <out>_gt/meshes/<name>.ply      each object's final mesh in world coordinates

Same conventions as scripts/blender_room.py: Blender +Z up, world frame +Y up with the
floor at y = 0, objects' pass_index = their index in objects.json (structure 101-106).
blender_room.py --scene blender06 builds it through build_bedroom and write_gt and adds the
walks, GT render passes and the video.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
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

SCENE = "blender06"
RX, RY, RH = 3.6, 4.2, 2.6  # Blender x (width), y (length), z (height)
HX, HY = RX / 2, RY / 2
REPO = Path(__file__).resolve().parents[1]

# The window in the north wall (y = HY): x range, sill and head height, reveal depth.
WIN_X0, WIN_X1, WIN_Z0, WIN_Z1, WIN_DEPTH = -0.40, 0.90, 0.85, 2.30, 0.26
# The door in the south wall (y = -HY): x range and height, hinge on the east side.
DOOR_X0, DOOR_X1, DOOR_H = 0.72, 1.58, 2.05
WALL_TINT = (0.93, 0.91, 0.87)  # warm white
ACCENT_TINT = (0.60, 0.67, 0.56)  # sage green, the west wall behind the bed

# Preview views: name -> (eye, target), Blender metres, phone at chest or eye height.
VIEWS = {
    "from_door": ((1.15, -1.55, 1.55), (-0.9, 0.7, 0.85)),
    "bed": ((0.85, -0.75, 1.5), (-1.7, 0.35, 1.0)),
    "desk_window": ((-0.55, -1.35, 1.5), (1.3, 1.25, 1.0)),
    "dresser_door": ((0.35, 1.25, 1.55), (-0.3, -2.1, 0.85)),
    "dresser_close": ((-0.35, -0.95, 1.25), (-0.65, -1.95, 0.9)),
    "shelf_close": ((0.75, 0.9, 1.65), (1.8, 0.9, 1.75)),
    "desk_close": ((0.55, 0.65, 1.3), (1.6, 1.05, 0.8)),
}


# ---------------------------------------------------------------- materials


def _tex_node(nt, path: Path, non_color: bool, coords, box: bool):
    bpy = _bpy()
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = bpy.data.images.load(str(path), check_existing=True)
    if non_color:
        node.image.colorspace_settings.name = "Non-Color"
    if box:
        node.projection, node.projection_blend = "BOX", 0.25
    nt.links.new(coords, node.inputs["Vector"])
    return node


def pbr_mat(name: str, folder: Path, tile: float, tint=None, sat: float = 1.0, uv: bool = False,
            rough_add: float = 0.0, sheen: float = 0.0, bump: float = 1.0):  # fmt: skip
    """Principled material from Poly Haven maps (diffuse, GL normal, roughness), tile =
    metres per texture repeat. Box projection on object coordinates (objects keep unit
    scale, so the texture keeps its real size) or, with uv, the mesh UVs. tint multiplies
    the colour after sat scales its saturation."""
    bpy = _bpy()
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / tile,) * 3
    nt.links.new(tc.outputs["UV" if uv else "Object"], mp.inputs["Vector"])
    co = mp.outputs["Vector"]
    files = {k: next(folder.glob(f"*_{k}_*.jpg")) for k in ("diff", "nor_gl", "rough")}
    col = _tex_node(nt, files["diff"], False, co, not uv).outputs["Color"]
    if sat != 1.0:
        hs = nt.nodes.new("ShaderNodeHueSaturation")
        hs.inputs["Saturation"].default_value = sat
        nt.links.new(col, hs.inputs["Color"])
        col = hs.outputs["Color"]
    if tint is not None:
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        mix.inputs[7].default_value = (*tint, 1.0)  # B (colour)
        nt.links.new(col, mix.inputs[6])  # A (colour)
        col = mix.outputs[2]
    nt.links.new(col, bsdf.inputs["Base Color"])
    rough = _tex_node(nt, files["rough"], True, co, not uv).outputs["Color"]
    if rough_add:
        m = nt.nodes.new("ShaderNodeMath")
        m.operation, m.use_clamp = "ADD", True
        m.inputs[1].default_value = rough_add
        nt.links.new(rough, m.inputs[0])
        rough = m.outputs[0]
    nt.links.new(rough, bsdf.inputs["Roughness"])
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.inputs["Strength"].default_value = bump
    nt.links.new(_tex_node(nt, files["nor_gl"], True, co, not uv).outputs["Color"],
                 nmap.inputs["Color"])  # fmt: skip
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    if sheen:
        bsdf.inputs["Sheen Weight"].default_value = sheen
    return mat


def flat_mat(name: str, color, rough: float = 0.5, metal: float = 0.0, coat: float = 0.0,
             noise: float = 0.0, emit: float = 0.0):  # fmt: skip
    """Plain Principled material. noise adds a faint large-scale value variation so painted
    surfaces are not perfectly uniform (also gives SfM a little texture)."""
    bpy = _bpy()
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Coat Weight"].default_value = coat
    if emit:
        bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emit
    if noise:
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = 6.0
        tex.inputs["Detail"].default_value = 8.0
        ramp = nt.nodes.new("ShaderNodeMapRange")
        ramp.inputs["To Min"].default_value = 1.0 - noise
        ramp.inputs["To Max"].default_value = 1.0 + noise
        nt.links.new(tex.outputs["Fac"], ramp.inputs["Value"])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        mix.inputs[6].default_value = (*color, 1.0)
        nt.links.new(ramp.outputs["Result"], mix.inputs[7])
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    return mat


def image_mat(name: str, path: Path, rough: float = 0.6, emit: float = 0.0):
    """Material showing an image on the mesh UVs (posters, the laptop screen)."""
    bpy = _bpy()
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(path), check_existing=True)
    tex.extension = "CLIP"
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = rough
    if emit:
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = emit
        bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
        nt.links.remove(bsdf.inputs["Base Color"].links[0])
        bsdf.inputs["Roughness"].default_value = 0.15
    return mat


def glass_mat(name: str):
    """Window glass that barely reflects: mostly a straight-through transparent surface with
    a 3 % gloss. Real glass mirrored the bright ceiling and the pendant lamp as a white
    patch over the view, and such view-dependent reflections are hard for splats anyway."""
    bpy = _bpy()
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    clear.inputs["Color"].default_value = (0.96, 0.98, 0.97, 1.0)
    gloss = nt.nodes.new("ShaderNodeBsdfGlossy")
    gloss.inputs["Roughness"].default_value = 0.02
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = 0.03
    nt.links.new(clear.outputs[0], mix.inputs[1])
    nt.links.new(gloss.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    return mat


# ---------------------------------------------------------------- geometry


class Parts:
    """Collects primitives into one bmesh, each with a material slot; build() makes the
    object. Matrices are mathutils 4x4 in the object's frame (= world, origin at 0)."""

    def __init__(self, name: str):
        import bmesh  # noqa: PLC0415

        self.name = name
        self.bm = bmesh.new()
        self.mats: list = []

    def _slot(self, mat) -> int:
        if mat not in self.mats:
            self.mats.append(mat)
        return self.mats.index(mat)

    def _tag(self, verts, mat, smooth: bool) -> None:
        mi = self._slot(mat)
        vs = set(verts)
        for f in {f for v in vs for f in v.link_faces}:
            if all(v in vs for v in f.verts):
                f.material_index, f.smooth = mi, smooth

    def box(self, lo, hi, mat, rot=None, pivot=None):
        """Axis-aligned box lo..hi, optionally rotated (Euler XYZ, radians) about pivot
        (default its centre)."""
        import bmesh  # noqa: PLC0415
        from mathutils import Euler, Matrix, Vector  # noqa: PLC0415

        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        c, s = (lo + hi) / 2, hi - lo
        m = Matrix.Translation(Vector(c)) @ Matrix.Diagonal(Vector((*s, 1.0)))
        if rot is not None:
            p = Vector(c if pivot is None else pivot)
            m = Matrix.Translation(p) @ Euler(rot).to_matrix().to_4x4() @ \
                Matrix.Translation(-p) @ m  # fmt: skip
        res = bmesh.ops.create_cube(self.bm, size=1.0, matrix=m)
        self._tag(res["verts"], mat, False)
        return res["verts"]

    def cyl(self, centre, r1, r2, depth, mat, rot=None, segs: int = 24, caps: bool = True):
        """Cone or cylinder along z (before rot), centred at centre."""
        import bmesh  # noqa: PLC0415
        from mathutils import Euler, Matrix, Vector  # noqa: PLC0415

        m = Matrix.Translation(Vector(centre))
        if rot is not None:
            m = m @ Euler(rot).to_matrix().to_4x4()
        res = bmesh.ops.create_cone(self.bm, cap_ends=caps, segments=segs, radius1=r1,
                                    radius2=r2, depth=depth, matrix=m)  # fmt: skip
        self._tag(res["verts"], mat, True)
        return res["verts"]

    def sphere(self, centre, radii, mat, segs: int = 24, rot=None):
        import bmesh  # noqa: PLC0415
        from mathutils import Euler, Matrix, Vector  # noqa: PLC0415

        m = Matrix.Translation(Vector(centre))
        if rot is not None:
            m = m @ Euler(rot).to_matrix().to_4x4()
        m = m @ Matrix.Diagonal(Vector((*radii, 1.0)))
        res = bmesh.ops.create_uvsphere(self.bm, u_segments=segs, v_segments=segs // 2,
                                        radius=1.0, matrix=m)  # fmt: skip
        self._tag(res["verts"], mat, True)
        return res["verts"]

    def torus(self, centre, major: float, minor: float, mat, rot=None, arc: float = 2 * math.pi,
              n1: int = 24, n2: int = 10):  # fmt: skip
        """Torus (or an arc of one, from angle 0) in the xz plane before rot."""
        from mathutils import Euler, Matrix, Vector  # noqa: PLC0415

        m = Matrix.Translation(Vector(centre))
        if rot is not None:
            m = m @ Euler(rot).to_matrix().to_4x4()
        closed = arc >= 2 * math.pi - 1e-6
        n_a = n1 if closed else n1 + 1
        rings = []
        for i in range(n_a):
            a = arc * i / n1
            ring = []
            for j in range(n2):
                b = 2 * math.pi * j / n2
                p = Vector(((major + minor * math.cos(b)) * math.cos(a), 0.0,
                            (major + minor * math.cos(b)) * math.sin(a)))  # fmt: skip
                p.y = minor * math.sin(b)
                ring.append(self.bm.verts.new(m @ p))
            rings.append(ring)
        verts = [v for r in rings for v in r]
        pairs = list(zip(rings, rings[1:] + ([rings[0]] if closed else []), strict=False))
        for r0, r1 in pairs:
            for j in range(n2):
                self.bm.faces.new((r0[j], r0[(j + 1) % n2], r1[(j + 1) % n2], r1[j]))
        self._tag(verts, mat, True)
        return verts

    def build(self, bevel: float = 0.0, segments: int = 2):
        bpy = _bpy()
        me = bpy.data.meshes.new(self.name)
        self.bm.normal_update()
        self.bm.to_mesh(me)
        self.bm.free()
        obj = bpy.data.objects.new(self.name, me)
        bpy.context.collection.objects.link(obj)
        for m in self.mats:
            me.materials.append(m)
        if bevel:
            mod = obj.modifiers.new("bevel", "BEVEL")
            mod.width, mod.segments = bevel, segments
            mod.limit_method, mod.angle_limit = "ANGLE", math.radians(40)
            mod.miter_outer = "MITER_ARC"
        return obj


def quads_object(name: str, quads: list, mats: list, mat_index=None):
    """Mesh object from planar quads (corners counter-clockwise seen from the lit side)."""
    import bmesh  # noqa: PLC0415

    bpy = _bpy()
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new()
    for i, q in enumerate(quads):
        f = bm.faces.new([bm.verts.new(p) for p in q])
        f.material_index = 0 if mat_index is None else mat_index[i]
        for loop, (u, v) in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1)), strict=False):
            loop[uv].uv = (u, v)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    for m in mats:
        me.materials.append(m)
    return obj


def rect_x(y: float, x0, x1, z0, z1, facing: float):
    """Quad in the plane y = const spanning x0..x1, z0..z1, normal along +-y."""
    q = [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
    return q if facing < 0 else [q[1], q[0], q[3], q[2]]


def rect_y(x: float, y0, y1, z0, z1, facing: float):
    """Quad in the plane x = const spanning y0..y1, z0..z1, normal along +-x."""
    q = [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]
    return q if facing > 0 else [q[1], q[0], q[3], q[2]]


def apply_modifiers(obj) -> None:
    """Bake the modifier stack into the mesh, so GT meshes match what is rendered."""
    bpy = _bpy()
    if not obj.modifiers:
        return
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg), depsgraph=dg)
    old = obj.data
    obj.modifiers.clear()
    obj.data = me
    me.name = obj.name
    bpy.data.meshes.remove(old)


def join(name: str, objs: list):
    bpy = _bpy()
    for o in objs:
        apply_modifiers(o)
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = obj.data.name = name
    return obj


def place(obj, x: float, y: float, z: float = 0.0, yaw: float = 0.0, scale: float = 1.0):
    obj.location = (x, y, z)
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (0.0, 0.0, math.radians(yaw))
    obj.scale = (scale,) * 3
    _bpy().context.view_layer.update()
    return obj


def bake_transform(obj) -> None:
    """Move the object transform into the mesh (origin stays at the world origin)."""
    from mathutils import Matrix  # noqa: PLC0415

    obj.data.transform(obj.matrix_world)
    obj.matrix_world = Matrix.Identity(4)


def drop_material_faces(obj, part: str) -> None:
    """Delete the faces whose material name contains part."""
    import bmesh  # noqa: PLC0415

    idx = {i for i, m in enumerate(obj.data.materials) if m and part in m.name}
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.material_index in idx], context="FACES")
    bm.to_mesh(obj.data)
    bm.free()


def top_z(obj) -> float:
    return float(world_bbox(obj)[1][2])


# ---------------------------------------------------------------- room shell


def build_shell(assets: Path, m: dict) -> dict:
    """Floor, walls (window hole and reveal in the north wall), ceiling, skirting."""
    h = RH
    floor = quads_object("floor", [[(-HX, -HY, 0), (HX, -HY, 0), (HX, HY, 0), (-HX, HY, 0)]],
                         [m["floor"]])  # fmt: skip
    ceiling = quads_object(
        "ceiling", [[(-HX, -HY, h), (-HX, HY, h), (HX, HY, h), (HX, -HY, h)]], [m["ceiling"]]
    )
    yw, yo = HY, HY + WIN_DEPTH
    quads = [
        rect_y(-HX, -HY, HY, 0, h, +1),  # west (accent)
        rect_y(HX, -HY, HY, 0, h, -1),  # east
        rect_x(-HY, -HX, HX, 0, h, +1),  # south (the door is a slab on it)
        # north wall around the window hole
        rect_x(yw, -HX, WIN_X0, 0, h, -1),
        rect_x(yw, WIN_X1, HX, 0, h, -1),
        rect_x(yw, WIN_X0, WIN_X1, 0, WIN_Z0, -1),
        rect_x(yw, WIN_X0, WIN_X1, WIN_Z1, h, -1),
        # reveal: the hole's sides, head and bottom (the sill board covers the bottom)
        rect_y(WIN_X0, yw, yo, WIN_Z0, WIN_Z1, +1),
        rect_y(WIN_X1, yw, yo, WIN_Z0, WIN_Z1, -1),
        [(WIN_X0, yw, WIN_Z1), (WIN_X0, yo, WIN_Z1), (WIN_X1, yo, WIN_Z1), (WIN_X1, yw, WIN_Z1)],
        [(WIN_X0, yw, WIN_Z0), (WIN_X1, yw, WIN_Z0), (WIN_X1, yo, WIN_Z0), (WIN_X0, yo, WIN_Z0)],
    ]
    walls = quads_object("wall", quads, [m["wall"], m["accent"]], [1] + [0] * (len(quads) - 1))

    # Skirting boards, 8 cm high and 1.6 cm deep, except across the door.
    sk = Parts("skirting")
    t, hs = 0.016, 0.08
    sk.box((-HX, -HY, 0), (-HX + t, HY, hs), m["white_paint"])
    sk.box((HX - t, -HY, 0), (HX, HY, hs), m["white_paint"])
    sk.box((-HX, HY - t, 0), (HX, HY, hs), m["white_paint"])
    sk.box((-HX, -HY, 0), (DOOR_X0 - 0.07, -HY + t, hs), m["white_paint"])
    sk.box((DOOR_X1 + 0.07, -HY, 0), (HX, -HY + t, hs), m["white_paint"])
    skirting = sk.build(bevel=0.003)
    walls = join("wall", [walls, skirting])
    for obj, key in ((floor, "floor"), (walls, "wall"), (ceiling, "ceiling")):
        obj.pass_index = STRUCTURE[key]
    return {"floor": floor, "wall": walls, "ceiling": ceiling}


def build_window(m: dict):
    """Two-casement white window in the reveal, glass, interior sill, plus curtains and
    their rod (curtains are a separate object)."""
    p = Parts("window")
    w = m["white_plastic"]
    yf0, yf1 = HY + 0.10, HY + 0.17  # frame depth inside the reveal
    x0, x1, z0, z1 = WIN_X0, WIN_X1, WIN_Z0, WIN_Z1
    f = 0.06  # outer frame width
    p.box((x0, yf0, z0), (x1, yf1, z0 + f), w)
    p.box((x0, yf0, z1 - f), (x1, yf1, z1), w)
    p.box((x0, yf0, z0), (x0 + f, yf1, z1), w)
    p.box((x1 - f, yf0, z0), (x1, yf1, z1), w)
    xm = (x0 + x1) / 2
    s = 0.055  # sash width
    for a, b in ((x0 + f, xm), (xm, x1 - f)):
        ys0, ys1 = yf0 - 0.012, yf1 - 0.02
        p.box((a, ys0, z0 + f), (b, ys1, z0 + f + s), w)
        p.box((a, ys0, z1 - f - s), (b, ys1, z1 - f), w)
        p.box((a, ys0, z0 + f), (a + s, ys1, z1 - f), w)
        p.box((b - s, ys0, z0 + f), (b, ys1, z1 - f), w)
        p.box((a + s, yf0 + 0.02, z0 + f + s), (b - s, yf0 + 0.026, z1 - f - s), m["glass"])
    # Handle on the right casement, at its left (meeting) stile.
    hx = xm + 0.028
    p.box((hx - 0.014, yf0 - 0.024, 1.52), (hx + 0.014, yf0 - 0.012, 1.60), m["white_plastic"])
    p.cyl((hx, yf0 - 0.03, 1.56), 0.007, 0.007, 0.02, m["chrome"], rot=(math.pi / 2, 0, 0))
    p.box((hx - 0.009, yf0 - 0.045, 1.44), (hx + 0.009, yf0 - 0.032, 1.57), m["white_plastic"])
    # Interior sill board, 3 cm proud of the wall.
    p.box((x0 - 0.04, HY - 0.035, z0 - 0.025), (x1 + 0.04, yf0, z0), m["sill"])
    win = p.build(bevel=0.002)
    win.pass_index = STRUCTURE["window"]

    # Curtain rod and rings, curtains gathered at both sides, floor length.
    rod = Parts("curtain_rod")
    yr, zr = HY - 0.10, 2.45
    rod.cyl((0.25, yr, zr), 0.011, 0.011, 2.2, m["black_metal"], rot=(0, math.pi / 2, 0))
    for xe in (-0.85, 1.35):
        rod.sphere((xe, yr, zr), (0.022, 0.022, 0.022), m["black_metal"], segs=16)
        xb = xe * 0.9 + 0.025
        rod.cyl((xb, HY - 0.05, zr), 0.008, 0.008, 0.1, m["black_metal"], rot=(math.pi / 2, 0, 0))
        rod.box((xb - 0.025, HY - 0.006, zr - 0.035), (xb + 0.025, HY, zr + 0.035),
                m["black_metal"])  # fmt: skip
    rods = rod.build()
    curtains = []
    for cx0, cx1 in ((-0.80, -0.34), (0.84, 1.30)):
        curtains.append(curtain_panel(cx0, cx1, yr, zr - 0.03, m))
        for i in range(9):
            x = cx0 + (cx1 - cx0) * i / 8
            curtains.append(_ring(x, yr, zr, m))
    curtain = join("curtains", [rods] + curtains)
    return win, curtain


def _ring(x, y, z, m):
    p = Parts("ring")
    p.torus((x, y, z), 0.018, 0.0035, m["black_metal"], rot=(0, 0, math.pi / 2), n1=16, n2=6)
    return p.build()


def curtain_panel(x0: float, x1: float, y: float, ztop: float, m: dict):
    """Gathered curtain: a pleated sheet, folds deeper towards the hem, hem 2 cm above the
    floor, with solidify for thickness."""
    bpy = _bpy()
    import bmesh  # noqa: PLC0415

    nx, nz = 120, 60
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new()
    rng = np.random.default_rng(int(x0 * 100) & 0xFFFF)
    phase = rng.uniform(0, 2 * math.pi, 3)
    width = x1 - x0
    fold = 0.11  # pleat spacing at the top (m)
    grid = []
    for k in range(nz + 1):
        v = k / nz  # 0 bottom .. 1 top
        z = 0.02 + v * (ztop - 0.02)
        row = []
        for i in range(nx + 1):
            u = i / nx
            x = x0 + u * width
            a = 2 * math.pi * u * width / fold
            depth = 0.035 + 0.025 * (1 - v)
            wob = 0.008 * math.sin(3.1 * u * 6.28 + phase[0] + 4 * v)
            dy = depth * math.sin(a + 0.4 * math.sin(phase[1] + 2.3 * v)) + wob
            x += 0.012 * (1 - v) * math.sin(phase[2] + 7 * u)
            row.append(bm.verts.new((x, y - 0.035 + dy, z)))
        grid.append(row)
    for k in range(nz):
        for i in range(nx):
            f = bm.faces.new((grid[k][i], grid[k][i + 1], grid[k + 1][i + 1], grid[k + 1][i]))
            for loop, (du, dv) in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1)), strict=False):
                loop[uv].uv = ((i + du) / nx * width * 1.6, (k + dv) / nz * ztop)
            f.smooth = True
    me = bpy.data.meshes.new("curtain")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("curtain", me)
    bpy.context.collection.objects.link(obj)
    me.materials.append(m["curtain"])
    sol = obj.modifiers.new("solidify", "SOLIDIFY")
    sol.thickness, sol.offset = 0.004, 0.0
    return obj


def build_door(m: dict):
    """Closed white flush door on the south wall: casing, slab, lever handle, hinges."""
    p = Parts("door")
    y = -HY
    cw, cd, d = 0.07, 0.018, 0.04
    w = m["white_paint"]
    p.box((DOOR_X0 - cw, y, 0), (DOOR_X0, y + cd, DOOR_H + cw), w)
    p.box((DOOR_X1, y, 0), (DOOR_X1 + cw, y + cd, DOOR_H + cw), w)
    p.box((DOOR_X0, y, DOOR_H), (DOOR_X1, y + cd, DOOR_H + cw), w)
    p.box((DOOR_X0 + 0.003, y, 0.008), (DOOR_X1 - 0.003, y + d - 0.012, DOOR_H - 0.003),
          m["door"])  # fmt: skip
    # Lever handle and escutcheons on the west (latch) side.
    hx, hz, yf = DOOR_X0 + 0.065, 1.02, y + d - 0.012
    p.box((hx - 0.022, yf, hz - 0.085), (hx + 0.022, yf + 0.008, hz + 0.03), m["steel"])
    p.cyl((hx, yf + 0.03, hz), 0.009, 0.009, 0.05, m["steel"], rot=(math.pi / 2, 0, 0))
    p.cyl((hx + 0.06, yf + 0.055, hz), 0.0085, 0.0085, 0.13, m["steel"], rot=(0, math.pi / 2, 0))
    p.cyl((hx, yf + 0.01, hz - 0.06), 0.006, 0.006, 0.006, m["black_metal"],
          rot=(math.pi / 2, 0, 0))  # keyhole  # fmt: skip
    for zh in (0.22, 1.82):  # hinges on the east edge
        p.cyl((DOOR_X1 - 0.004, y + 0.03, zh), 0.007, 0.007, 0.1, m["steel"])
    door = p.build(bevel=0.002)
    door.pass_index = STRUCTURE["door"]
    return door


def build_radiator(m: dict):
    """Panel radiator under the window: ribbed front, top grille, valve and pipe."""
    p = Parts("radiator")
    x0, x1, z0, z1 = -0.25, 0.75, 0.16, 0.70
    y1 = HY - 0.045
    y0 = y1 - 0.10
    p.box((x0, y0 + 0.01, z0), (x1, y1, z1), m["radiator"])
    n = int((x1 - x0) / 0.033)
    for i in range(n):
        xc = x0 + (i + 0.5) * (x1 - x0) / n
        p.box((xc - 0.009, y0, z0 + 0.01), (xc + 0.009, y0 + 0.012, z1 - 0.01), m["radiator"])
    p.box((x0 + 0.01, y0 + 0.015, z1), (x1 - 0.01, y1 - 0.005, z1 + 0.004), m["grille"])
    p.box((x0 - 0.01, y0 + 0.02, z0 + 0.03), (x0, y1 - 0.02, z1 - 0.03), m["radiator"])
    p.box((x1, y0 + 0.02, z0 + 0.03), (x1 + 0.01, y1 - 0.02, z1 - 0.03), m["radiator"])
    # Thermostatic valve at the right end and the pipe down into the floor.
    p.cyl((x1 + 0.05, y0 + 0.05, z0 + 0.06), 0.009, 0.009, 0.1, m["radiator"],
          rot=(0, math.pi / 2, 0))  # fmt: skip
    p.cyl((x1 + 0.1, y0 + 0.05, z0 + 0.1), 0.026, 0.024, 0.09, m["white_plastic"], segs=20)
    p.cyl((x1 + 0.1, y0 + 0.05, z0 + 0.15), 0.024, 0.02, 0.012, m["white_plastic"], segs=20)
    p.cyl((x1 + 0.1, y0 + 0.05, z0 / 2 + 0.02), 0.009, 0.009, z0 + 0.04, m["radiator"])
    p.cyl((x0 - 0.04, y0 + 0.05, z0 / 2 + 0.05), 0.009, 0.009, z0 + 0.1, m["radiator"])
    return p.build(bevel=0.0025)


def build_electrics(m: dict):
    """Light switch by the door and three double sockets, one object."""
    p = Parts("electrics")
    w = m["white_plastic"]

    def plate(wall: str, a: float, z: float, wdt: float, sockets: int):
        def box(da0, dz0, d0, da1, dz1, d1, mat):
            if wall == "south":
                p.box((a + da0, -HY + d0, z + dz0), (a + da1, -HY + d1, z + dz1), mat)
            elif wall == "east":
                p.box((HX - d1, a + da0, z + dz0), (HX - d0, a + da1, z + dz1), mat)
            elif wall == "north":
                p.box((a + da0, HY - d1, z + dz0), (a + da1, HY - d0, z + dz1), mat)
            else:  # west
                p.box((-HX + d0, a + da0, z + dz0), (-HX + d1, a + da1, z + dz1), mat)

        box(-wdt / 2, -0.043, 0, wdt / 2, 0.043, 0.009, w)
        if sockets == 0:  # rocker switch
            box(-0.027, -0.027, 0.009, 0.027, 0.027, 0.014, w)
        for i in range(sockets):
            c = (i - (sockets - 1) / 2) * 0.072
            box(c - 0.024, -0.024, 0.009, c + 0.024, 0.024, 0.011, m["grey_plastic"])

    plate("south", DOOR_X0 - 0.17, 1.05, 0.086, 0)  # switch by the door
    plate("east", 0.33, 0.30, 0.158, 2)  # behind the desk
    plate("north", -1.0, 0.30, 0.158, 2)  # by the plant
    plate("west", -0.65, 0.30, 0.158, 2)  # south of the bed
    return p.build(bevel=0.0015)


# ---------------------------------------------------------------- furniture built here


def build_bed(m: dict):
    """120 x 200 cm bed: oak frame and legs, upholstered headboard, mattress in a fitted
    sheet. Returns (frame, mattress) separately for the cloth collisions."""
    p = Parts("bed")
    oak, b = m["oak"], 0.04
    x0, x1, y0, y1 = -HX + 0.02, 0.30, -0.44, 0.84
    zr0, zr1 = 0.12, 0.38
    for xl in (x0 + 0.01, x1 - 0.06):
        for yl in (y0 + 0.01, y1 - 0.06):
            p.box((xl, yl, 0), (xl + 0.05, yl + 0.05, zr0 + 0.01), oak)
    p.box((x0, y0, zr0), (x1, y0 + b, zr1), oak)  # side rails
    p.box((x0, y1 - b, zr0), (x1, y1, zr1), oak)
    p.box((x1 - b, y0, zr0), (x1, y1, zr1 + 0.02), oak)  # foot board
    p.box((x0 + b, y0 + b, 0.27), (x1 - b, y1 - b, 0.30), m["slats"])  # slatted base (hidden)
    p.box((x0, y0 - 0.005, zr0), (x0 + b, y1 + 0.005, 1.02), oak)  # headboard board
    frame = p.build(bevel=0.004)
    pad = Parts("headboard_pad")
    pad.box((x0 + b, y0 + 0.03, 0.56), (x0 + b + 0.07, y1 - 0.03, 0.98), m["headboard"])
    padobj = pad.build(bevel=0.025, segments=4)
    frame = join("bed", [frame, padobj])

    mt = Parts("mattress")
    mt.box((x0 + b, y0 + b, 0.30), (x1 - b, y1 - b, 0.53), m["sheet"])
    mattress = mt.build(bevel=0.045, segments=5)
    apply_modifiers(mattress)
    return frame, mattress


def pillow(name: str, centre, size, mat, rot=(0.0, 0.0, 0.0), seed: int = 0):
    """Pillow: a superellipsoid (boxy, thinner at the edges) with soft wrinkles."""
    bpy = _bpy()
    import bmesh  # noqa: PLC0415
    from mathutils import Euler, Matrix, Vector  # noqa: PLC0415

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=1.0)
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 6.28, 4)
    sx, sy, sz = (s / 2 for s in size)
    for v in bm.verts:
        x, y, z = v.co
        e = 0.35
        x2 = math.copysign(abs(x) ** e, x)
        y2 = math.copysign(abs(y) ** e, y)
        edge = max(abs(x2), abs(y2))
        z2 = z * (1.0 - 0.75 * edge**6)
        wr = 0.06 * math.sin(9 * x2 + ph[0]) * math.sin(7 * y2 + ph[1]) * edge**2
        v.co = Vector((x2 * sx, y2 * sy, z2 * sz * (1 + wr)))
    for f in bm.faces:
        f.smooth = True
    m = Matrix.Translation(Vector(centre)) @ Euler(rot).to_matrix().to_4x4()
    bmesh.ops.transform(bm, matrix=m, verts=bm.verts)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    me.materials.append(mat)
    sub = obj.modifiers.new("subsurf", "SUBSURF")
    sub.levels = sub.render_levels = 1
    return obj


def cloth_sheet(name: str, x0, x1, y0, y1, z, nx, ny, mat, colliders: list, frames: int,
                stiffness: float, mass: float, thickness: float, seed: int,
                pin_x: float | None = None, noise: float = 0.003):  # fmt: skip
    """Drop a sheet onto colliders (and the floor) with Blender's cloth solver and keep the
    last frame. Faint noise on the start shape breaks the symmetry so it folds naturally.
    Vertices with x < pin_x stay put (a duvet edge tucked under the pillow), so the weight
    hanging over the foot cannot pull the sheet off the bed."""
    bpy = _bpy()
    import bmesh  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new()
    grid = [[bm.verts.new((x0 + (x1 - x0) * i / nx, y0 + (y1 - y0) * k / ny,
                           z + rng.normal(0, noise)))
             for i in range(nx + 1)] for k in range(ny + 1)]  # fmt: skip
    for k in range(ny):
        for i in range(nx):
            f = bm.faces.new((grid[k][i], grid[k][i + 1], grid[k + 1][i + 1], grid[k + 1][i]))
            for loop, (du, dv) in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1)), strict=False):
                loop[uv].uv = ((x0 + (x1 - x0) * (i + du) / nx),
                               (y0 + (y1 - y0) * (k + dv) / ny))  # fmt: skip
            f.smooth = True
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    me.materials.append(mat)
    if pin_x is not None:
        vg = obj.vertex_groups.new(name="pin")
        vg.add([v.index for v in me.vertices if v.co.x < pin_x], 1.0, "REPLACE")

    floor = quads_object("cloth_floor", [[(-HX, -HY, 0), (HX, -HY, 0), (HX, HY, 0), (-HX, HY, 0)]],
                         [mat])  # fmt: skip
    colliders = [*colliders, floor]
    for c in colliders:
        c.modifiers.new("collision", "COLLISION")
        c.collision.thickness_outer = 0.004
        c.collision.cloth_friction = 8.0
    mod = obj.modifiers.new("cloth", "CLOTH")
    s = mod.settings
    s.quality = 10
    s.mass = mass
    s.tension_stiffness = s.compression_stiffness = 20.0
    s.shear_stiffness = 10.0
    s.bending_stiffness = stiffness
    s.air_damping = 1.0
    if pin_x is not None:
        s.vertex_group_mass = "pin"
    cs = mod.collision_settings
    cs.collision_quality = 4
    cs.distance_min = 0.006 + thickness / 2
    cs.use_self_collision = True
    cs.self_distance_min = 0.004
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, frames
    mod.point_cache.frame_start, mod.point_cache.frame_end = 1, frames
    for f in range(1, frames + 1):
        scene.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    baked = bpy.data.meshes.new_from_object(obj.evaluated_get(dg), depsgraph=dg)
    obj.modifiers.clear()
    obj.data = baked
    baked.name = name
    obj.vertex_groups.clear()
    lo, hi = world_bbox(obj)
    if lo[2] < -0.01 or abs(lo[0]) > HX or abs(hi[0]) > HX or abs(lo[1]) > HY or abs(hi[1]) > HY:
        raise RuntimeError(f"cloth {name} left the room ({lo} .. {hi}): the solver blew up")
    scene.frame_set(1)
    for c in colliders:
        for md in [md for md in c.modifiers if md.type == "COLLISION"]:
            c.modifiers.remove(md)
    bpy.data.meshes.remove(floor.data)
    sol = obj.modifiers.new("solidify", "SOLIDIFY")
    sol.thickness, sol.offset = thickness, 0.0
    sub = obj.modifiers.new("subsurf", "SUBSURF")
    sub.levels = sub.render_levels = 1
    apply_modifiers(obj)
    return obj


def build_dresser(m: dict):
    """Chest of four drawers, 80 x 45 x 85 cm, oak top and carcass, white fronts, black
    bar handles, against the south wall."""
    p = Parts("dresser")
    x0, x1, y0, y1, h = -1.12, -0.32, -HY + 0.01, -HY + 0.46, 0.85
    oak = m["oak"]
    p.box((x0, y0, 0.0), (x1, y1 - 0.02, 0.06), m["black_metal"])  # recessed plinth
    p.box((x0, y0, 0.06), (x0 + 0.018, y1, h - 0.022), oak)
    p.box((x1 - 0.018, y0, 0.06), (x1, y1, h - 0.022), oak)
    p.box((x0 - 0.008, y0, h - 0.022), (x1 + 0.008, y1 + 0.012, h), oak)  # top
    p.box((x0, y0, 0.06), (x1, y0 + 0.012, h - 0.022), oak)  # back
    zs = np.linspace(0.07, h - 0.03, 5)
    for za, zb in zip(zs[:-1], zs[1:], strict=True):
        p.box((x0 + 0.02, y1 - 0.003, za + 0.002), (x1 - 0.02, y1 + 0.016, zb - 0.002),
              m["white_front"])  # fmt: skip
        zc = (za + zb) / 2 + 0.02
        for xc in (x0 + 0.2, x1 - 0.2):
            p.box((xc - 0.004, y1 + 0.016, zc - 0.004), (xc + 0.004, y1 + 0.034, zc + 0.004),
                  m["black_metal"])  # fmt: skip
        p.box((x0 + 0.19, y1 + 0.03, zc - 0.006), (x1 - 0.19, y1 + 0.042, zc + 0.006),
              m["black_metal"])  # fmt: skip
    return p.build(bevel=0.0025)


def build_desk(m: dict):
    """120 x 60 cm desk against the east wall: oak top, black steel frame."""
    p = Parts("desk")
    x0, x1, y0, y1, h = HX - 0.61, HX - 0.01, 0.45, 1.65, 0.75
    p.box((x0, y0, h - 0.025), (x1, y1, h), m["oak"])
    s = 0.035
    for xl in (x0 + 0.04, x1 - 0.04 - s):
        for yl in (y0 + 0.04, y1 - 0.04 - s):
            p.box((xl, yl, 0), (xl + s, yl + s, h - 0.025), m["black_metal"])
        p.box((xl, y0 + 0.04, h - 0.1), (xl + s, y1 - 0.04, h - 0.025), m["black_metal"])
    p.box((x1 - 0.075, y0 + 0.04, h - 0.1), (x1 - 0.04, y1 - 0.04, h - 0.025), m["black_metal"])
    return p.build(bevel=0.002)


def build_shelf(m: dict):
    """Oak wall shelf above the desk on two black brackets."""
    p = Parts("wall_shelf")
    y0, y1, z, d = 0.45, 1.35, 1.75, 0.22
    p.box((HX - d, y0, z), (HX, y1, z + 0.025), m["oak"])
    for yb in (y0 + 0.12, y1 - 0.12):
        p.box((HX - 0.18, yb - 0.008, z - 0.004), (HX, yb + 0.008, z), m["black_metal"])
        p.box((HX - 0.006, yb - 0.008, z - 0.16), (HX, yb + 0.008, z), m["black_metal"])
        p.box((HX - 0.12, yb - 0.006, z - 0.1), (HX - 0.01, yb + 0.006, z - 0.09),
              m["black_metal"], rot=(0, -math.radians(40), 0))  # fmt: skip
    return p.build(bevel=0.002)


def build_bin(m: dict, x: float, y: float):
    """Waste paper bin under the desk: open white steel cylinder, a crumpled paper in it."""
    p = Parts("bin")
    r, h = 0.12, 0.30
    p.cyl((x, y, h / 2), r * 0.92, r, h, m["bin"], segs=40, caps=False)
    p.cyl((x, y, h / 2 + 0.002), r * 0.92 - 0.002, r - 0.002, h - 0.004, m["bin"], segs=40,
          caps=False)  # fmt: skip
    p.cyl((x, y, 0.003), r * 0.92, r * 0.92, 0.006, m["bin"], segs=40)
    p.torus((x, y, h), r - 0.001, 0.0025, m["bin"], rot=(math.pi / 2, 0, 0), n1=40, n2=6)
    p.sphere((x + 0.02, y - 0.01, 0.045), (0.045, 0.04, 0.038), m["mount"], segs=10)
    obj = p.build()
    for v in obj.data.vertices:  # crumple the paper ball
        if v.co.z < 0.09 and math.hypot(v.co.x - x - 0.02, v.co.y - y + 0.01) < 0.05:
            v.co.x += 0.008 * math.sin(91 * v.co.y + 13 * v.co.z)
            v.co.z += 0.008 * math.sin(77 * v.co.x)
    return obj


def build_backpack(m: dict, x: float, y: float, yaw: float):
    """Soft fabric backpack standing on the floor: body, front pocket, zips, top handle and
    shoulder straps, all heavily rounded."""
    p = Parts("backpack_body")
    fab, dark = m["backpack"], m["backpack_trim"]
    w, d, h = 0.29, 0.17, 0.42
    p.box((-w / 2, -d / 2, 0), (w / 2, d / 2, h), fab)
    body = p.build(bevel=0.06, segments=6)
    q = Parts("backpack_pocket")
    q.box((-0.115, -d / 2 - 0.045, 0.04), (0.115, -d / 2 + 0.02, 0.24), fab)
    pocket = q.build(bevel=0.03, segments=5)
    r = Parts("backpack_trim")
    r.box((-0.1, -d / 2 - 0.0465, 0.215), (0.1, -d / 2 - 0.04, 0.222), dark)  # pocket zip
    r.box((-0.012, -d / 2 - 0.05, 0.19), (0.012, -d / 2 - 0.045, 0.225), m["steel"])  # puller
    for sx in (-1, 1):  # shoulder straps on the back (+y), padded
        r.box((sx * 0.07 - 0.03, d / 2 - 0.005, 0.06), (sx * 0.07 + 0.03, d / 2 + 0.022, 0.39),
              dark)  # fmt: skip
    r.torus((0, 0.0, h - 0.005), 0.035, 0.007, dark, rot=(0, 0, 0), arc=math.pi, n1=12, n2=6)
    trim = r.build(bevel=0.006, segments=3)
    obj = join("backpack", [body, pocket, trim])
    place(obj, x, y, 0.0, yaw)
    bake_transform(obj)
    return obj


def build_rug(m: dict):
    p = Parts("rug")
    p.box((-1.0, -1.42, 0), (0.62, -0.28, 0.012), m["rug"])
    return p.build(bevel=0.004)


def build_poster(name: str, image: Path, centre, wall: str, height: float, m: dict, img_wh):
    """Framed print: thin black frame, white mount, the image, flat on a wall."""
    aspect = img_wh[0] / img_wh[1]
    ih, iw = height, height * aspect
    mount = 0.05
    fw, fd = 0.018, 0.022
    W, H = iw + 2 * mount + 2 * fw, ih + 2 * mount + 2 * fw
    p = Parts(name)
    # Built facing -y on the plane y = 0 (front at y = -fd), then rotated onto the wall.
    p.box((-W / 2, -fd, -H / 2), (W / 2, -fd + fw / 2, -H / 2 + fw), m["frame"])
    p.box((-W / 2, -fd, H / 2 - fw), (W / 2, -fd + fw / 2, H / 2), m["frame"])
    p.box((-W / 2, -fd, -H / 2), (-W / 2 + fw, -fd + fw / 2, H / 2), m["frame"])
    p.box((W / 2 - fw, -fd, -H / 2), (W / 2, -fd + fw / 2, H / 2), m["frame"])
    p.box((-W / 2, -fd + fw / 2, -H / 2), (W / 2, 0.0, H / 2), m["frame"])
    p.box((-W / 2 + fw, -fd + fw / 2 - 0.002, -H / 2 + fw), (W / 2 - fw, -fd + fw / 2, H / 2 - fw),
          m["mount"])  # fmt: skip
    frame = p.build(bevel=0.0015)
    art = quads_object(name + "_art", [[(-iw / 2, -fd + fw / 2 - 0.0025, -ih / 2),
                                        (iw / 2, -fd + fw / 2 - 0.0025, -ih / 2),
                                        (iw / 2, -fd + fw / 2 - 0.0025, ih / 2),
                                        (-iw / 2, -fd + fw / 2 - 0.0025, ih / 2)]],
                       [image_mat(name + "_print", image, rough=0.55)])  # fmt: skip
    obj = join(name, [frame, art])
    # Built facing -y; yaw turns that towards the room.
    yaw = {"north": 0.0, "south": 180.0, "west": 90.0, "east": -90.0}[wall]
    place(obj, *centre, yaw=yaw)
    bake_transform(obj)
    return obj


BOOK_COLOURS = [
    (0.42, 0.06, 0.05), (0.06, 0.14, 0.30), (0.12, 0.25, 0.14), (0.75, 0.60, 0.35),
    (0.08, 0.08, 0.08), (0.55, 0.32, 0.12), (0.80, 0.78, 0.72), (0.25, 0.12, 0.30),
    (0.62, 0.18, 0.12), (0.18, 0.34, 0.42),
]  # fmt: skip


def book_mats(m: dict) -> list:
    return [flat_mat(f"book_{i}", c, rough=0.55, noise=0.08) for i, c in enumerate(BOOK_COLOURS)]


def build_book_stack(name: str, x: float, y: float, z: float, yaw: float, covers: list,
                     m: dict, n: int, seed: int):  # fmt: skip
    """Books lying flat in a slightly messy stack."""
    rng = np.random.default_rng(seed)
    p = Parts(name)
    zc = z
    for _ in range(n):
        w, d, t = rng.uniform(0.15, 0.2), rng.uniform(0.21, 0.25), rng.uniform(0.02, 0.04)
        a = math.radians(rng.uniform(-8, 8))
        dx, dy = rng.uniform(-0.01, 0.01, 2)
        lo = (-w / 2 + dx, -d / 2 + dy, zc)
        _book_flat(p, lo, (w, d, t), covers[rng.integers(len(covers))], m, rot=(0, 0, a),
                   pivot=(dx, dy, zc))  # fmt: skip
        zc += t
    obj = p.build(bevel=0.0012)
    place(obj, x, y, 0.0, yaw)
    bake_transform(obj)
    return obj


def _book_flat(p: Parts, lo, size, cover, m, rot=None, pivot=None):
    lo = np.asarray(lo, float)
    w, d, t = size
    c = 0.0025
    p.box(lo, lo + (w, d, c), cover, rot, pivot)
    p.box(lo + (0, 0, t - c), lo + (w, d, t), cover, rot, pivot)
    p.box(lo, lo + (0.003, d, t), cover, rot, pivot)
    p.box(lo + (0.003, 0.003, c), lo + (w - 0.003, d - 0.003, t - c), m["pages"], rot, pivot)


def screen_image(path: Path) -> None:
    """A code editor on a dark theme, 1600 x 1000, saved as a PNG for the laptop screen."""
    w, h = 1600, 1000
    img = np.zeros((h, w, 3), np.float32)
    img[:] = (0.12, 0.13, 0.16)
    img[:, :56] = (0.09, 0.10, 0.12)  # activity bar
    img[:, 56:300] = (0.10, 0.11, 0.13)  # file tree
    img[:34] = (0.08, 0.08, 0.10)  # title bar
    img[34:70, 300:] = (0.10, 0.11, 0.13)  # tabs
    img[34:70, 300:520] = (0.12, 0.13, 0.16)
    img[h - 26 :] = (0.15, 0.35, 0.65)  # status bar
    rng = np.random.default_rng(6)
    palette = [(0.78, 0.47, 0.86), (0.38, 0.69, 0.94), (0.60, 0.80, 0.45), (0.86, 0.73, 0.45),
               (0.80, 0.82, 0.85), (0.47, 0.52, 0.58), (0.90, 0.45, 0.40)]  # fmt: skip
    for i in range(14):  # file tree entries
        y = 90 + i * 30
        img[y : y + 10, 90 + 18 * (i % 3) : 90 + 18 * (i % 3) + rng.integers(60, 160)] = (
            0.55, 0.57, 0.6)  # fmt: skip
    indent = 0
    for i in range(44):  # code lines
        y = 90 + i * 20
        if y > h - 40:
            break
        img[y : y + 9, 310:332] = (0.35, 0.38, 0.42)  # line numbers
        if rng.random() < 0.12:
            continue
        indent = int(np.clip(indent + rng.integers(-1, 2), 0, 4))
        x = 360 + indent * 36
        for _ in range(rng.integers(1, 6)):
            ln = int(rng.integers(20, 120))
            if x + ln > w - 40:
                break
            img[y : y + 9, x : x + ln] = palette[rng.integers(len(palette))]
            x += ln + 12
    img[90 + 13 * 20 : 90 + 13 * 20 + 18, 300:] *= 1.35  # current line
    bpy = _bpy()
    im = bpy.data.images.new("laptop_screen_src", w, h, alpha=True)
    rgba = np.concatenate([np.clip(img, 0, 1), np.ones((h, w, 1), np.float32)], axis=2)
    im.pixels.foreach_set(rgba[::-1].ravel())  # Blender stores rows bottom-up
    im.filepath_raw, im.file_format = str(path), "PNG"
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save()
    bpy.data.images.remove(im)


def build_laptop(m: dict, screen: Path, x: float, y: float, z: float, yaw: float):
    """Thin aluminium laptop, lid open ~110 deg, keys, trackpad, screen lit (emissive)."""
    p = Parts("laptop")
    W, D, T = 0.304, 0.212, 0.012
    al = m["aluminium"]
    p.box((-W / 2, -D / 2, 0.002), (W / 2, D / 2, T), al)
    p.box((-W / 2 + 0.02, -D / 2 + 0.01, 0), (W / 2 - 0.02, D / 2 - 0.01, 0.002),
          m["grey_plastic"])  # feet strip  # fmt: skip
    # Keyboard well and keys (6 rows), trackpad. Front edge at -y.
    kx0, kx1, ky0, ky1 = -0.125, 0.125, -0.01, 0.085
    p.box((kx0 - 0.004, ky0 - 0.004, T - 0.0006), (kx1 + 0.004, ky1 + 0.004, T + 0.0001),
          m["key_well"])  # fmt: skip
    rows, pitch = 6, (ky1 - ky0) / 6
    for r in range(rows):
        n = 14 if r < 5 else 9
        kw = (kx1 - kx0) / n
        for c in range(n):
            a = kx0 + c * kw + 0.0012
            b = a + kw - 0.0024
            if r == 5 and c == 4:
                b = a + kw * 1 - 0.0024
            yk = ky0 + r * pitch
            p.box((a, yk + 0.0012, T), (b, yk + pitch - 0.0012, T + 0.0011), m["key"])
    p.box((-0.06, -D / 2 + 0.012, T - 0.0002), (0.06, -0.022, T + 0.0003), m["trackpad"])
    # Lid: hinge along the back edge, tilted 110 deg from the base.
    lid = Parts("lid")
    LT = 0.006
    lid.box((-W / 2, 0, 0), (W / 2, LT, 0.205), al)
    lid.box((-W / 2 + 0.004, -0.0005, 0.004), (W / 2 - 0.004, 0.0, 0.201), m["bezel"])
    sm = 0.012
    lid_obj = lid.build(bevel=0.002)
    disp = quads_object("screen", [[(-W / 2 + sm, -0.0008, 0.012), (W / 2 - sm, -0.0008, 0.012),
                                    (W / 2 - sm, -0.0008, 0.195), (-W / 2 + sm, -0.0008, 0.195)]],
                        [image_mat("laptop_screen", screen, emit=1.6)])  # fmt: skip
    lid_obj = join("lid", [lid_obj, disp])
    from mathutils import Euler, Matrix, Vector  # noqa: PLC0415

    lid_obj.data.transform(Matrix.Translation(Vector((0, D / 2 - LT, T)))
                           @ Euler((math.radians(-20), 0, 0)).to_matrix().to_4x4())  # fmt: skip
    base = p.build(bevel=0.0015)
    obj = join("laptop", [base, lid_obj])
    place(obj, x, y, z, yaw)
    bake_transform(obj)
    return obj


def build_mouse(m: dict, x, y, z, yaw):
    p = Parts("mouse")
    p.sphere((0, 0, 0.0), (0.031, 0.057, 0.019), m["mouse"], segs=32)
    obj = p.build()
    bm_cut = obj.data
    for v in bm_cut.vertices:
        v.co.z = max(v.co.z, 0.0)
    place(obj, x, y, z, yaw)
    bake_transform(obj)
    return obj


def build_mug(m: dict, x, y, z, yaw):
    p = Parts("mug")
    r, h = 0.041, 0.095
    p.cyl((0, 0, h / 2), r, r, h, m["ceramic"], segs=40, caps=False)
    p.cyl((0, 0, h / 2 + 0.003), r - 0.005, r - 0.005, h - 0.006, m["ceramic"], segs=40,
          caps=False)  # fmt: skip
    p.cyl((0, 0, 0.004), r, r, 0.008, m["ceramic"], segs=40)
    p.torus((0, 0, h), r - 0.0025, 0.0025, m["ceramic"], rot=(math.pi / 2, 0, 0), n1=40, n2=6)
    p.cyl((0, 0, h - 0.025), r - 0.006, r - 0.006, 0.002, m["coffee"], segs=40)
    p.torus((r - 0.002, 0, h * 0.52), 0.026, 0.006, m["ceramic"], rot=(0, math.pi / 2, 0),
            arc=math.pi, n1=16, n2=8)  # fmt: skip
    obj = p.build()
    place(obj, x, y, z, yaw)
    bake_transform(obj)
    return obj


def build_notebook(m: dict, x, y, z, yaw):
    p = Parts("notebook")
    w, d = 0.148, 0.21
    p.box((-w / 2, -d / 2, 0), (w / 2, d / 2, 0.012), m["pages"])
    p.box((-w / 2 - 0.001, -d / 2 - 0.001, 0.012), (w / 2 + 0.001, d / 2 + 0.001, 0.0145),
          m["notebook"])  # fmt: skip
    p.box((-w / 2 - 0.001, -d / 2 - 0.001, -0.0005), (w / 2 + 0.001, d / 2 + 0.001, 0.001),
          m["notebook"])  # fmt: skip
    p.box((w / 2 - 0.03, -d / 2 - 0.0015, 0), (w / 2 - 0.022, d / 2 + 0.0015, 0.0148),
          m["black_metal"])  # elastic band  # fmt: skip
    # Pen lying on it.
    p.cyl((0.0, 0.0, 0.0195), 0.0045, 0.0045, 0.14, m["pen"], rot=(math.pi / 2, 0, 0.4), segs=16)
    p.cyl((math.sin(0.4) * -0.075, math.cos(0.4) * 0.075, 0.0195), 0.0045, 0.001, 0.012,
          m["steel"], rot=(math.pi / 2, 0, 0.4 + math.pi), segs=16)  # fmt: skip
    obj = p.build(bevel=0.001)
    place(obj, x, y, z, yaw)
    bake_transform(obj)
    return obj


def build_robot(m: dict):
    """Retro tin wind-up robot toy, ~17 cm tall, red with silver details."""
    p = Parts("robot")
    red, sil = m["tin_red"], m["chrome"]
    for sx in (-1, 1):  # feet and legs
        p.box((sx * 0.022 - 0.014, -0.022, 0), (sx * 0.022 + 0.014, 0.02, 0.012), sil)
        p.box((sx * 0.022 - 0.011, -0.012, 0.012), (sx * 0.022 + 0.011, 0.012, 0.055), red)
    p.box((-0.042, -0.026, 0.055), (0.042, 0.026, 0.125), red)  # body
    p.box((-0.03, -0.0275, 0.07), (0.03, -0.026, 0.11), m["tin_panel"])  # chest panel
    for i, c in enumerate((m["tin_yellow"], m["tin_blue"], m["tin_yellow"])):
        p.cyl((-0.016 + i * 0.016, -0.029, 0.098), 0.0055, 0.0055, 0.004, c,
              rot=(math.pi / 2, 0, 0), segs=16)  # fmt: skip
    for k in range(4):
        p.box((-0.022, -0.0285, 0.076 + k * 0.004), (0.022, -0.0275, 0.0775 + k * 0.004), sil)
    for sx in (-1, 1):  # arms
        p.cyl((sx * 0.05, 0, 0.115), 0.01, 0.01, 0.014, sil, rot=(0, math.pi / 2, 0), segs=16)
        p.box((sx * 0.05 - 0.008, -0.009, 0.065), (sx * 0.05 + 0.008, 0.009, 0.115), red,
              rot=(math.radians(-25 * sx), 0, 0), pivot=(sx * 0.05, 0, 0.115))  # fmt: skip
    p.cyl((0, 0, 0.13), 0.012, 0.012, 0.01, sil, segs=16)  # neck
    p.box((-0.03, -0.024, 0.135), (0.03, 0.024, 0.18), red)  # head
    p.box((-0.024, -0.0255, 0.142), (0.024, -0.024, 0.172), m["tin_panel"])
    for sx in (-1, 1):
        p.cyl((sx * 0.012, -0.027, 0.16), 0.0065, 0.0065, 0.004, m["tin_yellow"],
              rot=(math.pi / 2, 0, 0), segs=16)  # fmt: skip
        p.cyl((sx * 0.033, 0, 0.158), 0.007, 0.007, 0.008, sil, rot=(0, math.pi / 2, 0), segs=12)
    p.box((-0.012, -0.0262, 0.146), (0.012, -0.0245, 0.149), sil)  # mouth grille
    p.cyl((0, 0, 0.19), 0.0018, 0.0018, 0.02, sil, segs=8)  # antenna
    p.sphere((0, 0, 0.2), (0.005, 0.005, 0.005), m["tin_yellow"], segs=12)
    p.cyl((0, 0.032, 0.095), 0.004, 0.004, 0.014, sil, rot=(math.pi / 2, 0, 0), segs=8)  # key
    p.box((-0.012, 0.038, 0.09), (0.012, 0.041, 0.1), sil)
    return p.build(bevel=0.0015)


def _figure(name: str, m: dict, suit, trim, skin, helmet: bool):
    """A 15 cm articulated action figure on a round base, built from capsules."""
    p = Parts(name)
    p.cyl((0, 0, 0.004), 0.035, 0.035, 0.008, m["base_black"], segs=32)
    for sx in (-1, 1):  # boots, legs (slight stance)
        p.box((sx * 0.012 - 0.009, -0.014, 0.008), (sx * 0.012 + 0.009, 0.01, 0.02), trim)
        p.cyl((sx * 0.012, -0.002, 0.045), 0.0075, 0.0085, 0.05, suit, rot=(0, sx * 0.06, 0))
        p.sphere((sx * 0.012, -0.002, 0.071), (0.0085, 0.0085, 0.0085), suit, segs=12)
        p.cyl((sx * 0.012, -0.002, 0.09), 0.0085, 0.0095, 0.038, suit)
    p.box((-0.02, -0.011, 0.104), (0.02, 0.011, 0.112), trim)  # belt
    p.cyl((0, 0, 0.128), 0.017, 0.022, 0.034, suit, segs=20)  # torso, chest wider
    p.box((-0.006, -0.0215, 0.122), (0.006, -0.019, 0.136), trim)  # chest emblem
    for sx in (-1, 1):  # shoulders, arms (one raised)
        p.sphere((sx * 0.026, 0, 0.142), (0.009, 0.009, 0.009), suit, segs=12)
        ang = 0.25 if sx < 0 else 2.3  # from hanging down; the right arm is raised
        d = np.array((sx * math.sin(ang), 0.0, -math.cos(ang)))
        sh = np.array((sx * 0.026, 0.0, 0.142))
        p.cyl(sh + 0.025 * d, 0.0065, 0.006, 0.05, suit, rot=(0, -sx * ang, 0), segs=12)
        p.sphere(sh + 0.053 * d, (0.0065, 0.0065, 0.0065), trim if helmet else skin, segs=12)
    p.cyl((0, 0, 0.151), 0.006, 0.006, 0.008, skin, segs=12)  # neck
    if helmet:
        p.sphere((0, 0, 0.166), (0.017, 0.017, 0.018), suit, segs=24)
        p.sphere((0, -0.006, 0.167), (0.0135, 0.013, 0.012), m["visor"], segs=24)
        p.box((-0.016, 0.012, 0.115), (0.016, 0.026, 0.14), suit)  # backpack
    else:
        p.sphere((0, 0, 0.165), (0.0125, 0.0135, 0.015), skin, segs=24)
        p.sphere((0, 0.002, 0.171), (0.0132, 0.0138, 0.0112), m["hair"], segs=24)
        p.box((-0.026, 0.017, 0.035), (0.026, 0.019, 0.15), trim,
              rot=(math.radians(10), 0, 0), pivot=(0, 0.018, 0.15))  # cape  # fmt: skip
    return p.build(bevel=0.0008)


def build_figures(m: dict):
    astro = _figure("astronaut_figure", m, m["suit_white"], m["suit_grey"], m["skin"], True)
    hero = _figure("hero_figure", m, m["suit_blue"], m["suit_red"], m["skin"], False)
    return astro, hero


# ---------------------------------------------------------------- scene


def materials(assets: Path) -> dict:
    a = assets
    return {
        "floor": pbr_mat("floor", a / "wood_floor", 1.7, tint=(0.92, 0.88, 0.82)),
        "wall": pbr_mat(
            "wall", a / "white_stucco", 2.0, tint=WALL_TINT, rough_add=0.15, bump=0.4
        ),  # fmt: skip
        "accent": pbr_mat(
            "accent_wall", a / "white_stucco", 2.0, tint=ACCENT_TINT, rough_add=0.15, bump=0.4
        ),  # fmt: skip
        "ceiling": pbr_mat(
            "ceiling", a / "white_stucco", 2.0, tint=(0.95, 0.95, 0.94), rough_add=0.2, bump=0.3
        ),  # fmt: skip
        "oak": pbr_mat("oak", a / "white_oak_veneer", 0.6, tint=(1.0, 0.93, 0.82)),
        "slats": flat_mat("slats", (0.55, 0.42, 0.28), rough=0.7),
        "headboard": pbr_mat(
            "headboard",
            a / "poly_wool_herringbone",
            0.27,
            sat=0.3,
            tint=(0.55, 0.55, 0.53),
            sheen=0.4,
        ),  # fmt: skip
        "sheet": pbr_mat(
            "sheet", a / "cotton_jersey", 0.26, sat=0.0, tint=(0.95, 0.95, 0.95), sheen=0.3
        ),  # fmt: skip
        "duvet": pbr_mat("duvet", a / "rough_linen", 0.27, uv=True, sat=0.75, sheen=0.4),
        "pillow": pbr_mat(
            "pillow_case", a / "rough_linen", 0.27, sat=0.0, tint=(0.93, 0.93, 0.92), sheen=0.3
        ),  # fmt: skip
        "sweater": pbr_mat(
            "sweater",
            a / "cotton_jersey",
            0.26,
            uv=True,
            sat=0.0,
            tint=(0.42, 0.43, 0.45),
            sheen=0.5,
        ),  # fmt: skip
        "throw": pbr_mat("throw", a / "waffle_pique_cotton", 0.28, uv=True, sheen=0.3),
        "rug": pbr_mat(
            "rug", a / "poly_wool_herringbone", 0.5, sat=0.6, tint=(0.95, 0.92, 0.88), sheen=0.5
        ),  # fmt: skip
        "curtain": pbr_mat(
            "curtain", a / "rough_linen", 0.27, uv=True, sat=0.0, tint=(0.92, 0.88, 0.80), sheen=0.3
        ),  # fmt: skip
        "white_paint": flat_mat("white_paint", (0.86, 0.86, 0.84), rough=0.35, noise=0.03),
        "door": flat_mat("door_paint", (0.85, 0.85, 0.83), rough=0.3, noise=0.03),
        "white_front": flat_mat("drawer_fronts", (0.84, 0.84, 0.82), rough=0.3, noise=0.03),
        "white_plastic": flat_mat("white_plastic", (0.88, 0.88, 0.87), rough=0.25),
        "grey_plastic": flat_mat("grey_plastic", (0.45, 0.45, 0.45), rough=0.4),
        "radiator": flat_mat("radiator_paint", (0.88, 0.88, 0.87), rough=0.2, coat=0.3),
        "grille": flat_mat("grille", (0.35, 0.35, 0.35), rough=0.5),
        "sill": flat_mat("sill", (0.80, 0.79, 0.76), rough=0.25, noise=0.06),
        "glass": glass_mat("glass"),
        "chrome": flat_mat("chrome", (0.9, 0.9, 0.9), rough=0.1, metal=1.0),
        "steel": flat_mat("steel", (0.72, 0.72, 0.72), rough=0.28, metal=1.0),
        "black_metal": flat_mat("black_metal", (0.03, 0.03, 0.03), rough=0.45, metal=0.6),
        "frame": flat_mat("frame_black", (0.02, 0.02, 0.02), rough=0.4),
        "mount": flat_mat("mount", (0.92, 0.91, 0.88), rough=0.8),
        "pages": flat_mat("pages", (0.88, 0.85, 0.76), rough=0.8, noise=0.05),
        "gold": flat_mat("spine_print", (0.80, 0.68, 0.40), rough=0.35, metal=0.6),
        "aluminium": flat_mat("aluminium", (0.68, 0.69, 0.70), rough=0.3, metal=1.0),
        "key_well": flat_mat("key_well", (0.05, 0.05, 0.055), rough=0.5),
        "key": flat_mat("keys", (0.07, 0.07, 0.075), rough=0.55),
        "trackpad": flat_mat("trackpad", (0.62, 0.63, 0.64), rough=0.25, metal=0.8),
        "bezel": flat_mat("bezel", (0.015, 0.015, 0.015), rough=0.2),
        "mouse": flat_mat("mouse", (0.85, 0.85, 0.84), rough=0.3),
        "ceramic": flat_mat("mug_glaze", (0.12, 0.25, 0.36), rough=0.12, coat=0.4),
        "coffee": flat_mat("coffee", (0.05, 0.025, 0.012), rough=0.05),
        "notebook": flat_mat("notebook", (0.12, 0.12, 0.13), rough=0.6),
        "pen": flat_mat("pen", (0.05, 0.12, 0.35), rough=0.3),
        "bin": flat_mat("bin_steel", (0.85, 0.85, 0.84), rough=0.35, metal=0.3),
        "backpack": flat_mat("backpack_fabric", (0.07, 0.13, 0.20), rough=0.8, noise=0.1),
        "backpack_trim": flat_mat("backpack_trim", (0.03, 0.03, 0.035), rough=0.7),
        "tin_red": flat_mat("tin_red", (0.62, 0.05, 0.03), rough=0.25, metal=0.5, coat=0.5),
        "tin_panel": flat_mat("tin_panel", (0.15, 0.15, 0.16), rough=0.35, metal=0.6),
        "tin_yellow": flat_mat("tin_yellow", (0.95, 0.72, 0.08), rough=0.25, coat=0.5),
        "tin_blue": flat_mat("tin_blue", (0.08, 0.3, 0.75), rough=0.25, coat=0.5),
        "base_black": flat_mat("figure_base", (0.02, 0.02, 0.02), rough=0.3),
        "suit_white": flat_mat("suit_white", (0.85, 0.85, 0.83), rough=0.45),
        "suit_grey": flat_mat("suit_grey", (0.35, 0.36, 0.38), rough=0.45),
        "visor": flat_mat("visor", (0.75, 0.5, 0.12), rough=0.05, metal=1.0),
        "suit_blue": flat_mat("suit_blue", (0.04, 0.12, 0.45), rough=0.35),
        "suit_red": flat_mat("suit_red", (0.55, 0.04, 0.04), rough=0.35),
        "skin": flat_mat("figure_skin", (0.80, 0.58, 0.45), rough=0.45),
        "hair": flat_mat("figure_hair", (0.06, 0.04, 0.03), rough=0.4),
    }


def build_bedroom(assets: Path, out: Path, warm_lamps: bool = False) -> tuple[dict, list]:
    """Build the whole room in the open (empty) Blender file. Returns the structure objects
    (floor, wall, ceiling, window, door) and the object records (name, noun, asset,
    support, obj) in GT index order."""
    bpy = _bpy()
    m = materials(assets)
    covers = book_mats(m)
    structure = build_shell(assets, m)
    window, curtains = build_window(m)
    structure["window"] = window
    structure["door"] = build_door(m)

    def asset(name: str, folder: str, x, y, z=0.0, yaw=0.0, scale=1.0):
        obj = import_asset(assets / folder / f"{folder}.gltf", name)
        place(obj, x, y, z, yaw, scale)
        return obj

    rec: list = []

    def add(obj, noun: str, src: str | None, support: str | None = None):
        rec.append((obj.name, noun, src, support, obj))
        return obj

    # Bed with mattress, pillows and the cloth-simulated duvet and throw.
    bed, mattress = build_bed(m)
    nightstand = asset("nightstand", "painted_wooden_nightstand", -HX + 0.27, 1.2, yaw=90)
    p1 = pillow("pillow", (-1.36, 0.20, 0.53 + 0.065), (0.48, 0.72, 0.15), m["pillow"], seed=1)
    apply_modifiers(p1)
    print("cloth: duvet")
    duvet = cloth_sheet("duvet", -0.86, 0.62, -0.62, 1.02, 0.58, 60, 60, m["duvet"],
                        [bed, mattress, p1, nightstand], frames=90, stiffness=2.5, mass=1.1,
                        thickness=0.045, seed=3, pin_x=-0.80, noise=0.006)  # fmt: skip
    print("cloth: throw")
    throw = cloth_sheet("throw", -0.32, 0.20, -0.55, 0.95, 0.70, 30, 80, m["throw"],
                        [bed, mattress, duvet], frames=80, stiffness=1.0, mass=0.4,
                        thickness=0.008, seed=4)  # fmt: skip
    bed_all = join("bed", [bed, mattress])
    add(bed_all, "bed", None)
    add(duvet, "duvet", None, "bed")
    add(throw, "blanket", "waffle_pique_cotton (texture)", "bed")
    add(p1, "pillow", None, "bed")
    add(nightstand, "nightstand", "painted_wooden_nightstand")

    # Desk corner.
    desk = add(build_desk(m), "desk", None)
    dz = top_z(desk)
    chair = add(asset("desk_chair", "painted_wooden_chair_01", 0.86, 1.0, yaw=90 + 8), "chair",
                "painted_wooden_chair_01")  # fmt: skip
    clo, chi = world_bbox(chair)
    print("cloth: sweater")
    back_x = clo[0] + 0.05  # the chair back is its -x side (it faces the desk, +x)
    sweater = cloth_sheet("sweater", back_x - 0.36, back_x + 0.34, 0.80, 1.20, chi[2] + 0.02,
                          42, 24, m["sweater"], [chair], frames=70, stiffness=3.0, mass=0.5,
                          thickness=0.008, seed=5, noise=0.004)  # fmt: skip
    add(sweater, "sweater", "cotton_jersey (texture)", "chair")
    add(build_backpack(m, 1.02, 1.76, yaw=200), "backpack", None)
    shelf = add(build_shelf(m), "shelf", None)
    add(asset("desk_lamp", "desk_lamp_arm_01", 1.62, 1.50, dz, yaw=-120), "desk lamp",
        "desk_lamp_arm_01", "desk")  # fmt: skip
    screen = out / "laptop_screen.png"
    screen_image(screen)
    add(build_laptop(m, screen, 1.47, 1.02, dz, yaw=-90 + 4), "laptop", None, "desk")
    add(build_mouse(m, 1.36, 0.73, dz, yaw=-90 + 10), "mouse", None, "desk")
    add(build_mug(m, 1.50, 0.55, dz, yaw=30), "mug", None, "desk")
    add(build_notebook(m, 1.36, 1.38, dz, yaw=-80), "notebook", None, "desk")
    add(asset("succulent", "potted_plant_04", 1.68, 0.70, dz, yaw=20), "succulent",
        "potted_plant_04", "desk")  # fmt: skip
    sz = top_z(shelf)
    add(asset("encyclopedias", "book_encyclopedia_set_01", HX - 0.11, 0.80, sz, yaw=-90,
              scale=0.7), "books", "book_encyclopedia_set_01", "shelf")  # fmt: skip
    # The model's front is its +x side. Its glass is opaque here (the glTF JPG has no alpha)
    # and hid the photo, so the glass faces go.
    frame = asset("photo_frame", "standing_picture_frame_01", HX - 0.1, 1.18, sz, yaw=195,
                  scale=0.9)  # fmt: skip
    drop_material_faces(frame, "glass")
    add(frame, "picture frame", "standing_picture_frame_01", "shelf")

    # Nightstand top.
    nz = top_z(nightstand)
    add(asset("alarm_clock", "alarm_clock_01", -1.62, 1.06, nz, yaw=110, scale=0.6),
        "alarm clock", "alarm_clock_01", "nightstand")  # fmt: skip
    add(build_book_stack("bedside_books", -1.55, 1.32, nz, 80, covers, m, 3, seed=11), "books",
        None, "nightstand")  # fmt: skip

    # Chest of drawers with the toys, clock above, basket beside.
    dresser = add(build_dresser(m), "chest of drawers", None)
    tz = top_z(dresser)
    robot = add(build_robot(m), "toy robot", None, "chest of drawers")
    place(robot, -0.98, -1.85, tz, yaw=200)  # built facing -y, turned to the room
    astro, hero = build_figures(m)
    add(place(astro, -0.80, -1.80, tz, yaw=170), "action figure", None, "chest of drawers")
    add(place(hero, -0.66, -1.86, tz, yaw=195), "action figure", None, "chest of drawers")
    add(asset("elephant", "carved_wooden_elephant", -0.42, -1.99, tz, yaw=160, scale=1.6),
        "wooden elephant", "carved_wooden_elephant", "chest of drawers")  # fmt: skip
    add(build_book_stack("dresser_books", -0.50, -1.72, tz, 2, covers, m, 2, seed=17), "books",
        None, "chest of drawers")  # fmt: skip
    for o in (robot, astro, hero):
        bake_transform(o)
    clock = asset("wall_clock", "wall_clock", -0.72, -HY, 1.72, yaw=180)
    clock.location.y = -HY + (clock.location.y - world_bbox(clock)[0][1])
    add(clock, "wall clock", "wall_clock")
    add(asset("laundry_basket", "wicker_basket_02", -1.52, -1.86, yaw=10), "laundry basket",
        "wicker_basket_02")  # fmt: skip

    # Plant, rug, ukulele, ceiling lamp, radiator, curtains, electrics, posters.
    add(asset("plant", "potted_plant_02", -1.42, 1.78, yaw=40), "potted plant", "potted_plant_02")
    add(build_rug(m), "rug", "poly_wool_herringbone (texture)")
    add(build_bin(m, 1.52, 0.78), "waste bin", None)
    uke = asset("ukulele", "Ukulele_01", HX - 0.09, -0.75, 0.0, yaw=-90)
    uke.rotation_euler = (math.radians(-12), 0.0, math.radians(-90))
    bpy.context.view_layer.update()
    lo, hi = world_bbox(uke)
    uke.location.z -= lo[2]
    uke.location.x += HX - 0.004 - hi[0]
    add(uke, "ukulele", "Ukulele_01")
    lamp = asset("ceiling_lamp", "modern_ceiling_lamp_01", 0.15, -0.15, 0.0, scale=0.7)
    lamp.location.z = RH - top_z(lamp)
    add(lamp, "ceiling lamp", "modern_ceiling_lamp_01")
    add(build_radiator(m), "radiator", None)
    add(curtains, "curtains", "rough_linen (texture)")
    add(build_electrics(m), "socket", None)
    posters = assets / "posters"
    wave = posters / "poster_great_wave.jpg"
    earth = posters / "poster_earthrise.jpg"
    add(build_poster("poster_wave", wave, (-HX, 0.20, 1.55), "west", 0.56, m,
                     bpy.data.images.load(str(wave)).size), "poster",
        "Great Wave off Kanagawa (Hokusai, public domain)")  # fmt: skip
    add(build_poster("poster_earthrise", earth, (HX, 0.92, 1.30), "east", 0.42, m,
                     bpy.data.images.load(str(earth)).size), "poster",
        "Earthrise (NASA, Apollo 8, public domain)")  # fmt: skip

    for idx, (_, _, _, _, obj) in enumerate(rec, start=1):
        if obj.modifiers:
            apply_modifiers(obj)
        obj.pass_index = idx
    check_overlaps(rec)
    for obj in structure.values():
        apply_modifiers(obj)
    build_lights(assets, warm_lamps, lamp, rec)
    return structure, rec


CLOTH = {"duvet", "throw", "sweater"}  # draped by the solver, they rest on each other


def _low_bbox(obj, slab: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """xy bbox of the object's bottom slab: what can clip into a neighbour on the same top
    (a lamp arm high above the notebook does not)."""
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    m = np.array(obj.matrix_world)
    w = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
    w = w[w[:, 2] < w[:, 2].min() + slab]
    return w.min(0)[:2], w.max(0)[:2]


def check_overlaps(rec: list, tol: float = 0.003) -> None:
    """Fail the build when two rigid items on the same support clip: the xy bboxes of their
    bottom 15 cm overlap by more than tol on both axes."""
    boxes: dict = {}
    for name, _, _, support, obj in rec:
        if support and name not in CLOTH:
            boxes.setdefault(support, []).append((name, *_low_bbox(obj)))
    bad = []
    for items in boxes.values():
        for i, (na, la, ha) in enumerate(items):
            for nb, lb, hb in items[i + 1 :]:
                over = np.minimum(ha, hb) - np.maximum(la, lb)
                if (over > tol).all():
                    bad.append(f"{na} / {nb} ({over[0]:.3f} x {over[1]:.3f} m)")
    if bad:
        raise RuntimeError("items clip: " + "; ".join(bad))


def build_lights(assets: Path, warm_lamps: bool, lamp, rec) -> None:
    """HDRI sky through the window (light portal in the opening) and a soft ceiling fill.
    With warm_lamps, bulbs in the pendant and the desk lamp too."""
    bpy = _bpy()
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    nt = world.node_tree
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(next((assets / "eilenriede_park").glob("*.hdr"))))
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (0, 0, math.radians(200))
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    bg = nt.nodes["Background"]
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 3.0
    bpy.context.scene.world = world

    portal = bpy.data.lights.new("window_portal", "AREA")
    portal.shape = "RECTANGLE"
    portal.size, portal.size_y = WIN_X1 - WIN_X0, WIN_Z1 - WIN_Z0
    portal.cycles.is_portal = True
    obj = bpy.data.objects.new("window_portal", portal)
    obj.location = ((WIN_X0 + WIN_X1) / 2, HY + WIN_DEPTH - 0.01, (WIN_Z0 + WIN_Z1) / 2)
    obj.rotation_euler = (math.radians(-90), 0, 0)  # faces -y, into the room
    bpy.context.collection.objects.link(obj)

    fill = bpy.data.lights.new("ceiling_fill", "AREA")
    fill.shape, fill.size, fill.size_y = "RECTANGLE", RX - 0.4, RY - 0.4
    fill.energy, fill.color = 160.0, (1.0, 0.96, 0.9)
    obj = bpy.data.objects.new("ceiling_fill", fill)
    # Not seen in reflections: the window glass showed it as a pale patch over the view.
    obj.visible_glossy = False
    obj.location = (0.0, 0.0, RH - 0.01)
    bpy.context.collection.objects.link(obj)

    if warm_lamps:
        lo, hi = world_bbox(lamp)
        for name, loc, power in (
            ("pendant_bulb", ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + 0.06), 40.0),
        ):
            light = bpy.data.lights.new(name, "POINT")
            light.energy, light.shadow_soft_size = power, 0.03
            light.color = (1.0, 0.78, 0.55)
            o = bpy.data.objects.new(name, light)
            o.location = loc
            bpy.context.collection.objects.link(o)
        desk_lamp = next(o for n, *_, o in rec if n == "desk_lamp")
        lo, hi = world_bbox(desk_lamp)
        spot = bpy.data.lights.new("desk_lamp_bulb", "SPOT")
        spot.energy, spot.spot_size, spot.shadow_soft_size = 15.0, math.radians(80), 0.02
        spot.color = (1.0, 0.8, 0.6)
        o = bpy.data.objects.new("desk_lamp_bulb", spot)
        o.location = (hi[0] - 0.05, (lo[1] + hi[1]) / 2, hi[2] - 0.08)
        bpy.context.collection.objects.link(o)


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
             "blender_bedroom.py", "support": support, "mesh": f"meshes/{name}.ply",
             "bbox_min": [round(float(v), 4) for v in lo],
             "bbox_max": [round(float(v), 4) for v in hi]}
        )  # fmt: skip
    for key, obj in structure.items():
        _export_mesh(obj, meshes / f"{key}.ply")
        objects.append(
            {"index": STRUCTURE[key], "id": key, "name": key, "noun": key, "kind": "structure",
             "asset": "built in blender_bedroom.py", "support": None, "mesh": f"meshes/{key}.ply"}
        )  # fmt: skip
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
                # blender_room.world_openings format: world x, z (Blender x, -y), heights
                # and the normal into the room
                "openings": [
                    {
                        "kind": "window",
                        "index": STRUCTURE["window"],
                        "wall": "north",
                        "outline_xz": [[WIN_X0, -HY], [WIN_X1, -HY]],
                        "bottom": WIN_Z0,
                        "top": WIN_Z1,
                        "normal_xz": [0.0, 1.0],
                    },
                    {
                        "kind": "door",
                        "index": STRUCTURE["door"],
                        "wall": "south",
                        "outline_xz": [[DOOR_X0, HY], [DOOR_X1, HY]],
                        "bottom": 0.0,
                        "top": DOOR_H,
                        "normal_xz": [0.0, -1.0],
                    },
                ],  # fmt: skip
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
    scene.render.use_persistent_data = True
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
    ap.add_argument("--warm-lamps", action="store_true", help="light the pendant and desk lamp")
    args = ap.parse_args(argv)
    bpy = _bpy()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    gt = out.parent / f"{out.name}_gt"
    blend = out / f"{SCENE}.blend"
    stages = set(args.stage.split(","))
    if "build" in stages:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        structure, rec = build_bedroom(args.assets.resolve(), out, args.warm_lamps)
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
