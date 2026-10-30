"""Synthetic Blender room with ground truth (scene blender01): build, render, export.

    "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b -P scripts/blender_room.py \
        -- --out data/raw/blender01                       # build + render both walks + GT + mp4
    ... -- --out data/raw/blender01 --walk test --frames 0:30 --stage render,gt   # quick check

A synthetic ground-truth scene for benchmarks and a fallback demo; the real phone capture
stays the main demo. Assets are CC0 from Poly Haven (scripts/blender_assets.py, docs/ASSETS.md);
the floor lamp and the room shell are built here.

Outputs (out = data/raw/blender01):
  <out>/blender01.blend, <out>/<walk>/rgb/000000.png ...    working files (noise-free renders)
  <out>.mp4 (main walk, 38 s), <out>_test.mp4 (test walk, 20 s)   H.264 with sensor noise
  <out>_gt/objects.json                 index -> id, noun, asset, support, world bbox
  <out>_gt/meshes/<name>.ply            each object's mesh in world coordinates
  <out>_gt/<walk>/cameras.json          K and cam_to_world per frame (OpenCV camera)
  <out>_gt/<walk>/depth/000000.png      uint16 z-depth in mm (0 = no hit)
  <out>_gt/<walk>/masks/000000.png      uint8 object index (0 none, 101-103 floor, wall, ceiling)
Frame i of each video is GT frame i (time i / 30 s). cameras.json holds the shutter-centre
pose; Cycles writes depth and index from one sample per pixel, at a random time within the
0.3-frame shutter, so GT edges can be off by ~1-2 px and depth by ~2 mm on fast motion.

World frame: metres, +Y up, floor at y = 0 (the scene contract). Blender is +Z up, so
world_from_blender maps (x, y, z) -> (x, z, -y), the same axes glTF uses.

The pure helpers (paths, conventions, PNG/PLY writers) import without bpy so tests can use
them; everything touching bpy is inside functions.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

SCENE_ID = "blender01"
FPS = 30
WIDTH, HEIGHT = 1920, 1080
FOCAL_MM, SENSOR_MM = 26.0, 36.0  # ~69 deg horizontal FOV, a phone main camera
ROOM_X, ROOM_Y, ROOM_H = 4.0, 5.0, 2.6  # Blender x (width), y (length), z (height)
STRUCTURE = {"floor": 101, "wall": 102, "ceiling": 103}

WORLD_FROM_BLENDER = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)
# Blender cameras look down -Z with +Y up; OpenCV cameras look down +Z with +Y down.
BLENDER_CAM_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0])

# name, noun, asset (None = built here), position (Blender x, y), yaw deg, support.
# Poly Haven models face -Y; yaw 0 faces the south wall (y = -2.5).
OBJECTS = [
    ("sofa", "sofa", "sofa_02", (0.0, 2.07), 0, None),  # back to the north wall, never filmed
    ("coffee_table", "coffee table", "modern_coffee_table_01", (0.0, 0.95), 90, None),
    ("armchair", "armchair", "ArmChair_01", (-1.45, 0.4), 90, None),
    ("side_table", "side table", "side_table_01", (1.38, 2.2), 0, None),
    ("floor_lamp", "floor lamp", None, (-1.5, 2.1), 0, None),
    ("cabinet", "cabinet", "painted_wooden_cabinet", (0.4, -2.17), 180, None),  # back to wall
    ("plant", "potted plant", "potted_plant_01", (1.5, -1.95), 30, None),
    ("books", "books", "book_encyclopedia_set_01", (0.35, -2.2), 180, "cabinet"),
    ("vase", "vase", "ceramic_vase_01", (1.38, 2.2), 0, "side_table"),
    ("succulent", "succulent", "potted_plant_04", (-0.3, 0.95), 0, "coffee_table"),
    ("bowl", "bowl", "wooden_bowl_01", (0.3, 0.92), 0, "coffee_table"),
]

WALKS = {"main": 38.0, "test": 20.0}


# ---------------------------------------------------------------- pure helpers (no bpy)


def intrinsics(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """OpenCV K (pixel centres at integer coordinates), horizontal sensor fit."""
    f = FOCAL_MM / SENSOR_MM * width
    return np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]])


def cam_to_world_cv(blender_cam_matrix: np.ndarray) -> np.ndarray:
    """Blender camera matrix_world -> OpenCV cam_to_world in the project world frame."""
    return WORLD_FROM_BLENDER @ np.asarray(blender_cam_matrix) @ BLENDER_CAM_FROM_CV


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation of a Blender camera at eye looking at target, Z-up world, no roll."""
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x = x / np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry),
                              math.cos(rz), math.sin(rz))  # fmt: skip
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def rounded_rect(s: float, x0: float, x1: float, y0: float, y1: float, r: float) -> np.ndarray:
    """Point at arc-length fraction s of a rounded rectangle, counter-clockwise, starting
    at the middle of the bottom (y0) edge."""
    pts = []
    corners = [
        (x1 - r, y0 + r, -90),
        (x1 - r, y1 - r, 0),
        (x0 + r, y1 - r, 90),
        (x0 + r, y0 + r, 180),
    ]
    pts.append(((x0 + x1) / 2, y0))
    for cx, cy, a0 in corners:
        for a in np.linspace(a0, a0 + 90, 16):
            pts.append((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
    pts.append(((x0 + x1) / 2, y0))
    p = np.array(pts)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    d = (s % 1.0) * cum[-1]
    i = min(int(np.searchsorted(cum, d, side="right")) - 1, len(seg) - 1)
    u = (d - cum[i]) / seg[i]
    return p[i] + u * (p[i + 1] - p[i])


def smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class Shake:
    """Handheld motion: sums of sines with fixed random frequencies and phases, plus a
    a 1.8 Hz step surge of 12 mm along the view axis (camera +Z; the blender01 renders were
    made with it there). Positions in metres, angles in radians, in the camera frame."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.pos = [(rng.uniform(0.2, 1.2, 3), rng.uniform(0, 2 * np.pi, 3)) for _ in range(3)]
        self.rot = [(rng.uniform(0.3, 1.5, 3), rng.uniform(0, 2 * np.pi, 3)) for _ in range(3)]
        self.jit = [(rng.uniform(4.0, 7.0, 3), rng.uniform(0, 2 * np.pi, 3)) for _ in range(3)]

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        dpos = sum(np.sin(2 * np.pi * f * t + p) for f, p in self.pos) * 0.004
        dpos = dpos + np.array([0.0, 0.0, 0.012 * math.sin(2 * np.pi * 1.8 * t)])
        drot = sum(np.sin(2 * np.pi * f * t + p) for f, p in self.rot) * math.radians(0.35)
        drot = drot + sum(np.sin(2 * np.pi * f * t + p) for f, p in self.jit) * math.radians(0.05)
        return dpos, drot


def _main_eye_target(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Main walk: 25 s perimeter loop at 1.5 m facing inwards, 2 s crouch, 11 s knee-height
    loop (0.6 m) closer in, also facing inwards."""
    t_loop, t_trans = 25.0, 2.0

    def high(tt: float):
        x, y = rounded_rect(tt / t_loop, -1.3, 1.3, -1.7, 1.6, 0.5)
        wander = np.array([0.5 * math.sin(2 * np.pi * tt / 9), 0.5 * math.sin(2 * np.pi * tt / 13)])
        d = np.array([0.0, 0.3]) + wander - np.array([x, y])
        d = 2.2 * d / np.linalg.norm(d)  # look 2.2 m ahead, down at 0.6 m: ~22 deg pitch
        return np.array([x, y, 1.5]), np.array([x + d[0], y + d[1], 0.6])

    def knee(tt: float):
        x, y = rounded_rect(tt / 11.0, -0.85, 0.85, -1.5, 0.5, 0.35)
        d = np.array([0.0, -0.5]) - np.array([x, y])
        d = d / max(np.linalg.norm(d), 1e-6)
        eye = np.array([x, y, 0.6])
        return eye, eye + np.array([2.5 * d[0], 2.5 * d[1], -0.25])

    if t <= t_loop:
        return high(t)
    if t <= t_loop + t_trans:
        u = smoothstep((t - t_loop) / t_trans)
        (e0, g0), (e1, g1) = high(t), knee(0.0)
        return (1 - u) * e0 + u * e1, (1 - u) * g0 + u * g1
    return knee(t - t_loop - t_trans)


def _test_eye_target(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Test walk: figure eight through the middle at 1.2 m, looking at a point that circles
    the furniture ring once (20 s), so every side is seen from off the main path."""
    w = 2 * np.pi / 20.0
    eye = np.array([0.8 * math.sin(2 * w * t), -0.4 + 1.0 * math.sin(w * t), 1.2])
    eye[2] += 0.05 * math.sin(2 * np.pi * t / 7)
    a = w * t + np.pi / 2
    return eye, np.array([1.5 * math.cos(a), 0.2 + 1.9 * math.sin(a), 0.45])


def camera_matrix(walk: str, t: float) -> np.ndarray:
    """Blender matrix_world (4x4) of the handheld camera at time t of a walk."""
    eye, target = (_main_eye_target if walk == "main" else _test_eye_target)(t)
    dpos, drot = Shake(seed=7 if walk == "main" else 11)(t)
    rot = look_at(eye, target) @ euler_xyz(*drot)
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = eye + rot @ dpos
    return m


def write_png(path: Path, img: np.ndarray) -> None:
    """Write a 2D uint8 or uint16 array as a grayscale PNG (no colour management)."""
    img = np.asarray(img)
    if img.ndim != 2 or img.dtype not in (np.uint8, np.uint16):
        raise ValueError("expected a 2D uint8 or uint16 array")
    h, w = img.shape
    depth = 8 if img.dtype == np.uint8 else 16
    raw = img.astype(">u2") if depth == 16 else img
    rows = np.concatenate([np.zeros((h, 1), np.uint8), raw.view(np.uint8).reshape(h, -1)], axis=1)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", w, h, depth, 0, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
        + chunk(b"IEND", b"")
    )


def write_mesh_ply(path: Path, verts: np.ndarray, tris: np.ndarray) -> None:
    """Binary little-endian PLY with float xyz vertices and triangle faces."""
    verts = np.asarray(verts, dtype="<f4")
    face = np.zeros(len(tris), dtype=[("n", "u1"), ("v", "<i4", (3,))])
    face["n"], face["v"] = 3, tris
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {len(tris)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.encode("ascii") + verts.tobytes() + face.tobytes())


def depth_to_mm(depth_m: np.ndarray) -> np.ndarray:
    """Metres -> uint16 millimetres; no hit or beyond 65.5 m -> 0."""
    d = np.nan_to_num(np.asarray(depth_m, dtype=np.float64), nan=0.0, posinf=0.0)
    mm = np.round(d * 1000.0)
    mm[(mm <= 0) | (mm > 65535)] = 0
    return mm.astype(np.uint16)


def parse_frames(spec: str | None, n: int) -> range:
    if not spec:
        return range(n)
    a, b, step = (spec.split(":") + ["", ""])[:3]
    return range(int(a or 0), min(int(b) if b else n, n), int(step or 1))


# ---------------------------------------------------------------- Blender scene


def _bpy():
    import bpy  # noqa: PLC0415 - only available inside Blender

    return bpy


def setup_render(samples: int, device: str, denoise: bool = True) -> None:
    bpy = _bpy()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    used = "CPU"
    for kind in [device] if device != "AUTO" else ["OPTIX", "CUDA"]:
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
        except TypeError:
            continue
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            used = kind
            break
    scene.cycles.device = "GPU" if used != "CPU" else "CPU"
    print(f"render device: {used}")
    c = scene.cycles
    c.samples = samples
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.03
    # OpenImageDenoise is a learned network, but it runs here, offline, to make test data;
    # no product stage calls it. --samples 128 --no-denoise renders without it.
    c.use_denoising = denoise
    c.denoiser = "OPENIMAGEDENOISE"
    c.denoising_use_gpu = used != "CPU"
    c.max_bounces, c.diffuse_bounces, c.glossy_bounces = 6, 3, 2
    c.transmission_bounces, c.transparent_max_bounces = 2, 8
    c.use_animated_seed = True
    r = scene.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = WIDTH, HEIGHT, 100
    r.fps = FPS
    r.use_motion_blur = True
    r.motion_blur_shutter = 0.3  # 1/100 s at 30 fps
    r.use_persistent_data = True
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0  # fixed exposure for the whole walk
    vl = scene.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_object_index = True


def _new_mat(name: str, color, rough: float = 0.6, metal: float = 0.0, emit: float = 0.0):
    bpy = _bpy()
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if emit:
        bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emit
    return mat


def _pbr_mat(name: str, folder: Path):
    """Principled material from Poly Haven diffuse, roughness and GL normal maps."""
    bpy = _bpy()
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    def tex(pattern: str, non_color: bool):
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(str(next(folder.glob(pattern))))
        if non_color:
            node.image.colorspace_settings.name = "Non-Color"
        return node

    nt.links.new(tex("*_diff_*.jpg", False).outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex("*_rough_*.jpg", True).outputs["Color"], bsdf.inputs["Roughness"])
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(tex("*_nor_gl_*.jpg", True).outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _quad_object(name: str, quads: list, tile: float, mat):
    """Mesh object from quads (4 corner points each); UVs are metres / tile along the two
    edges of each quad, so textures keep their real-world size."""
    bpy = _bpy()
    import bmesh  # noqa: PLC0415

    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new()
    for q in quads:
        q = [np.array(p, dtype=float) for p in q]
        f = bm.faces.new([bm.verts.new(p) for p in q])
        e1, e2 = q[1] - q[0], q[3] - q[0]
        e1n, e2n = e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)
        for loop in f.loops:
            d = np.array(loop.vert.co) - q[0]
            loop[uv].uv = (float(d @ e1n) / tile, float(d @ e2n) / tile)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    me.materials.append(mat)
    return obj


def build_room(assets: Path) -> dict:
    bpy = _bpy()
    hx, hy, h = ROOM_X / 2, ROOM_Y / 2, ROOM_H
    floor = _quad_object(
        "floor",
        [[(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0)]],
        1.7,
        _pbr_mat("floor", assets / "laminate_floor_02"),
    )
    walls = _quad_object(
        "wall",
        [  # corners counter-clockwise seen from inside, so normals face the room
            [(-hx, -hy, 0), (-hx, hy, 0), (-hx, hy, h), (-hx, -hy, h)],
            [(hx, hy, 0), (hx, -hy, 0), (hx, -hy, h), (hx, hy, h)],
            [(hx, -hy, 0), (-hx, -hy, 0), (-hx, -hy, h), (hx, -hy, h)],
            [(-hx, hy, 0), (hx, hy, 0), (hx, hy, h), (-hx, hy, h)],
        ],
        3.0,
        _pbr_mat("wall", assets / "beige_wall_001"),
    )
    ceiling = _quad_object(
        "ceiling",
        [[(-hx, -hy, h), (-hx, hy, h), (hx, hy, h), (hx, -hy, h)]],
        3.0,
        _new_mat("ceiling", (0.8, 0.8, 0.78), rough=0.9),
    )
    for obj, key in ((floor, "floor"), (walls, "wall"), (ceiling, "ceiling")):
        obj.pass_index = STRUCTURE[key]

    # Even light: two soft ceiling panels plus the floor lamp's bulb.
    for i, y in enumerate((1.0, -1.2)):
        light = bpy.data.lights.new(f"ceiling_panel_{i}", "AREA")
        light.size, light.energy, light.color = 1.0, 160.0, (1.0, 0.95, 0.88)
        obj = bpy.data.objects.new(light.name, light)
        obj.location = (0.0, y, h - 0.02)
        bpy.context.collection.objects.link(obj)
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    bpy.context.scene.world = world
    return {"floor": floor, "wall": walls, "ceiling": ceiling}


def build_floor_lamp():
    """Floor lamp (base, pole, fabric shade) as one mesh, plus a bulb light."""
    bpy = _bpy()
    import bmesh  # noqa: PLC0415
    from mathutils import Matrix  # noqa: PLC0415

    me = bpy.data.meshes.new("floor_lamp")
    bm = bmesh.new()
    parts = [  # (r1, r2, depth, centre z, caps, material index)
        (0.16, 0.15, 0.03, 0.015, True, 0),
        (0.012, 0.012, 1.40, 0.73, True, 0),
        (0.23, 0.15, 0.30, 1.50, False, 1),
    ]
    for r1, r2, depth, z, caps, mi in parts:
        res = bmesh.ops.create_cone(
            bm, cap_ends=caps, segments=40, radius1=r1, radius2=r2, depth=depth,
            matrix=Matrix.Translation((0, 0, z)),
        )  # fmt: skip
        vs = set(res["verts"])
        for f in bm.faces:
            if all(v in vs for v in f.verts):
                f.material_index = mi
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("floor_lamp", me)
    bpy.context.collection.objects.link(obj)
    me.materials.append(_new_mat("lamp_metal", (0.05, 0.05, 0.05), rough=0.35, metal=0.9))
    me.materials.append(_new_mat("lamp_shade", (0.85, 0.78, 0.62), rough=0.8, emit=1.5))
    for p in me.polygons:
        p.use_smooth = True
    bulb = bpy.data.lights.new("lamp_bulb", "POINT")
    bulb.energy, bulb.shadow_soft_size, bulb.color = 45.0, 0.05, (1.0, 0.8, 0.55)
    return obj, bulb


def import_asset(path: Path, name: str):
    """Import a glTF asset as one mesh object with its bottom centre at the origin."""
    bpy = _bpy()
    from mathutils import Matrix, Vector  # noqa: PLC0415

    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    # Read every world matrix before un-parenting anything: resetting a parent first would
    # move its children.
    world = {o.name: o.matrix_world.copy() for o in meshes}
    for o in meshes:
        o.parent = None
        o.data = o.data.copy()  # instanced meshes must not be transformed twice
        o.data.transform(world[o.name])
        o.matrix_world = Matrix.Identity(4)
    for o in new:
        if o.type != "MESH":
            bpy.data.objects.remove(o)
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = obj.data.name = name
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lo, hi = co.min(0), co.max(0)
    obj.data.transform(
        Matrix.Translation(Vector((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])))
    )
    return obj


def world_bbox(obj) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bbox of an object in Blender world coordinates."""
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    m = np.array(obj.matrix_world)
    w = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
    return w.min(0), w.max(0)


def build_scene(assets: Path, gt: Path) -> None:
    bpy = _bpy()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    structure = build_room(assets)
    placed: dict = {}
    records = []
    for idx, (name, noun, asset, (x, y), yaw, support) in enumerate(OBJECTS, start=1):
        if asset is None:
            obj, bulb = build_floor_lamp()
        else:
            obj = import_asset(assets / asset / f"{asset}.gltf", name)
        z = world_bbox(placed[support])[1][2] if support else 0.0
        obj.location = (x, y, z)
        obj.rotation_mode = "XYZ"  # the glTF importer leaves objects in quaternion mode
        obj.rotation_euler = (0.0, 0.0, math.radians(yaw))
        obj.pass_index = idx
        bpy.context.view_layer.update()
        if asset is None:
            bobj = bpy.data.objects.new("lamp_bulb", bulb)
            bobj.location = (x, y, 1.47)
            bpy.context.collection.objects.link(bobj)
        placed[name] = obj
        records.append((idx, name, noun, asset, support, obj))

    meshes = gt / "meshes"
    objects = []
    for idx, name, noun, asset, support, obj in records:
        _export_mesh(obj, meshes / f"{name}.ply")
        lo, hi = _world_bbox_project(obj)
        objects.append(
            {
                "index": idx,
                "id": f"{noun.replace(' ', '_')}#{idx}",
                "name": name,
                "noun": noun,
                "kind": "item" if support else "furniture",
                "asset": asset or "built in blender_room.py",
                "support": support,
                "mesh": f"meshes/{name}.ply",
                "bbox_min": [round(float(v), 4) for v in lo],
                "bbox_max": [round(float(v), 4) for v in hi],
            }
        )
    for key, obj in structure.items():
        _export_mesh(obj, meshes / f"{key}.ply")
        objects.append(
            {"index": STRUCTURE[key], "id": key, "name": key, "noun": key, "kind": "structure",
             "asset": "built in blender_room.py", "support": None, "mesh": f"meshes/{key}.ply"}
        )  # fmt: skip
    (gt / "objects.json").write_text(
        json.dumps(
            {
                "scene": SCENE_ID,
                "synthetic": True,
                "units": "m",
                "world": "+Y up, floor at y = 0 (scene contract)",
                "world_from_blender": WORLD_FROM_BLENDER.tolist(),
                "background_index": 0,
                "room_size_m": [ROOM_X, ROOM_H, ROOM_Y],
                "objects": objects,
            },
            indent=2,
        )
    )


def _project_verts(obj) -> tuple[np.ndarray, np.ndarray]:
    me = obj.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    m = WORLD_FROM_BLENDER @ np.array(obj.matrix_world)
    verts = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
    me.calc_loop_triangles()
    tris = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", tris)
    return verts, tris.reshape(-1, 3)


def _world_bbox_project(obj) -> tuple[np.ndarray, np.ndarray]:
    v, _ = _project_verts(obj)
    return v.min(0), v.max(0)


def _export_mesh(obj, path: Path) -> None:
    verts, tris = _project_verts(obj)
    write_mesh_ply(path, verts, tris)


def animate_camera(walk: str, n: int):
    bpy = _bpy()
    from mathutils import Matrix  # noqa: PLC0415

    cam_data = bpy.data.cameras.new("phone")
    cam_data.lens, cam_data.sensor_width, cam_data.sensor_fit = FOCAL_MM, SENSOR_MM, "HORIZONTAL"
    cam_data.clip_start, cam_data.clip_end = 0.05, 50.0
    cam = bpy.data.objects.new("phone", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.rotation_mode = "QUATERNION"
    prev = None
    for i in range(n):
        cam.matrix_world = Matrix(camera_matrix(walk, i / FPS).tolist())
        q = cam.rotation_quaternion.copy()
        if prev is not None and q.dot(prev) < 0:
            q.negate()
            cam.rotation_quaternion = q
        prev = q
        cam.keyframe_insert("location", frame=i)
        cam.keyframe_insert("rotation_quaternion", frame=i)
    return cam


def setup_compositor(raw: Path) -> None:
    bpy = _bpy()
    scene = bpy.context.scene
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Image"], comp.inputs["Image"])
    out = nt.nodes.new("CompositorNodeOutputFile")
    out.base_path = str(raw)
    out.format.file_format = "OPEN_EXR"
    out.format.color_depth = "32"
    out.format.color_mode = "BW"
    out.format.exr_codec = "ZIP"
    out.file_slots.clear()
    out.file_slots.new("depth/######")
    out.file_slots.new("index/######")
    nt.links.new(rl.outputs["Depth"], out.inputs[0])
    nt.links.new(rl.outputs["IndexOB"], out.inputs[1])


def render_walk(out: Path, gt: Path, walk: str, frames: range, force: bool) -> float:
    bpy = _bpy()
    scene = bpy.context.scene
    n = int(round(WALKS[walk] * FPS))
    animate_camera(walk, n)
    rgb, raw = out / walk / "rgb", out / walk / "raw"
    setup_compositor(raw)
    scene.frame_start, scene.frame_end, scene.frame_step = (
        frames.start,
        frames.stop - 1,
        frames.step,
    )
    scene.render.filepath = str(rgb / "######")
    scene.render.use_overwrite = force
    scene.render.use_placeholder = not force
    t0 = time.time()
    bpy.ops.render.render(animation=True)
    seconds = time.time() - t0
    K = intrinsics()
    cams = {
        "scene": SCENE_ID,
        "walk": walk,
        "synthetic": True,
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "K": K.tolist(),
        "convention": "cam_to_world, OpenCV camera (+X right, +Y down, +Z forward), "
        "world +Y up with floor at y = 0, metres; K has pixel centres at integers "
        "(add 0.5 to cx, cy for COLMAP)",
        "world_from_blender": WORLD_FROM_BLENDER.tolist(),
        "frames": [
            {
                "index": i,
                "time_s": round(i / FPS, 6),
                "cam_to_world": cam_to_world_cv(camera_matrix(walk, i / FPS)).round(7).tolist(),
            }
            for i in range(n)
        ],  # fmt: skip
    }
    (gt / walk).mkdir(parents=True, exist_ok=True)
    (gt / walk / "cameras.json").write_text(json.dumps(cams, indent=1))
    log = out / walk / "render_log.json"
    entries = json.loads(log.read_text()) if log.exists() else []
    entries.append({"frames": [frames.start, frames.stop], "seconds": round(seconds, 1),
                    "samples": scene.cycles.samples, "device": scene.cycles.device})  # fmt: skip
    log.write_text(json.dumps(entries, indent=2))
    print(f"{walk}: rendered frames {frames.start}..{frames.stop - 1} in {seconds:.0f} s")
    return seconds


def _read_exr(path: Path) -> np.ndarray:
    bpy = _bpy()
    img = bpy.data.images.load(str(path), check_existing=False)
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(h, w, 4)[::-1, :, 0]  # Blender images are stored bottom-up


def convert_gt(out: Path, gt: Path, walk: str, keep_exr: bool) -> int:
    raw = out / walk / "raw"
    done = 0
    for exr in sorted((raw / "depth").glob("*.exr")):
        i = int(exr.stem)
        write_png(gt / walk / "depth" / f"{i:06d}.png", depth_to_mm(_read_exr(exr)))
        idx = np.round(_read_exr(raw / "index" / exr.name)).clip(0, 255).astype(np.uint8)
        write_png(gt / walk / "masks" / f"{i:06d}.png", idx)
        if not keep_exr:
            exr.unlink()
            (raw / "index" / exr.name).unlink()
        done += 1
    print(f"{walk}: converted {done} GT frames")
    return done


def encode(out: Path, walk: str) -> Path:
    mp4 = out.parent / (out.name + ("" if walk == "main" else "_test") + ".mp4")
    noise = "noise=c0s=6:c0f=t+u:c1s=3:c1f=t+u:c2s=3:c2f=t+u"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS),
         "-i", str(out / walk / "rgb" / "%06d.png"),
         "-vf", f"format=yuv420p,{noise}", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)],
        check=True,
    )  # fmt: skip
    print(f"{walk}: wrote {mp4}")
    return mp4


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--out", type=Path, required=True, help="working folder, e.g. data/raw/blender01"
    )
    ap.add_argument("--assets", type=Path, default=Path("data/assets"))
    ap.add_argument("--walk", default="main,test", help="main, test or both (comma separated)")
    ap.add_argument("--stage", default="build,render,gt,encode")
    ap.add_argument("--frames", default=None, help="frame range a:b[:step] (default: whole walk)")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--device", default="AUTO", choices=["AUTO", "OPTIX", "CUDA", "CPU"])
    ap.add_argument("--no-denoise", action="store_true", help="skip OpenImageDenoise")
    ap.add_argument("--keep-exr", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-render frames that exist")
    args = ap.parse_args(argv)
    out = args.out.resolve()
    gt = out.parent / (out.name + "_gt")
    stages = set(args.stage.split(","))
    walks = [w for w in args.walk.split(",") if w]
    out.mkdir(parents=True, exist_ok=True)
    gt.mkdir(parents=True, exist_ok=True)

    if stages & {"build", "render"}:
        build_scene(args.assets.resolve(), gt)
        setup_render(args.samples, args.device, not args.no_denoise)
        _bpy().ops.wm.save_as_mainfile(filepath=str(out / f"{SCENE_ID}.blend"))
    for walk in walks:
        if "render" in stages:
            # A fresh camera per walk; drop the previous walk's camera.
            bpy = _bpy()
            for o in [o for o in bpy.data.objects if o.type == "CAMERA"]:
                bpy.data.objects.remove(o)
            render_walk(
                out, gt, walk, parse_frames(args.frames, int(WALKS[walk] * FPS)), args.force
            )
        if "gt" in stages:
            convert_gt(out, gt, walk, args.keep_exr)
        if "encode" in stages:
            encode(out, walk)
    return 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    sys.exit(main(argv))
