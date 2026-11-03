"""Blender script: one transparent PNG thumbnail per library .glb. scripts/build_library.py runs it.

    blender -b -P scripts/library_thumbs.py -- --dir data/library --ids sofa_02 ArmChair_01
"""

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SIZE_PX = 256
VIEW = Vector((0.65, -1.0, 0.55)).normalized()  # three-quarter view of the +Z (glTF) front


def setup(engine: str) -> None:
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = scene.render.resolution_y = SIZE_PX
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    if engine == "CYCLES":
        scene.cycles.samples = 32
    world = bpy.data.worlds.new("thumb")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.9
    scene.world = world


def clear() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def bounds() -> tuple[Vector, float]:
    pts = [
        o.matrix_world @ Vector(c)
        for o in bpy.context.scene.objects
        if o.type == "MESH"
        for c in o.bound_box
    ]
    lo = Vector(min(p[i] for p in pts) for i in range(3))
    hi = Vector(max(p[i] for p in pts) for i in range(3))
    return (lo + hi) / 2, (hi - lo).length / 2


def render(glb: Path, out: Path) -> None:
    clear()
    bpy.ops.import_scene.gltf(filepath=str(glb))
    bpy.context.view_layer.update()
    centre, radius = bounds()
    cam_data = bpy.data.cameras.new("thumb")
    cam_data.lens = 70
    cam = bpy.data.objects.new("thumb", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    half_fov = cam_data.angle / 2
    cam.location = centre + VIEW * (radius * 1.05 / math.sin(half_fov))
    cam.rotation_euler = (-VIEW).to_track_quat("-Z", "Y").to_euler()
    cam_data.clip_end = 1000
    bpy.context.scene.camera = cam
    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy = 2.5
    sun = bpy.data.objects.new("sun", sun_data)
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    bpy.context.scene.collection.objects.link(sun)
    bpy.context.scene.render.filepath = str(out)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--engine", default="BLENDER_EEVEE_NEXT")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    folder = Path(args.dir)
    setup(args.engine)
    for name in args.ids:
        render(folder / f"{name}.glb", folder / "thumbs" / f"{name}.png")
        print(f"thumb {name}", file=sys.stderr)


main()
