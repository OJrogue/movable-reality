"""Agent tools: pure functions on a Scene (ARCHITECTURE stage 7).

Each tool takes the scene plus JSON arguments and returns a JSON-able result. Edits go through
Scene.apply_edit, the same path the viewer UI and scripts/edit_scene.py use. Tools never touch
PLY files. Sizes and positions are world metres, +Y up, floor at y = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..scene import Scene, SceneObject

EDIT_TOOLS = {
    "move_object",
    "place_object",
    "rotate_object",
    "remove_object",
    "restore_object",
    "undo",
    "insert_object",
}


@dataclass
class RoomBounds:
    """Floor rectangle of the room in world x and z, from the background splats."""

    x: tuple[float, float]
    z: tuple[float, float]
    height: float

    def as_dict(self) -> dict[str, Any]:
        return {"x": list(self.x), "z": list(self.z), "height": self.height}


def room_bounds_from_points(points: np.ndarray, pct: float = 1.0) -> RoomBounds:
    """Robust room box from world-frame points: percentiles drop floaters outside the walls."""
    lo = np.percentile(points, pct, axis=0)
    hi = np.percentile(points, 100 - pct, axis=0)
    return RoomBounds(
        (round(float(lo[0]), 3), round(float(hi[0]), 3)),
        (round(float(lo[2]), 3), round(float(hi[2]), 3)),
        round(float(hi[1]), 3),
    )


def room_bounds(scene: Scene, folder: Path) -> RoomBounds | None:
    """Room box from the background PLY (world frame), or from the object boxes as a fallback."""
    ply = folder / (scene.plane_bake.source_background_ply if scene.plane_bake else "")
    if not scene.plane_bake:
        ply = folder / scene.background_ply
    if ply.exists():
        from plyfile import PlyData  # noqa: PLC0415 - only needed when a PLY exists

        v = PlyData.read(str(ply), mmap=False)["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        if len(xyz) >= 100:
            M = np.asarray(scene.world_from_recon, dtype=np.float64)
            world = xyz @ M[:3, :3].T + M[:3, 3]
            return room_bounds_from_points(world)
    if not scene.objects:
        return None
    mins = np.array([o.bbox_min for o in scene.objects])
    maxs = np.array([o.bbox_max for o in scene.objects])
    lo, hi = mins.min(axis=0) - 0.5, maxs.max(axis=0) + 0.5
    return RoomBounds((float(lo[0]), float(hi[0])), (float(lo[2]), float(hi[2])), float(hi[1]))


def _r(v: float) -> float:
    return round(float(v), 3)


def describe(obj: SceneObject) -> dict[str, Any]:
    mn, mx = obj.world_aabb()
    return {
        "id": obj.id,
        "label": obj.label,
        "visible": obj.visible,
        "source": obj.source,
        "center": [_r((a + b) / 2) for a, b in zip(mn, mx, strict=True)],
        "size_m": [_r(v) for v in obj.size],  # width x, height y, depth z, before yaw
        "yaw_deg": _r(obj.transform.yaw_deg),
        "footprint": {"x": [_r(mn[0]), _r(mx[0])], "z": [_r(mn[2]), _r(mx[2])]},
    }


def footprint_gap(a: tuple, b: tuple) -> float:
    """Shortest floor distance between two world AABBs (0 when they overlap in x and z)."""
    (amin, amax), (bmin, bmax) = a, b
    dx = max(bmin[0] - amax[0], amin[0] - bmax[0], 0.0)
    dz = max(bmin[2] - amax[2], amin[2] - bmax[2], 0.0)
    return math.hypot(dx, dz)


def fits(
    scene: Scene,
    object_id: str,
    x: float | None,
    z: float | None,
    bounds: RoomBounds | None,
    clearance: float = 0.05,
    yaw_deg: float | None = None,
) -> dict[str, Any]:
    """Would the object, with its centre at (x, z), stay inside the room and clear of the
    other visible objects? Items standing on the object move with it, so a collision with an
    object stacked on top is not counted (its footprint is inside)."""
    obj = scene.get(object_id).model_copy(deep=True)
    if x is not None and z is not None:
        cx, _, cz = obj.center
        obj.transform.translation = (x - cx, obj.transform.translation[1], z - cz)
    if yaw_deg is not None:
        obj.transform.yaw_deg = yaw_deg
    box = obj.world_aabb()
    (mn, mx) = box
    collisions = []
    for other in scene.objects:
        if other.id == object_id or not other.visible:
            continue
        omn, omx = other.world_aabb()
        overlap_y = omn[1] < mx[1] - clearance and mn[1] < omx[1] - clearance
        if not overlap_y:
            continue  # stacked on top or below, e.g. a vase on the table
        gap = footprint_gap(box, (omn, omx))
        inside_x = mn[0] < omx[0] + clearance and omn[0] < mx[0] + clearance
        inside_z = mn[2] < omx[2] + clearance and omn[2] < mx[2] + clearance
        if inside_x and inside_z:
            collisions.append({"id": other.id, "gap_m": _r(gap)})
    walls = []
    if bounds:
        for axis, i, (lo, hi) in (("x", 0, bounds.x), ("z", 2, bounds.z)):
            if mn[i] < lo - clearance:
                walls.append(f"{_r(lo - mn[i])} m past the {axis}-min wall")
            if mx[i] > hi + clearance:
                walls.append(f"{_r(mx[i] - hi)} m past the {axis}-max wall")
    return {
        "fits": not collisions and not walls,
        "collides_with": collisions,
        "outside_room": walls,
        "footprint": {"x": [_r(mn[0]), _r(mx[0])], "z": [_r(mn[2]), _r(mx[2])]},
    }


def measure(scene: Scene, id_a: str, id_b: str | None = None) -> dict[str, Any]:
    a = scene.get(id_a)
    out: dict[str, Any] = {"a": describe(a)}
    if id_b:
        b = scene.get(id_b)
        out["b"] = describe(b)
        out["floor_gap_m"] = _r(footprint_gap(a.world_aabb(), b.world_aabb()))
        ca, cb = np.array(describe(a)["center"]), np.array(describe(b)["center"])
        out["centre_distance_m"] = _r(float(np.linalg.norm((ca - cb)[[0, 2]])))
    return out


# ---------------------------------------------------------------- OpenAI tool schema


def _fn(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


_ID = {"type": "string", "description": "object id from list_objects, e.g. 'armchair#2'"}
_NUM = {"type": "number"}

TOOL_SCHEMAS = [
    _fn("list_objects", "List every object with id, label, centre, size and visibility.", {}, []),
    _fn(
        "move_object",
        "Move an object by a world offset in metres. dy is usually 0 (objects stay on the floor).",
        {"object_id": _ID, "dx": _NUM, "dy": _NUM, "dz": _NUM},
        ["object_id", "dx", "dz"],
    ),
    _fn(
        "place_object",
        "Put an object's centre at world floor position (x, z) in metres.",
        {"object_id": _ID, "x": _NUM, "z": _NUM},
        ["object_id", "x", "z"],
    ),
    _fn(
        "rotate_object",
        "Turn an object about the vertical axis by yaw_deg (positive = counter-clockwise seen "
        "from above).",
        {"object_id": _ID, "yaw_deg": _NUM},
        ["object_id", "yaw_deg"],
    ),
    _fn("remove_object", "Hide an object (undoable).", {"object_id": _ID}, ["object_id"]),
    _fn("restore_object", "Show a hidden object again.", {"object_id": _ID}, ["object_id"]),
    _fn("undo", "Revert the most recent edit.", {}, []),
    _fn(
        "measure",
        "Size of one object, or the floor gap and centre distance between two objects, metres.",
        {"id_a": _ID, "id_b": _ID},
        ["id_a"],
    ),
    _fn(
        "fits",
        "Check whether an object would fit with its centre at (x, z): no collision with other "
        "objects and inside the room walls. Omit x and z to check the current position.",
        {"object_id": _ID, "x": _NUM, "z": _NUM, "yaw_deg": _NUM},
        ["object_id"],
    ),
    _fn(
        "insert_object",
        "Generate a new piece of furniture from a short description and place it at (x, z). "
        "Slow (about a minute) and costs money; use only when the user asks to add something.",
        {"prompt": {"type": "string"}, "x": _NUM, "z": _NUM},
        ["prompt", "x", "z"],
    ),
]


def run_tool(
    scene: Scene,
    name: str,
    args: dict[str, Any],
    bounds: RoomBounds | None,
    instruction: str | None = None,
    inserter=None,
) -> dict[str, Any]:
    """Execute one tool call. Errors come back as {"error": ...} so the model can recover."""
    try:
        if name == "list_objects":
            return {
                "objects": [describe(o) for o in scene.objects],
                "room": bounds.as_dict() if bounds else None,
            }
        if name == "measure":
            return measure(scene, args["id_a"], args.get("id_b"))
        if name == "fits":
            return fits(
                scene, args["object_id"], args.get("x"), args.get("z"), bounds,
                yaw_deg=args.get("yaw_deg"),
            )  # fmt: skip
        if name == "insert_object":
            if inserter is None:
                return {"error": "insert is not available on this server"}
            return inserter(scene, args["prompt"], float(args["x"]), float(args["z"]), instruction)
        op = {
            "move_object": "move",
            "place_object": "place",
            "rotate_object": "rotate",
            "remove_object": "remove",
            "restore_object": "restore",
            "undo": "undo",
        }.get(name)
        if op is None:
            return {"error": f"unknown tool {name}"}
        oid = args.get("object_id")
        edit_args = {k: float(v) for k, v in args.items() if k != "object_id"}
        scene.apply_edit(op, oid, edit_args, instruction)
        result: dict[str, Any] = {"ok": True}
        if oid:
            result["object"] = describe(scene.get(oid))
            if op in ("move", "place", "rotate"):
                check = fits(scene, oid, None, None, bounds)
                if not check["fits"]:
                    result["warning"] = check
        return result
    except (KeyError, ValueError, TypeError) as exc:
        return {"error": str(exc).strip("'\"")}
