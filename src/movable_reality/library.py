"""Library of premade meshes that can be added to any scene, by the viewer or by the agent.

scripts/build_library.py packs the curated CC0 models below (Poly Haven, downloaded into
data/assets by scripts/blender_assets.py) into data/library/: one .glb per item, with its
textures embedded and reduced, standing on y = 0 and centred on x = z = 0, plus catalog.json
with the measured sizes. Nothing here is committed. A library object in a scene has no splats:
it is a SceneObject with source "library" and library_id, edited like every other object.
"""

from __future__ import annotations

import io
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel

from .config import LIBRARY_DIR
from .scene import Scene, SceneObject, Vec3

CATALOG = "catalog.json"


@dataclass(frozen=True)
class Curated:
    asset: str  # folder under data/assets, also the item's id
    label: str
    category: str
    on_surface: bool = False  # small things stand on a table or shelf rather than the floor


CURATED: tuple[Curated, ...] = (
    Curated("sofa_02", "sofa", "Seating"),
    Curated("sofa_03", "large sofa", "Seating"),
    Curated("ArmChair_01", "armchair", "Seating"),
    Curated("modern_arm_chair_01", "modern armchair", "Seating"),
    Curated("dining_chair_02", "dining chair", "Seating"),
    Curated("painted_wooden_chair_01", "painted chair", "Seating"),
    Curated("Ottoman_01", "ottoman", "Seating"),
    Curated("modern_coffee_table_01", "coffee table", "Tables"),
    Curated("coffee_table_round_01", "round coffee table", "Tables"),
    Curated("side_table_01", "side table", "Tables"),
    Curated("wooden_table_02", "wooden table", "Tables"),
    Curated("painted_wooden_cabinet", "cabinet", "Storage and beds"),
    Curated("modern_wooden_cabinet", "modern cabinet", "Storage and beds"),
    Curated("painted_wooden_nightstand", "nightstand", "Storage and beds"),
    Curated("wooden_bookshelf_worn", "bookshelf", "Storage and beds"),
    Curated("vintage_day_bed", "day bed", "Storage and beds"),
    Curated("potted_plant_01", "tall potted plant", "Plants"),
    Curated("potted_plant_02", "potted plant", "Plants"),
    Curated("potted_plant_04", "succulent", "Plants", on_surface=True),
    Curated("wicker_basket_02", "wicker basket", "Decor"),
    Curated("desk_lamp_arm_01", "desk lamp", "Decor", on_surface=True),
    Curated("ceramic_vase_01", "ceramic vase", "Decor", on_surface=True),
    Curated("brass_vase_01", "brass vase", "Decor", on_surface=True),
    Curated("wooden_bowl_01", "wooden bowl", "Decor", on_surface=True),
    Curated("book_encyclopedia_set_01", "books", "Decor", on_surface=True),
    Curated("throw_pillows_01", "throw pillows", "Decor", on_surface=True),
    Curated("mantel_clock_01", "mantel clock", "Decor", on_surface=True),
    Curated("horse_statue_01", "horse statue", "Decor", on_surface=True),
    Curated("carved_wooden_elephant", "wooden elephant", "Decor", on_surface=True),
)


class LibraryItem(BaseModel):
    id: str
    label: str
    category: str
    on_surface: bool = False
    size: Vec3  # width x, height y, depth z, metres
    triangles: int = 0
    glb: str  # relative to the library folder
    thumb: str | None = None  # PNG, relative to the library folder
    author: str = ""
    source_url: str = ""
    license: str = ""


def catalog(folder: Path | None = None) -> dict[str, LibraryItem]:
    """The built catalogue by id; empty when scripts/build_library.py has not run here."""
    path = (folder or LIBRARY_DIR) / CATALOG
    if not path.exists():
        return {}
    items = (LibraryItem.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8")))
    return {item.id: item for item in items}


def get_item(library_id: str, folder: Path | None = None) -> LibraryItem:
    items = catalog(folder)
    if library_id not in items:
        known = ", ".join(items) or "none"
        raise KeyError(f"No library item '{library_id}'. Known items: {known}")
    return items[library_id]


def inside(v: float, span: tuple[float, float], half: float) -> float:
    """v moved so that v +- half stays inside the span (centred when it cannot)."""
    lo, hi = span[0] + half, span[1] - half
    return (span[0] + span[1]) / 2 if lo > hi else min(max(v, lo), hi)


def add_to_scene(
    scene: Scene,
    item: LibraryItem,
    x: float | None = None,
    z: float | None = None,
    yaw_deg: float = 0.0,
    on_object_id: str | None = None,
    instruction: str | None = None,
) -> SceneObject:
    """Stand `item` on the floor at (x, z), or on top of another object (at its centre unless
    x and z are given, and always over its top). The caller saves the scene."""
    base_y = 0.0
    if on_object_id:
        parent = scene.get(on_object_id)
        if not parent.visible:
            raise ValueError(f"'{on_object_id}' has been removed")
        mn, mx = parent.world_aabb()
        base_y = mx[1]
        yaw = math.radians(yaw_deg)
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        w, d = item.size[0], item.size[2]
        x = (mn[0] + mx[0]) / 2 if x is None else x
        z = (mn[2] + mx[2]) / 2 if z is None else z
        x = inside(x, (mn[0], mx[0]), (c * w + s * d) / 2)
        z = inside(z, (mn[2], mx[2]), (s * w + c * d) / 2)
    if x is None or z is None:
        raise ValueError("give x and z, or on_object_id")
    return scene.add_library_object(
        item.id, item.label, item.size, float(x), float(z), float(yaw_deg), base_y, instruction
    )


# ---------------------------------------------------------------- packing


def _node_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T  # glTF: column-major
    qx, qy, qz, qw = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    rot = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )
    out = np.eye(4)
    out[:3, :3] = rot * np.array(node.get("scale", (1.0, 1.0, 1.0)))
    out[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return out


def scene_bounds(gltf: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """World box of the default scene from the POSITION accessors' min and max (glTF requires
    them), through the node hierarchy. Rotated nodes give a slightly loose box."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)

    def visit(index: int, parent: np.ndarray) -> None:
        nonlocal lo, hi
        node = gltf["nodes"][index]
        world = parent @ _node_matrix(node)
        if "mesh" in node:
            for prim in gltf["meshes"][node["mesh"]]["primitives"]:
                acc = gltf["accessors"][prim["attributes"]["POSITION"]]
                amin, amax = acc["min"], acc["max"]
                corners = np.array(
                    [[a, b, c, 1.0] for a in (amin[0], amax[0]) for b in (amin[1], amax[1])
                     for c in (amin[2], amax[2])]  # fmt: skip
                )
                p = (corners @ world.T)[:, :3]
                lo, hi = np.minimum(lo, p.min(axis=0)), np.maximum(hi, p.max(axis=0))
        for child in node.get("children", []):
            visit(child, world)

    for root in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        visit(root, np.eye(4))
    if not np.all(np.isfinite(lo)):
        raise ValueError("the scene has no mesh")
    return lo, hi


def triangle_count(gltf: dict[str, Any]) -> int:
    total = 0
    for mesh in gltf.get("meshes", []):
        for prim in mesh["primitives"]:
            key = prim["indices"] if "indices" in prim else prim["attributes"]["POSITION"]
            total += gltf["accessors"][key]["count"] // 3
    return total


def _colour_images(gltf: dict[str, Any]) -> set[int]:
    """Images used as base colour: they keep more resolution than normal and roughness maps."""
    out = set()
    for mat in gltf.get("materials", []):
        ref = mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if ref:
            out.add(gltf["textures"][ref["index"]]["source"])
    return out


def _encode_image(path: Path, max_px: int) -> tuple[bytes, str]:
    with Image.open(path) as im:
        im.load()
        im.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
        has_alpha = im.mode in ("RGBA", "LA") and im.getchannel("A").getextrema()[0] < 255
        buf = io.BytesIO()
        if has_alpha:
            im.convert("RGBA").save(buf, "PNG", optimize=True)
            return buf.getvalue(), "image/png"
        im.convert("RGB").save(buf, "JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"


def pack_model(
    src: Path, out: Path, colour_px: int = 1024, other_px: int = 512
) -> tuple[Vec3, int]:
    """One .gltf with external files -> one .glb standing on y = 0, centred on x = z = 0. Returns
    its size (x, y, z, metres) and triangle count."""
    gltf = json.loads(src.read_text(encoding="utf-8"))
    blob = bytearray()

    def append(data: bytes) -> int:
        blob.extend(b"\0" * (-len(blob) % 8))  # keeps every accessor's alignment
        start = len(blob)
        blob.extend(data)
        return start

    starts = [append((src.parent / b["uri"]).read_bytes()) for b in gltf["buffers"]]
    for view in gltf.get("bufferViews", []):
        view["byteOffset"] = view.get("byteOffset", 0) + starts[view.get("buffer", 0)]
        view["buffer"] = 0

    colour = _colour_images(gltf)
    views = gltf.setdefault("bufferViews", [])
    for i, image in enumerate(gltf.get("images", [])):
        max_px = colour_px if i in colour else other_px
        data, mime = _encode_image(src.parent / image["uri"], max_px)
        views.append({"buffer": 0, "byteOffset": append(data), "byteLength": len(data)})
        gltf["images"][i] = {"bufferView": len(views) - 1, "mimeType": mime}

    lo, hi = scene_bounds(gltf)
    scene = gltf["scenes"][gltf.get("scene", 0)]
    gltf["nodes"].append(
        {
            "name": "library_root",
            "children": scene["nodes"],
            "translation": [-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2],
        }
    )
    scene["nodes"] = [len(gltf["nodes"]) - 1]
    blob.extend(b"\0" * (-len(blob) % 4))
    gltf["buffers"] = [{"byteLength": len(blob)}]

    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    total = 12 + 8 + len(text) + 8 + len(blob)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(text), b"JSON") + text)
        f.write(struct.pack("<I4s", len(blob), b"BIN\0") + bytes(blob))
    size = hi - lo
    if not all(math.isfinite(float(v)) and v > 0 for v in size):
        raise ValueError(f"{src}: degenerate size {size}")
    return (float(size[0]), float(size[1]), float(size[2])), triangle_count(gltf)


def read_glb_json(path: Path) -> dict[str, Any]:
    """The JSON chunk of a .glb."""
    with path.open("rb") as f:
        f.seek(12)
        length, _ = struct.unpack("<I4s", f.read(8))
        return json.loads(f.read(length))
