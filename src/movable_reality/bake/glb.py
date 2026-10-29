"""Minimal binary glTF 2.0 (.glb) writer for one textured quad, plus a reader for tests.

Written by hand (spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) because
one quad needs about 60 lines and no new dependency. The material is unlit
(KHR_materials_unlit) and opaque, so three.js's GLTFLoader builds a MeshBasicMaterial and
the texture shows exactly the baked colours, independent of scene lighting.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

GLB_MAGIC = 0x46546C67  # "glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
FLOAT, USHORT = 5126, 5123
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
LINEAR, LINEAR_MIPMAP_LINEAR, CLAMP_TO_EDGE = 9729, 9987, 33071


def _pad(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def quad_glb(
    corners: np.ndarray,
    image: bytes,
    mime: str,
    *,
    name: str = "plane",
    extras: dict | None = None,
) -> bytes:
    """GLB with one quad. corners: (4, 3) in order c0, c0+u, c0+u+v, c0+v; c0 gets texture
    coordinate (0, 0), the first texel of the image's first row. With u x v pointing out
    of the room, triangles (0, 2, 1) and (0, 3, 2) are counter-clockwise (front-facing) seen
    from inside the room; the material is double-sided anyway."""
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 3)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    idx = np.array([0, 2, 1, 0, 3, 2], dtype=np.uint16)
    pos_b, uv_b, idx_b = corners.tobytes(), uv.tobytes(), _pad(idx.tobytes())
    img_off = len(pos_b) + len(uv_b) + len(idx_b)
    blob = _pad(pos_b + uv_b + idx_b + image)
    views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_b), "target": ARRAY_BUFFER},
        {"buffer": 0, "byteOffset": len(pos_b), "byteLength": len(uv_b), "target": ARRAY_BUFFER},
        {
            "buffer": 0,
            "byteOffset": len(pos_b) + len(uv_b),
            "byteLength": 12,
            "target": ELEMENT_ARRAY_BUFFER,
        },
        {"buffer": 0, "byteOffset": img_off, "byteLength": len(image)},
    ]
    doc = {
        "asset": {"version": "2.0", "generator": "movable-reality bake"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [
            {
                "name": name,
                "primitives": [
                    {"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}
                ],
            }
        ],
        "materials": [
            {
                "name": name,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "alphaMode": "OPAQUE",
                "doubleSided": True,
                "extensions": {"KHR_materials_unlit": {}},
            }
        ],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [
            {
                "magFilter": LINEAR,
                "minFilter": LINEAR_MIPMAP_LINEAR,
                "wrapS": CLAMP_TO_EDGE,
                "wrapT": CLAMP_TO_EDGE,
            }
        ],
        "images": [{"bufferView": 3, "mimeType": mime}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": FLOAT,
                "count": 4,
                "type": "VEC3",
                "min": corners.min(axis=0).tolist(),
                "max": corners.max(axis=0).tolist(),
            },
            {"bufferView": 1, "componentType": FLOAT, "count": 4, "type": "VEC2"},
            {"bufferView": 2, "componentType": USHORT, "count": 6, "type": "SCALAR"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    if extras:
        doc["nodes"][0]["extras"] = extras
    js = _pad(json.dumps(doc, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(js) + 8 + len(blob)
    out = struct.pack("<III", GLB_MAGIC, 2, total)
    out += struct.pack("<II", len(js), CHUNK_JSON) + js
    out += struct.pack("<II", len(blob), CHUNK_BIN) + blob
    return out


def write_quad_glb(path: Path, corners: np.ndarray, image: bytes, mime: str, **kw) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".glb.tmp")
    tmp.write_bytes(quad_glb(corners, image, mime, **kw))
    tmp.replace(path)


def read_glb(data: bytes) -> tuple[dict, bytes]:
    """(json document, binary chunk) of a GLB."""
    magic, version, total = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != 2 or total != len(data):
        raise ValueError("not a glTF 2.0 binary")
    jlen, jtype = struct.unpack_from("<II", data, 12)
    if jtype != CHUNK_JSON:
        raise ValueError("first chunk is not JSON")
    doc = json.loads(data[20 : 20 + jlen])
    blen, btype = struct.unpack_from("<II", data, 20 + jlen)
    if btype != CHUNK_BIN:
        raise ValueError("second chunk is not BIN")
    return doc, data[28 + jlen : 28 + jlen + blen]


def accessor(doc: dict, blob: bytes, i: int) -> np.ndarray:
    a = doc["accessors"][i]
    v = doc["bufferViews"][a["bufferView"]]
    dtype = {FLOAT: np.float32, USHORT: np.uint16}[a["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
    start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    arr = np.frombuffer(blob, dtype=dtype, count=a["count"] * width, offset=start)
    return arr.reshape(a["count"], width) if width > 1 else arr


def image_bytes(doc: dict, blob: bytes, i: int = 0) -> bytes:
    v = doc["bufferViews"][doc["images"][i]["bufferView"]]
    start = v.get("byteOffset", 0)
    return blob[start : start + v["byteLength"]]
