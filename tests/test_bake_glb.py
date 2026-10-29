import io
import struct

import numpy as np
from PIL import Image

from movable_reality.bake import glb


def png(w=3, h=2):
    buf = io.BytesIO()
    Image.fromarray(np.arange(w * h * 3, dtype=np.uint8).reshape(h, w, 3)).save(buf, "PNG")
    return buf.getvalue()


def test_glb_structure_is_valid_gltf_binary(tmp_path):
    corners = np.array([[0, 0, 0], [2, 0, 0], [2, 0, -1], [0, 0, -1]], dtype=float)
    image = png()
    data = glb.quad_glb(corners, image, "image/png", name="floor", extras={"label": "floor"})
    magic, version, total = struct.unpack_from("<III", data)
    assert (magic, version, total) == (glb.GLB_MAGIC, 2, len(data))
    assert len(data) % 4 == 0
    doc, blob = glb.read_glb(data)
    assert doc["asset"]["version"] == "2.0"
    assert doc["buffers"][0]["byteLength"] == len(blob)
    for v in doc["bufferViews"]:
        assert v["byteOffset"] % 4 == 0
        assert v["byteOffset"] + v["byteLength"] <= len(blob)
    pos = glb.accessor(doc, blob, 0)
    assert np.allclose(pos, corners)
    assert doc["accessors"][0]["min"] == [0, 0, -1] and doc["accessors"][0]["max"] == [2, 0, 0]
    assert glb.accessor(doc, blob, 1).tolist() == [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert glb.image_bytes(doc, blob) == image
    assert doc["nodes"][0]["extras"] == {"label": "floor"}
    glb.write_quad_glb(tmp_path / "x" / "floor.glb", corners, image, "image/png")
    assert (tmp_path / "x" / "floor.glb").read_bytes() == glb.quad_glb(corners, image, "image/png")


def test_triangles_face_the_room():
    # u = +x, v = -z, so u x v = +y... that is, pointing up; the room is below (ceiling).
    corners = np.array([[0, 2.5, 0], [1, 2.5, 0], [1, 2.5, -1], [0, 2.5, -1]], dtype=float)
    doc, blob = glb.read_glb(glb.quad_glb(corners, png(), "image/png"))
    idx = glb.accessor(doc, blob, 2).reshape(-1, 3)
    for tri in idx:
        a, b, c = corners[tri]
        face_normal = np.cross(b - a, c - a)
        assert face_normal[1] < 0  # counter-clockwise seen from below = from inside
