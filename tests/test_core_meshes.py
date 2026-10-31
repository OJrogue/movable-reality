import numpy as np

from movable_reality.bake.glb import read_glb
from movable_reality.meshes import (
    Z_TO_Y,
    linear_to_srgb,
    mesh_glb,
    read_mesh_glb,
    srgb_to_linear,
    to_gaussian_frame,
)


def _box():
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 2) for z in (0, 3)], dtype=float)
    f = np.array([[0, 1, 2], [1, 3, 2], [4, 6, 5], [5, 6, 7]])
    return v, f


def test_round_trip_keeps_vertices_colours_faces():
    v, f = _box()
    c = np.linspace(0, 1, 24).reshape(8, 3)
    doc, _ = read_glb(mesh_glb(v, c, f))
    assert doc["materials"][0]["extensions"] == {"KHR_materials_unlit": {}}
    v2, c2, f2 = read_mesh_glb(mesh_glb(v, c, f))
    assert np.allclose(v2, v, atol=1e-6)
    assert np.allclose(c2, c, atol=1e-5)  # stored linear, read back as sRGB
    assert (f2 == f).all()


def test_srgb_conversion_inverts():
    c = np.linspace(0, 1, 11)
    assert np.allclose(linear_to_srgb(srgb_to_linear(c)), c, atol=1e-9)
    assert srgb_to_linear(np.array([0.5]))[0] < 0.5


def test_node_transform_is_applied():
    import json
    import struct

    v, f = _box()
    data = mesh_glb(v, np.full((8, 3), 0.5), f)
    doc, blob = read_glb(data)
    doc["nodes"][0]["translation"] = [10, 0, 0]
    js = json.dumps(doc).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 28 + len(js) + len(blob))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942)
    v2, _, _ = read_mesh_glb(out + blob)
    assert np.allclose(v2, v + [10, 0, 0])


def test_frame_found_from_the_gaussians():
    rng = np.random.default_rng(1)
    gauss = rng.uniform([-0.2, -0.4, -0.1], [0.2, 0.4, 0.1], size=(3000, 3))  # z-up frame
    mesh_v = gauss[:2000] @ Z_TO_Y  # what to_glb does
    R, name, d = to_gaussian_frame(mesh_v, gauss)
    assert name == "y_to_z"
    assert np.allclose(mesh_v @ R.T, gauss[:2000])
    assert d["y_to_z"] < 0.02 < d["identity"]


def test_symmetric_object_keeps_the_default_frame():
    rng = np.random.default_rng(2)
    cube = rng.uniform(-0.3, 0.3, size=(3000, 3))  # every candidate fits about as well
    _, name, d = to_gaussian_frame(cube[:2000] @ Z_TO_Y, cube)
    assert name == "y_to_z" and set(d) == {"y_to_z", "z_to_y", "identity"}


def _trimesh_style_glb() -> bytes:
    """Interleaved POSITION (float) + COLOR_0 (normalised UBYTE VEC4) with byteStride 16,
    USHORT indices, a parent node with a translation and a child with rotation and scale."""
    import json
    import struct

    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    col = np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8)
    inter = b"".join(p.tobytes() + c.tobytes() for p, c in zip(pos, col, strict=True))
    idx = np.array([0, 1, 2], dtype=np.uint16).tobytes() + b"\x00\x00"
    blob = inter + idx
    h = np.sqrt(0.5)  # 90 degrees about +Z: x -> y
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"translation": [10, 0, 0], "children": [1]},
            {"mesh": 0, "rotation": [0, 0, h, h], "scale": [2, 2, 2]},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {
                "bufferView": 0,
                "byteOffset": 12,
                "componentType": 5121,
                "normalized": True,
                "count": 3,
                "type": "VEC4",
            },
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(inter), "byteStride": 16},
            {"buffer": 0, "byteOffset": len(inter), "byteLength": 6},
        ],
        "buffers": [{"byteLength": len(blob)}],
    }
    js = json.dumps(doc).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 28 + len(js) + len(blob))
    return (
        out
        + struct.pack("<II", len(js), 0x4E4F534A)
        + js
        + struct.pack("<II", len(blob), 0x004E4942)
        + blob
    )


def test_reads_trimesh_style_layout():
    v, c, f = read_mesh_glb(_trimesh_style_glb())
    assert np.allclose(v, [[10, 0, 0], [10, 2, 0], [8, 0, 0]], atol=1e-6)
    assert np.allclose(c, np.eye(3))
    assert (f == [[0, 1, 2]]).all()


def test_mesh_field_survives_a_scene_round_trip(tmp_path):
    from movable_reality.scene import Scene, SceneObject

    s = Scene(scene_id="t")
    s.objects.append(SceneObject(id="chair#1", label="chair", ply="objects/obj_chair_1.ply",
                                 bbox_min=(0, 0, 0), bbox_max=(1, 1, 1),
                                 mesh="objects/obj_chair_1_mesh.glb"))  # fmt: skip
    s.save(tmp_path / "scene.json")
    assert Scene.load(tmp_path / "scene.json").objects[0].mesh == "objects/obj_chair_1_mesh.glb"


def test_decimate_keeps_shape_and_colours():
    from movable_reality.meshes import decimate

    n = 60  # grid of 2 * 59 * 59 = 6962 triangles on a slightly curved sheet
    u, w = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    v = np.stack([u.ravel(), 0.1 * np.sin(3 * u.ravel()), w.ravel()], 1)
    c = np.where(u.ravel()[:, None] < 0.5, [1.0, 0, 0], [0, 0, 1.0])
    i = np.arange(n * n).reshape(n, n)
    a, b, d, e = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, :-1].ravel(), i[1:, 1:].ravel()
    f = np.concatenate([np.stack([a, b, d], 1), np.stack([b, e, d], 1)])
    v2, c2, f2 = decimate(v, c, f, target_faces=1000)
    assert len(f2) <= 1100 and f2.max() < len(v2)
    assert np.allclose(v2.min(0), v.min(0), atol=0.02) and np.allclose(
        v2.max(0), v.max(0), atol=0.02
    )
    left = v2[:, 0] < 0.3
    assert (c2[left, 0] > 0.9).all() and (c2[v2[:, 0] > 0.7, 2] > 0.9).all()
    same = decimate(v, c, f, target_faces=10_000)
    assert same[2] is f
