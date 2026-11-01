"""Triangle meshes of completed objects (SAM 3D Objects' mesh decoder) as vertex-coloured GLB.

SAM 3D Objects exports the mesh through trimesh, in the GLB's y-up canonical frame. These
functions read that file, find the rotation back to the frame of the object's Gaussians,
move the vertices into the reconstruction frame with the same matrix that registered the
Gaussians, and write a GLB the viewer draws next to the splats (recon frame, like the PLYs).

The written material is unlit (KHR_materials_unlit), so three.js draws the vertex colours
as they are, like the splats, with no scene lights. glTF defines COLOR_0 as linear, and the
decoder's colours are sRGB image colours, so they are converted on write.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .bake.glb import CHUNK_BIN, CHUNK_JSON, GLB_MAGIC, _pad, read_glb

BYTE, UBYTE, SHORT, USHORT, UINT, FLOAT = 5120, 5121, 5122, 5123, 5125, 5126
_DTYPE = {
    BYTE: np.int8, UBYTE: np.uint8, SHORT: np.int16, USHORT: np.uint16, UINT: np.uint32,
    FLOAT: np.float32,
}  # fmt: skip
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

# to_glb (sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py) turns the
# z-up decoder frame into the GLB's y-up one with v_glb = v @ Z_TO_Y (row vectors).
Z_TO_Y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _accessor(doc: dict, blob: bytes, i: int) -> np.ndarray:
    """Accessor i as float64 (normalised integers to 0..1), any stride."""
    a = doc["accessors"][i]
    v = doc["bufferViews"][a["bufferView"]]
    dt = np.dtype(_DTYPE[a["componentType"]])
    width = _WIDTH[a["type"]]
    if a["count"] == 0:
        return np.zeros((0, width)) if width > 1 else np.zeros(0)
    start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = v.get("byteStride") or dt.itemsize * width
    raw = np.frombuffer(blob, dtype=np.uint8, count=stride * (a["count"] - 1) + dt.itemsize * width,
                        offset=start)  # fmt: skip
    rows = np.lib.stride_tricks.as_strided(
        raw, shape=(a["count"], dt.itemsize * width), strides=(stride, 1)
    )
    out = np.ascontiguousarray(rows).view(dt).reshape(a["count"], width).astype(np.float64)
    if a.get("normalized") and a["componentType"] != FLOAT:
        out = np.maximum(out / np.iinfo(dt).max, -1.0)  # spec rule, signed types included
    return out if width > 1 else out[:, 0]


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T  # column-major
    M = np.eye(4)
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        M[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    M[:3, :3] = M[:3, :3] * np.asarray(node.get("scale", [1, 1, 1]))
    M[:3, 3] = node.get("translation", [0, 0, 0])
    return M


def read_mesh_glb(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vertices (n, 3), sRGB colours (n, 3) in 0..1, faces (m, 3)) of every triangle
    primitive in a GLB, node transforms applied. Vertices without colours come out grey."""
    doc, blob = read_glb(data)
    verts, cols, faces, n = [], [], [], 0
    nodes = doc.get("nodes", [])
    roots = doc["scenes"][doc.get("scene", 0)]["nodes"] if doc.get("scenes") else range(len(nodes))
    stack = [(i, np.eye(4)) for i in roots]
    while stack:
        i, parent = stack.pop()
        node = nodes[i]
        M = parent @ _node_matrix(node)
        stack += [(c, M) for c in node.get("children", [])]
        if "mesh" not in node:
            continue
        for prim in doc["meshes"][node["mesh"]]["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            attrs = prim["attributes"]
            p = _accessor(doc, blob, attrs["POSITION"])
            p = p @ M[:3, :3].T + M[:3, 3]
            if "COLOR_0" in attrs:
                c = _accessor(doc, blob, attrs["COLOR_0"])[:, :3]
                if doc["accessors"][attrs["COLOR_0"]]["componentType"] == FLOAT:
                    c = linear_to_srgb(c)  # float colours follow the spec (linear)
                # trimesh writes its 8-bit colours as they are, which here are sRGB.
            else:
                c = np.full_like(p, 0.6)
            f = (
                _accessor(doc, blob, prim["indices"]).astype(np.int64).reshape(-1, 3)
                if "indices" in prim
                else np.arange(len(p)).reshape(-1, 3)
            )
            verts.append(p)
            cols.append(c)
            faces.append(f + n)
            n += len(p)
    if not verts:
        raise ValueError("GLB has no triangle mesh")
    return np.concatenate(verts), np.concatenate(cols), np.concatenate(faces)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def mesh_glb(vertices: np.ndarray, colours: np.ndarray, faces: np.ndarray, name: str = "object",
             extras: dict | None = None) -> bytes:  # fmt: skip
    """GLB with one unlit, double-sided, vertex-coloured triangle mesh. colours are sRGB in
    0..1 and are stored linear (COLOR_0, float)."""
    pos = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    col = srgb_to_linear(np.asarray(colours, dtype=np.float64)).astype(np.float32).reshape(-1, 3)
    idx = np.asarray(faces, dtype=np.uint32).reshape(-1)
    if len(col) != len(pos):
        raise ValueError(f"{len(col)} colours for {len(pos)} vertices")
    if idx.size and idx.max() >= len(pos):
        raise ValueError("face index out of range")
    pos_b, col_b, idx_b = pos.tobytes(), col.tobytes(), idx.tobytes()
    blob = _pad(pos_b + col_b + idx_b)
    views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_b), "target": 34962},
        {"buffer": 0, "byteOffset": len(pos_b), "byteLength": len(col_b), "target": 34962},
        {
            "buffer": 0,
            "byteOffset": len(pos_b) + len(col_b),
            "byteLength": len(idx_b),
            "target": 34963,
        },
    ]
    doc = {
        "asset": {"version": "2.0", "generator": "movable-reality meshes"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name, **({"extras": extras} if extras else {})}],
        "meshes": [
            {
                "name": name,
                "primitives": [
                    {"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2, "material": 0}
                ],
            }
        ],
        "materials": [
            {
                "name": name,
                "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0},
                "doubleSided": True,
                "extensions": {"KHR_materials_unlit": {}},
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": FLOAT,
                "count": len(pos),
                "type": "VEC3",
                "min": pos.min(axis=0).tolist() if len(pos) else [0, 0, 0],
                "max": pos.max(axis=0).tolist() if len(pos) else [0, 0, 0],
            },
            {"bufferView": 1, "componentType": FLOAT, "count": len(col), "type": "VEC3"},
            {"bufferView": 2, "componentType": UINT, "count": len(idx), "type": "SCALAR"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    js = _pad(json.dumps(doc, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(js) + 8 + len(blob)
    out = struct.pack("<III", GLB_MAGIC, 2, total)
    out += struct.pack("<II", len(js), CHUNK_JSON) + js
    out += struct.pack("<II", len(blob), CHUNK_BIN) + blob
    return out


def write_mesh_glb(path: Path, vertices, colours, faces, **kw) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".glb.tmp")
    tmp.write_bytes(mesh_glb(vertices, colours, faces, **kw))
    tmp.replace(path)


def _mean_nearest(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> float:
    """Mean distance from each point of a to its nearest point of b (brute force)."""
    d = [np.sqrt(((a[i : i + chunk, None] - b[None]) ** 2).sum(-1)).min(1) for i in
         range(0, len(a), chunk)]  # fmt: skip
    return float(np.concatenate(d).mean())


def to_gaussian_frame(
    mesh_v: np.ndarray, gauss_xyz: np.ndarray, n: int = 1500, seed: int = 0, margin: float = 0.5
) -> tuple[np.ndarray, str, dict[str, float]]:
    """3x3 rotation R with v_gauss = v_mesh @ R.T. The default is the inverse of to_glb's
    z-up to y-up turn ("y_to_z"). Another candidate (the turn itself, identity) replaces it
    only when its symmetric mean nearest-point distance is under `margin` times the
    default's, so a near-cubic or symmetric object cannot flip the mesh by a small margin.
    Returns R, its name and every candidate's distance (canonical units) for the report."""
    rng = np.random.default_rng(seed)
    a = mesh_v[rng.choice(len(mesh_v), min(n, len(mesh_v)), replace=False)]
    b = gauss_xyz[rng.choice(len(gauss_xyz), min(n, len(gauss_xyz)), replace=False)]
    # v_glb = v_z @ Z_TO_Y gives v_z = v_glb @ Z_TO_Y.T, i.e. R = Z_TO_Y.
    cands = {"y_to_z": Z_TO_Y, "z_to_y": Z_TO_Y.T, "identity": np.eye(3)}
    dist = {}
    for name, R in cands.items():
        moved = a @ R.T
        dist[name] = round(0.5 * (_mean_nearest(moved, b) + _mean_nearest(b, moved)), 5)
    name = min(dist, key=dist.get)
    if name != "y_to_z" and dist[name] >= margin * dist["y_to_z"]:
        name = "y_to_z"
    elif name != "y_to_z":
        print(f"  mesh frame: {name} fits clearly better than y_to_z {dist}")
    return cands[name], name, dist


def decimate(
    vertices: np.ndarray, colours: np.ndarray, faces: np.ndarray, target_faces: int = 100_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadric edge collapse (fast-simplification, MIT, no learned weights) down to about
    target_faces. Each kept vertex takes the mean colour of the vertices merged into it.
    A mesh already at or under the target comes back unchanged."""
    if len(faces) <= target_faces:
        return vertices, colours, faces
    import fast_simplification as fs  # noqa: PLC0415 - only needed for large meshes

    p = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    _, _, collapses = fs.simplify(p, f, target_count=target_faces, return_collapses=True)
    p2, f2, mapping = fs.replay_simplification(p, f, collapses)
    keep = mapping >= 0
    n = len(p2)
    counts = np.bincount(mapping[keep], minlength=n).astype(np.float64)
    c2 = (
        np.stack(
            [np.bincount(mapping[keep], weights=colours[keep, k], minlength=n) for k in range(3)], 1
        )
        / np.maximum(counts, 1)[:, None]
    )
    return p2.astype(np.float64), c2, f2.astype(np.int64)


def read_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vertices, sRGB colours in 0..1, faces) of a triangle PLY such as open3d writes."""
    from plyfile import PlyData  # noqa: PLC0415

    ply = PlyData.read(str(path))
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    names = v.data.dtype.names
    if "red" in names:
        rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float64)
        rgb = rgb / 255.0 if rgb.max() > 1.0 or v["red"].dtype == np.uint8 else rgb
    else:
        rgb = np.full_like(xyz, 0.6)
    faces = np.stack(ply["face"]["vertex_indices"]).astype(np.int64).reshape(-1, 3)
    return xyz, rgb, faces
