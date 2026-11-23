"""Real blender07 data for the pipeline animation: sampled splats in world space, object ids, cameras."""
import json
import struct

import numpy as np
from plyfile import PlyData

D = "data/scenes/blender07/"
m = json.load(open(D + "scene.json", encoding="utf8"))
W = np.array(m["world_from_recon"], dtype=float)
if W.size == 16:
    W = W.reshape(4, 4)
print("world_from_recon", W)

v = PlyData.read(D + "splat/scene.ply")["vertex"].data
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(float)
rgb = np.clip(0.5 + 0.28209479 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), 0, 1)
op = 1 / (1 + np.exp(-v["opacity"]))
size = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).max(1)
ids = np.load(D + "ids/gaussian_ids.npy")
print("gaussians", len(xyz), "ids", ids.shape, ids.dtype, np.unique(ids)[:8])
world = xyz @ W[:3, :3].T + W[:3, 3]
print("world min", world.min(0), "max", world.max(0))

obj_labels = json.load(open(D + "ids/object_labels.json", encoding="utf8"))
visible = {o["id"] for o in m["objects"] if o.get("visible", True)}
lut = {}
names = []
for oid, insts in obj_labels.items():
    if oid not in visible:
        continue
    names.append(oid)
    for k in insts:
        lut[int(k) - 1] = len(names)  # object_labels counts from 1, gaussian_ids from 0
print("objects", len(names), names[:6])
oidx = np.zeros(len(xyz), dtype=np.int16)
for k, n in lut.items():
    oidx[ids == k] = n
print("assigned to objects", (oidx > 0).mean())

rng = np.random.default_rng(7)
keep = op > 0.5
idx = np.flatnonzero(keep)
rng.shuffle(idx)
idx = idx[:900000]
print("kept", len(idx))

# cameras from COLMAP images.bin: cam-to-world centre and forward in world space
cams = []
with open(D + "recon/sparse/images.bin", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    for _ in range(n):
        iid, qw, qx, qy, qz, tx, ty, tz, _cid = struct.unpack("<idddddddi", f.read(64))
        name = b""
        while (c := f.read(1)) != b"\0":
            name += c
        np2 = struct.unpack("<Q", f.read(8))[0]
        f.seek(24 * np2, 1)
        R = np.array([
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ])
        cams.append((name.decode(), R.T, -R.T @ np.array([tx, ty, tz])))
cams.sort()
C2W = np.zeros((len(cams), 4, 3))
for i, (_, Rc, c) in enumerate(cams):
    Rw = W[:3, :3] @ Rc
    C2W[i, :3, :] = Rw
    C2W[i, 3, :] = W[:3, :3] @ c + W[:3, 3]
print("cams", len(cams), cams[0][0], C2W[:, 3, :].min(0), C2W[:, 3, :].max(0))
np.savez("data/broll/_cut/anim_data.npz", xyz=world[idx].astype(np.float32), rgb=rgb[idx].astype(np.float32),
         oidx=oidx[idx], size=size[idx].astype(np.float32), cams=C2W.astype(np.float32), names=np.array(names))
