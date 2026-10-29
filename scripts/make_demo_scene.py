"""Write a synthetic scene (floor, wall, three box objects) so the agent and viewer can be
developed before real captures exist.

    uv run python scripts/make_demo_scene.py            # -> data/scenes/demo/
    cd web && npm run dev                               # http://localhost:5173/?scene=demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from movable_reality.config import scene_dir
from movable_reality.lod.split import object_ply_name
from movable_reality.scene import Scene, SceneObject

SH_C0 = 0.28209479177387814
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def write_gaussian_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, size: float = 0.02) -> None:
    """Standard 3DGS PLY (degree-0 SH): log scales, logit opacity, wxyz quaternions."""
    n = len(xyz)
    fields = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    data = np.zeros(n, dtype=[(f, "f4") for f in fields])
    data["x"], data["y"], data["z"] = xyz.T
    dc = (rgb - 0.5) / SH_C0
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = dc.T
    data["opacity"] = np.log(0.95 / 0.05)
    for k in ("scale_0", "scale_1", "scale_2"):
        data[k] = np.log(size)
    data["rot_0"] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def box_surface(rng, lo, hi, n) -> np.ndarray:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    pts = rng.uniform(lo, hi, size=(n, 3))
    axis = rng.integers(0, 3, n)
    side = rng.integers(0, 2, n)
    pts[np.arange(n), axis] = np.where(side == 1, hi[axis], lo[axis])
    return pts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="demo")
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    out = scene_dir(args.scene)

    g = np.linspace(-2.0, 2.0, 120)
    fx, fz = np.meshgrid(g, g - 1.5)
    floor = np.stack([fx.ravel(), np.zeros(fx.size), fz.ravel()], 1)
    wy, wx = np.meshgrid(np.linspace(0, 2.5, 80), g)
    wall = np.stack([wx.ravel(), wy.ravel(), np.full(wx.size, -3.5)], 1)
    bg = np.concatenate([floor, wall])
    bg_rgb = np.concatenate(
        [np.tile([0.55, 0.45, 0.35], (len(floor), 1)), np.tile([0.85, 0.85, 0.8], (len(wall), 1))]
    )
    write_gaussian_ply(out / "objects" / "background.ply", bg, bg_rgb, size=0.025)

    specs = [
        ("chair#1", "chair", (-1.1, 0.0, -1.8), (-0.6, 0.9, -1.3), (0.2, 0.35, 0.7)),
        ("table#1", "table", (0.2, 0.0, -2.2), (1.4, 0.75, -1.4), (0.45, 0.3, 0.15)),
        ("floor lamp#1", "floor lamp", (1.6, 0.0, -3.0), (1.8, 1.6, -2.8), (0.9, 0.8, 0.3)),
    ]
    objects = []
    for oid, label, lo, hi, color in specs:
        pts = box_surface(rng, lo, hi, 4000)
        rgb = np.clip(np.asarray(color) + rng.normal(0, 0.03, (len(pts), 3)), 0, 1)
        ply = object_ply_name(oid)
        write_gaussian_ply(out / ply, pts, rgb, size=0.015)
        objects.append(
            SceneObject(
                id=oid, label=label, ply=ply, bbox_min=tuple(pts.min(0)), bbox_max=tuple(pts.max(0))
            )
        )

    scene = Scene(scene_id=args.scene, world_from_recon=IDENTITY, objects=objects)
    scene.save(out / "scene.json")
    print(f"wrote {out} ({len(objects)} objects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
