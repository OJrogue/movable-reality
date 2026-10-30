"""Split a trained scene PLY into background and per-object PLYs by Gaussian id.

Stage 5 (lift_ids) decides the id of every Gaussian (-1 = background); this module only
does the bookkeeping: group vertices, keep every PLY property (SH, `generated`, ...), and
compute each object's bbox in the world frame for scene.json. Pure numpy, CPU only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from movable_reality.scene import Vec3

BACKGROUND = -1


def object_ply_name(object_id: str) -> str:
    """objects/obj_<safe id>.ply, e.g. "floor lamp#1" -> objects/obj_floor_lamp_1.ply."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", object_id).strip("_") or "object"
    return f"objects/obj_{safe}.ply"


def split_by_id(vertices: np.ndarray, ids: np.ndarray) -> dict[int, np.ndarray]:
    """Group a structured vertex array by integer id. Order within a group is kept."""
    ids = np.asarray(ids)
    if ids.shape != (len(vertices),):
        raise ValueError(f"ids has shape {ids.shape}, expected ({len(vertices)},)")
    return {int(i): vertices[ids == i] for i in np.unique(ids)}


def world_bbox(vertices: np.ndarray, world_from_recon: list[list[float]]) -> tuple[Vec3, Vec3]:
    """Axis-aligned bbox of the splat centres after world_from_recon (row-major 4x4)."""
    if len(vertices) == 0:
        raise ValueError("cannot take the bbox of zero splats")
    m = np.asarray(world_from_recon, dtype=np.float64)
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float64)
    world = xyz @ m[:3, :3].T + m[:3, 3]
    lo, hi = world.min(axis=0), world.max(axis=0)
    return (float(lo[0]), float(lo[1]), float(lo[2])), (float(hi[0]), float(hi[1]), float(hi[2]))


def write_vertices(path: Path, vertices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))


def write_split(
    scene_ply: Path,
    ids: np.ndarray,
    names: Mapping[int, str],
    out_dir: Path,
    world_from_recon: list[list[float]],
) -> dict[str, tuple[str, int, tuple[Vec3, Vec3]]]:
    """Write objects/background.ply and one objects/obj_<id>.ply per object id.

    `names` maps integer ids to scene object ids ("chair#1"); ids not in `names` go to the
    background. Returns {object id or "background": (relative ply, splat count, bbox)}; the
    background bbox is included for completeness. Objects with no splats are left out.
    """
    vertices = PlyData.read(str(scene_ply), mmap=False)["vertex"].data
    ids = np.asarray(ids).copy()
    ids[~np.isin(ids, list(names))] = BACKGROUND
    out: dict[str, tuple[str, int, tuple[Vec3, Vec3]]] = {}
    for i, group in split_by_id(vertices, ids).items():
        if len(group) == 0:
            continue
        key = "background" if i == BACKGROUND else names[i]
        rel = "objects/background.ply" if i == BACKGROUND else object_ply_name(names[i])
        write_vertices(out_dir / rel, group)
        out[key] = (rel, len(group), world_bbox(group, world_from_recon))
    return out
