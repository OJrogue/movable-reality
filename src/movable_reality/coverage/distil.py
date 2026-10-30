"""C3, last step: distil Muse Image repairs of gap views back into splats.

CPU side of `modal run modal_jobs/coverage.py --scene <id> --job distil`:

1. `load_views`: every coverage/repair/<view>/ with a repaired image (repaired.webp from
   coverage_repair.py --call) that passed the alignment check (check.json: Muse Image kept
   the viewpoint), its camera (request.json, world frame) and gap mask, the repaired image
   resized to the camera.
2. `seed_points`: candidate positions for the new Gaussians, jittered inside the gap's
   voxels (coverage/gaps.npz) where at least one repaired view of that gap shows them
   inside its mask, coloured from that view.
3. `pack_inputs`: cameras and seeds moved to the reconstruction frame (the frame of
   splat/scene.ply and of gsplat's training), as one .npz for the GPU job.

On the GPU (modal_jobs/coverage.py, `distil_repairs`), the scanned Gaussians stay frozen.
Only the new ones are fitted: masked L1 against each repaired view, plus L1 against the
real training frames, so a new splat that would show up where a real frame saw something
else is pushed away or faded out. Splats left nearly transparent are dropped. The result is
coverage/generated.ply, every splat flagged generated = 1.

4. `finish`: records coverage.generated_ply and generated_splats in scene.json and sets
   the repaired gaps to status "generated" (the gap overlay hides them, the viewer's
   "show generated" toggle hides the splats).
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..scene import Scene
from .cameras import Camera, similarity_parts, transform_points
from .ply import read_generated
from .repair import camera_from_dict
from .voxels import VoxelGrid

GENERATED_PLY = "coverage/generated.ply"


@dataclass
class RepairView:
    gap_id: str
    name: str  # folder name under coverage/repair/
    camera: Camera  # world frame
    image: np.ndarray  # (H, W, 3) uint8, repaired, at the camera's size
    mask: np.ndarray  # (H, W) bool, pixels Muse Image was asked to fill


def load_views(folder: Path) -> list[RepairView]:
    out = []
    for req_path in sorted((Path(folder) / "coverage" / "repair").glob("*/request.json")):
        d = req_path.parent
        repaired = d / "repaired.webp"
        if not repaired.exists() or not (d / "mask.png").exists():
            continue
        check = d / "check.json"  # repair.write_check: is the repair aligned with the view?
        if not check.exists() or not json.loads(check.read_text()).get("accepted", False):
            continue
        req = json.loads(req_path.read_text())
        cam = camera_from_dict(req["camera"], name=d.name)
        img = Image.open(repaired).convert("RGB")
        if img.size != (cam.width, cam.height):
            img = img.resize((cam.width, cam.height), Image.LANCZOS)
        mask = np.asarray(Image.open(d / "mask.png").convert("L")) > 127
        out.append(RepairView(req["gap_id"], d.name, cam, np.asarray(img), mask))
    return out


def gap_index(gap_id: str) -> int:
    """Label index of a gap in coverage/gaps.npz (build_gaps names gaps gap_{label:03d})."""
    return int(gap_id.rsplit("_", 1)[1])


def seed_points(
    grid: VoxelGrid,
    labels: np.ndarray,
    views: list[RepairView],
    per_voxel: int = 24,
    max_seeds: int = 60_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points (M, 3) world, rgb (M, 3) float in [0, 1], gap label (M,)) jittered inside
    the voxels of every repaired gap, kept where a repaired view of the same gap sees them
    inside its mask. Thinned evenly to `max_seeds`."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).ravel()
    pts_all, rgb_all, lab_all = [], [], []
    for gid in sorted({v.gap_id for v in views}):
        g = gap_index(gid)
        idx = np.nonzero(labels == g)[0]
        if idx.size == 0:
            continue
        centres = np.repeat(grid.centres(idx).astype(np.float64), per_voxel, axis=0)
        pts = centres + rng.uniform(-0.5, 0.5, centres.shape) * grid.size
        rgb = np.zeros((len(pts), 3))
        hit = np.zeros(len(pts), bool)
        for v in (v for v in views if v.gap_id == gid):
            uv, z = v.camera.project(pts)
            u = np.floor(uv[:, 0]).astype(int)
            w = np.floor(uv[:, 1]).astype(int)
            ok = (z > 0.05) & (u >= 0) & (u < v.camera.width) & (w >= 0) & (w < v.camera.height)
            ok[ok] &= v.mask[w[ok], u[ok]]
            new = ok & ~hit
            rgb[new] = v.image[w[new], u[new]] / 255.0
            hit |= ok
        pts_all.append(pts[hit])
        rgb_all.append(rgb[hit])
        lab_all.append(np.full(int(hit.sum()), g, np.int32))
    if not pts_all:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int32)
    pts, rgb, lab = np.concatenate(pts_all), np.concatenate(rgb_all), np.concatenate(lab_all)
    if len(pts) > max_seeds:
        keep = np.linspace(0, len(pts) - 1, max_seeds).round().astype(int)
        pts, rgb, lab = pts[keep], rgb[keep], lab[keep]
    return pts, rgb, lab


def load_gap_grid(folder: Path) -> tuple[VoxelGrid, np.ndarray]:
    z = np.load(Path(folder) / "coverage" / "gaps.npz")
    grid = VoxelGrid.from_dict(json.loads(str(z["grid"])))
    return grid, z["labels"].ravel()


def pack_inputs(
    views: list[RepairView],
    seeds: np.ndarray,
    seed_rgb: np.ndarray,
    voxel: float,
    world_from_recon: np.ndarray,
) -> bytes:
    """One .npz for the GPU job, everything in the reconstruction frame: per view K (3, 3),
    viewmat (4, 4) cam_from_recon, image, mask, plus seeds, their colours, a start scale."""
    W = np.asarray(world_from_recon, np.float64)
    recon_from_world = np.linalg.inv(W)
    scale, _ = similarity_parts(W)
    arrays: dict[str, np.ndarray] = {
        "seeds": transform_points(recon_from_world, seeds).astype(np.float32),
        "seed_rgb": np.asarray(seed_rgb, np.float32),
        "seed_scale": np.float32(0.5 * voxel / scale),
        "n_views": np.int32(len(views)),
    }
    for i, v in enumerate(views):
        c = v.camera.in_frame(recon_from_world)
        vm = np.eye(4)
        vm[:3, :3], vm[:3, 3] = c.R, c.t
        arrays[f"K_{i}"] = c.K.astype(np.float32)
        arrays[f"viewmat_{i}"] = vm.astype(np.float32)
        arrays[f"image_{i}"] = v.image
        arrays[f"mask_{i}"] = v.mask
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def finish(folder: Path, gap_ids: list[str]) -> Scene:
    """Point scene.json at coverage/generated.ply, count its splats and mark the repaired
    gaps generated."""
    folder = Path(folder)
    path = folder / "scene.json"
    scene = Scene.load(path)
    if scene.coverage is None:
        raise ValueError("scene.json has no coverage section: score the scene first")
    ply = folder / GENERATED_PLY
    scene.coverage.generated_ply = GENERATED_PLY
    scene.coverage.generated_splats = int(read_generated(ply).sum())
    done = set(gap_ids)
    scene.coverage.gaps = [
        g.model_copy(update={"status": "generated"}) if g.id in done else g
        for g in scene.coverage.gaps
    ]
    scene.save(path)
    return scene
