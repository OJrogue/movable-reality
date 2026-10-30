"""Score one scene folder: geometric coverage (+ residual uncertainty when available) ->
voxel states -> gaps -> next-best views -> coverage section of scene.json.

Inputs (data/scenes/<id>/): scene.json, splat/scene.ply (or the per-object PLYs listed in
the manifest), recon/ (COLMAP model). Optional: coverage/uncertainty.npz from
modal_jobs/coverage.py, aligned with splat/scene.ply.

Outputs: coverage/gaussians.npz (per Gaussian), coverage/gaps.npz (voxel grid, states,
scores, gap labels, exposure), coverage/report.json (counts, timings), and the `coverage`
section of scene.json.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from ..scene import Coverage, Scene
from .cameras import similarity_parts, transform_points
from .colmap import read_model
from .gaps import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    GapParams,
    build_gaps,
    candidates,
    carry_over,
    gap_voxels,
    voxel_coverage,
    voxel_gap_score,
    voxel_poor_share,
    voxel_states,
)
from .geometry import DepthParams, compute_coverage, geometric_score
from .nbv import NbvParams, WalkParams, exposure, eyes_at, gap_targets, rank_views
from .nbv import walkable_positions as walkable
from .ply import Splats, count_generated, read_splats
from .uncertainty import seen_uncertainty
from .voxels import VoxelGrid, occupancy, room_bounds


@dataclass(frozen=True)
class ScoreParams:
    voxel: float = 0.1  # metres
    views_ok: float = 3.0
    spread_ok: float = 10.0  # degrees
    uncertainty_ref: float = 0.25  # residual-uncertainty level that counts as "not covered"
    min_opacity: float = 0.3
    exposure_dirs: int = 6000
    depth: DepthParams = field(default_factory=DepthParams)
    gaps: GapParams = field(default_factory=GapParams)
    walk: WalkParams = field(default_factory=WalkParams)
    nbv: NbvParams = field(default_factory=NbvParams)
    # Eye height from the capture: the median camera height replaces walk.eye_height (and
    # scales the crouch height), so a scene whose metric scale or phone height differs from
    # 1.6 m still gets standing positions inside the scored room.
    eye_from_cameras: bool = True


@dataclass
class ScoreResult:
    grid: VoxelGrid
    state: np.ndarray
    score: np.ndarray
    labels: np.ndarray
    exposed: np.ndarray
    gauss_cov: np.ndarray
    view_count: np.ndarray
    spread: np.ndarray
    coverage: Coverage
    report: dict


def coarse_subset(xz: np.ndarray, spacing: float, coarse: float) -> np.ndarray:
    """Every k-th standing position along both grid axes (k = coarse / spacing), so the
    exposure pass uses a coarser grid than next-best-view ranking."""
    k = max(1, int(round(coarse / spacing)))
    if k == 1 or len(xz) == 0:
        return xz
    idx = np.rint((xz - xz.min(axis=0)) / spacing).astype(int)
    sel = (idx % k == 0).all(axis=1)
    return xz[sel] if sel.any() else xz


def recon_dir(folder: Path) -> Path:
    """recon_full/ (cameras rescaled to the full-resolution frames, as gsplat trains on
    them) when present, else MapAnything's recon/ model."""
    full = Path(folder) / "recon_full"
    return full if (full / "sparse").is_dir() else Path(folder) / "recon"


def walk_for_capture(walk: WalkParams, cam_heights: np.ndarray, top: float, voxel: float):
    """WalkParams with the eye height set to the median camera height, kept half a voxel
    below the top of the grid, and the crouch heights scaled by the same factor."""
    if len(cam_heights) == 0:
        return walk
    eye = float(np.clip(np.median(cam_heights), voxel, top - 0.5 * voxel))
    k = eye / walk.eye_height
    heights = tuple(round(min(h * k, eye), 3) for h in walk.exposure_heights)
    return replace(walk, eye_height=round(eye, 3), exposure_heights=heights)


def scene_ply_paths(folder: Path, scene: Scene) -> list[Path]:
    """splat/scene.ply when present, else background + object PLYs from the manifest."""
    full = folder / "splat" / "scene.ply"
    if full.exists():
        return [full]
    paths = [folder / scene.background_ply] + [folder / o.ply for o in scene.objects]
    return [p for p in paths if p.exists()]


def load_splats(folder: Path, scene: Scene) -> tuple[Splats, list[Path]]:
    paths = scene_ply_paths(folder, scene)
    if not paths:
        raise FileNotFoundError(f"no splat PLY under {folder} (splat/scene.ply or objects/)")
    return Splats.concat([read_splats(p) for p in paths]), paths


def score_scene(folder: Path, params: ScoreParams | None = None) -> ScoreResult:
    p = params or ScoreParams()
    folder = Path(folder)
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    scene = Scene.load(folder / "scene.json")
    splats, ply_paths = load_splats(folder, scene)
    world_from_recon = np.asarray(scene.world_from_recon, dtype=np.float64)
    scale, _ = similarity_parts(world_from_recon)
    means = transform_points(world_from_recon, splats.means)
    scales = splats.scales * np.float32(scale)
    recon_cams = read_model(recon_dir(folder))
    cams = [c.in_frame(world_from_recon) for c in recon_cams]
    t["load_s"] = time.perf_counter() - t0

    lo, hi = room_bounds(means, splats.opacity, min_opacity=p.min_opacity)
    grid = VoxelGrid.from_bounds(lo, hi, p.voxel)
    if p.eye_from_cameras:
        top = float(grid.origin[1] + grid.dims[1] * grid.size)
        heights = np.array([c.center[1] for c in cams])
        p = replace(p, walk=walk_for_capture(p.walk, heights, top, grid.size))
    keep = np.all(
        (means >= grid.origin) & (means < grid.origin + np.asarray(grid.dims) * grid.size), 1
    )

    t0 = time.perf_counter()
    gc, free = compute_coverage(
        means, splats.opacity, scales, cams, p.depth, voxel_centres=grid.centres()
    )
    t["geometric_s"] = time.perf_counter() - t0
    view_count, spread = gc.view_count, gc.spread_deg
    geo = geometric_score(view_count, spread, p.views_ok, p.spread_ok)

    unc_path = folder / "coverage" / "uncertainty.npz"
    used_uncertainty = False
    gauss_cov = geo
    if unc_path.exists():
        unc = np.load(unc_path)
        if "coeffs" in unc and len(unc["coeffs"]) == len(geo):
            u = seen_uncertainty(unc["coeffs"], splats.means, recon_cams)
        else:
            u = unc["u_mean"].astype(np.float32)
        if len(u) == len(geo):
            gauss_cov = geo * (1.0 - np.clip(u / p.uncertainty_ref, 0.0, 1.0))
            used_uncertainty = True

    t0 = time.perf_counter()
    occ, gvox = occupancy(grid, means, splats.opacity, p.min_opacity)
    state = voxel_states(occ, free)
    weight = splats.opacity * keep
    vcov = voxel_coverage(grid, gvox, gauss_cov, weight)
    score = voxel_gap_score(state, vcov)
    poor = voxel_poor_share(grid, gvox, gauss_cov, weight, p.gaps.cov_threshold)
    cand = candidates(state, poor, p.gaps)
    # The grid rounds up past `hi`, so its last layers lie behind the walls and above the
    # ceiling. Keep candidates whose centre is within half a voxel of the room box.
    vc = grid.centres()
    half = 0.5 * grid.size
    cand &= np.all((vc >= lo - half) & (vc <= hi + half), axis=1)
    stand = walkable(grid, state, p.walk)
    eyes = eyes_at(
        coarse_subset(stand, p.walk.spacing, p.walk.exposure_spacing), p.walk.exposure_heights
    )
    # Rays stop at well-covered surfaces only: a VR user sees into poorly covered or
    # never-observed regions (floaters, holes), so those count as exposed.
    solid = (state == OCCUPIED) & ~cand
    exp_count = exposure(grid, solid, state != FREE, eyes, n_dirs=p.exposure_dirs)
    exposed = exp_count > 0
    t["voxels_and_exposure_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    labels = gap_voxels(cand, exposed, grid, p.gaps)
    gaps = build_gaps(grid, labels, score, scene, p.gaps)
    rng = np.random.default_rng(p.nbv.seed)
    for g in gaps:
        gi = int(g.id.split("_")[1])
        tg = gap_targets(
            grid,
            labels,
            gi,
            means,
            gvox,
            gauss_cov,
            state,
            p.gaps.cov_threshold,
            p.nbv.max_targets,
            rng,
        )
        g.next_best_views = rank_views(grid, state, labels, gi, tg, stand, p.walk, p.nbv)
    t["gaps_and_nbv_s"] = time.perf_counter() - t0

    exposed_solid = exposed & (state != FREE)
    gap_share = float((labels[exposed_solid] >= 0).mean()) if exposed_solid.any() else 0.0
    old = scene.coverage.gaps if scene.coverage else []
    cov = Coverage(
        gap_map="coverage/gaps.npz",
        gaps=carry_over(gaps, old),
        generated_splats=count_generated(ply_paths),
        gap_share=round(gap_share, 4),
    )
    report = {
        "gaussians": int(len(means)),
        "views": len(cams),
        "voxel_size": p.voxel,
        "grid_dims": list(grid.dims),
        "voxels": {
            "free": int((state == FREE).sum()),
            "occupied": int((state == OCCUPIED).sum()),
            "unknown": int((state == UNKNOWN).sum()),
            "exposed": int(exposed.sum()),
            "gap": int((labels >= 0).sum()),
        },
        "walkable_positions": int(len(stand)),
        "exposure_eyes": int(len(eyes)),
        "gaps": len(gaps),
        "gap_share": cov.gap_share,
        "used_uncertainty": used_uncertainty,
        "gaussians_seen_by_0_views": int((view_count == 0).sum()),
        "gaussians_seen_by_1_view": int((view_count == 1).sum()),
        "median_view_count": float(np.median(view_count)),
        "median_spread_deg": float(np.median(spread[view_count >= 2]))
        if (view_count >= 2).any()
        else 0.0,
        "timings_s": {k: round(v, 2) for k, v in t.items()} | {"geometric_detail": gc.timings},
        "params": _params_dict(p),
    }
    return ScoreResult(
        grid=grid,
        state=state,
        score=score,
        labels=labels,
        exposed=exposed,
        gauss_cov=gauss_cov,
        view_count=view_count,
        spread=spread,
        coverage=cov,
        report=report,
    )


def _params_dict(p: ScoreParams) -> dict:
    d = asdict(p)
    d["walk"]["exposure_heights"] = list(p.walk.exposure_heights)
    return d


def write_outputs(folder: Path, res: ScoreResult) -> None:
    out = Path(folder) / "coverage"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "gaussians.npz",
        view_count=res.view_count,
        spread_deg=res.spread.astype(np.float16),
        coverage=res.gauss_cov.astype(np.float16),
    )
    np.savez_compressed(
        out / "gaps.npz",
        grid=json.dumps(res.grid.to_dict()),
        state=res.state.reshape(res.grid.dims),
        score=res.score.reshape(res.grid.dims).astype(np.float16),
        labels=res.labels.reshape(res.grid.dims).astype(np.int16),
        exposed=res.exposed.reshape(res.grid.dims),
    )
    (out / "report.json").write_text(json.dumps(res.report, indent=2))
    path = Path(folder) / "scene.json"
    scene = Scene.load(path)  # reload: the agent may have edited it meanwhile
    scene.coverage = res.coverage
    scene.save(path)
