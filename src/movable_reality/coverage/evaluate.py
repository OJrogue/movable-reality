"""Precision and recall of the gap map against the synthetic ground truth.

Ground truth per voxel (from synthetic/ground_truth.npz, exact ray-box visibility of every
true surface sample):
  truth gap  the voxel holds a true surface sample a VR user would see (vr_visible) that
             the capture saw from fewer than `min_views` views;
  scored     the voxel holds any VR-visible true surface sample.

Matching uses a tolerance of `tol` voxels (26-neighbourhood), as in boundary F-scores: a
gap drawn one voxel in front of the surface it stands for still counts.
  precision = gap voxels within tol of a truth-gap voxel / gap voxels within tol of a
              scored voxel (gap voxels in empty space far from any true surface are not
              scored either way, and are reported as `gap_voxels_unscored`)
  recall    = truth-gap voxels within tol of a gap voxel / truth-gap voxels
Also reported per named hidden region (behind the sofa, under the desk, ...).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .voxels import VoxelGrid, dilate


def load_truth(folder: Path) -> tuple[dict, dict]:
    gt = np.load(Path(folder) / "synthetic" / "ground_truth.npz")
    meta = json.loads((Path(folder) / "synthetic" / "regions.json").read_text())
    return {k: gt[k] for k in gt.files}, meta


def truth_voxels(
    grid: VoxelGrid, points: np.ndarray, views: np.ndarray, vr: np.ndarray, min_views: int
) -> tuple[np.ndarray, np.ndarray]:
    """(truth_gap, scored) flat bool masks over the grid."""
    flat, inside = grid.index(points)
    sel = inside & vr
    bad = np.bincount(flat[sel & (views < min_views)], minlength=grid.n) > 0
    scored = np.bincount(flat[sel], minlength=grid.n) > 0
    return bad, scored


def _near(mask: np.ndarray, grid: VoxelGrid, tol: int) -> np.ndarray:
    return (dilate(mask.reshape(grid.dims), tol) if tol else mask.reshape(grid.dims)).ravel()


def precision_recall(
    grid: VoxelGrid, gap: np.ndarray, truth_gap: np.ndarray, scored: np.ndarray, tol: int = 1
) -> dict[str, float]:
    """`gap`, `truth_gap`, `scored`: flat bool masks over the grid."""
    near_truth = _near(truth_gap, grid, tol)
    near_scored = _near(scored, grid, tol)
    near_gap = _near(gap, grid, tol)
    tp_gap = int((gap & near_truth).sum())
    scored_gap = int((gap & near_scored).sum())
    hit_truth = int((truth_gap & near_gap).sum())
    n_truth = int(truth_gap.sum())
    prec = tp_gap / scored_gap if scored_gap else 0.0
    rec = hit_truth / n_truth if n_truth else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "gap_voxels": int(gap.sum()),
        "gap_voxels_scored": scored_gap,
        "gap_voxels_unscored": int(gap.sum()) - scored_gap,
        "truth_gap_voxels": n_truth,
    }


def evaluate_labels(
    folder: Path, grid: VoxelGrid, labels: np.ndarray, min_views: int = 3, tol: int = 1
) -> dict:
    truth, meta = load_truth(folder)
    pts, views, vr = truth["points"], truth["views"], truth["vr_visible"].astype(bool)
    bad, scored = truth_voxels(grid, pts, views, vr, min_views)
    gap = labels.ravel() >= 0
    out = {
        "min_views": min_views,
        "tolerance_voxels": tol,
        "overall": precision_recall(grid, gap, bad, scored, tol),
        "regions": {},
    }
    near_gap = _near(gap, grid, tol)
    for i, name in enumerate(meta["regions"].keys()):
        sel = (truth["region"] == i) & vr & (views < min_views)
        flat, inside = grid.index(pts[sel])
        vox = np.unique(flat[inside])
        out["regions"][name] = {
            "truth_gap_voxels": int(len(vox)),
            "recall": round(float(near_gap[vox].mean()), 3) if len(vox) else None,
        }
    out["truth"] = {
        "vr_visible_samples": int(vr.sum()),
        "vr_visible_under_min_views": int((vr & (views < min_views)).sum()),
        "scored_voxels": int(scored.sum()),
    }
    return out


def evaluate_scene(folder: Path, min_views: int = 3, tol: int = 1) -> dict:
    """Precision/recall from the files score_coverage wrote (coverage/gaps.npz)."""
    g = np.load(Path(folder) / "coverage" / "gaps.npz")
    grid = VoxelGrid.from_dict(json.loads(str(g["grid"])))
    return evaluate_labels(folder, grid, g["labels"].ravel().astype(np.int32), min_views, tol)
