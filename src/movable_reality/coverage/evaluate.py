"""Precision and recall of the gap map against the synthetic ground truth.

Ground truth per true surface sample comes from exact ray-box visibility
(synthetic/ground_truth.npz): its view count and angle spread over the training cameras,
turned into the same geometric score the estimator uses (geometry.geometric_score). A
sample is poorly covered when that exact score is below the gap threshold; the
views-only variant (fewer than 3 views) is reported too.

Per voxel:
  truth gap  holds a poorly covered true sample that a VR user would see (vr_visible);
  scored     holds any VR-visible true sample.
Matching uses a tolerance of `tol` voxels (26-neighbourhood), as in boundary F-scores:
  precision = gap voxels within tol of a truth-gap voxel / gap voxels within tol of a
              scored voxel (gap voxels in empty space away from any true surface are not
              scored either way; reported as `gap_voxels_unscored`)
  recall    = truth-gap voxels within tol of a gap voxel / truth-gap voxels
Also reported per named hidden region (behind the sofa, under the desk, ...).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import geometric_score
from .voxels import VoxelGrid, dilate


def load_truth(folder: Path) -> tuple[dict, dict]:
    gt = np.load(Path(folder) / "synthetic" / "ground_truth.npz")
    meta = json.loads((Path(folder) / "synthetic" / "regions.json").read_text())
    return {k: gt[k] for k in gt.files}, meta


def poorly_covered(
    truth: dict, mode: str = "score", cov_threshold: float = 0.35, min_views: int = 3
) -> np.ndarray:
    """Per true sample: "score" uses the exact geometric score (views and spread),
    "views" only the exact view count."""
    if mode == "views":
        return truth["views"] < min_views
    exact = geometric_score(truth["views"], truth["spread"])
    return exact < cov_threshold


def truth_voxels(
    grid: VoxelGrid, points: np.ndarray, poor: np.ndarray, vr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(truth_gap, scored) flat bool masks over the grid."""
    flat, inside = grid.index(points)
    sel = inside & vr
    bad = np.bincount(flat[sel & poor], minlength=grid.n) > 0
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
    folder: Path,
    grid: VoxelGrid,
    labels: np.ndarray,
    tol: int = 1,
    cov_threshold: float = 0.35,
    min_views: int = 3,
) -> dict:
    truth, meta = load_truth(folder)
    pts, vr = truth["points"], truth["vr_visible"].astype(bool)
    gap = labels.ravel() >= 0
    near_gap = _near(gap, grid, tol)
    out: dict = {"tolerance_voxels": tol, "cov_threshold": cov_threshold, "min_views": min_views}
    for mode in ("score", "views"):
        poor = poorly_covered(truth, mode, cov_threshold, min_views)
        bad, scored = truth_voxels(grid, pts, poor, vr)
        out[mode] = precision_recall(grid, gap, bad, scored, tol)
    poor = poorly_covered(truth, "views", cov_threshold, min_views)
    out["regions"] = {}
    for i, name in enumerate(meta["regions"].keys()):
        flat, inside = grid.index(pts[(truth["region"] == i) & vr & poor])
        vox = np.unique(flat[inside])
        out["regions"][name] = {
            "truth_gap_voxels": int(len(vox)),
            "recall": round(float(near_gap[vox].mean()), 3) if len(vox) else None,
        }
    out["truth"] = {
        "vr_visible_samples": int(vr.sum()),
        "vr_visible_poor_score": int((vr & poorly_covered(truth, "score", cov_threshold)).sum()),
        "vr_visible_under_min_views": int((vr & poor).sum()),
    }
    return out


def evaluate_scene(folder: Path, tol: int = 1, cov_threshold: float = 0.35) -> dict:
    """Precision/recall from the files score_coverage wrote (coverage/gaps.npz)."""
    g = np.load(Path(folder) / "coverage" / "gaps.npz")
    grid = VoxelGrid.from_dict(json.loads(str(g["grid"])))
    labels = g["labels"].ravel().astype(np.int32)
    return evaluate_labels(folder, grid, labels, tol, cov_threshold)
