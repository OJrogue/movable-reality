"""Object completion (stage 6). Pick a view and register SAM 3D Objects' output to the scan.

Pure functions. scripts/complete_objects.py does the I/O and the Modal call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import splat_ops as so


def best_frame(masks_dir: Path, value: int, border_penalty: float = 0.5) -> tuple[int, int]:
    """Frame index and pixel count of the largest mask with label `value` in
    ids/masks/{i:06d}.png. Masks that touch the image border (object cut off) count half."""
    best = (-1, 0, 0.0)
    for path in sorted(Path(masks_dir).glob("[0-9]*.png")):
        m = np.asarray(Image.open(path)) == value
        n = int(m.sum())
        if n == 0:
            continue
        touches = m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
        score = n * (border_penalty if touches else 1.0)
        if score > best[2]:
            best = (int(path.stem), n, score)
    if best[0] < 0:
        raise ValueError(f"label {value} appears in no mask under {masks_dir}")
    return best[0], best[1]


def coarse_scale(full: np.ndarray, scan: np.ndarray, scale_range=(0.5, 2.0)) -> float:
    """Uniform scale from the median ratio of the box extents. The median ignores the one
    axis that a partial scan cut short."""
    flo, fhi = so.robust_bounds(full)
    slo, shi = so.robust_bounds(scan)
    return float(np.clip(np.median((shi - slo) / np.maximum(fhi - flo, 1e-6)), *scale_range))


def coarse_starts(full: np.ndarray, scan: np.ndarray, s: float) -> list[np.ndarray]:
    """Candidate 4x4 similarities that put `full` (world frame, +Y up, already roughly
    oriented) over `scan`. Each uses scale `s` and matches box bottoms in y (both stand on
    the floor), and in x and z it matches box centres, min faces or max faces. A partial scan
    has its true extent on the sides it saw, so one of the face pairings is right."""
    flo, fhi = so.robust_bounds(full * s)
    slo, shi = so.robust_bounds(scan)
    picks = []
    for ax in (0, 2):
        picks.append(
            [
                (slo[ax] + shi[ax]) / 2 - (flo[ax] + fhi[ax]) / 2,
                slo[ax] - flo[ax],
                shi[ax] - fhi[ax],
            ]
        )
    return [
        so.similarity(np.eye(3), [dx, slo[1] - flo[1], dz], s) for dx in picks[0] for dz in picks[1]
    ]


def clean_scan(p: np.ndarray, alpha: np.ndarray | None = None, k: float = 4.0) -> np.ndarray:
    """Opaque scan points without floaters. Per axis, it drops points more than k robust standard
    deviations (1.4826 MAD) from the median."""
    if alpha is not None and (alpha >= 0.5).sum() >= 100:
        p = p[alpha >= 0.5]
    med = np.median(p, axis=0)
    mad = 1.4826 * np.median(np.abs(p - med), axis=0)
    keep = (np.abs(p - med) <= k * np.maximum(mad, 1e-3)).all(axis=1)
    return p[keep] if keep.sum() >= 50 else p


def register(
    local: np.ndarray,
    recon_from_local: np.ndarray,
    scan_recon: np.ndarray,
    world_from_recon: np.ndarray,
    n_points: int = 2000,
    scan_alpha: np.ndarray | None = None,
    max_scale_change: float = 0.25,
) -> tuple[np.ndarray, dict]:
    """Recon-frame splats of the completed object, aligned to the scanned object.

    `recon_from_local` is the first guess from SAM 3D's pose and the frame's camera, and it
    fixes orientation. Scale comes from the boxes, unless it departs from the (metric) guess
    by more than `max_scale_change`. Floaters are dropped from the scan first (`clean_scan`).
    ICP on the splat centres (scan matched to the model, trimmed, similarity) runs from the
    pose guess and from `coarse_starts`, and the start with the lowest trimmed RMS wins.
    Returns splats with `generated` = 1 and a report (scale, ICP RMS before and after, m,
    and the final recon_from_local matrix)."""
    W = np.asarray(world_from_recon, dtype=np.float64)
    W0 = W @ recon_from_local
    alpha = so.opacity(local)
    full = so.xyz(local)[alpha >= 0.5] if (alpha >= 0.5).sum() >= 100 else so.xyz(local)
    full_w = so.subsample(full @ W0[:3, :3].T + W0[:3, 3], n_points)
    scan_w = clean_scan(scan_recon @ W[:3, :3].T + W[:3, 3], scan_alpha)
    scan_w = so.subsample(scan_w, n_points, 1)
    s = coarse_scale(full_w, scan_w)
    if abs(s - 1.0) > max_scale_change:  # boxes disagree with the metric pose guess: trust it
        s = 1.0
    c = full_w.mean(0)  # first start: the pose guess itself, rescaled about its centroid
    starts = [so.similarity(np.eye(3), c - s * c, s), *coarse_starts(full_w, scan_w, s)]
    best = None
    for C in starts:
        moved = full_w @ C[:3, :3].T + C[:3, 3]
        M_icp, rms0, rms = so.icp(moved, scan_w)
        if best is None or rms < best[3]:
            best = (M_icp @ C, rms0, len(starts), rms)
    M, rms0, n_starts, rms = best
    recon_final = np.linalg.inv(W) @ M @ W0
    out = so.apply_matrix(so.standard(local, generated=1), recon_final)
    return out, {
        "coarse_scale": round(s, 4),
        "total_scale": round(float(so.decompose(M)[2]), 4),
        "starts": n_starts,
        "icp_rms_before_m": round(rms0, 4),
        "icp_rms_after_m": round(rms, 4),
        "recon_from_local": recon_final.tolist(),  # the mesh of the same run uses it too
    }
