"""Refine MapAnything's cameras with classical SfM (COLMAP via pycolmap, no learned weights).

    python -m movable_reality.sfm --frames frames --ma recon_ma --out recon [--fx .. --cy ..]

MapAnything gives metric cameras, but on blender01 (31 Oct) its poses were 0.29 m off after a
Sim(3) fit, which blurs every splat. SIFT + incremental mapping on the same 152 frames gave 2.9
mm ATE against GT (3.3 mm after the fit into the MapAnything frame) and 0.08 deg median
rotation error. COLMAP has no metric scale, so its model is fitted to MapAnything's camera
centres by one Sim(3) (Umeyama). The output lands in recon/ in MapAnything's own convention
(model-resolution intrinsics), so every later stage reads it unchanged. If too few frames
register, MapAnything's model is copied through instead.
Needs pycolmap 4 (API of 4.2.1). Version 3.13, which gsplat's requirements install in the
training image, registered none of the 152 blender01 frames, so this runs in its own CPU image.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from .coverage.cameras import Camera
from .coverage.colmap import find_sparse_dir, read_model, write_model
from .recon import full_res_camera, model_res_camera
from .splat_ops import similarity, umeyama

MIN_REGISTERED = 0.5  # fraction of frames SfM must register; the rest are dropped


def fit_to_reference(cams: list[Camera], ref: list[Camera]) -> tuple[np.ndarray, dict]:
    """4x4 similarity ref_from_sfm that maps SfM camera centres onto the reference ones
    (matched by image name), plus the fit residuals in reference units."""
    by_name = {c.name: c for c in ref}
    pairs = [(c.center, by_name[c.name].center) for c in cams if c.name in by_name]
    if len(pairs) < 3:
        raise ValueError(f"only {len(pairs)} cameras shared with the reference")
    src, dst = (np.array(x) for x in zip(*pairs, strict=True))
    R, t, s = umeyama(src, dst)
    res = np.linalg.norm((s * src @ R.T + t) - dst, axis=1)
    # One refit without the pairs far off the median, so a few bad reference poses do not
    # bend the whole fit. The residuals reported are those of every pair.
    keep = res <= 3 * np.median(res) + 1e-9
    if 3 <= keep.sum() < len(res):
        R, t, s = umeyama(src[keep], dst[keep])
        res = np.linalg.norm((s * src @ R.T + t) - dst, axis=1)
    stats = {
        "shared": len(pairs),
        "trimmed": int((~keep).sum()),
        "scale": s,
        "residual_rms": float(np.sqrt((res**2).mean())),
        "residual_max": float(res.max()),
    }
    return similarity(R, t, s), stats


def run_colmap(frames: Path, work: Path, K: np.ndarray | None, fix_k: bool):
    """SIFT features, sequential matching, incremental mapping. Returns the largest model."""
    import pycolmap  # noqa: PLC0415 - pycolmap 4 (sfm image in modal_jobs/reconstruct.py)

    print("pycolmap", pycolmap.__version__, flush=True)

    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    db.unlink(missing_ok=True)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    if K is not None:
        reader.camera_params = f"{K[0, 0]},{K[1, 1]},{K[0, 2]},{K[1, 2]}"
    extract = pycolmap.FeatureExtractionOptions()
    extract.sift.max_num_features = 8192
    # Plain painted walls and ceilings have little contrast: a lower peak threshold keeps
    # their weak corners (default 0.0067).
    extract.sift.peak_threshold = 0.002
    pycolmap.extract_features(
        db, frames, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
        extraction_options=extract,
    )  # fmt: skip
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = 15
    pairing.quadratic_overlap = True  # also frames 16, 32, 64 ... apart: closes walk loops
    pycolmap.match_sequential(db, pairing_options=pairing)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_num_matches = 10  # default 15
    opts.mapper.abs_pose_min_num_inliers = 15  # default 30
    if fix_k:
        opts.ba_refine_focal_length = False
        opts.ba_refine_principal_point = False
        opts.ba_refine_extra_params = False
    models = pycolmap.incremental_mapping(db, frames, work, options=opts)
    if not models:
        return None
    return max(models.values(), key=lambda r: r.num_reg_images())


def to_cameras(rec) -> tuple[list[Camera], np.ndarray, np.ndarray]:
    """pycolmap Reconstruction -> Cameras (full resolution), point xyz, point rgb in 0..1."""
    cams = []
    for img in rec.images.values():
        if not img.has_pose:
            continue
        cam = rec.cameras[img.camera_id]
        Rt = img.cam_from_world().matrix()
        cams.append(
            Camera(img.name, cam.width, cam.height, cam.calibration_matrix(), Rt[:, :3], Rt[:, 3])
        )
    pts = rec.points3D.values()
    xyz = np.array([p.xyz for p in pts]).reshape(-1, 3)
    rgb = np.array([p.color for p in pts]).reshape(-1, 3) / 255.0
    return sorted(cams, key=lambda c: c.name), xyz, rgb


def refine(frames: Path, ma: Path, out: Path, K: np.ndarray | None) -> dict:
    t0 = time.time()
    ma_cams = read_model(ma)
    small_w, small_h = ma_cams[0].width, ma_cams[0].height
    from PIL import Image  # noqa: PLC0415

    W, H = Image.open(frames / ma_cams[0].name).size
    fix_k = K is not None
    if K is None:  # MapAnything's self-calibrated camera as the starting point, refined by BA
        K = full_res_camera(ma_cams[0], W, H).K
    report: dict = {"frames": len(ma_cams), "fixed_intrinsics": fix_k}
    shutil.rmtree(out, ignore_errors=True)
    with tempfile.TemporaryDirectory() as tmp:  # SQLite database on local disk, not a volume
        rec = run_colmap(frames, Path(tmp), K, fix_k)
    n_reg = rec.num_reg_images() if rec else 0
    report["registered"] = n_reg
    if n_reg < MIN_REGISTERED * len(ma_cams):
        report["used"] = "mapanything (too few frames registered)"
        shutil.copytree(find_sparse_dir(ma), out / "sparse")
    else:
        report["mean_reprojection_px"] = float(rec.compute_mean_reprojection_error())
        cams, xyz, rgb = to_cameras(rec)
        M, fit = fit_to_reference(cams, ma_cams)
        report["fit_to_mapanything"] = fit
        xyz = xyz @ M[:3, :3].T + M[:3, 3]
        moved = [model_res_camera(c.in_frame(M), small_w, small_h) for c in cams]
        # Frames SfM could not register are left out of training. Placing them with
        # MapAnything's pose relative to the nearest registered frame was tried on the
        # blender01 ceiling turn (31 Oct): up to 1.6 m and 178 deg off, worse than no frame.
        have = {c.name for c in cams}
        report["dropped"] = [c.name for c in ma_cams if c.name not in have]
        write_model(out / "sparse", moved, xyz, rgb)
        report["used"] = "sfm"
        report["K_full_res"] = cams[0].K.tolist()
        report["points"] = len(xyz)
    if (ma / "images").exists():
        (out / "images").symlink_to(Path("..") / ma.name / "images")
    for name in ("mapanything.json",):
        if (ma / name).exists():
            shutil.copyfile(ma / name, out / name)
    report["sfm_s"] = round(time.time() - t0, 1)
    (out / "sfm.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--ma", type=Path, required=True, help="MapAnything's COLMAP model")
    ap.add_argument("--out", type=Path, required=True)
    for k in ("fx", "fy", "cx", "cy"):
        ap.add_argument(f"--{k}", type=float, default=None)
    a = ap.parse_args()
    K = None if a.fx is None else np.array([[a.fx, 0, a.cx], [0, a.fy, a.cy], [0, 0, 1.0]])
    refine(a.frames, a.ma, a.out, K)


if __name__ == "__main__":
    main()
