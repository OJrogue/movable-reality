"""Read and write COLMAP models (cameras, images, points3D), binary or text.

MapAnything's `demo_colmap.py` exports `recon/sparse/{cameras,images,points3D}.bin`; gsplat
also accepts `sparse/0/`. Only what coverage needs is kept: pinhole intrinsics (distortion
parameters are dropped; coverage works on a coarse depth map where they do not matter),
poses and image names. Format reference: https://colmap.github.io/format.html
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .cameras import Camera, qvec_to_rotmat, rotmat_to_qvec

# model id -> (name, number of params, index of fx, fy, cx, cy; fy == fx for SIMPLE_*)
_MODELS: dict[int, tuple[str, int, tuple[int, int, int, int]]] = {
    0: ("SIMPLE_PINHOLE", 3, (0, 0, 1, 2)),
    1: ("PINHOLE", 4, (0, 1, 2, 3)),
    2: ("SIMPLE_RADIAL", 4, (0, 0, 1, 2)),
    3: ("RADIAL", 5, (0, 0, 1, 2)),
    4: ("OPENCV", 8, (0, 1, 2, 3)),
    5: ("OPENCV_FISHEYE", 8, (0, 1, 2, 3)),
    6: ("FULL_OPENCV", 12, (0, 1, 2, 3)),
    7: ("FOV", 5, (0, 1, 2, 3)),
    8: ("SIMPLE_RADIAL_FISHEYE", 4, (0, 0, 1, 2)),
    9: ("RADIAL_FISHEYE", 5, (0, 0, 1, 2)),
    10: ("THIN_PRISM_FISHEYE", 12, (0, 1, 2, 3)),
}
_MODEL_IDS = {name: mid for mid, (name, _, _) in _MODELS.items()}


def find_sparse_dir(recon: Path) -> Path:
    """recon/sparse/0 or recon/sparse (or recon itself) holding cameras + images."""
    recon = Path(recon)
    for cand in (recon / "sparse" / "0", recon / "sparse", recon):
        for ext in ("bin", "txt"):
            if (cand / f"cameras.{ext}").exists() and (cand / f"images.{ext}").exists():
                return cand
    raise FileNotFoundError(f"no COLMAP model (cameras + images .bin/.txt) under {recon}")


def _intrinsics(model_id: int, params: list[float]) -> np.ndarray:
    _, n, (ifx, ify, icx, icy) = _MODELS[model_id]
    if len(params) != n:
        raise ValueError(f"camera model {model_id} expects {n} params, got {len(params)}")
    return np.array(
        [[params[ifx], 0.0, params[icx]], [0.0, params[ify], params[icy]], [0.0, 0.0, 1.0]]
    )


def _read_cameras_bin(path: Path) -> dict[int, tuple[int, int, np.ndarray]]:
    out = {}
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            n_params = _MODELS[model_id][1]
            params = list(struct.unpack(f"<{n_params}d", f.read(8 * n_params)))
            out[cam_id] = (int(w), int(h), _intrinsics(model_id, params))
    return out


def _read_images_bin(path: Path) -> list[tuple[int, str, np.ndarray, np.ndarray, int]]:
    out = []
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            img_id, qw, qx, qy, qz, tx, ty, tz, cam_id = struct.unpack("<idddddddi", f.read(64))
            name = bytearray()
            while (c := f.read(1)) != b"\x00":
                if not c:
                    raise ValueError(f"truncated image name in {path}")
                name += c
            (n_pts,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * n_pts, 1)  # x, y (double), point3D_id (int64)
            q, t = np.array([qw, qx, qy, qz]), np.array([tx, ty, tz])
            out.append((img_id, name.decode("utf-8"), q, t, cam_id))
    return out


def _read_cameras_txt(path: Path) -> dict[int, tuple[int, int, np.ndarray]]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        cam_id, model, w, h = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        params = [float(p) for p in parts[4:]]
        out[cam_id] = (w, h, _intrinsics(_MODEL_IDS[model], params))
    return out


def _read_images_txt(path: Path) -> list[tuple[int, str, np.ndarray, np.ndarray, int]]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    out = []
    for line in lines[::2]:  # every image line is followed by its 2D points line
        p = line.split()
        q = np.array([float(v) for v in p[1:5]])
        t = np.array([float(v) for v in p[5:8]])
        out.append((int(p[0]), " ".join(p[9:]), q, t, int(p[8])))
    return out


def read_model(recon: Path) -> list[Camera]:
    """All registered images as Cameras (cam_from_recon), sorted by image name."""
    sparse = find_sparse_dir(recon)
    if (sparse / "cameras.bin").exists():
        cams, images = (
            _read_cameras_bin(sparse / "cameras.bin"),
            _read_images_bin(sparse / "images.bin"),
        )
    else:
        cams, images = (
            _read_cameras_txt(sparse / "cameras.txt"),
            _read_images_txt(sparse / "images.txt"),
        )
    out = []
    for _, name, q, t, cam_id in sorted(images, key=lambda r: r[1]):
        w, h, K = cams[cam_id]
        out.append(Camera(name=name, width=w, height=h, K=K, R=qvec_to_rotmat(q), t=t))
    return out


def write_model(
    sparse: Path,
    cameras: list[Camera],
    points: np.ndarray | None = None,
    colors: np.ndarray | None = None,
) -> None:
    """Write PINHOLE cameras, images (no 2D points) and optional points3D (empty tracks)
    as COLMAP binary files. One COLMAP camera per distinct intrinsics."""
    sparse = Path(sparse)
    sparse.mkdir(parents=True, exist_ok=True)
    intr: dict[tuple, int] = {}
    cam_ids = []
    for cam in cameras:
        key = (cam.width, cam.height, *np.round(cam.K[[0, 1, 0, 1], [0, 1, 2, 2]], 9))
        cam_ids.append(intr.setdefault(key, len(intr) + 1))
    with (sparse / "cameras.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(intr)))
        for (w, h, fx, fy, cx, cy), cid in intr.items():
            f.write(struct.pack("<iiQQ", cid, _MODEL_IDS["PINHOLE"], w, h))
            f.write(struct.pack("<4d", fx, fy, cx, cy))
    with (sparse / "images.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for i, (cam, cid) in enumerate(zip(cameras, cam_ids, strict=True), start=1):
            q = rotmat_to_qvec(cam.R)
            f.write(struct.pack("<idddddddi", i, *q, *np.asarray(cam.t, float), cid))
            f.write(cam.name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))
    pts = np.zeros((0, 3)) if points is None else np.asarray(points, dtype=np.float64)
    rgb = (
        np.full((len(pts), 3), 128, np.uint8)
        if colors is None
        else np.clip(np.asarray(colors) * 255, 0, 255).astype(np.uint8)
    )
    with (sparse / "points3D.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for i, (p, c) in enumerate(zip(pts, rgb, strict=True), start=1):
            f.write(struct.pack("<Q3d3BdQ", i, *p, *c, 0.0, 0))
