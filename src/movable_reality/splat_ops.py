"""Gaussian splat geometry for M4 (insert and object completion), numpy only.

Splats live in structured arrays with the standard 3DGS PLY layout (gsplat, Spark,
SAM 3D Objects' `save_ply`): x y z, nx ny nz, f_dc_0..2, f_rest_*, opacity (logit),
scale_0..2 (log), rot_0..3 (quaternion w x y z), plus the `generated` flag (uchar).

A similarity transform x' = s R x + t moves the centres, multiplies the scales by s and
left-multiplies the quaternions by the quaternion of R. Higher SH bands (f_rest) are not
rotated. SAM 3D Objects writes degree 0 only, so generated splats have none.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from .fsutil import replace_file

GENERATED = "generated"
CORE = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
TAIL = ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
# OpenCV camera (x right, y down, z forward) <-> PyTorch3D camera (x left, y up, z forward).
P3D_TO_CV = np.diag([-1.0, -1.0, 1.0])


# ---------------------------------------------------------------- io


def read_vertices(src: Path | bytes) -> np.ndarray:
    data = io.BytesIO(src) if isinstance(src, bytes | bytearray) else str(src)
    return PlyData.read(data, mmap=False)["vertex"].data.copy()


def standard(v: np.ndarray, generated: int | None = None) -> np.ndarray:
    """Copy into the standard layout (float32), dropping unknown properties. Missing normals
    become 0. `generated` (0 or 1) sets the flag for every splat, and None keeps an existing one."""
    names = v.dtype.names or ()
    rest = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    fields = CORE + rest + TAIL
    keep_flag = generated is not None or GENERATED in names
    dtype = [(f, "f4") for f in fields] + ([(GENERATED, "u1")] if keep_flag else [])
    out = np.zeros(len(v), dtype=dtype)
    for f in fields:
        if f in names:
            out[f] = v[f]
        elif f not in ("nx", "ny", "nz"):
            raise ValueError(f"PLY lacks property '{f}'")
    if generated is not None:
        out[GENERATED] = generated
    elif keep_flag:
        out[GENERATED] = v[GENERATED]
    return out


def write_vertices(path: Path, v: np.ndarray) -> None:
    """Binary little-endian PLY, written to a temp file and renamed (viewer never sees half)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".ply.tmp")
    PlyData([PlyElement.describe(v, "vertex")], byte_order="<").write(str(tmp))
    replace_file(tmp, path)


def xyz(v: np.ndarray) -> np.ndarray:
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


def opacity(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))


# ---------------------------------------------------------------- rotations


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (..., 4) quaternions in w x y z order."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Unit quaternion (w x y z, w >= 0) of a proper rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12))
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


# ---------------------------------------------------------------- transforms


def transform(v: np.ndarray, R: np.ndarray, t: np.ndarray, s: float = 1.0) -> np.ndarray:
    """Splats moved by x' = s R x + t (R proper rotation, s > 0). Returns a new array."""
    R = np.asarray(R, dtype=np.float64)
    if abs(np.linalg.det(R) - 1.0) > 1e-4:
        raise ValueError("R must be a proper rotation (det +1)")
    out = v.copy()
    p = s * xyz(v) @ R.T + np.asarray(t, dtype=np.float64)
    out["x"], out["y"], out["z"] = p.T.astype(np.float32)
    q = np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    q = quat_mul(rotmat_to_quat(R)[None], q)
    for i in range(4):
        out[f"rot_{i}"] = q[:, i].astype(np.float32)
    for i in range(3):
        out[f"scale_{i}"] = (np.asarray(v[f"scale_{i}"], np.float64) + math.log(s)).astype(
            np.float32
        )
    if "nx" in (v.dtype.names or ()):
        n = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float64) @ R.T
        out["nx"], out["ny"], out["nz"] = n.T.astype(np.float32)
    return out


def similarity(R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = s * np.asarray(R, dtype=np.float64)
    M[:3, 3] = t
    return M


def decompose(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(R, t, s) of a 4x4 similarity matrix (uniform scale, no reflection)."""
    M = np.asarray(M, dtype=np.float64)
    A = M[:3, :3]
    det = np.linalg.det(A)
    if det <= 0:
        raise ValueError("matrix has a reflection or is singular")
    s = det ** (1.0 / 3.0)
    R = A / s
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-4):
        raise ValueError("matrix is not a similarity transform")
    return R, M[:3, 3].copy(), s


def apply_matrix(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    R, t, s = decompose(M)
    return transform(v, R, t, s)


def sam3d_cam_from_local(
    rotation: list[float], translation: list[float], scale: list[float]
) -> tuple[np.ndarray, np.ndarray, float]:
    """SAM 3D Objects pose to OpenCV camera coordinates: (R, t, s) with x_cv = s R x + t.

    The pipeline composes a PyTorch3D Transform3d (row vectors): x_p3d = (s x) @ R(q) + t,
    i.e. x_p3d = s R(q)^T x + t in column form. Then x_cv = diag(-1, -1, 1) x_p3d."""
    Rq = quat_to_rotmat(np.asarray(rotation[:4], dtype=np.float64))
    s = float(np.mean(scale[:3])) if len(scale) >= 3 else float(scale[0])
    R = P3D_TO_CV @ Rq.T
    t = P3D_TO_CV @ np.asarray(translation[:3], dtype=np.float64)
    return R, t, s


# ---------------------------------------------------------------- placement


def robust_bounds(
    p: np.ndarray, alpha: np.ndarray | None = None, pct: float = 1.0, min_alpha: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile box of the opaque splats (all splats when too few are opaque)."""
    sel = p if alpha is None else p[alpha >= min_alpha]
    if len(sel) < 16:
        sel = p
    return np.percentile(sel, pct, axis=0), np.percentile(sel, 100 - pct, axis=0)


def upright_rotation(
    cam_from_local: np.ndarray, snap_deg: float = 40.0, up_axis: int | None = None
) -> tuple[np.ndarray, float]:
    """Rotation world_from_local for an object seen by a roughly level camera.

    World up (+Y) is taken as camera up (OpenCV -Y) expressed in the local frame. When a
    local axis lies within `snap_deg` of it, that axis is used instead, since SAM 3D Objects'
    canonical frame is axis-aligned for upright objects (local +Z up in the first live run,
    seen from 25 degrees above, 30 Oct). World +Z points from the object
    towards the camera (horizontally), so the object faces the default viewer.
    `up_axis` forces that local axis (sign chosen towards camera up): SAM 3D Objects'
    canonical frame is Z-up (upstream `_fix_gaussian_alignment`), so generated objects pass 2.
    Returns (R, angle in degrees between camera up and the chosen up axis)."""
    A = np.asarray(cam_from_local, dtype=np.float64)
    up = A.T @ np.array([0.0, -1.0, 0.0])
    back = A.T @ np.array([0.0, 0.0, -1.0])
    k = int(np.argmax(np.abs(up))) if up_axis is None else up_axis
    axis = np.zeros(3)
    axis[k] = np.sign(up[k]) or 1.0
    angle = math.degrees(math.acos(min(1.0, abs(up[k]))))
    if angle <= snap_deg or up_axis is not None:
        up = axis
    up /= np.linalg.norm(up)
    f = back - up * (back @ up)
    if np.linalg.norm(f) < 1e-6:  # looking straight down: any horizontal direction
        f = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 0.0, 1.0])
    f /= np.linalg.norm(f)
    x = np.cross(up, f)
    return np.stack([x, up, f]), angle


def fit_scale(extent: np.ndarray, target: tuple[float, float, float] | None) -> float:
    """Uniform scale that maps an object's world extent (x, y, z) to a real size
    (width, height, depth). Height counts as much as the larger footprint side, because a
    model's width and depth may be swapped relative to the estimate."""
    if target is None:
        return 1.0
    w, h, d = (float(a) for a in target)
    ex, ey, ez = (max(float(a), 1e-6) for a in extent)
    by_height = h / ey
    by_footprint = max(w, d) / max(ex, ez)
    return math.sqrt(by_height * by_footprint)


def floor_placement(
    p: np.ndarray, alpha: np.ndarray | None, x: float, z: float, scale: float = 1.0
) -> np.ndarray:
    """Translation t so that scale * p + t stands on y = 0 with its box centre at (x, z)."""
    lo, hi = robust_bounds(p * scale, alpha)
    return np.array([x - (lo[0] + hi[0]) / 2, -lo[1], z - (lo[2] + hi[2]) / 2])


# ---------------------------------------------------------------- registration


def umeyama(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = True
) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares (R, t, s) with dst ~ s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = (a**2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var) if with_scale and var > 0 else 1.0
    return R, mu_d - s * R @ mu_s, s


def nearest(query: np.ndarray, ref: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Index into `ref` and distance of the nearest neighbour of every query point."""
    idx = np.empty(len(query), dtype=np.int64)
    dist = np.empty(len(query))
    rr = (ref**2).sum(1)
    for i in range(0, len(query), chunk):
        q = query[i : i + chunk]
        d2 = (q**2).sum(1)[:, None] - 2 * q @ ref.T + rr[None]
        j = np.argmin(d2, axis=1)
        idx[i : i + chunk] = j
        dist[i : i + chunk] = np.sqrt(np.maximum(d2[np.arange(len(q)), j], 0))
    return idx, dist


def icp(
    full: np.ndarray,
    partial: np.ndarray,
    iters: int = 20,
    scale_range: tuple[float, float] = (0.8, 1.25),
    trim: float = 0.9,
) -> tuple[np.ndarray, float, float]:
    """Register a complete model `full` to a partial scan `partial` (both N x 3).

    Every scanned point is matched to its nearest model point (the scan is a subset of the
    object's surface, so this direction has no false matches from unseen sides). The worst
    (1 - trim) matches are dropped and a similarity is fitted. The accumulated scale stays in
    `scale_range`. RMS is over the best `trim` share of the scan, so floaters do not decide
    it. Returns (best 4x4 M with partial ~ M full, trimmed RMS before, trimmed RMS of M)."""

    def trimmed_rms(pts: np.ndarray) -> float:
        d = nearest(partial, pts)[1]
        return float(np.sqrt(np.mean(np.sort(d)[: max(1, int(len(d) * trim))] ** 2)))

    M = np.eye(4)
    cur = full.copy()
    rms0 = rms = trimmed_rms(cur)
    best = (M.copy(), rms)
    for _ in range(iters):
        j, d = nearest(partial, cur)
        keep = d <= np.quantile(d, trim)
        R, t, s = umeyama(cur[j[keep]], partial[keep])
        total = (np.linalg.det(M[:3, :3]) ** (1 / 3)) * s
        lo, hi = scale_range
        if not lo <= total <= hi:
            s = float(np.clip(total, lo, hi)) / (np.linalg.det(M[:3, :3]) ** (1 / 3))
            t = partial[keep].mean(0) - s * R @ cur[j[keep]].mean(0)
        M = similarity(R, t, s) @ M
        cur = cur @ (s * R).T + t
        new = trimmed_rms(cur)
        if new < best[1]:
            best = (M.copy(), new)
        if rms - new < 1e-5:
            break
        rms = new
    return best[0], rms0, best[1]


def subsample(p: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(p) <= n:
        return p
    return p[np.random.default_rng(seed).choice(len(p), n, replace=False)]
