"""Gaussian splat PLY io and per-Gaussian geometry (standard 3DGS layout, as gsplat writes it).

Properties: x y z, f_dc_0..2, f_rest_*, opacity (logit), scale_0..2 (log), rot_0..3 (wxyz).
Reading keeps the raw structured array, so a subset can be written back with every property
(including extras such as `generated`) unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814


@dataclass
class Gaussians:
    means: np.ndarray  # (N, 3)
    quats: np.ndarray  # (N, 4) wxyz, not necessarily normalised
    log_scales: np.ndarray  # (N, 3)
    opacity_logits: np.ndarray  # (N,)
    f_dc: np.ndarray  # (N, 3) degree-0 SH coefficients

    def __len__(self) -> int:
        return len(self.means)

    def subset(self, idx: np.ndarray) -> Gaussians:
        return Gaussians(
            self.means[idx],
            self.quats[idx],
            self.log_scales[idx],
            self.opacity_logits[idx],
            self.f_dc[idx],
        )

    @property
    def opacities(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.opacity_logits))

    @property
    def rgb(self) -> np.ndarray:
        """Degree-0 colour as 3DGS renders it (SH DC + 0.5, clamped at 0)."""
        return np.maximum(SH_C0 * self.f_dc + 0.5, 0.0)

    def covariances(self) -> np.ndarray:
        """(N, 3, 3) world covariances R S^2 R^T."""
        r = quat_to_rotmat(self.quats)
        s2 = np.exp(2.0 * self.log_scales)
        return np.einsum("nij,nj,nkj->nik", r, s2, r)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz quaternions to (N, 3, 3) rotation matrices; normalises first."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def rotmat_to_quat(r: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices to (N, 4) wxyz unit quaternions with w >= 0.

    Shepperd's method: divide by the largest of the four candidates, so rotations near
    180 degrees stay accurate.
    """
    r = np.asarray(r, dtype=np.float64).reshape(-1, 3, 3)
    m = {(i, j): r[:, i, j] for i in range(3) for j in range(3)}
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    cand = np.stack([tr, m[0, 0], m[1, 1], m[2, 2]], 1)
    case = np.argmax(cand, 1)
    q = np.empty((len(r), 4))
    for k in range(4):
        sel = case == k
        if not sel.any():
            continue
        g = {key: val[sel] for key, val in m.items()}
        # Antisymmetric parts carry w * (x, y, z), symmetric parts carry products of x, y, z.
        wx, wy, wz = g[2, 1] - g[1, 2], g[0, 2] - g[2, 0], g[1, 0] - g[0, 1]
        xy, xz, yz = g[0, 1] + g[1, 0], g[0, 2] + g[2, 0], g[1, 2] + g[2, 1]
        if k == 0:
            s = 2.0 * np.sqrt(1.0 + tr[sel])
            q[sel] = np.stack([s / 4, wx / s, wy / s, wz / s], 1)
        elif k == 1:
            s = 2.0 * np.sqrt(1.0 + g[0, 0] - g[1, 1] - g[2, 2])
            q[sel] = np.stack([wx / s, s / 4, xy / s, xz / s], 1)
        elif k == 2:
            s = 2.0 * np.sqrt(1.0 - g[0, 0] + g[1, 1] - g[2, 2])
            q[sel] = np.stack([wy / s, xy / s, s / 4, yz / s], 1)
        else:
            s = 2.0 * np.sqrt(1.0 - g[0, 0] - g[1, 1] + g[2, 2])
            q[sel] = np.stack([wz / s, xz / s, yz / s, s / 4], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    return q


# ---------------------------------------------------------------- io


def read_ply(path: Path) -> PlyData:
    return PlyData.read(str(path), mmap=False)


def vertices(ply: PlyData) -> np.ndarray:
    return ply["vertex"].data


def gaussians_from_vertices(v: np.ndarray) -> Gaussians:
    def cols(*names: str) -> np.ndarray:
        return np.stack([np.asarray(v[n], dtype=np.float64) for n in names], axis=1)

    return Gaussians(
        means=cols("x", "y", "z"),
        quats=cols("rot_0", "rot_1", "rot_2", "rot_3"),
        log_scales=cols("scale_0", "scale_1", "scale_2"),
        opacity_logits=np.asarray(v["opacity"], dtype=np.float64),
        f_dc=cols("f_dc_0", "f_dc_1", "f_dc_2"),
    )


def write_subset(path: Path, ply: PlyData, keep: np.ndarray) -> int:
    """Write the vertices of `ply` selected by the bool mask `keep` (all properties kept)."""
    data = vertices(ply)[keep]
    elements = [PlyElement.describe(data, "vertex")]
    elements += [e for e in ply.elements if e.name != "vertex"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".ply.tmp")
    PlyData(
        elements,
        text=False,
        byte_order=ply.byte_order if ply.byte_order != "=" else "<",
        comments=list(ply.comments),
        obj_info=list(ply.obj_info),
    ).write(str(tmp))
    tmp.replace(path)
    return len(data)


def write_gaussians(path: Path, g: Gaussians, sh_rest: int = 9) -> None:
    """Standard 3DGS PLY (float32, binary little endian); higher SH bands written as zeros."""
    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    fields += [f"f_rest_{i}" for i in range(sh_rest)]
    fields += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros(len(g), dtype=[(f, "<f4") for f in fields])
    data["x"], data["y"], data["z"] = g.means.T
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = g.f_dc.T
    data["opacity"] = g.opacity_logits
    data["scale_0"], data["scale_1"], data["scale_2"] = g.log_scales.T
    data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = g.quats.T
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False, byte_order="<").write(str(path))


def _xyz_bits(v: np.ndarray) -> np.ndarray:
    keys = np.empty(len(v), dtype=[("x", "<u4"), ("y", "<u4"), ("z", "<u4")])
    for n in "xyz":
        keys[n] = np.ascontiguousarray(v[n], dtype="<f4").view("<u4")
    return keys


def match_positions(v: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Bool mask over the vertices `v` whose float32 x, y, z equal those of a vertex in `targets`.

    Finds the same Gaussians in two PLYs written from one source (scene.ply and
    background.ply) without relying on their order. Exact bit match, no tolerance.
    """
    if len(targets) == 0:
        return np.zeros(len(v), dtype=bool)
    return np.isin(_xyz_bits(v), _xyz_bits(targets))
