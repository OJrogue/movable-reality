"""3DGS PLY files: read the fields coverage needs, write synthetic splats, and read or write
the per-splat `generated` flag (PLY property `generated`, uchar 0 or 1, see AGENTS.md).

Standard 3DGS layout (gsplat `--save_ply`, Spark): x y z, nx ny nz, f_dc_0..2, f_rest_*,
opacity (logit), scale_0..2 (log), rot_0..3 (quaternion w x y z).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from ..fsutil import replace_file

SH_C0 = 0.28209479177387814
GENERATED = "generated"


@dataclass
class Splats:
    means: np.ndarray  # (N, 3) float32, PLY frame
    opacity: np.ndarray  # (N,) float32 in [0, 1]
    scales: np.ndarray  # (N, 3) float32, metres
    rgb: np.ndarray  # (N, 3) float32 base colour from the DC term, [0, 1]
    generated: np.ndarray  # (N,) uint8, 0 = scanned, 1 = generated

    def __len__(self) -> int:
        return len(self.means)

    @staticmethod
    def concat(parts: list[Splats]) -> Splats:
        return Splats(
            *(np.concatenate([getattr(p, f) for p in parts]) for f in Splats.__dataclass_fields__)
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float32)))).astype(np.float32)


def read_splats(path: Path) -> Splats:
    v = PlyData.read(str(path), mmap=False)["vertex"].data
    names = v.dtype.names or ()
    n = len(v)
    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    opacity = _sigmoid(v["opacity"]) if "opacity" in names else np.ones(n, np.float32)
    if "scale_0" in names:
        scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], axis=1)).astype(np.float32)
    else:
        scales = np.full((n, 3), 0.01, np.float32)
    if "f_dc_0" in names:
        dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float32)
        rgb = np.clip(dc * SH_C0 + 0.5, 0.0, 1.0)
    else:
        rgb = np.full((n, 3), 0.5, np.float32)
    gen = v[GENERATED].astype(np.uint8) if GENERATED in names else np.zeros(n, np.uint8)
    return Splats(means=means, opacity=opacity, scales=scales, rgb=rgb, generated=gen)


def write_splats(
    path: Path,
    means: np.ndarray,
    rgb: np.ndarray,
    scales: np.ndarray | float = 0.02,
    opacity: np.ndarray | float = 0.95,
    generated: np.ndarray | None = None,
    quats: np.ndarray | None = None,
) -> None:
    """Write a degree-0 3DGS PLY (log scales, logit opacity, rotations as w x y z quaternions,
    identity when not given). The `generated` property is written only when given."""
    n = len(means)
    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    fields += ["scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    dtype = [(f, "f4") for f in fields] + ([(GENERATED, "u1")] if generated is not None else [])
    data = np.zeros(n, dtype=dtype)
    data["x"], data["y"], data["z"] = np.asarray(means, np.float32).T
    dc = (np.asarray(rgb, np.float32) - 0.5) / SH_C0
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = dc.T
    op = np.clip(np.broadcast_to(np.asarray(opacity, np.float32), (n,)), 1e-4, 1 - 1e-4)
    data["opacity"] = np.log(op / (1 - op))
    sc = np.broadcast_to(np.asarray(scales, np.float32), (n,) if np.ndim(scales) <= 1 else (n, 3))
    sc = np.repeat(sc[:, None], 3, axis=1) if sc.ndim == 1 else sc
    for i in range(3):
        data[f"scale_{i}"] = np.log(sc[:, i])
    if quats is None:
        data["rot_0"] = 1.0
    else:
        q = np.asarray(quats, np.float32)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        for i in range(4):
            data[f"rot_{i}"] = q[:, i]
    if generated is not None:
        data[GENERATED] = np.asarray(generated, np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


# ---------------------------------------------------------------- generated flag


def read_generated(path: Path) -> np.ndarray:
    """Per-splat generated flag (uint8). All zeros when the PLY has no such property."""
    v = PlyData.read(str(path), mmap=False)["vertex"].data
    if GENERATED in (v.dtype.names or ()):
        return v[GENERATED].astype(np.uint8)
    return np.zeros(len(v), np.uint8)


def write_generated(path: Path, flags: np.ndarray, out: Path | None = None) -> Path:
    """Set the `generated` property of every splat (adds it if missing), keeping all other
    properties and their order. Writes atomically to `out` (default: in place)."""
    path = Path(path)
    out = Path(out) if out is not None else path
    ply = PlyData.read(str(path), mmap=False)
    v = ply["vertex"].data
    flags = np.asarray(flags).astype(np.uint8)
    if flags.shape != (len(v),):
        raise ValueError(f"expected {len(v)} flags, got shape {flags.shape}")
    if not np.isin(flags, (0, 1)).all():
        raise ValueError("generated flags must be 0 or 1")
    names = v.dtype.names or ()
    if GENERATED in names:
        data = v.copy()
    else:
        data = np.empty(len(v), dtype=v.dtype.descr + [(GENERATED, "u1")])
        for name in names:
            data[name] = v[name]
    data[GENERATED] = flags
    others = [e for e in ply.elements if e.name != "vertex"]
    new = PlyData(
        [PlyElement.describe(data, "vertex"), *others],
        text=ply.text,
        byte_order=ply.byte_order,
        comments=list(ply.comments),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    new.write(str(tmp))
    replace_file(tmp, out)
    return out


def count_generated(paths: list[Path]) -> int:
    """Number of splats flagged generated over all existing PLYs in `paths`."""
    return int(sum(int(read_generated(p).sum()) for p in paths if Path(p).exists()))


def append_generated(path: Path, new: Splats, out: Path | None = None) -> Path:
    """Append generated splats (e.g. distilled from repaired views) to a degree-0 PLY written
    by `write_splats`, flagging them generated = 1 and keeping existing flags."""
    old = read_splats(path)
    flags = np.concatenate([old.generated, np.ones(len(new), np.uint8)])
    out = Path(out) if out is not None else Path(path)
    write_splats(
        out,
        np.concatenate([old.means, new.means]),
        np.concatenate([old.rgb, new.rgb]),
        scales=np.concatenate([old.scales, new.scales]),
        opacity=np.concatenate([old.opacity, new.opacity]),
        generated=flags,
    )
    return out
