"""Client for SAM 3D Objects on Modal (modal_jobs/sam3d.py), with a disk cache.

The Modal app must be deployed once: `uv run modal deploy modal_jobs/sam3d.py`. Each result
is cached under data/cache/sam3d/ keyed by the hash of image, mask and seed, so reruns cost
nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from . import splat_ops as so
from .config import CACHE_DIR, Models

APP_NAME = "movable-reality-sam3d"


class Sam3DUnavailable(RuntimeError):
    """The Modal app is not deployed or cannot be reached."""


@dataclass
class Sam3DResult:
    vertices: np.ndarray  # canonical-frame splats, standard layout
    rotation: list[float]  # pose, see splat_ops.sam3d_cam_from_local
    translation: list[float]
    scale: list[float]
    meta: dict
    glb: bytes | None = None  # mesh with vertex colours, y-up canonical frame (run(mesh=True))

    def cam_from_local(self) -> tuple[np.ndarray, np.ndarray, float]:
        return so.sam3d_cam_from_local(self.rotation, self.translation, self.scale)


def _key(image: bytes, mask: bytes, seed: int, dense: bool, mesh: bool = False) -> str:
    h = hashlib.sha256()
    opts = f"{seed}:{dense}" + (":mesh" if mesh else "")  # keys without a mesh stay as before
    for part in (Models.SAM3D_OBJECTS.encode(), image, mask, opts.encode()):
        h.update(hashlib.sha256(part).digest())
    return h.hexdigest()[:24]


def run_sam3d(
    image: bytes,
    mask_png: bytes,
    seed: int = 42,
    dense: bool = False,
    force: bool = False,
    mesh: bool = False,
) -> Sam3DResult:
    """Image (PNG/JPEG bytes) + mask (PNG, white = object) -> Gaussians and pose.
    dense=True: 32 splats per voxel (about 1M for a stool) instead of 4 (about 120k).
    mesh=True: also the object as a vertex-coloured triangle mesh (GLB bytes). The same seed
    gives the same Gaussians and pose, so the mesh run can replace a cached Gaussian run."""
    cache = CACHE_DIR / "sam3d"
    key = _key(image, mask_png, seed, dense, mesh)
    ply_path, meta_path = cache / f"{key}.ply", cache / f"{key}.json"
    glb_path = cache / f"{key}.glb"
    if not (ply_path.exists() and meta_path.exists()) or force:
        import modal  # noqa: PLC0415 - only needed on a cache miss

        try:
            cls = modal.Cls.from_name(APP_NAME, "Sam3D")
            res = cls().run.remote(image, mask_png, seed, dense, mesh)
        except modal.exception.NotFoundError as exc:
            raise Sam3DUnavailable(
                f"Modal app '{APP_NAME}' not deployed: uv run modal deploy modal_jobs/sam3d.py"
            ) from exc
        cache.mkdir(parents=True, exist_ok=True)
        ply_path.write_bytes(res.pop("ply"))
        glb = res.pop("glb", None)
        if glb:
            glb_path.write_bytes(glb)
        meta_path.write_text(json.dumps(res, indent=2))
    meta = json.loads(meta_path.read_text())
    return Sam3DResult(
        vertices=so.standard(so.read_vertices(ply_path)),
        rotation=meta["rotation"],
        translation=meta["translation"],
        scale=meta["scale"],
        meta=meta,
        glb=glb_path.read_bytes() if glb_path.exists() else None,
    )


def mask_png(mask: np.ndarray) -> bytes:
    import io  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    buf = io.BytesIO()
    Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255).save(buf, format="PNG")
    return buf.getvalue()
