"""Reconstruction helpers: map MapAnything's model-resolution cameras back to the frames.

MapAnything's COLMAP export (demo_colmap.py) stores images and intrinsics at model input
resolution (518 x 294 for 16:9). Its preprocessing (mapanything/utils/cropping.py, pinned
SHA in modal_jobs/reconstruct.py) scales by s = max(w / W, h / H), floors the resized size,
then centre-crops with left = (W' - w) // 2 and top = (H' - h) // 2. `full_res_camera`
inverts that, so masks, depth and training can use the full-resolution frames.
"""

from __future__ import annotations

import math
from dataclasses import replace

from .coverage.cameras import Camera


def full_res_camera(cam: Camera, width: int, height: int) -> Camera:
    """The same camera with intrinsics for the original width x height frame."""
    s = max(cam.width / width, cam.height / height) + 1e-8
    rw, rh = math.floor(width * s), math.floor(height * s)
    left, top = (rw - cam.width) // 2, (rh - cam.height) // 2
    kx, ky = width / rw, height / rh  # full pixels per resized pixel
    K = cam.K.copy()
    K[0, 0] *= kx
    K[1, 1] *= ky
    K[0, 2] = (cam.K[0, 2] + left + 0.5) * kx - 0.5
    K[1, 2] = (cam.K[1, 2] + top + 0.5) * ky - 0.5
    return replace(cam, width=width, height=height, K=K)
