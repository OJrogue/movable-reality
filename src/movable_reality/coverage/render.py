"""CPU preview renderer: z-buffered square footprints of Gaussian centres, no blending.

Good enough for synthetic frames, gap previews and reference matching; final-quality
renders of real scenes come from gsplat on Modal.
"""

from __future__ import annotations

import numpy as np

from .cameras import Camera


def render_points(
    cam: Camera,
    means: np.ndarray,
    rgb: np.ndarray,
    scales: np.ndarray | None = None,
    opacity: np.ndarray | None = None,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    min_opacity: float = 0.3,
    footprint_sigma: float = 1.2,
    max_radius: int = 4,
    near: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Render world-frame points with a camera. Returns (image (H, W, 3) float32 in [0, 1],
    z-depth (H, W) float32, +inf where nothing was drawn)."""
    means = np.asarray(means, dtype=np.float32)
    keep = np.ones(len(means), bool) if opacity is None else np.asarray(opacity) >= min_opacity
    uv, z = cam.project(means[keep], near)
    u = np.floor(uv[:, 0]).astype(np.int32)
    v = np.floor(uv[:, 1]).astype(np.int32)
    ok = (z > near) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    u, v, z = u[ok], v[ok], z[ok]
    col = np.asarray(rgb, dtype=np.float32)[keep][ok]
    if scales is None:
        r = np.ones(len(z), np.int32)
    else:
        s = np.asarray(scales, dtype=np.float32)
        s = s.max(axis=1) if s.ndim == 2 else s
        r = np.rint(footprint_sigma * s[keep][ok] * float(cam.K[0, 0]) / z)
        r = np.clip(r, 0, max_radius).astype(np.int32)
    W, H = cam.width, cam.height
    depth = np.full(W * H, np.inf, np.float32)
    passes = []
    for rr in np.unique(r):
        sel = np.nonzero(r == rr)[0]
        for dy in range(-int(rr), int(rr) + 1):
            for dx in range(-int(rr), int(rr) + 1):
                uu, vv = u[sel] + dx, v[sel] + dy
                inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
                pix = vv[inb] * W + uu[inb]
                np.minimum.at(depth, pix, z[sel][inb])
                passes.append((pix, sel[inb]))
    image = np.empty((W * H, 3), np.float32)
    image[:] = np.asarray(background, np.float32)
    for pix, idx in passes:
        win = z[idx] <= depth[pix]
        image[pix[win]] = col[idx[win]]
    return image.reshape(H, W, 3), depth.reshape(H, W)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
