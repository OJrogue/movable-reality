"""Turn a baked (premultiplied colour, alpha) pair into an opaque texture.

Texels the splats covered only partly are composited over a fill colour, and holes (no
splat coverage, e.g. the floor under a sofa that the phone never saw) get the fill colour.
The fill is push-pull (Gortler et al. 1996, "The Lumigraph"): an alpha-weighted image
pyramid is built down to one texel and pulled back up, so small holes take their colour
from the texels right next to them and large holes get a smooth blend of their border.
"""

from __future__ import annotations

import numpy as np


def _down(s: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, wd = w.shape
    ph, pw = h % 2, wd % 2
    if ph or pw:
        s = np.pad(s, ((0, ph), (0, pw), (0, 0)))
        w = np.pad(w, ((0, ph), (0, pw)))
    s2 = s[0::2, 0::2] + s[1::2, 0::2] + s[0::2, 1::2] + s[1::2, 1::2]
    w2 = w[0::2, 0::2] + w[1::2, 0::2] + w[0::2, 1::2] + w[1::2, 1::2]
    over = w2 > 1.0
    s2[over] /= w2[over, None]
    w2 = np.minimum(w2, 1.0)
    return s2, w2


def _blur3(x: np.ndarray) -> np.ndarray:
    p = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="edge")
    rows = p[:-2] + p[1:-1] + p[2:]
    return (rows[:, :-2] + rows[:, 1:-1] + rows[:, 2:]) / 9.0


def _up(c: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    big = np.repeat(np.repeat(c, 2, axis=0), 2, axis=1)[: shape[0], : shape[1]]
    return _blur3(big)


def push_pull(premult: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """(H, W, C) premultiplied colour and (H, W) alpha to an (H, W, C) opaque colour.

    Where alpha is 1 the result is the baked colour; where it is 0 the result is the fill;
    in between it is the baked colour composited over the fill.
    """
    s = np.asarray(premult, dtype=np.float64)
    w = np.clip(np.asarray(alpha, dtype=np.float64), 0.0, 1.0)
    if w.max() <= 0:
        return np.zeros_like(s)
    levels = [(s, w)]
    while max(levels[-1][1].shape) > 1:
        levels.append(_down(*levels[-1]))
    s_top, w_top = levels[-1]
    colour = s_top / np.maximum(w_top, 1e-12)[..., None]
    for s_l, w_l in reversed(levels[:-1]):
        up = _up(colour, w_l.shape)
        colour = s_l + (1.0 - w_l)[..., None] * up
    return colour


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def hole_fraction(alpha: np.ndarray, threshold: float = 0.5) -> float:
    """Share of texels whose colour comes mostly from the fill."""
    return float(np.mean(np.asarray(alpha) < threshold))
