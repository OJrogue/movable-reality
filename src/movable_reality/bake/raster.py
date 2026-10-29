"""CPU reference rasterizer for an orthographic bake of Gaussians onto a plane.

The camera looks at the plane from the room side (along -normal). Pixel (row i, col j) has
its centre at origin + (j + 0.5) * texel * u + (i + 0.5) * texel * v, the same convention
as gsplat's orthographic camera with fx = fy = 1 / texel and cx = cy = 0 (see
`ortho_camera`), so the CPU and GPU paths produce the same image.

Per Gaussian, as gsplat's classic mode does it: 2D covariance J Sigma J^T plus eps2d on the
diagonal, alpha = min(0.999, opacity * exp(-0.5 d^T Sigma2^-1 d)), fragments below 1/255
dropped, opacity-aware extent, front-to-back compositing that stops once transmittance
falls to 1e-4. Output is the premultiplied colour sum (no background) and the alpha.

Vectorised: Gaussians are binned into square pixel tiles; each tile expands its Gaussians
into fragments, sorts them by pixel (stable, so depth order holds inside a pixel) and
composites them with a segmented cumulative product. Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.999
T_MIN = 1e-4
EXTENT = 3.33


@dataclass(frozen=True)
class OrthoView:
    origin: np.ndarray  # (3,) world position of pixel corner (0, 0)
    u: np.ndarray  # (3,) unit, +column direction
    v: np.ndarray  # (3,) unit, +row direction
    normal: np.ndarray  # (3,) unit, towards the viewer; u x v = -normal
    texel: float  # metres per pixel
    width: int
    height: int


def project(means: np.ndarray, covs: np.ndarray, view: OrthoView):
    """World means (N, 3) and covariances (N, 3, 3) to pixel means (N, 2), pixel covariances
    (N, 2, 2) and the distance towards the viewer (N,), larger = nearer."""
    d = np.asarray(means, dtype=np.float64) - view.origin
    j = np.stack([view.u, view.v]) / view.texel  # (2, 3)
    means2d = d @ j.T
    cov2d = np.einsum("ij,njk,lk->nil", j, covs, j)
    toward = d @ view.normal
    return means2d, cov2d, toward


def ortho_camera(view: OrthoView, height: float) -> tuple[np.ndarray, np.ndarray]:
    """gsplat camera for the same image: world-to-camera 4x4 and intrinsics 3x3.

    Camera axes (OpenCV): x = u, y = v, z = -normal; it sits `height` metres in front of
    the plane origin, so a Gaussian at distance s towards the viewer has depth height - s.
    gsplat's orthographic projection is mean2d = (fx * x + cx, fy * y + cy).
    """
    r = np.stack([view.u, view.v, -view.normal])
    cam = view.origin + height * view.normal
    viewmat = np.eye(4)
    viewmat[:3, :3] = r
    viewmat[:3, 3] = -r @ cam
    k = np.array([[1.0 / view.texel, 0.0, 0.0], [0.0, 1.0 / view.texel, 0.0], [0.0, 0.0, 1.0]])
    return viewmat, k


def rasterize(
    means2d: np.ndarray,
    cov2d: np.ndarray,
    toward: np.ndarray,
    opacities: np.ndarray,
    colors: np.ndarray,
    width: int,
    height: int,
    *,
    eps2d: float = 0.3,
    tile: int = 128,
    max_fragments: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gaussians front to back (largest `toward` first).

    Returns (colour (H, W, C) premultiplied sum, alpha (H, W)).
    """
    n = len(means2d)
    nc = colors.shape[1]
    out_c = np.zeros((height * width, nc))
    out_t = np.ones(height * width)
    if n == 0:
        return out_c.reshape(height, width, nc), np.zeros((height, width))

    a = cov2d[:, 0, 0] + eps2d
    b = 0.5 * (cov2d[:, 0, 1] + cov2d[:, 1, 0])
    c = cov2d[:, 1, 1] + eps2d
    det = a * c - b * b
    op = np.asarray(opacities, dtype=np.float64)
    ok = (det > 0) & (op >= ALPHA_MIN)
    extent = np.zeros(n)
    extent[ok] = np.minimum(EXTENT, np.sqrt(2.0 * np.log(op[ok] / ALPHA_MIN)))
    rx = np.ceil(extent * np.sqrt(np.maximum(a, 0)))
    ry = np.ceil(extent * np.sqrt(np.maximum(c, 0)))
    mx, my = means2d[:, 0], means2d[:, 1]
    # Pixels whose centre (j + 0.5) lies within the radius.
    x0 = np.maximum(np.ceil(mx - rx - 0.5), 0).astype(np.int64)
    x1 = np.minimum(np.floor(mx + rx - 0.5), width - 1).astype(np.int64)
    y0 = np.maximum(np.ceil(my - ry - 0.5), 0).astype(np.int64)
    y1 = np.minimum(np.floor(my + ry - 0.5), height - 1).astype(np.int64)
    ok &= (x1 >= x0) & (y1 >= y0) & (rx > 0) & (ry > 0)

    # Front to back: nearest the viewer first; stable for ties.
    order = np.argsort(-np.asarray(toward, dtype=np.float64), kind="stable")
    order = order[ok[order]]
    if len(order) == 0:
        return out_c.reshape(height, width, nc), np.zeros((height, width))
    ca, cb, cc = c / det, -b / det, a / det  # conic of the inverse covariance

    # Bin (Gaussian, tile) pairs; pairs stay in depth order inside each tile.
    tx0, tx1 = x0[order] // tile, x1[order] // tile
    ty0, ty1 = y0[order] // tile, y1[order] // tile
    ntx = (width + tile - 1) // tile
    nw, nh = tx1 - tx0 + 1, ty1 - ty0 + 1
    counts = nw * nh
    pair_g = np.repeat(np.arange(len(order)), counts)
    local = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    pair_tx = tx0[pair_g] + local % nw[pair_g]
    pair_ty = ty0[pair_g] + local // nw[pair_g]
    pair_tile = pair_ty * ntx + pair_tx
    srt = np.argsort(pair_tile, kind="stable")
    pair_tile, pair_g = pair_tile[srt], order[pair_g[srt]]
    bounds = np.flatnonzero(np.diff(pair_tile)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(pair_tile)]])

    col = np.asarray(colors, dtype=np.float64)
    for s, e in zip(starts, ends, strict=True):
        t_id = int(pair_tile[s])
        px0, py0 = (t_id % ntx) * tile, (t_id // ntx) * tile
        tw, th = min(tile, width - px0), min(tile, height - py0)
        g = pair_g[s:e]
        gx0 = np.maximum(x0[g], px0)
        gx1 = np.minimum(x1[g], px0 + tw - 1)
        gy0 = np.maximum(y0[g], py0)
        gy1 = np.minimum(y1[g], py0 + th - 1)
        w_g = gx1 - gx0 + 1
        frag_n = w_g * (gy1 - gy0 + 1)
        tile_c = np.zeros((tw * th, nc))
        tile_t = np.ones(tw * th)
        done = np.zeros(tw * th, dtype=bool)
        # Chunks of Gaussians, still front to back, bounded in fragment count.
        cum = np.cumsum(frag_n)
        chunk_edges = [0]
        while chunk_edges[-1] < len(g):
            base = cum[chunk_edges[-1] - 1] if chunk_edges[-1] > 0 else 0
            nxt = int(np.searchsorted(cum, base + max_fragments, side="right"))
            chunk_edges.append(max(nxt, chunk_edges[-1] + 1))
        for c0, c1 in zip(chunk_edges[:-1], chunk_edges[1:], strict=True):
            gi = g[c0:c1]
            fn = frag_n[c0:c1]
            fg = np.repeat(np.arange(len(gi)), fn)
            off = np.arange(fn.sum()) - np.repeat(np.cumsum(fn) - fn, fn)
            fx = gx0[c0:c1][fg] + off % w_g[c0:c1][fg]
            fy = gy0[c0:c1][fg] + off // w_g[c0:c1][fg]
            gg = gi[fg]
            dx = fx + 0.5 - mx[gg]
            dy = fy + 0.5 - my[gg]
            sigma = 0.5 * (ca[gg] * dx * dx + cc[gg] * dy * dy) + cb[gg] * dx * dy
            alpha = np.minimum(ALPHA_MAX, op[gg] * np.exp(-sigma))
            keep = (sigma >= 0) & (alpha >= ALPHA_MIN)
            pix = (fy - py0) * tw + (fx - px0)
            pix, alpha, gg = pix[keep], alpha[keep], gg[keep]
            keep_pix = ~done[pix]
            pix, alpha, gg = pix[keep_pix], alpha[keep_pix], gg[keep_pix]
            if len(pix) == 0:
                continue
            srt = np.argsort(pix, kind="stable")
            pix, alpha, gg = pix[srt], alpha[srt], gg[srt]
            log1m = np.log1p(-alpha)
            excl = np.cumsum(log1m) - log1m
            first = np.concatenate([[True], pix[1:] != pix[:-1]])
            seg_start = np.maximum.accumulate(np.where(first, np.arange(len(pix)), 0))
            t_before = tile_t[pix] * np.exp(excl - excl[seg_start])
            include = t_before * (1.0 - alpha) > T_MIN
            # Once a pixel is done, later fragments in it are excluded too.
            stop = np.where(~include, np.arange(len(pix)), len(pix))
            seg_id = np.cumsum(first) - 1
            first_stop = np.full(int(seg_id[-1]) + 1, len(pix))
            np.minimum.at(first_stop, seg_id, stop)
            include = np.arange(len(pix)) < first_stop[seg_id]
            wgt = (t_before * alpha)[include]
            ip, gi_inc = pix[include], gg[include]
            for k in range(nc):
                tile_c[:, k] += np.bincount(ip, weights=wgt * col[gi_inc, k], minlength=tw * th)
            absorbed = np.bincount(ip, weights=np.log1p(-alpha[include]), minlength=tw * th)
            tile_t *= np.exp(absorbed)
            stopped_pix = np.unique(pix[first_stop[seg_id] < len(pix)])
            done[stopped_pix] = True
        rows = (np.arange(th)[:, None] + py0) * width + (np.arange(tw)[None, :] + px0)
        out_c[rows.ravel()] = tile_c
        out_t[rows.ravel()] = tile_t
    return out_c.reshape(height, width, nc), 1.0 - out_t.reshape(height, width)


def reference_rasterize(
    means2d, cov2d, toward, opacities, colors, width, height, *, eps2d: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel loop with the same rules as `rasterize`. Slow; for tests only."""
    order = np.argsort(-np.asarray(toward, dtype=np.float64), kind="stable")
    out_c = np.zeros((height, width, colors.shape[1]))
    out_a = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            t = 1.0
            acc = np.zeros(colors.shape[1])
            for gi in order:
                a = cov2d[gi, 0, 0] + eps2d
                b = 0.5 * (cov2d[gi, 0, 1] + cov2d[gi, 1, 0])
                c = cov2d[gi, 1, 1] + eps2d
                det = a * c - b * b
                o = float(opacities[gi])
                if det <= 0 or o < ALPHA_MIN:
                    continue
                ext = min(EXTENT, np.sqrt(2.0 * np.log(o / ALPHA_MIN)))
                rx, ry = np.ceil(ext * np.sqrt(a)), np.ceil(ext * np.sqrt(c))
                dx = j + 0.5 - means2d[gi, 0]
                dy = i + 0.5 - means2d[gi, 1]
                if abs(dx) > rx or abs(dy) > ry:
                    continue
                sigma = 0.5 * (c * dx * dx + a * dy * dy) / det - b * dx * dy / det
                alpha = min(ALPHA_MAX, o * np.exp(-sigma))
                if sigma < 0 or alpha < ALPHA_MIN:
                    continue
                if t * (1 - alpha) <= T_MIN:
                    break
                acc += t * alpha * colors[gi]
                t *= 1 - alpha
            out_c[i, j] = acc
            out_a[i, j] = 1 - t
    return out_c, out_a
