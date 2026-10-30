"""U-3DGS-style per-primitive photometric uncertainty, as pure numpy functions.

Method: Galappaththige et al., "Predictive Photometric Uncertainty in Gaussian Splatting
for Novel View Synthesis", arXiv 2603.22786 (https://chumsy0725.github.io/3DGS-Uncertainty/).
The trained scene stays frozen. Every Gaussian k gets an uncertainty channel
u_k(d) = sum_i c_ki Y_i(d) (real spherical harmonics, degree 3 -> 16 coefficients), rendered
with the same alpha blending as colour:

    U(x) = sum_k u_k(d) alpha_k(x) T_k(x)                         (paper Eq. 2)

and fitted to the per-pixel training residual L_x = (1 - l) L1_x + l DSSIM_x, l = 0.2, by
Bayesian-regularised linear least squares (A_jk = alpha_k(x_j) T_k(x_j) is fixed):

    L' = sum_x (L_x - U_x)^2 + lambda_reg * sum_k int_S2 (b - u_k(r))^2 dr

The regulariser is a Gaussian prior centred at maximal uncertainty b = 1, so directions no
training view constrained are pulled to "uncertain". The paper approximates the sphere
integral by quadrature (Gauss-Legendre) and minimises with the same first-order optimiser as
3DGS training; both are here (`regulariser`, `adam_fit`), plus the exact solution of the
normal equations for small problems (`solve_normal_equations`), which tests compare against.

Conventions chosen here (the paper leaves them open, see docs/BUILD_LOG.md):
- the data term is a sum over all pixels of all training views, the regulariser a sum over
  Gaussians; a one-view minibatch scales the regulariser by 1 / n_views so the stochastic
  gradient is unbiased;
- directions d point from the camera centre to the Gaussian (as in gsplat's SH colour);
- SH basis and constants are the standard real, orthonormal ones used by 3DGS and gsplat.

The GPU job (modal_jobs/coverage.py) mirrors `sh_basis`, `pixel_residual` and the objective
in torch and checks them against these functions on the first view.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SSIM_LAMBDA = 0.2  # 3DGS: L = (1 - 0.2) L1 + 0.2 DSSIM
PRIOR_B = 1.0  # maximal normalised residual
SQRT_PI = float(np.sqrt(np.pi))

# Real SH constants (3DGS / gsplat convention).
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def n_coeffs(degree: int) -> int:
    return (degree + 1) ** 2


def sh_basis(dirs: np.ndarray, degree: int = 3) -> np.ndarray:
    """Real SH basis Y_i(d) for unit directions (..., 3) -> (..., (degree+1)^2), in the
    coefficient order 3DGS and gsplat use."""
    if not 0 <= degree <= 3:
        raise ValueError("degree must be 0..3")
    d = np.asarray(dirs, dtype=np.float64)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    out = [np.full(x.shape, C0)]
    if degree >= 1:
        out += [-C1 * y, C1 * z, -C1 * x]
    if degree >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        out += [
            C2[0] * xy,
            C2[1] * yz,
            C2[2] * (2.0 * zz - xx - yy),
            C2[3] * xz,
            C2[4] * (xx - yy),
        ]
    if degree >= 3:
        out += [
            C3[0] * y * (3 * xx - yy),
            C3[1] * xy * z,
            C3[2] * y * (4 * zz - xx - yy),
            C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
            C3[4] * x * (4 * zz - xx - yy),
            C3[5] * z * (xx - yy),
            C3[6] * x * (xx - 3 * yy),
        ]
    return np.stack(out, axis=-1)


def eval_uncertainty(coeffs: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """u_k(d) for coefficients (K, s) and directions (K, 3) or (K, Q, 3)."""
    degree = int(round(np.sqrt(coeffs.shape[-1]))) - 1
    Y = sh_basis(dirs, degree)
    if Y.ndim == 2:
        return np.einsum("ks,ks->k", coeffs, Y)
    return np.einsum("ks,kqs->kq", coeffs, Y)


def seen_uncertainty(
    coeffs: np.ndarray, means: np.ndarray, cameras: list, b: float = 1.0, chunk: int = 200_000
) -> np.ndarray:
    """Mean of u_k(d) over the training cameras whose frustum holds Gaussian k, d = mean -
    camera centre (the direction the fit used), normalised by b and clipped to [0, 1]; b
    (the prior) where no camera sees it. Means and cameras in the frame of the fit (recon).

    The spherical mean stays near the prior for any capture that covers a narrow arc,
    because the fit only lowers u in observed directions. Directions the capture missed are
    the geometric score's job (view count, angle spread); this term adds how badly the
    Gaussian renders where it was observed."""
    coeffs = np.asarray(coeffs, np.float32)
    means = np.asarray(means, np.float32)
    total = np.zeros(len(means), np.float64)
    count = np.zeros(len(means), np.int32)
    for cam in cameras:
        for s in range(0, len(means), chunk):
            m = means[s : s + chunk]
            vis = np.nonzero(cam.in_frustum(m))[0]
            if vis.size == 0:
                continue
            d = m[vis] - cam.center.astype(np.float32)
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-8)
            total[s + vis] += eval_uncertainty(coeffs[s + vis], d)
            count[s + vis] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(count > 0, total / np.maximum(count, 1), b)
    return np.clip(u / b, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------- sphere quadrature


def sphere_quadrature(n_theta: int = 8, n_phi: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes in cos(theta) times uniform phi: directions (Q, 3) and weights
    (Q,) summing to 4 pi. Exact for polynomials of degree < 2 n_theta in cos(theta) and
    trigonometric degree < n_phi in phi, so products of degree-3 SH are integrated exactly
    with the defaults."""
    xg, wg = np.polynomial.legendre.leggauss(n_theta)
    phi = (np.arange(n_phi) + 0.5) * (2 * np.pi / n_phi)
    ct, ph = np.meshgrid(xg, phi, indexing="ij")
    st = np.sqrt(1.0 - ct**2)
    dirs = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=-1).reshape(-1, 3)
    w = (wg[:, None] * np.full(n_phi, 2 * np.pi / n_phi)[None, :]).reshape(-1)
    return dirs, w


def regulariser(coeffs: np.ndarray, b: float = PRIOR_B, quad=None) -> np.ndarray:
    """Per-Gaussian l_k = int_S2 (b - u_k(r))^2 dr by quadrature (K,)."""
    dirs, w = quad if quad is not None else sphere_quadrature()
    Y = sh_basis(dirs, int(round(np.sqrt(coeffs.shape[-1]))) - 1)  # (Q, s)
    u = coeffs @ Y.T  # (K, Q)
    return ((b - u) ** 2) @ w


def regulariser_closed_form(coeffs: np.ndarray, b: float = PRIOR_B) -> np.ndarray:
    """Same integral from SH orthonormality: 4 pi b^2 - 4 sqrt(pi) b c_0 + |c|^2."""
    return 4 * np.pi * b * b - 4 * SQRT_PI * b * coeffs[:, 0] + (coeffs**2).sum(axis=1)


def regulariser_grad(coeffs: np.ndarray, b: float = PRIOR_B) -> np.ndarray:
    g = 2.0 * coeffs
    g[:, 0] -= 4 * SQRT_PI * b
    return g


def spherical_mean(coeffs: np.ndarray) -> np.ndarray:
    """Mean of u_k over the sphere (K,): only the constant term survives."""
    return coeffs[:, 0] * C0


def directional_max(coeffs: np.ndarray, n_dirs: int = 64) -> np.ndarray:
    """Max of u_k over a Fibonacci set of directions (K,)."""
    from .voxels import fibonacci_sphere

    Y = sh_basis(fibonacci_sphere(n_dirs), int(round(np.sqrt(coeffs.shape[-1]))) - 1)
    return (coeffs @ Y.T).max(axis=1)


# ---------------------------------------------------------------- residual


def _gauss_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    x = np.arange(size) - size // 2
    k = np.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def _filter2(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Separable 'same' filter with zero padding (as 3DGS's conv2d SSIM), (H, W)."""
    r = len(k) // 2
    p = np.pad(img, ((0, 0), (r, r)))
    tmp = sum(k[i] * p[:, i : i + img.shape[1]] for i in range(len(k)))
    p = np.pad(tmp, ((r, r), (0, 0)))
    return sum(k[i] * p[i : i + img.shape[0], :] for i in range(len(k)))


def ssim_map(a: np.ndarray, b: np.ndarray, size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Per-pixel SSIM (H, W) averaged over channels, images (H, W, C) in [0, 1]."""
    k = _gauss_kernel(size, sigma)
    c1, c2 = 0.01**2, 0.03**2
    out = np.zeros(a.shape[:2])
    for ch in range(a.shape[2]):
        x, y = a[..., ch].astype(np.float64), b[..., ch].astype(np.float64)
        mx, my = _filter2(x, k), _filter2(y, k)
        sxx = _filter2(x * x, k) - mx * mx
        syy = _filter2(y * y, k) - my * my
        sxy = _filter2(x * y, k) - mx * my
        num = (2 * mx * my + c1) * (2 * sxy + c2)
        den = (mx * mx + my * my + c1) * (sxx + syy + c2)
        out += num / den
    return out / a.shape[2]


def pixel_residual(render: np.ndarray, target: np.ndarray, lam: float = SSIM_LAMBDA) -> np.ndarray:
    """L_x = (1 - lam) * mean_c |render - target| + lam * (1 - SSIM_x) / 2, (H, W) in [0, 1]."""
    l1 = np.abs(render.astype(np.float64) - target.astype(np.float64)).mean(axis=2)
    dssim = np.clip((1.0 - ssim_map(render, target)) / 2.0, 0.0, 1.0)
    return (1.0 - lam) * l1 + lam * dssim


# ---------------------------------------------------------------- the least-squares problem


@dataclass
class Design:
    """Sparse design matrix in COO form: entry e couples pixel `pix[e]` (global pixel index
    over all views) with Gaussian `gauss[e]` through blending weight w = alpha * T and the
    view direction `dirs[e]` (camera to Gaussian, unit)."""

    pix: np.ndarray  # (E,) int
    gauss: np.ndarray  # (E,) int
    weight: np.ndarray  # (E,) float, alpha * T
    dirs: np.ndarray  # (E, 3)
    n_pix: int
    n_gauss: int
    degree: int = 3

    def rows(self) -> np.ndarray:
        """Per-entry SH-weighted rows (E, s): w * Y(d)."""
        return self.weight[:, None] * sh_basis(self.dirs, self.degree)


def render_u(design: Design, coeffs: np.ndarray) -> np.ndarray:
    """U_x for every pixel (n_pix,)."""
    vals = np.einsum("es,es->e", design.rows(), coeffs[design.gauss])
    return np.bincount(design.pix, weights=vals, minlength=design.n_pix)


def objective(
    design: Design, y: np.ndarray, coeffs: np.ndarray, lam: float, b: float = PRIOR_B
) -> float:
    r = y - render_u(design, coeffs)
    return float(r @ r + lam * regulariser_closed_form(coeffs, b).sum())


def gradient(
    design: Design, y: np.ndarray, coeffs: np.ndarray, lam: float, b: float = PRIOR_B
) -> np.ndarray:
    rows = design.rows()
    r = y - render_u(design, coeffs)
    g = np.zeros_like(coeffs)
    np.add.at(g, design.gauss, -2.0 * r[design.pix][:, None] * rows)
    return g + lam * regulariser_grad(coeffs, b)


def solve_normal_equations(
    design: Design, y: np.ndarray, lam: float, b: float = PRIOR_B
) -> np.ndarray:
    """Exact minimiser for small problems: (A^T A + lam I) c = A^T y + 2 sqrt(pi) lam b e_0
    per Gaussian's constant coefficient. Dense in K * s, so keep K * s below ~5000."""
    s = n_coeffs(design.degree)
    rows = design.rows()
    col = design.gauss[:, None] * s + np.arange(s)[None, :]
    A = np.zeros((design.n_pix, design.n_gauss * s))
    np.add.at(A, (np.repeat(design.pix, s), col.ravel()), rows.ravel())
    lhs = A.T @ A + lam * np.eye(A.shape[1])
    rhs = A.T @ y
    rhs[0::s] += 2 * SQRT_PI * lam * b
    return np.linalg.lstsq(lhs, rhs, rcond=None)[0].reshape(design.n_gauss, s)


def adam_fit(
    design: Design,
    y: np.ndarray,
    lam: float,
    b: float = PRIOR_B,
    steps: int = 2000,
    lr: float = 0.05,
    init: float | None = None,
) -> np.ndarray:
    """First-order fit with Adam (the GPU job's optimiser), full batch."""
    s = n_coeffs(design.degree)
    c = np.zeros((design.n_gauss, s))
    c[:, 0] = (b if init is None else init) / C0  # start at the prior's constant level
    m = np.zeros_like(c)
    v = np.zeros_like(c)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        g = gradient(design, y, c, lam, b)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        c -= lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    return c


def normalised_uncertainty(coeffs: np.ndarray, b: float = PRIOR_B) -> dict[str, np.ndarray]:
    """Per-Gaussian summaries in [0, 1] for coverage: spherical mean and directional max of
    u_k / b."""
    return {
        "u_mean": np.clip(spherical_mean(coeffs) / b, 0.0, 1.0).astype(np.float32),
        "u_max": np.clip(directional_max(coeffs) / b, 0.0, 1.0).astype(np.float32),
    }
