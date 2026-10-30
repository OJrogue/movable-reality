import numpy as np
import pytest

from movable_reality.coverage import uncertainty as U


def random_dirs(rng, n):
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def test_sh_basis_is_orthonormal_on_the_sphere():
    dirs, w = U.sphere_quadrature()
    Y = U.sh_basis(dirs, 3)
    gram = (Y * w[:, None]).T @ Y
    assert w.sum() == pytest.approx(4 * np.pi)
    assert np.allclose(gram, np.eye(16), atol=1e-10)


def test_sh_basis_degrees_are_prefixes():
    d = random_dirs(np.random.default_rng(0), 5)
    full = U.sh_basis(d, 3)
    for deg in range(3):
        assert np.allclose(U.sh_basis(d, deg), full[:, : U.n_coeffs(deg)])


def test_regulariser_quadrature_matches_closed_form():
    c = np.random.default_rng(1).normal(size=(7, 16))
    assert np.allclose(U.regulariser(c, 1.0), U.regulariser_closed_form(c, 1.0))
    # u == b everywhere has zero penalty.
    flat = np.zeros((1, 16))
    flat[0, 0] = 1.0 / U.C0
    assert U.regulariser_closed_form(flat, 1.0)[0] == pytest.approx(0.0, abs=1e-12)


def make_design(rng, n_pix=60, n_gauss=4, degree=3, per_pix=3):
    pix = np.repeat(np.arange(n_pix), per_pix)
    gauss = rng.integers(0, n_gauss, pix.size)
    weight = rng.uniform(0.05, 0.6, pix.size)
    dirs = random_dirs(rng, pix.size)
    return U.Design(pix, gauss, weight, dirs, n_pix, n_gauss, degree)


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    d = make_design(rng)
    y = rng.uniform(0, 0.5, d.n_pix)
    c = rng.normal(scale=0.3, size=(d.n_gauss, 16))
    g = U.gradient(d, y, c, lam=0.3)
    eps = 1e-6
    for k, i in [(0, 0), (1, 3), (3, 15), (2, 8)]:
        cp, cm = c.copy(), c.copy()
        cp[k, i] += eps
        cm[k, i] -= eps
        fd = (U.objective(d, y, cp, 0.3) - U.objective(d, y, cm, 0.3)) / (2 * eps)
        assert g[k, i] == pytest.approx(fd, rel=1e-5, abs=1e-7)


def test_normal_equations_minimise_the_objective():
    rng = np.random.default_rng(3)
    d = make_design(rng)
    y = rng.uniform(0, 0.5, d.n_pix)
    c = U.solve_normal_equations(d, y, lam=0.2)
    assert np.abs(U.gradient(d, y, c, 0.2)).max() < 1e-8
    f0 = U.objective(d, y, c, 0.2)
    for _ in range(5):
        assert U.objective(d, y, c + rng.normal(scale=1e-3, size=c.shape), 0.2) > f0


def test_adam_reaches_the_exact_solution():
    rng = np.random.default_rng(4)
    d = make_design(rng, degree=1)
    y = rng.uniform(0, 0.5, d.n_pix)
    exact = U.solve_normal_equations(d, y, lam=0.5)
    fit = U.adam_fit(d, y, lam=0.5, steps=4000, lr=0.02)
    assert U.objective(d, y, fit, 0.5) == pytest.approx(U.objective(d, y, exact, 0.5), rel=1e-4)


def test_unobserved_gaussian_goes_to_the_prior_and_observed_one_follows_residuals():
    """Gaussian 0 covers every pixel from one side with residual 0.1; Gaussian 1 is never
    rendered. With the prior, u_1 = b everywhere; u_0 is pulled to the residual along the
    observed direction but stays near b in the opposite direction."""
    n = 200
    d0 = np.tile([0.0, 0.0, 1.0], (n, 1)) + np.random.default_rng(5).normal(0, 0.05, (n, 3))
    d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
    design = U.Design(np.arange(n), np.zeros(n, int), np.full(n, 0.9), d0, n, 2, 3)
    y = np.full(n, 0.9 * 0.1)  # U_x = w * u, so u = 0.1 along +z
    c = U.solve_normal_equations(design, y, lam=0.5)
    u_seen = U.eval_uncertainty(c[:1], np.array([[0.0, 0.0, 1.0]]))[0]
    u_back = U.eval_uncertainty(c[:1], np.array([[0.0, 0.0, -1.0]]))[0]
    assert u_seen == pytest.approx(0.1, abs=0.02)
    assert u_back > 0.6
    q = random_dirs(np.random.default_rng(6), 20)
    assert np.allclose(U.eval_uncertainty(np.repeat(c[1:], 20, 0), q), 1.0, atol=1e-8)


def test_lambda_zero_recovers_a_constant_uncertainty_exactly():
    """Degree 0, one Gaussian per pixel, y = w * u_true: the unregularised fit gives
    u_true back for every Gaussian."""
    rng = np.random.default_rng(7)
    n_pix, n_gauss = 200, 5
    gauss = np.arange(n_pix) % n_gauss
    weight = rng.uniform(0.1, 0.9, n_pix)
    d = U.Design(np.arange(n_pix), gauss, weight, random_dirs(rng, n_pix), n_pix, n_gauss, 0)
    u_true = np.array([0.0, 0.1, 0.25, 0.5, 0.9])
    c = U.solve_normal_equations(d, weight * u_true[gauss], lam=0.0)
    assert np.allclose(U.spherical_mean(c), u_true, atol=1e-9)


def test_pixel_residual():
    rng = np.random.default_rng(8)
    a = rng.uniform(size=(24, 32, 3))
    assert np.allclose(U.pixel_residual(a, a), 0.0, atol=1e-9)
    b = np.clip(a + 0.1, 0, 1)
    r = U.pixel_residual(a, b)
    assert r.shape == (24, 32)
    assert (r > 0).all() and (r <= 1).all()
    # Pure L1 part for a constant offset on a flat image.
    flat = np.full((24, 32, 3), 0.4)
    assert U.pixel_residual(flat, flat + 0.1, lam=0.0) == pytest.approx(np.full((24, 32), 0.1))


def test_normalised_summaries():
    c = np.zeros((2, 16))
    c[0, 0] = 1.0 / U.C0  # u = 1 everywhere
    c[1, 0] = 0.2 / U.C0
    c[1, 3] = 0.1  # some directional variation
    s = U.normalised_uncertainty(c)
    assert s["u_mean"] == pytest.approx([1.0, 0.2])
    assert s["u_max"][1] > s["u_mean"][1]


def test_seen_uncertainty_averages_observed_directions_only():
    from movable_reality.coverage.cameras import look_at

    # Gaussian 0 at the origin, seen by one camera on +Z, Gaussian 1 behind that camera.
    cam = look_at(np.array([0.0, 0.0, 2.0]), np.zeros(3), 64, 48)
    means = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    coeffs = np.zeros((2, 16))
    coeffs[:, 0] = 1.0 / U.C0  # u = 1 everywhere
    # A degree-1 term lowers u towards -Z (the direction from the camera to Gaussian 0).
    coeffs[0, 2] = 0.6 / U.C1  # Y_2 = C1 * z, d = (0, 0, -1) -> u = 1 - 0.6
    u = U.seen_uncertainty(coeffs, means, [cam])
    assert u[0] == pytest.approx(0.4, abs=1e-5)
    assert u[1] == 1.0  # no camera sees it: the prior
