import numpy as np
import pytest

from movable_reality.bake import raster as R
from movable_reality.bake import texture as T


def random_set(rng, n, w, h):
    m2 = rng.uniform(-2, [w + 2, h + 2], (n, 2))
    lo = rng.normal(0, 1.5, (n, 2, 2))
    c2 = lo @ lo.transpose(0, 2, 1) + 0.2 * np.eye(2)
    toward = rng.normal(size=n)
    op = rng.uniform(0.0, 1.0, n)
    op[:5] = 0.999  # opaque ones so early termination happens
    col = rng.uniform(0, 1, (n, 3))
    return m2, c2, toward, op, col


@pytest.mark.parametrize("tile,max_fragments", [(4, 37), (7, 1000), (128, 2_000_000)])
def test_vectorised_matches_per_pixel_reference(tile, max_fragments):
    rng = np.random.default_rng(1)
    w, h = 24, 18
    args = random_set(rng, 60, w, h)
    c, a = R.rasterize(*args, w, h, tile=tile, max_fragments=max_fragments)
    c_ref, a_ref = R.reference_rasterize(*args, w, h)
    assert np.abs(c - c_ref).max() < 1e-12
    assert np.abs(a - a_ref).max() < 1e-12


def test_front_to_back_order_follows_toward():
    # Two opaque discs centred on pixel (5, 5); the one nearer the viewer (larger toward)
    # covers the other. Alpha is 0.999 at the centre; the far one would take transmittance
    # below 1e-4, so the pixel stops before it (gsplat's rule).
    m2 = np.array([[5.5, 5.5], [5.5, 5.5]])
    c2 = np.tile(np.eye(2) * 4.0, (2, 1, 1))
    op = np.array([0.999, 0.999])
    col = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    c, a = R.rasterize(m2, c2, np.array([0.0, 0.01]), op, col, 10, 10)
    assert c[5, 5, 2] == pytest.approx(0.999) and c[5, 5, 0] == 0.0
    assert a[5, 5] == pytest.approx(0.999)
    c, _ = R.rasterize(m2, c2, np.array([0.01, 0.0]), op, col, 10, 10)
    assert c[5, 5, 0] == pytest.approx(0.999) and c[5, 5, 2] == 0.0


def test_project_maps_world_to_texels():
    view = R.OrthoView(
        origin=np.array([1.0, 0.0, 2.0]),
        u=np.array([1.0, 0, 0]),
        v=np.array([0, 0, -1.0]),
        normal=np.array([0, 1.0, 0]),
        texel=0.01,
        width=100,
        height=100,
    )
    means = np.array([[1.255, 0.02, 1.5]])
    cov = np.diag([1e-4, 4e-6, 9e-4])[None]
    m2, c2, toward = R.project(means, cov, view)
    assert np.allclose(m2, [[25.5, 50.0]])
    assert np.allclose(c2[0], np.diag([1.0, 9.0]))
    assert np.allclose(toward, [0.02])


def test_ortho_camera_matches_gsplat_convention():
    """gsplat ortho: p_cam = viewmat @ p, mean2d = (fx x + cx, fy y + cy), depth = z."""
    rng = np.random.default_rng(2)
    a = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    u, v = a[:, 0], a[:, 1]
    n = -np.cross(u, v)
    view = R.OrthoView(rng.normal(size=3), u, v, n, 0.005, 300, 200)
    viewmat, k = R.ortho_camera(view, height=1.0)
    assert np.allclose(viewmat[:3, :3] @ viewmat[:3, :3].T, np.eye(3))
    assert np.linalg.det(viewmat[:3, :3]) == pytest.approx(1.0)
    pts = rng.normal(size=(20, 3))
    cov = np.einsum("nij,nkj->nik", *(2 * [rng.normal(0, 0.01, (20, 3, 3))]))
    pc = pts @ viewmat[:3, :3].T + viewmat[:3, 3]
    mean2d = pc[:, :2] * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
    j = np.array([[k[0, 0], 0, 0], [0, k[1, 1], 0]])
    rc = viewmat[:3, :3]
    cov2d = np.einsum("ij,njk,lk->nil", j @ rc, cov, j @ rc)
    m2, c2, toward = R.project(pts, cov, view)
    assert np.allclose(mean2d, m2)
    assert np.allclose(cov2d, c2)
    # gsplat sorts by increasing depth (front first); depth = height - toward.
    assert np.allclose(pc[:, 2], 1.0 - toward)


def test_push_pull_keeps_covered_and_fills_holes():
    h, w = 40, 60
    rgb = np.zeros((h, w, 3))
    rgb[:, : w // 2] = [1.0, 0.0, 0.0]
    rgb[:, w // 2 :] = [0.0, 0.0, 1.0]
    alpha = np.ones((h, w))
    alpha[10:20, 5:15] = 0.0  # hole inside the red half
    alpha[30:, 40:] = 0.0  # hole in the blue corner
    alpha[0, 0] = 0.5  # partly covered texel
    out = T.push_pull(rgb * alpha[..., None], alpha)
    full = alpha == 1.0
    assert np.allclose(out[full], rgb[full])
    assert np.all(out[12:18, 7:13, 0] > 0.95)  # red hole is red
    assert np.all(out[33:, 45:, 2] > 0.95)  # blue hole is blue
    assert out[0, 0, 0] == pytest.approx(1.0, abs=0.02)
    assert np.all((out >= -1e-9) & (out <= 1 + 1e-9))
    assert T.hole_fraction(alpha) == pytest.approx((100 + 200) / (h * w), abs=1e-9)
    # No coverage at all: black, not NaN.
    assert np.all(T.push_pull(np.zeros((4, 4, 3)), np.zeros((4, 4))) == 0)
