import math

import numpy as np
import pytest

from movable_reality.bake import planefit as P


def plane_points(rng, n, normal, offset, size=(4.0, 5.0), noise=0.003, outliers=0.0):
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    a, b = P._basis(normal)
    s = rng.uniform(-0.5, 0.5, (n, 2)) * size
    pts = s[:, :1] * a + s[:, 1:] * b + offset * normal + rng.normal(0, noise, (n, 1)) * normal
    k = int(outliers * n)
    if k:
        junk = rng.uniform(-3, 3, (k, 3))
        pts = np.concatenate([pts, junk])
    return pts


@pytest.mark.parametrize(
    "normal,offset",
    [((0, 1, 0), 0.0), ((0.05, 0.99, -0.12), 1.3), ((0, -1, 0), -2.5), ((1, 2, 3), -0.7)],
)
def test_fit_recovers_plane_with_outliers(normal, offset):
    rng = np.random.default_rng(3)
    pts = plane_points(rng, 20_000, normal, offset, outliers=0.3)
    n_true = np.asarray(normal, float) / np.linalg.norm(normal)
    interior = (offset + 1.0) * n_true  # 1 m on the normal side
    plane = P.fit_plane(pts, 0.02, interior=interior)
    assert P.angle_deg(plane.normal, n_true) < 0.1
    assert np.dot(plane.normal, n_true) > 0  # oriented towards the interior point
    assert abs(plane.offset - offset) < 1e-3
    assert plane.rms < 0.004
    assert plane.inliers[:20_000].mean() > 0.99
    assert plane.inliers[20_000:].mean() < 0.05


def test_tilt_prior_prefers_the_horizontal_plane():
    rng = np.random.default_rng(4)
    floor = plane_points(rng, 3000, (0, 1, 0), 0.0, size=(2, 2))
    wall = plane_points(rng, 6000, (1, 0, 0), 1.0, size=(2, 2.5))  # more points than the floor
    pts = np.concatenate([floor, wall])
    free = P.fit_plane(pts, 0.02)
    assert P.angle_deg(free.normal, (1, 0, 0)) < 1.0
    prior = P.fit_plane(pts, 0.02, up=np.array([0, 1, 0]), max_tilt_deg=20)
    assert P.angle_deg(prior.normal, (0, 1, 0)) < 0.5


def test_fit_needs_three_points():
    with pytest.raises(ValueError):
        P.fit_plane(np.zeros((2, 3)))


def test_rectangle_matches_rotated_room_and_ignores_outliers_and_islands():
    rng = np.random.default_rng(5)
    yaw = math.radians(27.0)
    u0 = np.array([math.cos(yaw), 0, math.sin(yaw)])
    v0 = np.array([-math.sin(yaw), 0, math.cos(yaw)])
    s = rng.uniform(0, 1, (30_000, 2)) * [5.0, 3.0]
    pts = np.outer(s[:, 0], u0) + np.outer(s[:, 1], v0) + [1.0, 0.0, -2.0]
    # Island 1 m beyond the long side (floor seen through a door), and sparse strays.
    island = np.outer(rng.uniform(6.0, 6.5, 400), u0) + np.outer(rng.uniform(1, 1.5, 400), v0)
    strays = rng.uniform(-8, 8, (40, 3)) * [1, 0, 1]
    pts = np.concatenate([pts, island + [1.0, 0.0, -2.0], strays])
    plane = P.Plane(normal=np.array([0, 1.0, 0]), offset=0.0, inliers=np.ones(len(pts), bool),
                    rms=0.0)  # fmt: skip
    r = P.bounding_rect(pts, plane)
    assert r.size_u == pytest.approx(5.0, abs=0.05)
    assert r.size_v == pytest.approx(3.0, abs=0.05)
    assert P.angle_deg(r.u, u0) < 1.0
    # u x v = -normal, so (u, v, -n) is a right-handed camera looking down at the floor.
    assert np.allclose(np.cross(r.u, r.v), -r.normal)
    c = r.corners()
    centre = c.mean(axis=0)
    assert np.allclose(centre, [1.0, 0, -2.0] + 2.5 * u0 + 1.5 * v0, atol=0.05)
    assert r.contains(pts[:30_000], margin=0.01).mean() > 0.999


def test_largest_component_drops_island():
    rng = np.random.default_rng(6)
    main = rng.uniform(0, 2, (2000, 2))
    island = rng.uniform(3, 3.5, (200, 2))
    keep = P.largest_component(np.concatenate([main, island]), cell=0.1)
    assert keep[:2000].all() and not keep[2000:].any()


def test_snap_to_texels_grows_about_centre_and_caps_size():
    r = P.Rect(
        origin=np.zeros(3),
        u=np.array([1.0, 0, 0]),
        v=np.array([0, 0, -1.0]),
        normal=np.array([0, 1.0, 0]),
        size_u=1.003,
        size_v=0.5,
    )
    out, texel, w, h = P.snap_to_texels(r, 0.005, 4096)
    assert (w, h) == (201, 100) and texel == 0.005
    assert out.size_u == pytest.approx(1.005)
    assert np.allclose(out.corners().mean(0), r.corners().mean(0))
    big = P.Rect(r.origin, r.u, r.v, r.normal, size_u=30.0, size_v=10.0)
    _, texel, w, h = P.snap_to_texels(big, 0.005, 4096)
    assert w == 4096 and h <= 4096 and texel == pytest.approx(30.0 / 4096)
