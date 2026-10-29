import numpy as np
import pytest

from movable_reality.coverage.cameras import (
    Camera,
    look_at,
    qvec_to_rotmat,
    rotmat_to_qvec,
    transform_points,
)
from movable_reality.coverage.colmap import read_model, write_model
from movable_reality.coverage.geometry import (
    DepthParams,
    compute_coverage,
    depth_slope,
    geometric_score,
    spread_from_sums,
)
from movable_reality.coverage.voxels import VoxelGrid, dilate, label_components


def test_look_at_projects_target_to_image_centre():
    cam = look_at([1.0, 1.6, 2.0], [0.0, 0.5, 0.0], 640, 480)
    uv, z = cam.project(np.array([[0.0, 0.5, 0.0]]))
    assert uv[0] == pytest.approx([320, 240], abs=1e-3)
    assert z[0] == pytest.approx(np.linalg.norm([1.0, 1.1, 2.0]), rel=1e-5)
    assert cam.center == pytest.approx([1.0, 1.6, 2.0])
    # +Y up in the world means "up" in the image is -v.
    uv_up, _ = cam.project(np.array([[0.0, 0.6, 0.0]]))
    assert uv_up[0, 1] < 240


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[0] >= 0 else -q
        assert rotmat_to_qvec(qvec_to_rotmat(q)) == pytest.approx(q, abs=1e-9)


def test_camera_frame_change_keeps_projections():
    cam = look_at([0.5, 1.2, 1.0], [0.0, 0.4, -1.0], 320, 240)
    pts = np.random.default_rng(1).uniform([-1, 0, -2], [1, 1, 0], (50, 3))
    # Similarity with scale 2: recon = world scaled, rotated and moved.
    ang = 0.7
    rot = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    A = np.eye(4)
    A[:3, :3] = 2.0 * rot
    A[:3, 3] = [0.3, -1.0, 2.0]
    cam2 = cam.in_frame(A)
    uv1, z1 = cam.project(pts)
    uv2, z2 = cam2.project(transform_points(A, pts))
    assert uv2 == pytest.approx(uv1, abs=1e-3)
    assert z2 == pytest.approx(2.0 * z1, rel=1e-5)


def test_colmap_roundtrip(tmp_path):
    cams = [
        look_at([np.cos(a), 1.3, np.sin(a)], [0, 0.5, 0], 640, 360, name=f"{i:06d}.jpg")
        for i, a in enumerate(np.linspace(0, 3, 4))
    ]
    write_model(tmp_path / "sparse" / "0", cams, np.zeros((3, 3)))
    back = read_model(tmp_path)
    assert [c.name for c in back] == [c.name for c in cams]
    for a, b in zip(cams, back, strict=True):
        assert b.K == pytest.approx(a.K)
        assert b.R == pytest.approx(a.R, abs=1e-9)
        assert b.t == pytest.approx(a.t, abs=1e-9)
        assert (b.width, b.height) == (640, 360)


def wall_and_box_scene():
    """A wall of points at z = -3 and a small opaque square at z = -1.5 in front of it."""
    g = np.linspace(-1, 1, 41)
    gx, gy = np.meshgrid(g, g + 1)
    wall = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, -3.0)], 1)
    b = np.linspace(-0.2, 0.2, 9)
    bx, by = np.meshgrid(b, b + 1)
    box = np.stack([bx.ravel(), by.ravel(), np.full(bx.size, -1.5)], 1)
    means = np.concatenate([wall, box]).astype(np.float32)
    n = len(means)
    return means, np.ones(n, np.float32), np.full((n, 3), 0.03, np.float32), len(wall)


def test_occluded_points_are_not_counted():
    means, opac, scales, n_wall = wall_and_box_scene()
    cam = look_at([0.0, 1.0, 0.0], [0.0, 1.0, -3.0], 320, 320, hfov_deg=60)
    gc, _ = compute_coverage(means, opac, scales, [cam], DepthParams(long_side=128))
    wall = means[:n_wall]
    behind_box = (np.abs(wall[:, 0]) < 0.08) & (np.abs(wall[:, 1] - 1.0) < 0.08)
    assert gc.view_count[n_wall:].min() == 1  # the box is seen
    assert gc.view_count[:n_wall][behind_box].max() == 0  # wall behind its middle is not
    assert gc.view_count[:n_wall][np.abs(wall[:, 0]) > 0.5].min() == 1


def test_points_outside_the_frustum_are_not_counted():
    means, opac, scales, _ = wall_and_box_scene()
    cam = look_at([0.0, 1.0, 0.0], [0.0, 1.0, 3.0], 320, 320)  # looking away
    gc, _ = compute_coverage(means, opac, scales, [cam])
    assert gc.view_count.max() == 0


def test_angle_spread_of_two_views():
    d1 = np.array([1.0, 0.0, 0.0])
    d2 = np.array([np.cos(np.radians(30)), np.sin(np.radians(30)), 0.0])
    s = spread_from_sums(np.array([2, 1, 0]), np.stack([d1 + d2, d1, np.zeros(3)]))
    assert s[0] == pytest.approx(30.0, abs=1e-3)
    assert s[1] == 0.0 and s[2] == 0.0


def test_geometric_score_saturates():
    s = geometric_score(np.array([0, 1, 3, 10]), np.array([0.0, 0.0, 10.0, 40.0]))
    assert s.tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0])


def test_depth_slope_ignores_occlusion_edges():
    d = np.full((5, 6), 2.0, np.float32)
    d[:, 3:] = 4.0  # a depth edge
    s = depth_slope(d.ravel(), 6, 5, cap=10.0).reshape(5, 6)
    assert s.max() == 0.0  # one-sided minimum: an edge is not a slope
    ramp = np.tile(np.arange(6, dtype=np.float32) * 0.1 + 2.0, (5, 1))
    assert depth_slope(ramp.ravel(), 6, 5, cap=10.0).reshape(5, 6)[2, 2] == pytest.approx(0.1)


def test_free_space_carving_marks_voxels_in_front_of_the_wall():
    means, opac, scales, _ = wall_and_box_scene()
    cam = look_at([0.0, 1.0, 0.0], [0.0, 1.0, -3.0], 320, 320, hfov_deg=60)
    grid = VoxelGrid.from_bounds(np.array([-1, 0, -3.2]), np.array([1, 2, 0.0]), 0.2)
    _, free = compute_coverage(means, opac, scales, [cam], voxel_centres=grid.centres())
    c = grid.centres()
    on_axis = (np.abs(c[:, 0]) < 0.11) & (np.abs(c[:, 1] - 1.0) < 0.11)
    between = on_axis & (c[:, 2] < -0.3) & (c[:, 2] > -1.3)
    behind_box = on_axis & (c[:, 2] < -1.8) & (c[:, 2] > -2.8)
    assert free[between].min() == 1
    assert free[behind_box].max() == 0


def test_label_components_26_connectivity():
    m = np.zeros((6, 6, 6), bool)
    m[0, 0, 0] = m[1, 1, 1] = True  # diagonal neighbours: one component
    m[4, 4, 4] = m[4, 4, 5] = True
    m[0, 5, 0] = True
    lab = label_components(m).reshape(m.shape)
    assert lab[0, 0, 0] == lab[1, 1, 1]
    assert len({lab[0, 0, 0], lab[4, 4, 4], lab[0, 5, 0]}) == 3
    assert lab[4, 4, 4] == lab[4, 4, 5]
    assert (lab[~m] == -1).all()


def test_label_components_long_snake():
    m = np.zeros((20, 3, 20), bool)
    m[:, 1, 0] = True
    m[19, 1, :] = True
    m[:, 1, 19] = True
    lab = label_components(m)
    assert set(np.unique(lab[lab >= 0]).tolist()) == {0}


def test_dilate():
    m = np.zeros((5, 5, 5), bool)
    m[2, 2, 2] = True
    assert dilate(m, 1).sum() == 27
    assert dilate(m, 2).sum() == 125


def test_grid_centres_and_index_agree():
    grid = VoxelGrid.from_bounds(np.array([-1.0, 0.0, -2.0]), np.array([1.0, 2.6, 2.0]), 0.1)
    flat = np.arange(0, grid.n, 97)
    back, inside = grid.index(grid.centres(flat))
    assert inside.all() and (back == flat).all()
    # Room bounds lie on voxel centres.
    assert grid.centres(np.array([0]))[0] == pytest.approx([-1.0, 0.0, -2.0])


def test_camera_scaled_keeps_projection_ratio():
    cam = Camera("a", 640, 360, np.array([[500.0, 0, 320], [0, 500, 180], [0, 0, 1]]),
                 np.eye(3), np.zeros(3))  # fmt: skip
    small = cam.scaled(160)
    assert (small.width, small.height) == (160, 90)
    p = np.array([[0.3, -0.2, 2.0]])
    assert small.project(p)[0] * 4 == pytest.approx(cam.project(p)[0], rel=1e-5)
