"""Room shell (mesh-only view): outline from floor hits, walls, projection. CPU only."""

import numpy as np

from movable_reality import roommesh, roomshell


def _area(p: np.ndarray) -> float:
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _hits(lo, hi, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, size=(n, 2))


def test_ortho_convex_fills_notch_keeps_l():
    m = np.zeros((10, 10), bool)
    m[2:8, 2:8] = True
    m[2:4, 4:6] = False  # notch along the top wall (furniture)
    assert roomshell.ortho_convex(m)[2:8, 2:8].all()
    ell = np.zeros((10, 10), bool)
    ell[1:9, 1:4] = True
    ell[6:9, 1:9] = True
    assert np.array_equal(roomshell.ortho_convex(ell), ell)


def test_rectify_snaps_and_merges_step():
    # A 4 x 3 m box with slightly tilted edges and a 10 cm step in one wall.
    poly = np.array([[0, 0], [2, 0.02], [2.0, 0.1], [4, 0.1], [4.03, 3], [0, 3.02]])
    out = roomshell.rectify(poly, step_m=0.4)
    assert len(out) == 4
    xs, ys = sorted(set(np.round(out[:, 0], 2))), sorted(set(np.round(out[:, 1], 2)))
    assert len(xs) == 2 and len(ys) == 2


def test_room_outline_rectangle_turned():
    th = np.radians(20)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    frames = [_hits([-2, -1.5], [2, 1.5], seed=s) @ rot.T for s in range(4)]
    poly = roomshell.room_outline(frames, [])
    assert len(poly) == 4
    assert _area(poly) > 0
    assert abs(_area(poly) - 12.0) < 0.8
    sides = sorted(np.linalg.norm(poly - np.roll(poly, -1, 0), axis=1))
    assert abs(sides[0] - 3) < 0.15 and abs(sides[-1] - 4) < 0.15


def test_room_outline_needs_min_frames_and_adds_boxes():
    frames = [_hits([0, 0], [3, 3], seed=s) for s in range(3)]
    frames[0] = np.concatenate([frames[0], _hits([3, 0], [4, 3], n=500)])  # one stray frame
    poly = roomshell.room_outline(frames, [(np.array([2.5, 0.0]), np.array([3.6, 3.0]))])
    assert poly[:, 0].max() > 3.5  # the box footprint widens the room
    poly2 = roomshell.room_outline(frames, [])
    assert poly2[:, 0].max() < 3.2  # the stray frame alone does not


def test_surfaces_normals_point_inward():
    poly = np.array([[0, 0], [4, 0], [4, 3], [0, 3.0]])
    surf = roomshell.surfaces(poly, 2.5)
    kinds = [s.kind for s in surf]
    assert kinds.count("wall") == 4 and "floor" in kinds and "ceiling" in kinds
    centre = np.array([2.0, 1.25, 1.5])
    for s in surf:
        mid = s.origin + 0.5 * s.u * s.size_u + 0.5 * s.v * s.size_v
        assert np.dot(centre - mid, s.normal) > 0, s.kind
        assert abs(np.dot(np.cross(s.u, s.v), s.normal)) > 0.99


def test_floor_points_hit_plane():
    K = np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]])
    # Camera 1.5 m above the floor looking straight down (+Z cam = -Y world).
    R = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
    centre = np.array([0.5, 1.5, -0.5])
    t = -R @ centre
    mask = np.ones((100, 100), bool)
    p = roomshell.floor_points(K, R, t, mask, stride=10)
    assert np.allclose(p.mean(0), [0.5, -0.5], atol=0.05)
    assert np.ptp(p[:, 0]) < 1.6  # 1.5 m away at 100 px focal length: about 1.5 m wide


def test_projector_takes_colour_and_skips_blocked():
    floor = roomshell.surfaces(np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]]), 2.5)[0]
    pts, w, h = floor.grid(0.1)
    K = np.array([[200.0, 0, 100], [0, 200, 100], [0, 0, 1]])
    R = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
    centre = np.array([0.5, 2.0, 0.5])
    img = np.zeros((200, 200, 3), np.uint8)
    img[..., 0] = 200
    blocked = np.zeros((200, 200), bool)
    blocked[:, :100] = True  # left half of the image hides the floor
    pr = roomshell.Projector(pts, floor.normal)
    pr.add(roomshell.View(K, R, -R @ centre, centre, img, blocked))
    rgb, seen = pr.result((h, w))
    assert 0.3 < seen.mean() < 0.7
    assert np.allclose(rgb[seen > 0][:, 0], 200 / 255)
    rgba = roomshell.finish(rgb, seen, floor.outline_uv, 0.1, (floor.size_u, floor.size_v))
    assert rgba.shape == (h, w, 4) and (rgba[..., 3] == 255).all()


def test_roommesh_helpers():
    m = np.diag([2.0, 2.0, 2.0, 1.0])
    assert abs(roommesh.unit_m(m.tolist()) - 2.0) < 1e-9
    s = {"background_ply": "objects/background.ply"}
    assert roommesh.source_background(s) == "objects/background.ply"
    s["plane_bake"] = {"source_background_ply": "background.ply"}
    assert roommesh.source_background(s) == "background.ply"


def test_room_outline_l_shape_and_angled_wall_kept():
    a = [np.concatenate([_hits([0, 0], [4, 2], seed=s), _hits([0, 2], [2, 4], seed=s + 9)])
         for s in range(3)]  # fmt: skip
    poly = roomshell.room_outline(a, [])
    assert len(poly) == 6 and abs(_area(poly) - 12.0) < 0.8
    # A 45 deg wall 1.4 m long across one corner stays a wall.
    pts = np.array([[0, 0], [3, 0], [3, 2], [2, 3], [0, 3.0]])
    assert len(roomshell.rectify(pts)) == 5


def test_projector_through_similarity():
    """room_shell.py composes recon cameras with world_from_recon (scale 2.4, a turn and a
    shift). A world point must land on the pixel its recon point lands on."""
    th = np.radians(30)
    W = np.eye(4)
    W[:3, :3] = 2.4 * np.array(
        [[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]
    )
    W[:3, 3] = [0.3, -0.2, 1.0]
    Winv = np.linalg.inv(W)
    K = np.array([[300.0, 0, 160], [0, 300, 120], [0, 0, 1]])
    Rc, tc = np.eye(3), np.array([0.0, 0.0, 2.0])  # recon camera
    p_recon = np.array([0.1, -0.05, 0.4])
    q = Rc @ p_recon + tc
    pix = (K @ q)[:2] / q[2]
    R, t = Rc @ Winv[:3, :3], Rc @ Winv[:3, 3] + tc
    p_world = W[:3, :3] @ p_recon + W[:3, 3]
    img = np.zeros((240, 320, 3), np.uint8)
    img[int(pix[1]), int(pix[0])] = (10, 200, 30)
    centre = W[:3, :3] @ (-Rc.T @ tc) + W[:3, 3]
    normal = (centre - p_world) / np.linalg.norm(centre - p_world)
    pr = roomshell.Projector(p_world[None], normal, k=1)
    assert pr.add(roomshell.View(K, R, t, centre, img, None, near=0.1 / 2.4)) == 1
    rgb, _ = pr.result((1, 1))
    assert np.allclose(rgb[0, 0] * 255, (10, 200, 30))


def test_unseen_ceiling_takes_the_upper_wall_colour():
    wall_rgb = np.zeros((10, 4, 3))
    wall_rgb[:5] = [0.8, 0.7, 0.6]  # upper half
    wall_rgb[5:] = [0.2, 0.2, 0.2]  # skirting, lower half
    upper = roomshell.upper_wall_colour([(wall_rgb, np.ones((10, 4)))])
    assert np.allclose(upper, [0.8, 0.7, 0.6])
    dark = roomshell.finish(np.zeros((6, 6, 3)), np.zeros((6, 6)), None, 0.01, (0.06, 0.06))
    lit = roomshell.finish(np.zeros((6, 6, 3)), np.zeros((6, 6)), None, 0.01, (0.06, 0.06),
                           fallback=upper)  # fmt: skip
    assert dark[..., :3].max() == 0
    assert np.allclose(lit[..., :3], lit[0, 0, :3]) and lit[0, 0, 0] > 150


# ---- surface artifacts (room_shell.py depth test and fill, room_surfaces.py)

SQ = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])


def test_surface_distance_inside_and_outside():
    pts = np.array([[2.0, 1.0, 1.5], [2.0, 0.02, 1.5], [0.05, 1.0, 1.5], [2.0, -0.1, 1.5],
                    [-0.3, 1.0, 1.5], [2.0, 2.45, 1.5]])  # fmt: skip
    d = roomshell.surface_distance(pts, SQ, 2.5)
    assert np.allclose(d, [1.0, 0.02, 0.05, -0.1, -0.3, 0.05])


def test_occluder_zbuffer_nearest_depth_and_grow():
    K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    pts = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])  # both on the principal ray
    z = roomshell.occluder_zbuffer(pts, K, (100, 100), cell=4, grow=1)
    assert z.shape == (100, 100)
    assert z[50, 50] == 2.0
    assert z[50, 54] == 2.0  # one cell away, closed by grow
    assert np.isinf(z[0, 0])


def test_projector_drops_texels_behind_an_occluder():
    K = np.array([[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1]])
    pts = np.array([[0.0, 0.0, 2.0], [0.6, 0.0, 2.0]])  # a wall 2 m ahead, normal to camera
    img = np.full((64, 64, 3), 200, np.uint8)
    view = roomshell.View(K, np.eye(3), np.zeros(3), np.zeros(3), img, None,
                          zbuf=np.full((64, 64), np.inf), z_margin=0.05)  # fmt: skip
    view.zbuf[:, :40] = 1.0  # a chair at 1 m covers the left of the image
    pr = roomshell.Projector(pts, np.array([0.0, 0.0, -1.0]))
    assert pr.add(view) == 1  # only the texel at x = 0.6 m (u = 47) is seen
    _, seen = pr.result((1, 2))
    assert seen.tolist() == [[0.0, 1.0]]


def test_hole_mask_includes_footprint_and_islands():
    seen = np.ones((20, 20))
    seen[5:12, 5:12] = 0
    seen[8, 8] = 1  # a stray sample in the hole
    keep = np.ones((20, 20), bool)
    foot = np.zeros((20, 20), bool)
    foot[15, 15] = True
    m = roomshell.hole_mask(seen, keep, foot, grow=0)
    assert m[8, 8] and m[15, 15] and not m[0, 0]


def test_footprint_mask_marks_floor_under_furniture():
    s = roomshell.surfaces(SQ, 2.5)[0]
    assert s.kind == "floor"
    shape = (30, 40)  # 0.1 m texels
    chair = np.array([[1.05, 0.3, 1.05], [1.05, 3.0, 2.05]])  # the second is above `reach`
    m = roomshell.footprint_mask(s, chair, shape, reach=1.2, grow_m=0.0)
    assert m.sum() == 9  # one texel grown by 1
    rel = chair[0] - s.origin
    v = int(rel @ s.v / s.size_v * shape[0])
    u = int(rel @ s.u / s.size_u * shape[1])
    assert m[v, u]


def test_copy_fill_continues_stripes():
    h, w = 40, 80  # room for a whole-period offset with its ring in bounds
    stripes = ((np.arange(w) // 4) % 2 * 120 + 60).astype(np.uint8)
    rgb = np.repeat(np.repeat(stripes[None, :, None], h, 0), 3, 2)
    holes = np.zeros((h, w), bool)
    holes[15:25, 20:32] = True
    damaged = rgb.copy()
    damaged[holes] = 0
    out = roomshell.copy_fill(damaged, holes, np.ones((h, w), bool), step=1)
    err = np.abs(out[holes].astype(int) - rgb[holes].astype(int)).mean()
    assert err < 5  # planks continue in line, not a blur


def test_surface_splats_reach_outside_and_haze():
    pts = np.array([[2.0, 1.2, 1.5], [2.0, 1.2, 0.3], [2.0, 1.2, 0.3], [-0.5, 1.0, 1.0],
                    [2.0, 1.2, 1.5]])  # fmt: skip
    size = np.array([0.02, 0.02, 0.15, 0.02, 0.2])  # the third reaches the wall at z = 0
    opacity = np.array([0.9, 0.9, 0.9, 0.9, 0.1])  # the last is a faint 20 cm blob: haze
    gone = roomshell.surface_splats(pts, size, opacity, SQ, 2.5)
    assert gone.tolist() == [False, False, True, True, True]


def test_neighbour_counts_flags_isolated_points():
    rng = np.random.default_rng(0)
    cluster = rng.uniform(0.0, 0.1, size=(50, 3))
    lone = np.array([[1.0, 1.0, 1.0]])
    n = roomshell.neighbour_counts(np.vstack([cluster, lone]), cell=0.1)
    assert n[-1] == 1 and n[:-1].min() >= 50
