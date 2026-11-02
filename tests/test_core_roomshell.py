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


def test_room_outline_needs_min_frames_and_clips_boxes():
    frames = [_hits([0, 0], [3, 3], seed=s) for s in range(3)]
    frames[0] = np.concatenate([frames[0], _hits([3, 0], [4, 3], n=500)])  # one stray frame
    poly2 = roomshell.room_outline(frames, [])
    assert poly2[:, 0].max() < 3.2  # the stray frame alone does not widen the room
    # A box 60 cm through the wall (blender02: a desk's lift box) widens it by at most the
    # 10 cm box reach.
    poly = roomshell.room_outline(frames, [(np.array([2.5, 0.0]), np.array([3.6, 3.0]))])
    assert poly[:, 0].max() < poly2[:, 0].max() + 0.15


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


def test_copy_fill_meets_a_soft_shadow_without_a_step():
    # Flat floor, a soft shadow darkening the left side (40 levels over 40 texels); the hole
    # covers part of it. The copy comes from the bright right, so its edge would be a step.
    h, w = 60, 160
    x = np.arange(w)
    shade = np.clip(40 - x, 0, 40)  # 40 levels darker at x = 0, none from x = 40
    rgb = np.repeat(np.repeat((180 - shade)[None, :, None], h, 0), 3, 2).astype(np.uint8)
    holes = np.zeros((h, w), bool)
    holes[20:40, 10:60] = True
    out = roomshell.copy_fill(rgb, holes, np.ones((h, w), bool), step=1).astype(int)
    assert abs(out[30, 10] - out[30, 9]).max() <= 8  # the left edge, inside the shadow
    assert abs(out[30, 45] - 180).max() <= 6  # deep inside: the clean copy


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


def test_finish_alpha_reaches_past_the_outline():
    # A 1 x 1 m surface at 5 cm texels, outline the lower-left triangle: the texels the
    # diagonal cuts stay opaque (no crack along a wall), the far corner stays clear.
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    rgba = roomshell.finish(np.full((20, 20, 3), 0.5), np.ones((20, 20)), tri, 0.05, (1.0, 1.0))
    keep = roomshell.outline_keep(tri, (1.0, 1.0), (20, 20))
    a = rgba[..., 3] > 0
    assert a[keep].all()
    assert (a & ~keep).any()  # grown past the diagonal
    assert not a[19, 19]


def test_surface_colour_gap_matches_the_texture_under_a_point():
    # 4 x 4 m room, 2.5 m high: the floor texture is grey on the left half, red on the right.
    poly = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    surfs = [s for s in roomshell.surfaces(poly, 2.5) if s.kind == "floor"]
    tex = np.full((40, 40, 4), 255, dtype=np.uint8)
    tex[:, :, :3] = 128
    tex[:, 20:, :3] = (200, 30, 30)
    pts = np.array([[1.0, 0.3, 2.0], [3.0, 0.3, 2.0], [1.0, 0.3, 2.0], [5.0, 0.3, 2.0]])
    grey, red = np.full(3, 128 / 255), np.array([200, 30, 30]) / 255
    dist, gap = roomshell.surface_colour_gap(pts, np.stack([grey, red, red, grey]), surfs, [tex])
    assert np.allclose(dist[:3], 0.3)
    assert gap[0] < 0.01 and gap[1] < 0.01  # a floor-coloured ghost over each half
    assert gap[2] > 0.3  # a red thing over the grey half is not
    assert np.isinf(dist[3]) and np.isinf(gap[3])  # over no surface


def test_main_angle_of_an_l_shaped_floor_with_a_corner_unseen():
    rng = np.random.default_rng(0)
    ell = np.array([[-2.25, 2.5], [2.25, 2.5], [2.25, -0.5], [0.75, -0.5], [0.75, -2.5],
                    [-2.25, -2.5]])  # fmt: skip
    p = rng.uniform(-3, 3, (60000, 2))
    p = p[roomshell.inside_polygon(p, ell)]
    p = p[~((p[:, 0] > 0) & (p[:, 1] > 1.5))]  # blender02: the far corner half seen
    for deg in (6.3, 30.0):
        t = np.radians(deg)
        rot = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        assert abs(np.degrees(roomshell.main_angle(p @ rot)) - deg) < 0.5


def test_copy_fill_takes_the_median_when_no_source_fits():
    # The hole covers most of the surface; its rim is a dark strip on one side only.
    h, w = 40, 200
    rgb = np.full((h, w, 3), 170, dtype=np.uint8)
    rgb[:, :4] = 60  # a dark corner strip
    holes = np.zeros((h, w), bool)
    holes[:, 4:180] = True
    out = roomshell.copy_fill(rgb, holes, np.ones((h, w), bool), step=4).astype(int)
    # Deep inside, the surface's own colour within the clamped correction (0.06 = 15 levels),
    # not the rim's push-pull average (115).
    assert abs(out[20, 100] - 170).max() <= 16


def test_base_mask_keeps_the_floor_between_legs_and_drops_a_painted_base():
    s = roomshell.surfaces(SQ, 2.5)[0]
    shape = (30, 40)  # 0.1 m texels
    rgb = np.full((*shape, 3), 120, np.uint8)
    rgb[::2] = 140  # planks
    good = np.ones(shape, bool)

    def at(x, z):
        rel = np.array([x, 0.0, z]) - s.origin
        return int(rel @ s.v / s.size_v * shape[0]), int(rel @ s.u / s.size_u * shape[1])

    # A desk: legs at its corners, the floor between them filmed in the floor's colour.
    legs = np.array([[x, 0.05, z] for x in (0.55, 1.45) for z in (0.55, 1.45)])
    black = np.array([[20.0, 20.0, 20.0]])
    desk = (np.array([0.5, 0.0, 0.5]), np.array([1.5, 0.75, 1.5]), legs, black)
    m, _ = roomshell.base_mask(s, [desk], rgb, good, contact=0.12, grow_m=0.0)
    assert m[at(0.55, 0.55)] and not m[at(1.0, 1.0)]
    # A bed whose wooden base was painted onto the floor under it: the whole box goes.
    v0, u0 = at(2.5, 0.5)
    v1, u1 = at(3.5, 1.5)
    rgb[min(v0, v1) : max(v0, v1) + 1, min(u0, u1) : max(u0, u1) + 1] = (230, 190, 140)
    rgb[at(2.55, 1.45)] = (100, 120, 170)  # a blue rug's corner reaches under the bed
    bed = (np.array([2.5, 0.0, 0.5]), np.array([3.5, 0.5, 1.5]), np.array([[3.45, 0.05, 1.45]]),
           np.array([[230.0, 190.0, 140.0]]))  # fmt: skip
    m, foreign = roomshell.base_mask(s, [bed], rgb, good, contact=0.12, grow_m=0.0)
    assert m[at(3.0, 1.0)] and m[at(2.55, 0.55)] and not m[at(1.0, 1.0)]
    assert foreign[at(3.0, 1.0)] and not foreign[at(1.0, 1.0)]
    assert foreign[at(2.55, 1.45)] and not m[at(2.55, 1.45)]  # the rug stays


def test_base_mask_keeps_a_rug_under_a_chair():
    # A grey-blue rug filmed between a white chair's legs: foreign to the floor, but not the
    # chair's colour, so only the legs go.
    s = roomshell.surfaces(SQ, 2.5)[0]
    shape = (30, 40)  # 0.1 m texels
    rgb = np.full((*shape, 3), (150, 100, 60), np.uint8)  # wood
    rgb[::2] = (130, 85, 50)
    good = np.ones(shape, bool)
    rug = np.zeros(shape, bool)
    rug[5:25, 5:25] = True
    rgb[rug] = (110, 120, 140)
    legs = np.array([[x, 0.05, z] for x in (1.05, 1.45) for z in (1.05, 1.45)])
    chair = (np.array([1.0, 0.0, 1.0]), np.array([1.5, 0.9, 1.5]), legs,
             np.array([[235.0, 235.0, 230.0]]))  # fmt: skip
    m, foreign = roomshell.base_mask(s, [chair], rgb, good, contact=0.12, grow_m=0.0)
    assert foreign[rug].mean() > 0.9
    assert (m & rug).sum() <= 4 * 9  # the legs, grown by a texel


def test_copy_fill_continues_stripes_across_a_hole_larger_than_any_source():
    # Planks along y; the hole covers most of the width, so no single offset fits it.
    h, w = 120, 200
    stripes = ((np.arange(w) // 6) % 2 * 100 + 70).astype(np.uint8)
    rgb = np.repeat(np.repeat(stripes[None, :, None], h, 0), 3, 2)
    holes = np.zeros((h, w), bool)
    holes[30:110, 10:190] = True
    damaged = rgb.copy()
    damaged[holes] = 0
    no_rug = np.zeros((h, w), bool)
    out = roomshell.copy_fill(damaged, holes, np.ones((h, w), bool), step=1, tile=20,
                              foreign=no_rug)  # fmt: skip
    err = np.abs(out[holes].astype(int) - rgb[holes].astype(int)).mean()
    assert err < 20  # planks in line (a flat median would be 50 off, shifted planks 100)


def test_copy_fill_keeps_a_rug_inside_its_edge():
    # Planks with a flat grey rug on the left and a wall (outside `keep`) on the right.
    h, w = 120, 200
    stripes = ((np.arange(w) // 6) % 2 * 60 + 90).astype(np.uint8)
    rgb = np.repeat(np.repeat(stripes[None, :, None], h, 0), 3, 2)
    rug = np.zeros((h, w), bool)
    rug[10:110, :60] = True
    rgb[rug] = 200
    keep = np.ones((h, w), bool)
    keep[:, 90:] = False
    # A hole beside the rug: past the seam correction at its edge, no rug.
    holes = np.zeros((h, w), bool)
    holes[30:90, 60:90] = True
    out = roomshell.copy_fill(rgb, holes, keep, step=1, tile=20, foreign=rug)
    deep = out[30:90, 72:90, 0].astype(int)
    assert (np.abs(deep - 200) < 15).mean() < 0.05
    # A hole inside the rug, larger than a piece: rug.
    holes = np.zeros((h, w), bool)
    holes[20:100, 5:45] = True
    out = roomshell.copy_fill(rgb, holes, keep, step=1, tile=20, foreign=rug)
    assert (np.abs(out[holes][:, 0].astype(int) - 200) < 15).mean() > 0.95
