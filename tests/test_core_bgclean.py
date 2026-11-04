"""Background clean-up: what moving an object reveals."""

from __future__ import annotations

import numpy as np

from movable_reality.bgclean import (
    WALL_BEHIND,
    clean,
    covered_grid,
    frame_agreement,
    in_grid,
    in_outline,
    mirror,
    repaint,
    unseen_splats,
    wall_coords,
    wall_tinted,
)


def test_mirror_takes_seen_splats_from_across_the_edge():
    x, y = np.meshgrid(np.arange(0, 2, 0.02), np.arange(0, 2, 0.02))
    uv = np.stack([x.ravel(), y.ravel()], 1)
    inside = (np.abs(uv[:, 0] - 1) < 0.2) & (np.abs(uv[:, 1] - 1) < 0.2)
    lo, grid = covered_grid(uv, uv[inside], grow=0.03)
    hidden = np.flatnonzero(in_grid(uv, lo, grid))
    seen = np.setdiff1d(np.arange(len(uv)), hidden)

    donor = mirror(uv[hidden], uv, seen, lo, grid)

    assert inside[hidden].sum() == inside.sum() and len(hidden) < 1.5 * inside.sum()
    assert not np.isin(donor, hidden).any()
    # A mirror image lies at most the area's half width outside it: nothing from far away.
    assert np.linalg.norm(uv[hidden] - uv[donor], axis=1).max() < 0.6


def test_covered_grid_with_nothing_covering():
    uv = np.random.default_rng(0).uniform(0, 1, (100, 2))
    assert covered_grid(uv, np.zeros((0, 2))) is None


def test_in_outline_handles_an_l_shaped_room():
    room = [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]]
    pts = np.array([[1.0, 1.0], [3.0, 1.0], [1.0, 3.0], [3.0, 3.0], [5.0, 1.0]])
    assert in_outline(pts, room).tolist() == [True, True, True, False, False]


def test_a_wall_of_an_l_room_does_not_reach_past_its_end():
    rng = np.random.default_rng(9)
    room = [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]]
    # In the room, on the far side of the line through the wall at z = 2 from x = 2 to 4.
    far = np.stack(
        [rng.uniform(0.2, 1.8, 3000), rng.uniform(0.5, 2.6, 3000), np.full(3000, 3.0)], 1
    )
    wall = np.stack([rng.uniform(2, 4, 6000), rng.uniform(0.1, 2.6, 6000), np.full(6000, 2.0)], 1)
    xyz = np.concatenate([far, wall])
    v = np.zeros(len(xyz), dtype=[(n, "f4") for n in ("f_dc_0", "f_rest_0", "opacity")])
    tv = np.stack([rng.uniform(0.5, 1.5, 500), rng.uniform(1, 1.6, 500), np.full(500, 3.3)], 1)
    unseen = np.ones(len(xyz), bool)

    done = clean(v, xyz, {"tv#0": tv}, room, 2.7, 1.0, unseen)

    assert not np.isin(done.drop, np.arange(3000)).any()


def test_a_splat_joined_at_a_corner_is_not_repainted_or_dropped_by_the_next_wall():
    rng = np.random.default_rng(10)
    a = np.stack([rng.uniform(0, 4, 20000), rng.uniform(0.1, 2.6, 20000), np.zeros(20000)], 1)
    b = np.stack([np.zeros(20000), rng.uniform(0.1, 2.6, 20000), rng.uniform(0, 4, 20000)], 1)
    xyz = np.concatenate([a, b])
    v = np.zeros(len(xyz), dtype=[(n, "f4") for n in ("f_dc_0", "f_rest_0", "opacity")])
    v["f_dc_0"] = 1.0
    corner = (xyz[:, 0] < 0.5) & (xyz[:, 2] < 0.05) & (np.abs(xyz[:, 1] - 1.4) < 0.3)
    v["f_dc_0"][corner] = -2.0  # the canvas
    before = v["f_dc_0"].copy()
    picture = np.stack(
        [rng.uniform(0, 0.5, 500), rng.uniform(1.1, 1.7, 500), np.full(500, 0.02)], 1
    )

    done = clean(
        v, xyz, {"picture#0": picture}, [[0, 0], [4, 0], [4, 4], [0, 4]], 2.7, 1.0,
        np.zeros(len(xyz), bool),
    )  # fmt: skip

    joined = done.joins["picture#0"]
    assert len(np.unique(joined)) == len(joined) and corner[joined].sum() > 0.9 * corner.sum()
    np.testing.assert_array_equal(v["f_dc_0"][joined], before[joined])
    assert not np.isin(done.drop, joined).any() and not np.isin(done.new_donor, joined).any()


def test_repaint_copies_colour_and_opacity_and_clears_view_dependence():
    names = ["x", "f_dc_0", "f_rest_0", "opacity", "scale_0"]
    v = np.zeros(2, dtype=[(n, "f4") for n in names])
    v["f_dc_0"], v["f_rest_0"], v["opacity"], v["scale_0"] = [1, 5], [9, 2], [0, 3], [7, 8]
    repaint(v, np.array([0]), np.array([1]))
    assert v["f_dc_0"].tolist() == [5, 5] and v["opacity"].tolist() == [3, 3]
    assert v["f_rest_0"].tolist() == [0, 2]
    assert v["scale_0"].tolist() == [7, 8]  # shape stays
    repaint(v, np.array([0]), np.array([1]), max_log=2.0)
    assert v["scale_0"].tolist() == [2, 8]  # capped on request


def test_wall_coords_follow_a_rotated_wall():
    a, b = (0.0, 0.0), (3.0, 4.0)  # 5 m long
    p = np.array([[1.5, 1.0, 2.0], [1.5, 2.0, 2.5]])
    uv, dist = wall_coords(p, a, b)
    np.testing.assert_allclose(uv[0], [2.5, 1.0])
    np.testing.assert_allclose(dist, [0, 0.3], atol=1e-9)


def test_clean_repaints_under_an_object_and_a_stray_splat_joins_it():
    rng = np.random.default_rng(1)
    floor = np.stack([rng.uniform(0, 4, 8000), np.zeros(8000), rng.uniform(0, 4, 8000)], 1)
    chair = np.stack(
        [rng.uniform(1.5, 2.5, 3000), rng.uniform(0.1, 0.9, 3000), rng.uniform(1.5, 2.5, 3000)], 1
    )
    stray = chair[0] + (0.01, 0, 0)
    far = np.array([3.5, 1.0, 3.5])
    xyz = np.concatenate([floor, [stray, far]])
    v = np.zeros(len(xyz), dtype=[(n, "f4") for n in ("f_dc_0", "f_rest_0", "opacity")])
    v["f_dc_0"] = 1.0
    under = (np.abs(floor[:, 0] - 2) < 0.4) & (np.abs(floor[:, 2] - 2) < 0.4)
    v["f_dc_0"][: len(floor)][under] = -3.0  # dark and blotchy under the chair
    v["f_rest_0"][: len(floor)][under] = 2.0

    outline = [[-1, -1], [5, -1], [5, 5], [-1, 5]]
    done = clean(v, xyz, {"chair#0": chair}, outline, 2.7)

    assert v["f_dc_0"][: len(floor)][under].min() == 1.0
    assert v["f_rest_0"][: len(floor)][under].max() == 0.0
    assert done.joins["chair#0"].tolist() == [len(floor)]  # the stray splat; the far one stays
    assert done.stats["joined"] == 1 and done.stats["floor"] >= under.sum()


def test_clean_fills_a_hole_behind_an_object_and_drops_what_lies_behind_the_wall():
    rng = np.random.default_rng(2)
    # A wall at z = 0 facing +z, with no splats where the television hangs.
    wall = np.stack([rng.uniform(0, 4, 20000), rng.uniform(0.1, 2.6, 20000), np.zeros(20000)], 1)
    hole = (np.abs(wall[:, 0] - 2) < 0.5) & (np.abs(wall[:, 1] - 1.4) < 0.3)
    wall = wall[~hole]
    junk = np.array([[2.0, 1.4, -0.3], [2.1, 1.3, -0.5]])  # seen through the hole
    far_junk = np.array([[3.8, 2.4, -0.3]])  # behind the wall, not behind the television
    xyz = np.concatenate([wall, junk, far_junk])
    v = np.zeros(len(xyz), dtype=[(n, "f4") for n in ("f_dc_0", "f_rest_0", "opacity")])
    tv = np.stack(
        [rng.uniform(1.5, 2.5, 3000), rng.uniform(1.1, 1.7, 3000), np.full(3000, 0.05)], 1
    )

    done = clean(v, xyz, {"tv#0": tv}, [[0, 0], [4, 0], [4, 4], [0, 4]], 2.7)

    assert set(done.drop.tolist()) == {len(wall), len(wall) + 1}
    assert len(done.new_xyz) > 300  # about 1 x 0.6 m of 3 cm cells
    np.testing.assert_allclose(done.new_xyz[:, 2], -WALL_BEHIND, atol=1e-6)  # just behind the wall
    assert np.all(np.abs(done.new_xyz[:, 0] - 2) < 0.65)
    assert np.all(done.new_donor < len(wall))


def test_wall_splats_showing_a_hung_picture_join_it_and_the_wall_is_filled_behind():
    rng = np.random.default_rng(3)
    xyz = np.stack([rng.uniform(0, 4, 20000), rng.uniform(0.1, 2.6, 20000), np.zeros(20000)], 1)
    on = (np.abs(xyz[:, 0] - 2) < 0.5) & (np.abs(xyz[:, 1] - 1.4) < 0.3)
    v = np.zeros(len(xyz), dtype=[(n, "f4") for n in ("f_dc_0", "f_rest_0", "opacity")])
    v["f_dc_0"] = 1.0
    v["f_dc_0"][on] = rng.choice([-2.0, 1.0], on.sum())  # canvas, some of it the wall's colour
    before = v["f_dc_0"].copy()
    frame = np.stack(
        [rng.uniform(1.5, 2.5, 500), rng.uniform(1.1, 1.7, 500), np.full(500, 0.02)], 1
    )
    unseen = on.copy()  # the masks say nothing shows that wall

    done = clean(v, xyz, {"picture#0": frame}, [[0, 0], [4, 0], [4, 4], [0, 4]], 2.7, unseen=unseen)

    joined = done.joins["picture#0"]
    assert on[joined].mean() > 0.8 and on[joined].sum() > 0.98 * on.sum()  # the canvas, unpainted
    np.testing.assert_array_equal(v["f_dc_0"][joined], before[joined])
    assert len(done.new_xyz) > 300 and np.all(done.new_xyz[:, 2] < 0)  # the picture draws over them
    assert np.all(v["f_dc_0"][done.new_donor] == 1.0)  # wall colour, not canvas
    assert done.stats["wall_joined"] == len(joined) and done.stats["joined"] == 0


def test_a_thick_object_by_the_wall_takes_only_what_is_not_wall_coloured():
    rng = np.random.default_rng(4)
    xyz = np.stack([rng.uniform(0, 4, 20000), rng.uniform(0.1, 2.6, 20000), np.zeros(20000)], 1)
    on = (np.abs(xyz[:, 0] - 2) < 0.3) & (np.abs(xyz[:, 1] - 1.0) < 0.3)
    v = np.zeros(len(xyz), dtype=[(n, "f4") for n in ("f_dc_0", "f_rest_0", "opacity")])
    v["f_dc_0"] = 1.0
    dirty = on & (rng.uniform(0, 1, len(xyz)) < 0.2)
    v["f_dc_0"][dirty] = -2.0  # the vase's colour on the wall behind it
    vase = np.stack(
        [rng.uniform(1.7, 2.3, 500), rng.uniform(0.7, 1.3, 500), rng.uniform(0.02, 0.3, 500)], 1
    )

    done = clean(v, xyz, {"vase#0": vase}, [[0, 0], [4, 0], [4, 4], [0, 4]], 2.7, unseen=on)

    assert set(done.joins["vase#0"].tolist()) == set(np.flatnonzero(dirty).tolist())
    assert v["f_dc_0"][on & ~dirty].min() == 1.0


def test_clean_leaves_what_frames_saw_and_a_flat_object_alone():
    rng = np.random.default_rng(3)
    floor = np.stack([rng.uniform(0, 4, 8000), np.zeros(8000), rng.uniform(0, 4, 8000)], 1)
    table = np.stack(
        [rng.uniform(1.5, 2.5, 2000), rng.uniform(0.4, 0.5, 2000), rng.uniform(1.5, 2.5, 2000)], 1
    )
    v = np.zeros(len(floor), dtype=[(n, "f4") for n in ("f_dc_0", "f_rest_0", "opacity")])
    v["f_dc_0"] = 1.0
    v["f_dc_0"][:100] = 7.0  # a seen splat under the table keeps its colour
    unseen = np.zeros(len(floor), bool)
    done = clean(
        v, floor, {"table#0": table}, [[-1, -1], [5, -1], [5, 5], [-1, 5]], 2.7, 1.0, unseen
    )
    assert done.stats["floor"] == 0 and v["f_dc_0"][:100].min() == 7.0

    rug = floor[:3000] + (0, 0.01, 0)
    got = clean(v, floor, {"rug#0": rug}, [[-1, -1], [5, -1], [5, 5], [-1, 5]], 2.7, flat={"rug#0"})
    assert got.stats["floor"] == 0 and len(got.new_xyz) == 0


def test_unseen_splats_uses_the_masks_of_the_frames(tmp_path):
    from PIL import Image

    from movable_reality.coverage.cameras import Camera

    # A camera at the origin looking along +z; the left half of its frame is an object's mask.
    mask = np.zeros((100, 200), np.uint8)
    mask[:, :100] = 5
    Image.fromarray(mask).save(tmp_path / "000000.png")
    K = np.array([[100.0, 0, 100], [0, 100.0, 50], [0, 0, 1]])
    cam = Camera("000000.jpg", 200, 100, K, np.eye(3), np.zeros(3))
    pts = np.array([[-0.5, 0, 1], [0.5, 0, 1], [50.0, 0, 1]])  # masked, clear, out of view

    out = unseen_splats(pts, [cam], tmp_path, {5}, min_seen=1, min_share=0.0)
    assert out.tolist() == [True, False, True]


def test_frame_agreement_finds_a_colour_no_frame_shows(tmp_path):
    from PIL import Image

    from movable_reality.coverage.cameras import Camera

    # Grey frame; the left half is an object's mask.
    mask = np.zeros((100, 200), np.uint8)
    mask[:, :100] = 5
    masks, frames = tmp_path / "m", tmp_path / "f"
    masks.mkdir()
    frames.mkdir()
    Image.fromarray(mask).save(masks / "000000.png")
    Image.fromarray(np.full((100, 200, 3), 128, np.uint8)).save(frames / "000000.jpg")
    K = np.array([[100.0, 0, 100], [0, 100.0, 50], [0, 0, 1]])
    cam = Camera("000000.jpg", 200, 100, K, np.eye(3), np.zeros(3))
    pts = np.array([[0.5, 0, 1], [0.5, 0, 1], [-0.5, 0, 1]])  # right, wrong, masked
    rgb = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])

    assert frame_agreement(pts, rgb, [cam], masks, frames, {5}, step=1).tolist() == [1, 1, 1]
    out = frame_agreement(pts, rgb, [cam] * 3, masks, frames, {5}, step=1)
    assert out.tolist() == [1.0, 0.0, 1.0]  # nothing shows the masked one: no evidence against it


def test_clean_keeps_a_window_view_and_drops_junk_near_an_object():
    rng = np.random.default_rng(7)
    wall = np.stack([rng.uniform(0, 4, 20000), rng.uniform(0.1, 2.6, 20000), np.zeros(20000)], 1)
    window = (np.abs(wall[:, 0] - 2) < 0.5) & (np.abs(wall[:, 1] - 1.4) < 0.3)
    wall = wall[~window]
    garden = np.array([[2.0, 1.4, -0.5], [2.1, 1.5, -2.0]])  # past the wall, seen through the glass
    clump = np.array([[2.0, 1.0, 0.45]])  # in the room, matches no frame
    leg = np.array([[2.0, 1.0, 0.46]])  # in the room, matches
    xyz = np.concatenate([wall, garden, clump, leg])
    n = len(wall)
    v = np.zeros(len(xyz), dtype=[(k, "f4") for k in ("f_dc_0", "f_rest_0", "opacity")])
    curtain = np.stack(
        [rng.uniform(1.5, 2.5, 3000), rng.uniform(0.9, 1.9, 3000), np.full(3000, 0.3)], 1
    )
    unseen = np.zeros(len(xyz), bool)
    unseen[(np.abs(xyz[:, 0] - 2) < 0.2) & (xyz[:, 2] == 0)] = True
    agreement = np.ones(len(xyz))
    agreement[n + 2] = 0.0

    done = clean(
        v, xyz, {"curtain#0": curtain}, [[0, 0], [4, 0], [4, 4], [0, 4]], 2.7, 1.0, unseen,
        agreement=agreement,
    )  # fmt: skip

    assert n not in done.drop and n + 1 not in done.drop
    again = np.ones(len(xyz))
    again[n] = 0.0  # past the wall, in view, matching no frame: junk behind a television
    v2 = np.zeros(len(xyz), dtype=v.dtype)
    outline = [[0, 0], [4, 0], [4, 4], [0, 4]]
    got = clean(v2, xyz, {"curtain#0": curtain}, outline, 2.7, 1.0, unseen, agreement=again)
    assert n in got.drop and n + 1 not in got.drop
    assert (n + 2) in done.drop and (n + 3) not in done.drop and done.stats["junk"] == 1
    assert len(done.new_xyz) > 0  # the covered wall's holes are filled, but not over the view
    assert np.linalg.norm(done.new_xyz[:, :2] - [2.0, 1.4], axis=1).min() > 0.05


def test_mirror_on_a_plain_surface_ignores_a_dirty_rim():
    rng = np.random.default_rng(6)
    uv = rng.uniform(0, 3, (6000, 2))
    inside = np.all(np.abs(uv - 1.5) < 0.5, axis=1)
    colours = np.zeros((len(uv), 3))
    colours[np.all(np.abs(uv - 1.5) < 0.65, axis=1)] = 5.0  # the object's colour bled onto its rim
    lo, grid = covered_grid(uv, uv[inside], grow=0.03)
    h = np.flatnonzero(inside)
    seen = np.flatnonzero(~inside & ~np.all(np.abs(uv - 1.5) < 0.53, axis=1))

    plain = mirror(uv[h], uv, seen, lo, grid, colours=colours, k=400)
    assert np.mean(colours[plain, 0] == 0) > 0.95


def test_wall_tinted_finds_the_wall_on_an_object_standing_by_it():
    rng = np.random.default_rng(8)
    wall = ((0.0, 0.0), (4.0, 0.0), np.array([0.5, 0.6, 0.6]))
    far_wall = ((0.0, 4.0), (4.0, 4.0), np.array([0.9, 0.9, 0.9]))
    pts = np.stack(
        [rng.uniform(1, 2, 400), rng.uniform(0.5, 0.6, 400), rng.uniform(0.2, 0.3, 400)], 1
    )
    rgb = np.full((400, 3), 0.1)  # a dark soundbar
    rgb[:50] = (0.5, 0.6, 0.6)  # with some wall in its mask

    assert wall_tinted(pts, rgb, [wall, far_wall]).tolist() == [True] * 50 + [False] * 350
    # A wall-coloured object keeps everything; so does a picture flat on the wall, and one far off.
    assert not wall_tinted(pts, np.full((400, 3), 0.55), [wall]).any()
    flat = pts * (1, 1, 0) + (0, 0, 0.02)
    assert not wall_tinted(flat, rgb, [wall]).any()
    assert not wall_tinted(pts + (0, 0, 1.0), rgb, [wall]).any()
