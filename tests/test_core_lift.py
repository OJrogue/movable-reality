import numpy as np
import pytest

from movable_reality.coverage.cameras import Camera, look_at
from movable_reality.lift import (
    LiftParams,
    assign,
    dominant_yaw,
    floor_alignment,
    largest_cluster,
    support,
    vote,
)


def _box(rng, lo, hi, n):
    return rng.uniform(lo, hi, size=(n, 3))


def _ring(n=12, r=3.0, h=1.5, w=160, hgt=120):
    cams = []
    for i in range(n):
        a = 2 * np.pi * i / n
        eye = np.array([r * np.cos(a), h, r * np.sin(a)])
        cams.append(look_at(eye, np.array([0, 0.4, 0]), w, hgt, name=f"{i:06d}.jpg"))
    return cams


def test_vote_assigns_visible_object_and_respects_occlusion():
    rng = np.random.default_rng(0)
    obj = _box(rng, [-0.3, 0, -0.3], [0.3, 0.8, 0.3], 3000)
    floor = np.c_[rng.uniform(-2, 2, 6000), np.zeros(6000), rng.uniform(-2, 2, 6000)]
    xyz = np.vstack([obj, floor]).astype(np.float32)
    truth = np.r_[np.zeros(len(obj), int), np.ones(len(floor), int)]
    cams = _ring()
    labels = []
    for cam in cams:  # perfect masks: rasterise the true labels (object drawn last = on top)
        lab = np.full((cam.height, cam.width), -1, np.int32)
        uv, z = cam.project(xyz)
        for t in (1, 0):
            m = (truth == t) & (z > 0.05)
            px = np.clip(uv[m].astype(int), 0, [cam.width - 1, cam.height - 1])
            lab[px[:, 1], px[:, 0]] = t
        labels.append(lab)
    p = LiftParams(depth_grid=80)
    votes, seen = vote(cams, labels, xyz, np.ones(len(xyz), np.float32), 2, p)
    ids = assign(votes, seen, p)
    vis = seen[: len(obj)] >= p.min_views  # the box is solid: inner points are never seen
    assert vis.mean() > 0.2
    assert (ids[: len(obj)][vis] == 0).mean() > 0.9
    assert (ids[len(obj) :] == 0).mean() < 0.05  # floor under the box is hidden, not voted


def test_floor_alignment_recovers_tilted_floor():
    rng = np.random.default_rng(1)
    floor = np.c_[rng.uniform(-2, 2, 5000), np.zeros(5000), rng.uniform(-1, 1, 5000)]
    a = np.radians(20)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    tilted = floor @ R.T + [0.3, 1.2, -0.4]
    cams = np.array([[0, 1.5, 0]]) @ R.T + [0.3, 1.2, -0.4]
    M = floor_alignment(tilted, cams)
    world = tilted @ M[:3, :3].T + M[:3, 3]
    assert np.abs(world[:, 1]).max() < 0.01
    assert (cams @ M[:3, :3].T + M[:3, 3])[0, 1] > 1.0  # cameras above the floor


def test_dominant_yaw_finds_rotated_rectangle():
    rng = np.random.default_rng(2)
    pts = rng.uniform([-2, -1], [2, 1], size=(4000, 2))
    a = np.radians(30)
    rot = pts @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
    yaw = dominant_yaw(rot)
    assert min(abs(yaw - a), abs(yaw - a + np.pi / 2), abs(yaw - a - np.pi / 2)) < np.radians(2)


def test_largest_cluster_drops_floaters():
    rng = np.random.default_rng(3)
    pts = np.vstack([_box(rng, 0, 0.5, 2000), [[3, 3, 3], [3.01, 3, 3]]])
    keep = largest_cluster(pts, 0.05)
    assert keep[:2000].all() and not keep[2000:].any()


def test_support_merges_items_and_loose_splats():
    rng = np.random.default_rng(4)
    table = _box(rng, [-0.5, 0, -0.3], [0.5, 0.45, 0.3], 2000)
    vase = _box(rng, [-0.05, 0.45, -0.05], [0.05, 0.75, 0.05], 300)
    book = _box(rng, [0.2, 0.45, 0.0], [0.35, 0.5, 0.2], 100)  # unlabelled
    chair = _box(rng, [1.0, 0, 1.0], [1.5, 0.9, 1.5], 800)
    world = np.vstack([table, vase, book, chair])
    ids = np.r_[np.zeros(2000), np.ones(300), -np.ones(100), np.full(800, 2)].astype(int)
    out, merged = support(world, ids, [0, 1, 2], LiftParams())
    assert merged == {1: 0}
    assert (out[2000:2400] == 0).all() and (out[2400:] == 2).all()


def test_support_merges_an_item_when_the_table_holds_splats_above_its_top():
    # The lift gives the table some splats of the items on it, so its 98th percentile sits
    # well above the top surface; the item standing on the top must still merge.
    rng = np.random.default_rng(5)
    table = _box(rng, [-0.6, 0, -0.3], [0.6, 0.40, 0.3], 2000)
    on_table = _box(rng, [0.2, 0.40, -0.1], [0.4, 0.75, 0.1], 300)  # bowl splats the table took
    plant = _box(rng, [-0.3, 0.55, -0.05], [-0.2, 0.70, 0.05], 300)  # SAM missed the pot
    world = np.vstack([table, on_table, plant])
    ids = np.r_[np.zeros(2300), np.ones(300)].astype(int)
    _, merged = support(world, ids, [0, 1], LiftParams())
    assert merged == {1: 0}


def test_full_res_camera_inverts_mapanything_crop():
    from movable_reality.recon import full_res_camera

    full = look_at(np.array([0, 1.5, 3.0]), np.array([0, 0.5, 0]), 1920, 1080, name="a")
    # what MapAnything's preprocessing does to the frame: scale, floor, centre-crop to 518x294
    s = max(518 / 1920, 294 / 1080) + 1e-8
    rw, rh = int(1920 * s), int(1080 * s)
    left, top = (rw - 518) // 2, (rh - 294) // 2
    K = full.K.copy()
    K[0, 0] *= rw / 1920
    K[1, 1] *= rh / 1080
    K[0, 2] = (full.K[0, 2] + 0.5) * rw / 1920 - 0.5 - left
    K[1, 2] = (full.K[1, 2] + 0.5) * rh / 1080 - 0.5 - top
    small = Camera(name="a", width=518, height=294, K=K, R=full.R, t=full.t)
    back = full_res_camera(small, 1920, 1080)
    assert np.allclose(back.K, full.K, atol=1e-6)


@pytest.mark.parametrize("yaw_deg", [10, 25, 40])
@pytest.mark.parametrize("opencv", [False, True])
def test_floor_alignment_squares_rotated_room(yaw_deg, opencv):
    rng = np.random.default_rng(5)
    floor = np.c_[rng.uniform(-2, 2, 6000), np.zeros(6000), rng.uniform(-1, 1, 6000)]
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    if opencv:  # +Y down, as MapAnything's frame: flip about x
        R = np.diag([1.0, -1.0, -1.0]) @ R
    pts = floor @ R.T
    cams = np.array([[0.0, 1.5, 0.0]]) @ R.T
    M = floor_alignment(pts, cams)
    world = pts @ M[:3, :3].T + M[:3, 3]
    assert np.abs(world[:, 1]).max() < 0.01
    lo, hi = world.min(axis=0), world.max(axis=0)
    area = (hi[0] - lo[0]) * (hi[2] - lo[2])
    assert area < 8.0 * 1.03  # axis-aligned 4 x 2 m, not a rotated box


def test_ceiling_scale_puts_ceiling_at_measured_height():
    from movable_reality.lift import ceiling_scale, scaled

    rng = np.random.default_rng(0)
    ceiling_y = 1.42 + rng.normal(0, 0.01, 500)  # recon came out 2.6 / 1.42 too small
    s = ceiling_scale(ceiling_y, 2.6)
    assert abs(s - 2.6 / 1.42) < 0.01
    M = np.eye(4)
    M[1, 3] = 0.3
    p = np.array([1.0, 1.12, -2.0, 1.0])  # 1.42 m above the floor after M
    out = scaled(M, s) @ p
    assert abs(out[1] - 2.6) < 0.02
    assert abs(out[0] - s) < 1e-9 and abs(out[2] + 2 * s) < 1e-9


def test_ceiling_scale_rejects_too_few_or_negative():
    import pytest

    from movable_reality.lift import ceiling_scale

    with pytest.raises(ValueError):
        ceiling_scale(np.ones(10), 2.5)
    with pytest.raises(ValueError):
        ceiling_scale(-np.ones(200), 2.5)


def test_ceiling_from_histogram_finds_the_dense_layer():
    from movable_reality.lift import ceiling_from_histogram

    rng = np.random.default_rng(1)
    clutter = rng.uniform(0, 1.4, 5000)  # furniture and walls below and around
    walls = rng.uniform(1.4, 1.5, 800)  # walls continue up to the ceiling
    ceiling = 1.45 + rng.normal(0, 0.003, 2000)
    y = np.concatenate([clutter, walls, ceiling])
    layer = ceiling_from_histogram(y, cam_y=np.array([0.7, 0.8]))
    assert abs(np.median(layer) - 1.45) < 0.01


def test_ceiling_from_histogram_refuses_a_plain_tail():
    from movable_reality.lift import ceiling_from_histogram

    rng = np.random.default_rng(2)
    tail = 1.2 + rng.exponential(0.2, 5000)  # blender01: no ceiling, heights only tail off
    assert len(ceiling_from_histogram(tail, cam_y=np.array([1.1]))) == 0


def test_sparse_label_counts_only_its_queried_views():
    # Floor masks exist only on every third view (lift_ids.SPARSE_NOUNS). Counted against
    # every view the floor's share stays near 1/3 and it is dropped; per-label views fix it.
    rng = np.random.default_rng(0)
    obj = _box(rng, [-0.3, 0, -0.3], [0.3, 0.8, 0.3], 3000)
    floor = np.c_[rng.uniform(-2, 2, 6000), np.zeros(6000), rng.uniform(-2, 2, 6000)]
    xyz = np.vstack([obj, floor]).astype(np.float32)
    truth = np.r_[np.zeros(len(obj), int), np.ones(len(floor), int)]
    cams = _ring()
    labels, queried = [], []
    for i, cam in enumerate(cams):
        asked = i % 3 == 0
        lab = np.full((cam.height, cam.width), -1, np.int32)
        uv, z = cam.project(xyz)
        for t in (1, 0) if asked else (0,):
            m = (truth == t) & (z > 0.05)
            px = np.clip(uv[m].astype(int), 0, [cam.width - 1, cam.height - 1])
            lab[px[:, 1], px[:, 0]] = t
        labels.append(lab)
        queried.append(np.array([True, asked]))
    p = LiftParams(depth_grid=80, min_views=2)
    ones = np.ones(len(xyz), np.float32)
    v_all, s_all = vote(cams, labels, xyz, ones, 2, p)
    v_q, s_q = vote(cams, labels, xyz, ones, 2, p, queried=queried)
    seen_floor = s_q[len(obj) :, 1] >= p.min_views
    assert seen_floor.mean() > 0.3
    assert (assign(v_all, s_all, p)[len(obj) :][seen_floor] == 1).mean() < 0.1
    assert (assign(v_q, s_q, p)[len(obj) :][seen_floor] == 1).mean() > 0.8
    # the object, asked in every view, keeps its label
    ids = assign(v_q, s_q, p)[: len(obj)]
    vis = s_q[: len(obj), 0] >= p.min_views
    assert (ids[vis] == 0).mean() > 0.9
