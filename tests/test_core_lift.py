import numpy as np

from movable_reality.coverage.cameras import look_at
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
