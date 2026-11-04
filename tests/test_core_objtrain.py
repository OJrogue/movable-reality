"""Per-object training, CPU side: installing the trained PLYs into the scene."""

from __future__ import annotations

import json

import numpy as np

from movable_reality.lod.split import write_vertices
from movable_reality.objtrain import install
from movable_reality.scene import Scene, SceneObject
from movable_reality.splat_ops import read_vertices


def _ply(path, xyz):
    names = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2")
    v = np.zeros(len(xyz), dtype=[(f, "f4") for f in names])
    v["x"], v["y"], v["z"] = np.asarray(xyz, float).T
    v["scale_0"] = v["scale_1"] = v["scale_2"] = np.log(0.01)
    write_vertices(path, v)


def test_install_replaces_plys_keeps_the_lift_and_refits_boxes(tmp_path):
    _ply(tmp_path / "objects" / "obj_chair_1.ply", [[0, 0, 0], [5, 5, 5]])
    _ply(tmp_path / "objects" / "background.ply", [[0, 0, 0]] * 3)
    trained = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(201)]  # 0..20, 0..40, 0..60
    _ply(tmp_path / "objects_trained" / "obj_chair_1.ply", trained)
    _ply(tmp_path / "objects_trained" / "background.ply", [[0, 0, 0]] * 2)
    m = np.eye(4)
    m[:3, 3] = (1, 0, 0)
    Scene(
        scene_id="t",
        world_from_recon=m.tolist(),
        objects=[SceneObject(id="chair#1", label="chair", ply="objects/obj_chair_1.ply",
                             lod="lod/x.rad", bbox_min=(0, 0, 0), bbox_max=(5, 5, 5))],
    ).save(tmp_path / "scene.json")  # fmt: skip

    counts = install(tmp_path, objects_from="trained")

    assert counts == {"chair#1": 201, "background": 2}
    assert len(read_vertices(tmp_path / "objects_lifted" / "obj_chair_1.ply")) == 2
    assert len(read_vertices(tmp_path / "objects" / "obj_chair_1.ply")) == 201
    obj = json.loads((tmp_path / "scene.json").read_text())["objects"][0]
    assert obj["lod"] is None
    np.testing.assert_allclose(obj["bbox_min"], [1.1, 0.2, 0.3], atol=1e-6)  # 0.5th pct
    np.testing.assert_allclose(obj["bbox_max"], [20.9, 39.8, 59.7], atol=1e-5)  # 99.5th pct


def test_fold_labels_adds_a_lost_sam_id_to_the_only_object_of_its_noun():
    from movable_reality.objtrain import fold_labels

    labels = {"coffee table#0": [6], "sofa#0": [1], "sofa#1": [2]}
    sam = {"1": "sofa#0", "2": "sofa#1", "3": "sofa#4", "6": "coffee table#0",
           "7": "coffee table#1", "15": "vase#1", "20": "floor#0"}  # fmt: skip
    out, folded = fold_labels(labels, sam)
    assert out["coffee table#0"] == [6, 7]
    assert out["sofa#0"] == [1] and out["sofa#1"] == [2]  # two sofas: sofa#4 stays out
    assert folded == {"coffee table#0": [7]}


def _surface(rng, centre, size, n, noise=0.002):
    """Points on the faces of a box: what a trained object's splat centres look like."""
    p = rng.uniform(-0.5, 0.5, (n, 3))
    axis = rng.integers(0, 3, n)
    p[np.arange(n), axis] = np.sign(p[np.arange(n), axis]) * 0.5
    return p * size + centre + rng.normal(0, noise, (n, 3))


def test_merge_duplicates_joins_copies_and_chains_to_the_largest():
    from movable_reality.objtrain import merge_duplicates

    rng = np.random.default_rng(0)
    sofa = _surface(rng, (0, 0.4, 0), (2.0, 0.8, 0.9), 20000)
    plant = _surface(rng, (3, 0.7, 0), (0.6, 1.4, 0.6), 12000)
    xyz = {
        "sofa#1": sofa,
        "armchair#0": sofa[rng.choice(len(sofa), 4000)] + rng.normal(0, 0.002, (4000, 3)),
        "plant#1": plant,
        # a re-acquired track: the upper half of the plant again, plus its own leaves
        "plant#7": np.concatenate([plant[plant[:, 1] > 0.7][:4000], plant[:1000] + (0, 0, 0.3)]),
        "table#0": _surface(rng, (6, 0.3, 0), (1.0, 0.6, 1.0), 5000),
    }
    xyz["plant#5"] = xyz["plant#7"][:3000] + rng.normal(0, 0.002, (3000, 3))
    assert merge_duplicates(xyz) == {"armchair#0": "sofa#1", "plant#7": "plant#1",
                                     "plant#5": "plant#1"}  # fmt: skip


def test_merge_duplicates_keeps_items_that_sit_inside_another_object():
    from movable_reality.objtrain import merge_duplicates

    rng = np.random.default_rng(1)
    shelf = _surface(rng, (0, 1.0, 0), (0.4, 2.0, 1.2), 30000)
    books = _surface(rng, (0, 1.2, 0.1), (0.25, 0.25, 0.4), 3000)  # on a shelf board
    assert merge_duplicates({"bookshelf#1": shelf, "book#9": books}) == {}


def test_install_joins_a_copy_and_shows_objects_an_old_merge_hid(tmp_path):
    rng = np.random.default_rng(2)
    plant = _surface(rng, (0, 0.7, 0), (0.6, 1.4, 0.6), 4000)
    extra = plant[:500] + (0, 0, 0.3)  # leaves only the second track saw
    copy = np.concatenate([plant[:2000], extra])
    shelf = _surface(rng, (3, 1.0, 0), (0.4, 2.0, 1.2), 8000)
    book = _surface(rng, (3, 1.2, 0.1), (0.25, 0.25, 0.4), 800)
    objs = []
    for oid, pts in (("plant#1", plant), ("plant#7", copy), ("bookshelf#1", shelf),
                     ("book#9", book)):  # fmt: skip
        name = f"obj_{oid.replace('#', '_')}.ply"
        _ply(tmp_path / "objects" / name, pts[:10])
        _ply(tmp_path / "objects_trained" / name, pts)
        objs.append(SceneObject(id=oid, label=oid.split("#")[0], ply=f"objects/{name}",
                                bbox_min=(0, 0, 0), bbox_max=(1, 1, 1),
                                visible=oid != "book#9"))  # fmt: skip
    Scene(scene_id="t", world_from_recon=np.eye(4).tolist(), objects=objs).save(
        tmp_path / "scene.json"
    )

    counts = install(tmp_path, objects_from="trained")

    shown = {
        o["id"]: o["visible"] for o in json.loads((tmp_path / "scene.json").read_text())["objects"]
    }
    assert shown == {"plant#1": True, "plant#7": False, "bookshelf#1": True, "book#9": True}
    # its own leaves joined (some land on the plant's surface), the copied half not
    assert 4000 + 300 <= counts["plant#1"] <= 4000 + 500


def test_high_needles_drops_only_long_thin_splats_above_the_cameras():
    from movable_reality.objtrain import high_needles

    names = ["x", "y", "z", "scale_0", "scale_1", "scale_2"]
    v = np.zeros(4, dtype=[(n, "f4") for n in names])
    v["y"] = [2.1, 2.1, 1.0, 2.1]  # world = recon here
    s = np.log([[0.40, 0.02, 0.01], [0.05, 0.05, 0.05], [0.40, 0.02, 0.01], [0.20, 0.10, 0.01]])
    for i, n in enumerate(["scale_0", "scale_1", "scale_2"]):
        v[n] = s[:, i]
    # needle up high, round splat up high, needle at camera height, wide flat splat up high
    assert high_needles(v, np.eye(4), cam_top=1.5).tolist() == [True, False, False, False]


def test_tame_splats_drops_haze_and_shortens_needles():
    from movable_reality.splat_ops import tame_splats

    names = ["x", "y", "z", "scale_0", "scale_1", "scale_2"]
    v = np.zeros(4, dtype=[(n, "f4") for n in names])
    v["x"] = [0, 1, 2, 3]
    s = np.array([[0.02, 0.02, 0.005], [0.40, 0.30, 0.20], [0.20, 0.01, 0.01], [0.10, 0.10, 0.01]])
    for i, n in enumerate(["scale_0", "scale_1", "scale_2"]):
        v[n] = np.log(s[:, i] / 2.0)  # two metres per recon unit

    out = tame_splats(v, world_per_unit=2.0, drop_sigma=0.3)

    assert out["x"].tolist() == [0, 2, 3]  # the 40 cm haze splat is gone
    got = np.exp(np.stack([out[f"scale_{i}"] for i in range(3)], 1)) * 2.0
    np.testing.assert_allclose(got[0], [0.02, 0.02, 0.005], rtol=1e-5)  # a normal splat stays
    np.testing.assert_allclose(got[1], [0.03, 0.01, 0.01], rtol=1e-5)  # needle: 3 x middle
    np.testing.assert_allclose(got[2], [0.08, 0.08, 0.01], rtol=1e-5)  # flat, capped at 8 cm


def test_merge_duplicates_joins_pieces_of_one_thing_only_under_one_noun():
    from movable_reality.objtrain import merge_duplicates

    rng = np.random.default_rng(3)
    # Upper and lower piece of a curtain: their boxes overlap, their surfaces do not touch.
    lower = _surface(rng, (0, 1.0, 0), (0.5, 2.0, 0.4), 8000)
    upper = _surface(rng, (0.05, 1.8, 0), (0.5, 1.6, 0.46), 4000)
    lamp = _surface(rng, (0.05, 1.4, 0), (0.3, 0.4, 0.3), 1000)  # a different noun inside
    far = _surface(rng, (3, 1.0, 0), (0.5, 2.0, 0.4), 6000)
    xyz = {"curtain#0": lower, "curtain#1": upper, "lamp#0": lamp, "curtain#2": far}
    labels = {o: o.split("#")[0] for o in xyz}
    assert merge_duplicates(xyz) == {}
    assert merge_duplicates(xyz, labels=labels) == {"curtain#1": "curtain#0"}


def test_install_default_keeps_lifted_splats_and_clears_the_shell(tmp_path):
    rng = np.random.default_rng(4)
    plant = _surface(rng, (0, 0.7, 0), (0.6, 1.4, 0.6), 4000)
    for name, pts in (("obj_plant_1.ply", plant[:50]), ("obj_plant_2.ply", plant[50:90])):
        _ply(tmp_path / "objects" / name, pts)
    _ply(tmp_path / "objects_trained" / "obj_plant_1.ply", plant)
    _ply(tmp_path / "objects_trained" / "obj_plant_2.ply", plant[:2000])  # a copy when trained
    _ply(tmp_path / "objects_trained" / "background.ply", [[0, 0, 0]] * 2)
    objs = [SceneObject(id=f"plant#{i}", label="plant", ply=f"objects/obj_plant_{i}.ply",
                        bbox_min=(0, 0, 0), bbox_max=(1, 1, 1)) for i in (1, 2)]  # fmt: skip
    Scene(scene_id="t", world_from_recon=np.eye(4).tolist(), objects=objs,
          room_planes=["room/shell_wall_0.glb"]).save(tmp_path / "scene.json")  # fmt: skip

    counts = install(tmp_path)

    scene = json.loads((tmp_path / "scene.json").read_text())
    # The duplicate is found on the trained splats, the output is the lifted ones, all of them.
    assert counts["plant#1"] == 90
    assert [o["visible"] for o in scene["objects"]] == [True, False]
    assert scene["room_planes"] == []
    assert scene["background_ply"] == "objects/background.ply"


def test_merge_duplicates_joins_end_to_end_pieces_only_for_a_noun_counted_once():
    from movable_reality.objtrain import merge_duplicates

    rng = np.random.default_rng(5)
    xyz = {
        "soundbar#1": _surface(rng, (0.5, 0.6, 0), (0.9, 0.15, 0.1), 3000),
        "soundbar#2": _surface(rng, (-0.2, 0.6, 0), (0.4, 0.15, 0.1), 1500),
        "pillow#1": _surface(rng, (3.0, 0.7, 0), (0.5, 0.5, 0.2), 2000),
        "pillow#2": _surface(rng, (3.6, 0.7, 0), (0.5, 0.5, 0.2), 2000),
        "vase#1": _surface(rng, (6.0, 0.8, 0), (0.2, 0.4, 0.2), 2000),
        "vase#2": _surface(rng, (6.3, 0.72, 0), (0.2, 0.24, 0.2), 1000),  # shorter: not one vase
    }
    labels = {o: o.split("#")[0] for o in xyz}
    got = merge_duplicates(xyz, labels=labels, single={"soundbar", "vase"})
    assert got == {"soundbar#2": "soundbar#1"}
    frames = {
        "picture frame#1": _surface(rng, (0.5, 1.5, 0), (0.8, 0.6, 0.05), 2000),
        "picture frame#2": _surface(rng, (1.5, 1.5, 0), (0.8, 0.6, 0.05), 2000),
    }
    labels = {o: "picture frame" for o in frames}  # side by side, as long as they are tall: two
    assert merge_duplicates(frames, labels=labels, single={"picture frame"}) == {}


def test_moved_items_are_the_background_splats_the_trainer_gave_the_object():
    from movable_reality.objtrain import moved_items

    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")]

    def verts(points):
        v = np.zeros(len(points), dtype=fields)
        v["x"], v["y"], v["z"] = np.asarray(points, "f4").T
        return v

    world = lambda v: np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)  # noqa: E731
    own, far = [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]
    book, halo, lump = [1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]
    trained = verts([own, book, halo, lump])
    trained["opacity"] = [1, 2, 3, 4]

    got = moved_items(trained, verts([own, far]), np.array([book, lump]), world)

    # The halo is on no background splat; the lump is on one, but outside the object's box.
    assert got["x"].tolist() == [1.0] and got["opacity"].tolist() == [2.0]
    assert got.dtype.names == tuple(n for n, _ in fields)


def test_end_to_end_joins_the_real_soundbar_boxes():
    from movable_reality.objtrain import end_to_end

    # blender07's trained boxes (5th to 95th percentile): stray splats inflate the second's depth.
    bar1 = np.array([[1.128, 0.564, -4.707], [1.929, 0.645, -4.338]])
    bar2 = np.array([[0.684, 0.575, -4.559], [1.008, 0.669, -4.163]])
    assert end_to_end(bar1, bar2, 0.25)
    basket = np.array([[-0.688, 0.142, -0.772], [-0.485, 0.752, -0.566]])
    assert not end_to_end(basket, basket + 0.3, 0.25)
