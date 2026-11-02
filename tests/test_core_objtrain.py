"""Per-object training, CPU side: installing the trained PLYs into the scene."""

from __future__ import annotations

import json

import numpy as np

from movable_reality.lod.split import write_vertices
from movable_reality.objtrain import install
from movable_reality.scene import Scene, SceneObject
from movable_reality.splat_ops import read_vertices


def _ply(path, xyz):
    v = np.zeros(len(xyz), dtype=[(f, "f4") for f in ("x", "y", "z", "opacity")])
    v["x"], v["y"], v["z"] = np.asarray(xyz, float).T
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

    counts = install(tmp_path)

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


def test_merge_duplicates_joins_one_thing_under_two_nouns():
    from movable_reality.objtrain import merge_duplicates

    rng = np.random.default_rng(0)

    def cloud(centre, n):
        v = np.zeros(n, dtype=[(f, "f4") for f in ("x", "y", "z")])
        xyz = rng.uniform(-0.4, 0.4, (n, 3)) + centre
        v["x"], v["y"], v["z"] = xyz.T
        return v

    verts = {"sofa#1": cloud((0, 0, 0), 900), "armchair#0": cloud((0.05, 0, 0), 300),
             "table#0": cloud((2, 0, 0), 500)}  # fmt: skip
    assert merge_duplicates(verts) == {"armchair#0": "sofa#1"}


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
