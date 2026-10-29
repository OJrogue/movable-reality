import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from movable_reality.lod.split import (
    object_ply_name,
    split_by_id,
    world_bbox,
    write_split,
)
from movable_reality.scene import OPENCV_TO_Y_UP

IDENTITY = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]


def synthetic_vertices(xyz: np.ndarray) -> np.ndarray:
    """3DGS-style vertices with an SH rest band and the coverage lane's `generated` flag."""
    fields = [(k, "f4") for k in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity")]
    fields += [(f"f_rest_{i}", "f4") for i in range(3)] + [("generated", "u1")]
    v = np.zeros(len(xyz), dtype=fields)
    v["x"], v["y"], v["z"] = np.asarray(xyz, dtype=np.float32).T
    v["f_rest_0"] = np.arange(len(xyz))  # tags each splat so order can be checked
    v["generated"] = np.arange(len(xyz)) % 2
    return v


def test_object_ply_name_is_filesystem_safe():
    assert object_ply_name("chair#1") == "objects/obj_chair_1.ply"
    assert object_ply_name("floor lamp#12") == "objects/obj_floor_lamp_12.ply"
    assert object_ply_name("gen#3") == "objects/obj_gen_3.ply"


def test_split_by_id_keeps_order_and_every_splat():
    v = synthetic_vertices(np.zeros((6, 3)))
    groups = split_by_id(v, np.array([-1, 0, 0, 1, -1, 0]))
    assert sorted(groups) == [-1, 0, 1]
    assert groups[0]["f_rest_0"].tolist() == [1, 2, 5]
    assert sum(len(g) for g in groups.values()) == 6
    with pytest.raises(ValueError, match="shape"):
        split_by_id(v, np.array([0, 1]))


def test_world_bbox_applies_world_from_recon():
    v = synthetic_vertices(np.array([[0.0, 1.0, 2.0], [1.0, 3.0, 4.0]]))
    assert world_bbox(v, IDENTITY) == ((0.0, 1.0, 2.0), (1.0, 3.0, 4.0))
    # OpenCV +Y down, +Z forward -> world +Y up: y and z flip sign.
    lo, hi = world_bbox(v, OPENCV_TO_Y_UP)
    assert lo == pytest.approx((0.0, -3.0, -4.0)) and hi == pytest.approx((1.0, -1.0, -2.0))


def test_write_split_roundtrip_keeps_all_properties(tmp_path):
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-1, 1, (300, 3))
    ids = np.full(300, -1)
    ids[:100] = 0
    ids[100:150] = 7
    ids[150:160] = 99  # an id the manifest does not know -> background
    scene_ply = tmp_path / "scene.ply"
    PlyData([PlyElement.describe(synthetic_vertices(xyz), "vertex")]).write(str(scene_ply))

    out = write_split(scene_ply, ids, {0: "chair#1", 7: "floor lamp#2"}, tmp_path, IDENTITY)

    assert set(out) == {"background", "chair#1", "floor lamp#2"}
    rel, n, (lo, hi) = out["chair#1"]
    assert rel == "objects/obj_chair_1.ply" and n == 100
    assert lo == pytest.approx(tuple(xyz[:100].min(0)), abs=1e-6)
    assert hi == pytest.approx(tuple(xyz[:100].max(0)), abs=1e-6)
    assert out["background"][1] == 300 - 150

    chair = PlyData.read(str(tmp_path / rel))["vertex"].data
    assert chair.dtype.names == synthetic_vertices(xyz[:1]).dtype.names
    assert chair["f_rest_0"].tolist() == list(range(100))
    assert chair["generated"].tolist() == [i % 2 for i in range(100)]
    total = sum(len(PlyData.read(str(tmp_path / r))["vertex"].data) for r, _, _ in out.values())
    assert total == 300
