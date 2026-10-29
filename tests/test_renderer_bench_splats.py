import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = load_script("bench_make_splats")


def test_room_has_exact_count_and_stays_in_the_room():
    room = bench.build_room(20_000, seed=3)
    assert len(room["xyz"]) == 20_000
    lo, hi = room["xyz"].min(0), room["xyz"].max(0)
    assert lo[1] >= -1e-6 and hi[1] <= bench.ROOM_H + 1e-6
    assert np.all(np.abs(lo[[0, 2]]) <= [bench.ROOM_X + 0.2, bench.ROOM_Z + 0.2])
    assert np.all(np.abs(hi[[0, 2]]) <= [bench.ROOM_X + 0.2, bench.ROOM_Z + 0.2])
    # A quarter or so of the splats sit on the floor.
    on_floor = np.mean(np.abs(room["xyz"][:, 1]) < 1e-4)
    assert 0.15 < on_floor < 0.3


def test_scales_opacities_and_quats_are_varied_and_valid():
    room = bench.build_room(20_000, seed=3)
    a = room["opacity"]
    assert a.min() > 0 and a.max() < 1
    assert np.mean(a < 0.5) > 0.05  # a tail of faint splats
    assert np.mean(a > 0.8) > 0.4  # mostly opaque
    s = room["scales"]
    assert s.min() > 0
    assert np.percentile(s.max(1), 99) / np.percentile(s.max(1), 1) > 10  # spread of sizes
    np.testing.assert_allclose(np.linalg.norm(room["quats"], axis=1), 1.0, atol=1e-5)
    assert np.all((room["rgb"] >= 0) & (room["rgb"] <= 1))


def test_surface_quats_put_the_thin_axis_on_the_normal():
    normals = np.array([[0, 1, 0], [0, -1, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1.0]])
    q = bench.quat_from_z_to(normals)
    w, x, y, z = q.T
    # Third column of the rotation matrix = R @ (0, 0, 1).
    col = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)
    np.testing.assert_allclose(col, normals, atol=1e-6)


def test_density_grows_with_n_in_the_same_room():
    small = bench.build_room(10_000, seed=0)["xyz"]
    large = bench.build_room(40_000, seed=0)["xyz"]
    np.testing.assert_allclose(small.min(0), large.min(0), atol=0.3)
    np.testing.assert_allclose(small.max(0), large.max(0), atol=0.3)


def test_writes_standard_ply_and_manifest(tmp_path):
    out = tmp_path / "bench_5000"
    assert bench.make_scene(5_000, out, seed=1)
    ply = PlyData.read(str(out / "objects" / "background.ply"))
    v = ply["vertex"]
    assert v.count == 5_000
    assert [p.name for p in v.properties] == bench.FIELDS
    assert not ply.text and ply.byte_order == "<"
    # Same seed, same file.
    first = (out / "objects" / "background.ply").read_bytes()
    assert not bench.make_scene(5_000, out, seed=1)  # skipped: exists
    assert bench.make_scene(5_000, out, seed=1, force=True)
    assert (out / "objects" / "background.ply").read_bytes() == first
    manifest = json.loads((out / "scene.json").read_text())
    assert manifest["scene_id"] == "bench_5000"
    assert manifest["background_ply"] == "objects/background.ply"
    assert manifest["objects"] == []
    assert manifest["world_from_recon"][1] == [0.0, 1.0, 0.0, 0.0]  # world frame, no flip


@pytest.mark.parametrize("n", [1, 7])
def test_tiny_counts(n):
    assert len(bench.build_room(n)["xyz"]) == n
