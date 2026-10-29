import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from movable_reality.lod import build
from movable_reality.lod.build import build_scene_lod, find_build_lod, rad_name, spark_version
from movable_reality.lod.rad import read_rad_header
from movable_reality.scene import Scene, SceneObject


def write_ply(path: Path, n: int) -> None:
    fields = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    fields += ["scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    v = np.zeros(n, dtype=[(f, "f4") for f in fields])
    rng = np.random.default_rng(n)
    v["x"], v["y"], v["z"] = rng.uniform(-1, 1, (n, 3)).T
    v["opacity"] = 2.0
    for k in ("scale_0", "scale_1", "scale_2"):
        v[k] = np.log(0.01)
    v["rot_0"] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")]).write(str(path))


def make_scene(root: Path, sizes: dict[str, int]) -> Path:
    write_ply(root / "objects" / "background.ply", 50)
    objects = []
    for oid, n in sizes.items():
        ply = f"objects/obj_{oid.replace('#', '_')}.ply"
        if n >= 0:
            write_ply(root / ply, n)
        objects.append(
            SceneObject(id=oid, label=oid, ply=ply, bbox_min=(0, 0, 0), bbox_max=(1, 1, 1))
        )
    path = root / "scene.json"
    Scene(scene_id="t", objects=objects).save(path)
    return path


@pytest.fixture
def fake_build_lod(tmp_path) -> Path:
    """Stand-in for Spark's build-lod: writes <stem>-lod.rad with a valid header, logs calls."""
    meta = json.dumps({"version": 1, "type": "gsplat", "count": 2, "lodTree": True}).encode()
    header = tmp_path / "header.rad"
    header.write_bytes(b"RAD0" + struct.pack("<I", len(meta)) + meta)
    script = tmp_path / "build-lod"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{tmp_path}/calls.log"\n'
        'for a; do f="$a"; done\n'
        f'cp "{header}" "${{f%.ply}}-lod.rad"\n'
    )
    script.chmod(0o755)
    return script


def calls(tmp_path: Path) -> int:
    log = tmp_path / "calls.log"
    return len(log.read_text().splitlines()) if log.exists() else 0


def test_rad_name_and_spark_version():
    assert rad_name("objects/obj_chair_1.ply") == "lod/obj_chair_1.rad"
    assert rad_name("objects/background.ply") == "lod/background.rad"
    assert spark_version() == "2.4.0"  # web/package.json pin; build-lod must match it


def test_find_build_lod_prefers_env(tmp_path, monkeypatch, fake_build_lod):
    monkeypatch.setenv("SPARK_BUILD_LOD", str(fake_build_lod))
    assert find_build_lod() == fake_build_lod
    monkeypatch.setenv("SPARK_BUILD_LOD", str(tmp_path / "missing"))
    assert find_build_lod() is None


def test_build_writes_trees_and_manifest_then_skips(tmp_path, fake_build_lod):
    scene_dir = tmp_path / "scene"
    path = make_scene(scene_dir, {"chair#1": 10, "lamp#2": 0, "ghost#3": -1})

    results = build_scene_lod(path, fake_build_lod, log=lambda _: None)

    status = {r.ply: r.status for r in results}
    assert status == {
        "objects/background.ply": "built",
        "objects/obj_chair_1.ply": "built",
        "objects/obj_lamp_2.ply": "empty",
        "objects/obj_ghost_3.ply": "missing",
    }
    scene = Scene.load(path)
    assert scene.background_lod == "lod/background.rad"
    assert scene.get("chair#1").lod == "lod/obj_chair_1.rad"
    assert scene.get("lamp#2").lod is None and scene.get("ghost#3").lod is None
    assert read_rad_header(scene_dir / "lod" / "obj_chair_1.rad").lod_tree
    assert calls(tmp_path) == 2
    assert "--quality" in (tmp_path / "calls.log").read_text()

    build_scene_lod(path, fake_build_lod, log=lambda _: None)
    assert calls(tmp_path) == 2  # idempotent
    build_scene_lod(path, fake_build_lod, force=True, method="quick", log=lambda _: None)
    assert calls(tmp_path) == 4
    assert "--quick" in (tmp_path / "calls.log").read_text().splitlines()[-1]


def test_stale_tree_is_rebuilt(tmp_path, fake_build_lod):
    path = make_scene(tmp_path / "scene", {"chair#1": 10})
    build_scene_lod(path, fake_build_lod, log=lambda _: None)
    ply = tmp_path / "scene" / "objects" / "obj_chair_1.ply"
    rad = tmp_path / "scene" / "lod" / "obj_chair_1.rad"
    later = rad.stat().st_mtime + 10
    os.utime(ply, (later, later))  # object re-split after the tree was built
    build_scene_lod(path, fake_build_lod, log=lambda _: None)
    assert calls(tmp_path) == 3


def test_manifest_edits_during_build_are_kept(tmp_path, fake_build_lod, monkeypatch):
    path = make_scene(tmp_path / "scene", {"chair#1": 10})
    real = build.build_rad

    def build_and_edit(*args, **kwargs):
        real(*args, **kwargs)
        s = Scene.load(path)
        s.move("chair#1", 0.5, 0.0, 0.0)  # the agent edits while trees are building
        s.save(path)

    monkeypatch.setattr(build, "build_rad", build_and_edit)
    build_scene_lod(path, fake_build_lod, log=lambda _: None)
    scene = Scene.load(path)
    assert scene.get("chair#1").transform.translation[0] == pytest.approx(1.0)
    assert scene.get("chair#1").lod == "lod/obj_chair_1.rad"


def test_failure_without_output_raises(tmp_path):
    silent = tmp_path / "build-lod"
    silent.write_text("#!/bin/sh\necho 'Stopping processing due to invalid splats!'\n")
    silent.chmod(0o755)
    path = make_scene(tmp_path / "scene", {})
    with pytest.raises(RuntimeError, match="invalid splats"):
        build_scene_lod(path, silent, log=lambda _: None)


REAL_BUILD_LOD = find_build_lod()


@pytest.mark.skipif(REAL_BUILD_LOD is None, reason="run scripts/build_lod.py --setup first")
def test_real_build_lod_on_synthetic_ply(tmp_path):
    assert REAL_BUILD_LOD is not None
    path = make_scene(tmp_path / "scene", {"chair#1": 2000})
    results = build_scene_lod(path, REAL_BUILD_LOD, log=lambda _: None)
    chair = next(r for r in results if r.ply.endswith("chair_1.ply"))
    header = read_rad_header(tmp_path / "scene" / chair.rad)
    assert header.lod_tree and header.stats["input_splat_count"] == 2000
    # Leaves plus interior nodes; build-lod prunes a few, but a tree is larger than its input.
    assert header.count > 2000
