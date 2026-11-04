"""run_pipeline.py: freshness and the plan, on fake scene folders (nothing runs)."""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look their module up
    spec.loader.exec_module(mod)
    return mod


rp = _load("run_pipeline")
ARGS = argparse.Namespace(scene="t", video=None, focal35=None, preset="quality",
                          ceiling_height=None, scale=None, skip="", real_capture=False,
                          group="")  # fmt: skip


def _touch(root: Path, rel: str, t: float) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    os.utime(p, (t, t))


def _complete(root: Path, t: float = 1000.0) -> None:
    """Every stage's outputs, all written at time t."""
    for st in rp.stages(ARGS):
        for o in st.outputs:
            _touch(root, f"{o}/f" if o in ("planes", "lod", "coverage") else o, t)
    _touch(root, "ids/gaussian_ids.npy", t)


def test_freshness_todo_stale_done(tmp_path):
    lift = next(st for st in rp.stages(ARGS) if st.name == "lift")
    assert rp.freshness(tmp_path, lift) == "todo"
    _complete(tmp_path)
    assert rp.freshness(tmp_path, lift) == "done"
    _touch(tmp_path, "inventory.json", 2000.0)  # an input newer than the outputs
    assert rp.freshness(tmp_path, lift) == "stale"


def test_up_to_date_scene_skips_everything(tmp_path):
    _complete(tmp_path)
    plan = rp.stages(ARGS)
    v = rp.decide(tmp_path, plan, 0, len(plan) - 1, set())
    assert all(x == "skip (up to date)" for x in v.values())


def test_a_stale_stage_forces_every_later_one(tmp_path):
    _complete(tmp_path)
    _touch(tmp_path, "ids/gaussian_ids.npy", 2000.0)  # lift output newer than objects' output
    plan = rp.stages(ARGS)
    forced: set[str] = set()
    v = rp.decide(tmp_path, plan, 0, len(plan) - 1, forced)
    assert v["lift"] == "skip (up to date)"
    assert v["objects"] == "run (stale)"
    assert v["complete"] == "run (forced)" and v["coverage"] == "run (forced)"
    assert {"complete", "shell", "bake", "lod", "coverage"} <= forced


def test_sam_does_not_force_reconstruct(tmp_path):
    # A scene with frames and an inventory but no lift yet: SAM is todo, reconstruct has run.
    _touch(tmp_path, "frames.json", 1000.0)
    _touch(tmp_path, "sam_input.mp4", 1000.0)
    _touch(tmp_path, "inventory.json", 1000.0)
    _touch(tmp_path, "splat/scene.ply", 1000.0)
    plan = rp.stages(ARGS)
    v = rp.decide(tmp_path, plan, 0, len(plan) - 1, set())
    assert v["sam"] == "run (todo)"
    assert v["reconstruct"] == "skip (up to date)"
    assert v["lift"] == "run (todo)"


def test_only_runs_one_stage(tmp_path):
    _complete(tmp_path)
    plan = rp.stages(ARGS)
    i = rp.NAMES.index("shell")
    v = rp.decide(tmp_path, plan, i, i, {"shell"})
    assert v["shell"] == "run (forced)"
    assert {k for k, x in v.items() if x.startswith("run")} == {"shell"}
    assert v["lod"] == "out of range"


def test_bake_runs_before_lod():
    assert rp.NAMES.index("bake") < rp.NAMES.index("lod")


def test_ceiling_height_reaches_the_room_shell():
    a = argparse.Namespace(**{**vars(ARGS), "ceiling_height": 2.7})
    shell = next(st for st in rp.stages(a) if st.name == "shell")
    assert shell.cmd[-2:] == ["--height", "2.7"]
    assert "--height" not in next(st for st in rp.stages(ARGS) if st.name == "shell").cmd


def test_real_capture_reaches_only_the_reconstruction():
    a = argparse.Namespace(**{**vars(ARGS), "real_capture": True})
    with_flag = [st.name for st in rp.stages(a) if "--real-capture" in st.cmd]
    assert with_flag == ["reconstruct"]
    assert not [st.name for st in rp.stages(ARGS) if "--real-capture" in st.cmd]


def test_group_reaches_only_the_lift():
    a = argparse.Namespace(**{**vars(ARGS), "group": "table=chair"})
    assert [st.name for st in rp.stages(a) if "--group" in st.cmd] == ["lift"]


def test_a_stage_that_is_off_forces_nothing(tmp_path):
    (tmp_path / "in.txt").write_text("x")
    plan = [
        rp.Stage("a", "", ["x"], ["in.txt"], ["a.txt"]),
        rp.Stage("b", "", ["x"], ["in.txt"], ["b.txt"]),
    ]
    (tmp_path / "b.txt").write_text("x")
    v = rp.decide(tmp_path, plan, 0, 1, set(), off=("a",))
    assert v == {"a": "off", "b": "skip (up to date)"}


def test_a_resume_keeps_room_splats_when_the_background_came_from_it(tmp_path):
    assert not rp.uses_room_splats(tmp_path)  # no scene.json
    scene = tmp_path / "scene.json"
    scene.write_text('{"background_ply": "objects/background.ply"}', encoding="utf-8")
    assert not rp.uses_room_splats(tmp_path)
    scene.write_text('{"background_ply": "objects/background_clean.ply"}', encoding="utf-8")
    assert rp.uses_room_splats(tmp_path)
