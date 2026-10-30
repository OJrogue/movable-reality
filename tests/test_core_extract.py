"""extract_frames.py helpers (no video needed)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_focal35_matches_blender01_camera():
    k = _load("extract_frames").intrinsics_from_focal35(26.0, 1920, 1080)
    assert abs(k["fx"] - 1386.67) < 0.01  # blender01 GT: fx = fy = 1386.67 at 1920 x 1080
    assert k["fy"] == k["fx"]
    assert (k["cx"], k["cy"]) == (960.0, 540.0)


def test_focal35_portrait_uses_long_side():
    k = _load("extract_frames").intrinsics_from_focal35(26.0, 1080, 1920)
    assert abs(k["fx"] - 1386.67) < 0.01
