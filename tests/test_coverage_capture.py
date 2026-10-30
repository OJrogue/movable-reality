"""Scoring on real captures: cameras from recon_full/, eye height from the capture."""

from pathlib import Path

import numpy as np

from movable_reality.coverage.nbv import WalkParams
from movable_reality.coverage.pipeline import recon_dir, walk_for_capture


def test_eye_height_follows_the_cameras_and_scales_the_crouch():
    w = walk_for_capture(WalkParams(), np.array([0.9, 1.0, 1.1]), top=1.5, voxel=0.1)
    assert w.eye_height == 1.0
    assert w.exposure_heights == (1.0, 0.625)  # 1.6 and 1.0 scaled by 1.0 / 1.6


def test_eye_height_stays_inside_the_grid():
    w = walk_for_capture(WalkParams(), np.array([1.6, 1.7]), top=1.4, voxel=0.1)
    assert w.eye_height == 1.35
    assert max(w.exposure_heights) <= w.eye_height


def test_no_cameras_keeps_the_defaults():
    assert walk_for_capture(WalkParams(), np.array([]), top=2.5, voxel=0.1) == WalkParams()


def test_recon_dir_prefers_full_resolution_model(tmp_path: Path):
    assert recon_dir(tmp_path) == tmp_path / "recon"
    (tmp_path / "recon_full" / "sparse" / "0").mkdir(parents=True)
    assert recon_dir(tmp_path) == tmp_path / "recon_full"
