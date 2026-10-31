"""SfM refinement helpers: model-resolution intrinsics and the Sim(3) fit to MapAnything."""

from __future__ import annotations

import numpy as np

from movable_reality.coverage.cameras import Camera, look_at
from movable_reality.recon import full_res_camera, model_res_camera
from movable_reality.sfm import fit_to_reference
from movable_reality.splat_ops import similarity


def _cam(name: str, eye, target=(0.0, 1.0, 0.0)) -> Camera:
    return look_at(np.asarray(eye, float), np.asarray(target, float), 1920, 1080, name=name)


def test_model_res_camera_inverts_full_res_camera():
    small = Camera(
        "a", 518, 294, np.array([[380.0, 0, 259.0], [0, 381.0, 147.0], [0, 0, 1]]),
        np.eye(3), np.zeros(3),
    )  # fmt: skip
    full = full_res_camera(small, 1920, 1080)
    back = model_res_camera(full, 518, 294)
    assert (back.width, back.height) == (518, 294)
    np.testing.assert_allclose(back.K, small.K, atol=1e-6)


def test_fit_to_reference_recovers_similarity_and_moves_cameras():
    sfm = [_cam(f"{i:06d}.jpg", (np.cos(i), 1.5, np.sin(i) * 2)) for i in range(8)]
    a = 0.4
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    M_true = similarity(R, np.array([0.3, -0.2, 1.0]), 0.25)
    ref = [c.in_frame(M_true) for c in sfm[1:]]  # one frame missing from the reference
    M, stats = fit_to_reference(sfm, ref)
    assert stats["shared"] == 7
    assert stats["residual_rms"] < 1e-9
    np.testing.assert_allclose(M, M_true, atol=1e-9)
    moved = sfm[3].in_frame(M)
    np.testing.assert_allclose(moved.center, ref[2].center, atol=1e-9)
    np.testing.assert_allclose(moved.R, ref[2].R, atol=1e-9)
