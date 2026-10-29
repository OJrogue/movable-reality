"""Pinhole cameras in the OpenCV convention (x right, y down, z forward) and frame changes.

A `Camera` stores cam_from_world as (R, t): x_cam = R @ x_world + t. Coverage works in the
world frame of scene.json (+Y up, floor at y = 0); COLMAP poses come in the reconstruction
frame and are moved once with `Camera.in_frame(world_from_recon)`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

WORLD_UP = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class Camera:
    name: str
    width: int
    height: int
    K: np.ndarray  # (3, 3) intrinsics
    R: np.ndarray  # (3, 3) rotation, cam_from_world
    t: np.ndarray  # (3,) translation, cam_from_world

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:
        """Viewing direction (camera +Z) in world coordinates."""
        return self.R[2].copy()

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        """World points (N, 3) to camera coordinates (N, 3), float32."""
        R = self.R.astype(np.float32)
        t = self.t.astype(np.float32)
        return np.asarray(points, dtype=np.float32) @ R.T + t

    def project(self, points: np.ndarray, near: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates (N, 2) and depth (N,) of world points. Depth <= near is invalid
        (pixel coordinates are then meaningless)."""
        pc = self.to_camera(points)
        z = pc[:, 2]
        safe = np.where(z > near, z, np.float32(1.0))
        fx, fy, cx, cy = (
            np.float32(v) for v in (self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2])
        )
        uv = np.stack([pc[:, 0] / safe * fx + cx, pc[:, 1] / safe * fy + cy], axis=1)
        return uv, z

    def in_frustum(self, points: np.ndarray, near: float = 0.05, far: float = 1e9) -> np.ndarray:
        uv, z = self.project(points, near)
        return (
            (z > near)
            & (z < far)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < self.width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < self.height)
        )

    def scaled(self, long_side: int) -> Camera:
        """Same camera with the image resampled so its long side is `long_side` pixels."""
        s = long_side / max(self.width, self.height)
        K = self.K.copy()
        K[:2] *= s
        w, h = max(1, round(self.width * s)), max(1, round(self.height * s))
        return replace(self, width=w, height=h, K=K)

    def in_frame(self, new_from_old: np.ndarray) -> Camera:
        """Express this camera in another frame, given the 4x4 similarity new_from_old
        (rotation, uniform scale, translation). Depths are scaled with the frame."""
        A = np.asarray(new_from_old, dtype=np.float64)
        scale, rot = similarity_parts(A)
        # x_old = rot.T @ (x_new - T) / scale ; x_cam_old = R x_old + t
        # Keep metric depth in the new frame: x_cam = scale * x_cam_old.
        R_new = self.R @ rot.T
        t_new = scale * self.t - R_new @ A[:3, 3]
        return replace(self, R=R_new, t=t_new)


def similarity_parts(A: np.ndarray) -> tuple[float, np.ndarray]:
    """Uniform scale and rotation of a 4x4 similarity transform."""
    M = np.asarray(A, dtype=np.float64)[:3, :3]
    scale = float(np.cbrt(np.linalg.det(M)))
    if scale <= 0:
        raise ValueError("transform must be a proper similarity (positive determinant)")
    rot = M / scale
    if not np.allclose(rot @ rot.T, np.eye(3), atol=1e-4):
        raise ValueError("transform is not a similarity (rotation x uniform scale)")
    return scale, rot


def transform_points(A: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N, 3) points; result float32."""
    A = np.asarray(A, dtype=np.float32)
    return np.asarray(points, dtype=np.float32) @ A[:3, :3].T + A[:3, 3]


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    width: int,
    height: int,
    hfov_deg: float = 65.0,
    name: str = "",
) -> Camera:
    """OpenCV camera at `eye` looking at `target` in a +Y-up world (no roll)."""
    eye = np.asarray(eye, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, WORLD_UP)
    if np.linalg.norm(right) < 1e-6:  # looking straight up or down
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    f = width / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
    K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
    return Camera(name=name, width=width, height=height, K=K, R=R, t=-R @ eye)


def camera_from_yaw_pitch(
    eye: np.ndarray, yaw_deg: float, pitch_deg: float, width: int, height: int, hfov_deg: float
) -> Camera:
    """Camera at `eye` looking along yaw (0 = -Z, positive turns toward +X) and pitch
    (positive looks up)."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    d = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])
    return look_at(eye, np.asarray(eye) + d, width, height, hfov_deg)


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) to a rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to a COLMAP quaternion (w, x, y, z) with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q if q[0] >= 0 else -q
