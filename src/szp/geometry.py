"""SE(3) primitives.

Small, dependency-free and deliberately strict.  Every rotation that goes into
a transform is checked for orthonormality on construction, because a silently
denormalised rotation matrix in an extrinsic file produces a systematic range
error of a few centimetres that looks exactly like sensor noise and is
miserable to find later.

Angle convention for ``rpy``: intrinsic Z-Y-X, i.e. R = Rz(yaw) Ry(pitch)
Rx(roll), which is the ROS convention.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

__all__ = [
    "rotation_from_rpy",
    "rpy_from_rotation",
    "rotation_from_quaternion",
    "quaternion_from_rotation",
    "se3",
    "se3_inverse",
    "transform_points",
    "transform_directions",
    "is_rotation",
    "project_points",
    "backproject",
    "so3_exp",
    "so3_log",
]

_ORTHONORMAL_TOL = 1e-6


def is_rotation(R: np.ndarray, tol: float = _ORTHONORMAL_TOL) -> bool:
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        return False
    if not np.allclose(R.T @ R, np.eye(3), atol=tol):
        return False
    return bool(abs(np.linalg.det(R) - 1.0) < max(tol, 1e-8))


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic Z-Y-X Euler angles, radians, to a rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def rpy_from_rotation(R: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of :func:`rotation_from_rpy`.  Gimbal lock resolved with roll = 0."""
    R = np.asarray(R, dtype=np.float64)
    sp = -R[2, 0]
    if abs(sp) > 1.0 - 1e-9:
        pitch = float(np.sign(sp) * np.pi / 2)
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]) * np.sign(sp))
        return roll, pitch, yaw
    pitch = float(np.arcsin(sp))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def rotation_from_quaternion(q: Sequence[float]) -> np.ndarray:
    """Quaternion (w, x, y, z) to a rotation matrix.  Normalised on the way in."""
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_from_rotation(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to quaternion (w, x, y, z), w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array(
            [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
        )
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def se3(R: np.ndarray, t: Iterable[float]) -> np.ndarray:
    """Build a 4x4 homogeneous transform, validating the rotation."""
    R = np.asarray(R, dtype=np.float64)
    if not is_rotation(R):
        raise ValueError(
            "rotation is not orthonormal with det +1; "
            f"max |RtR - I| = {np.abs(R.T @ R - np.eye(3)).max():.3e}, "
            f"det = {np.linalg.det(R):.9f}"
        )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(list(t), dtype=np.float64).reshape(3)
    return T


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Inverse of a homogeneous transform, without a general matrix inverse."""
    T = np.asarray(T, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply ``T`` to an (N, 3) array of points.  Returns (N, 3)."""
    points = np.asarray(points, dtype=np.float64)
    single = points.ndim == 1
    p = points.reshape(-1, 3)
    out = p @ np.asarray(T, dtype=np.float64)[:3, :3].T + np.asarray(T)[:3, 3]
    return out.reshape(3) if single else out


def transform_directions(T: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate an (N, 3) array of direction vectors, ignoring translation."""
    vectors = np.asarray(vectors, dtype=np.float64)
    single = vectors.ndim == 1
    v = vectors.reshape(-1, 3)
    out = v @ np.asarray(T, dtype=np.float64)[:3, :3].T
    return out.reshape(3) if single else out


def project_points(K: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    """Pinhole projection of (N, 3) camera-frame points to (N, 2) pixels.

    Points at or behind the image plane come back as NaN rather than as a
    plausible-looking pixel, because a person behind the camera projecting to a
    valid-looking coordinate is precisely the sort of thing that silently
    removes a detection.
    """
    p = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    z = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.asarray(K)[0, 0] * p[:, 0] / z + np.asarray(K)[0, 2]
        v = np.asarray(K)[1, 1] * p[:, 1] / z + np.asarray(K)[1, 2]
    bad = ~np.isfinite(z) | (z <= 1e-6)
    u = np.where(bad, np.nan, u)
    v = np.where(bad, np.nan, v)
    return np.stack([u, v], axis=1)


def backproject(K: np.ndarray, pixels: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Turn (N, 2) pixels plus (N,) metric depths into (N, 3) camera-frame points.

    ``depth`` is the z coordinate, not the range along the ray, which is what
    every RGB-D sensor reports.
    """
    px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    d = np.asarray(depth, dtype=np.float64).reshape(-1)
    K = np.asarray(K, dtype=np.float64)
    x = (px[:, 0] - K[0, 2]) / K[0, 0] * d
    y = (px[:, 1] - K[1, 2]) / K[1, 1] * d
    return np.stack([x, y, d], axis=1)


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues: a rotation vector to a rotation matrix."""
    w = np.asarray(w, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Inverse of :func:`so3_exp`."""
    R = np.asarray(R, dtype=np.float64)
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    if abs(theta - np.pi) < 1e-6:
        # Near pi the antisymmetric part vanishes; recover the axis from the
        # symmetric part instead.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(axis))
        axis = axis * np.sign(A[i]) * np.sign(axis[i]) if axis[i] > 0 else axis
        signs = np.sign([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        signs[signs == 0] = 1.0
        return axis * signs * theta
    return theta / (2.0 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )
