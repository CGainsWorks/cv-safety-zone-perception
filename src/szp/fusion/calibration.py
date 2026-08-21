"""Extrinsic calibration: camera to 2D lidar and camera to floor.

Camera-lidar
------------
A 2D lidar sees a line, not a surface, so the usual trick of matching a point
cloud to a mesh is not available.  The method used here is the planar-target
constraint of Zhang and Pless (IROS 2004), which needs nothing but a
checkerboard:

    the camera gives the board plane in its own frame as a unit normal ``n``
    and a distance ``d``; every lidar return that lands on the board must
    satisfy ``n . (R p + t) = d``.

One board pose gives one such constraint direction, so several poses at
genuinely different orientations are needed.  Because the lidar's points all
have ``z = 0`` in its own frame, the third column of ``R`` never appears in the
equations; it is recovered afterwards as ``r1 x r2``.  That is a real property
of the geometry rather than a shortcut and it is why this routine solves for
nine unknowns and not twelve.

The linear solution is a starting point only.  It is followed by a
Levenberg-Marquardt refinement on the same residual, parametrised minimally
over se(3), which is what the reported residual comes from.

Camera-floor
------------
``estimate_floor_plane`` fits a plane to the depth map by RANSAC and returns
the transform that puts the floor at ``z = 0`` with ``+z`` up.  Yaw and the
origin in the floor plane are not observable from a floor fit; they come from
one surveyed point, which ``world_from_floor`` takes as an argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from szp.geometry import backproject, se3, se3_inverse, so3_exp

__all__ = [
    "PlaneObservation",
    "CalibrationResult",
    "calibrate_camera_lidar",
    "estimate_floor_plane",
    "world_from_floor",
]


@dataclass(frozen=True)
class PlaneObservation:
    """One board pose.

    ``normal`` is the unit normal of the board plane in the camera frame and
    ``distance`` the perpendicular distance from the camera centre, both from
    the usual checkerboard pose estimate.  ``lidar_points`` is the (N, 2) set
    of scan returns that fall on the board, in the lidar frame, metres.
    """

    normal: np.ndarray
    distance: float
    lidar_points: np.ndarray

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(n))
        if norm < 1e-9:
            raise ValueError("plane normal has zero length")
        object.__setattr__(self, "normal", n / norm)
        pts = np.asarray(self.lidar_points, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            raise ValueError("a plane observation with no lidar points is not usable")
        object.__setattr__(self, "lidar_points", pts)


@dataclass(frozen=True)
class CalibrationResult:
    T_camera_lidar: np.ndarray
    #: Point-to-plane residuals in metres, one per lidar point, after refinement.
    residuals_m: np.ndarray
    #: Residuals from the closed-form initialisation, kept so the report can
    #: show that the refinement actually did something.
    linear_residuals_m: np.ndarray
    n_observations: int
    n_points: int
    converged: bool

    @property
    def rms_m(self) -> float:
        return float(np.sqrt(np.mean(self.residuals_m**2)))

    @property
    def max_abs_m(self) -> float:
        return float(np.max(np.abs(self.residuals_m)))

    def summary(self) -> dict:
        return {
            "rms_m": self.rms_m,
            "max_abs_m": self.max_abs_m,
            "linear_rms_m": float(np.sqrt(np.mean(self.linear_residuals_m**2))),
            "n_observations": self.n_observations,
            "n_points": self.n_points,
            "converged": self.converged,
            "translation_m": [float(v) for v in self.T_camera_lidar[:3, 3]],
        }


def _residuals(T: np.ndarray, observations: Sequence[PlaneObservation]) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out: List[np.ndarray] = []
    for obs in observations:
        p = np.concatenate([obs.lidar_points, np.zeros((len(obs.lidar_points), 1))], axis=1)
        cam = p @ R.T + t
        out.append(cam @ obs.normal - obs.distance)
    return np.concatenate(out) if out else np.zeros(0)


def _linear_initialisation(observations: Sequence[PlaneObservation]) -> np.ndarray:
    """Least squares for [r1; r2; t] under ``n . (x r1 + y r2 + t) = d``."""
    rows: List[np.ndarray] = []
    rhs: List[float] = []
    for obs in observations:
        n = obs.normal
        for x, y in obs.lidar_points:
            rows.append(np.concatenate([x * n, y * n, n]))
            rhs.append(obs.distance)
    A = np.asarray(rows, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    if A.shape[0] < 9:
        raise ValueError(
            f"need at least 9 lidar points on the board across all poses, got {A.shape[0]}"
        )
    rank = int(np.linalg.matrix_rank(A, tol=1e-8))
    if rank < 9:
        raise ValueError(
            "the board poses do not constrain the extrinsic: the design matrix has rank "
            f"{rank} of 9.  Use more poses at genuinely different orientations -- rotating "
            "the board about the lidar's scan axis alone does not add information."
        )
    solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    r1, r2, t = solution[0:3], solution[3:6], solution[6:9]

    # Project [r1 r2] onto the nearest pair of orthonormal columns, then take
    # the third column from the cross product.  A 2D lidar cannot observe it.
    M = np.stack([r1, r2], axis=1)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    Q = U @ Vt
    r1, r2 = Q[:, 0], Q[:, 1]
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    # Re-orthonormalise the whole thing; the cross product can leave a rotation
    # a few ulps off det = 1 and geometry.se3 is strict about that.
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return se3(R, t)


def calibrate_camera_lidar(
    observations: Sequence[PlaneObservation],
    *,
    max_iterations: int = 200,
) -> CalibrationResult:
    """Solve for ``T_camera_lidar`` from planar-target observations.

    Raises ``ValueError`` when the poses do not constrain the problem, rather
    than returning a confident-looking transform built from a rank-deficient
    system.
    """
    if len(observations) < 3:
        raise ValueError(
            f"at least 3 board poses are needed, got {len(observations)}; "
            "five or more at clearly different tilts is what the residual in "
            "docs/calibration.md was measured with"
        )
    T0 = _linear_initialisation(observations)
    linear_residuals = _residuals(T0, observations)

    R0, t0 = T0[:3, :3], T0[:3, 3]

    def unpack(x: np.ndarray) -> np.ndarray:
        return se3(so3_exp(x[:3]) @ R0, t0 + x[3:])

    def cost(x: np.ndarray) -> np.ndarray:
        return _residuals(unpack(x), observations)

    result = least_squares(cost, np.zeros(6), method="lm", max_nfev=max_iterations * 6)
    T = unpack(result.x)
    return CalibrationResult(
        T_camera_lidar=T,
        residuals_m=_residuals(T, observations),
        linear_residuals_m=linear_residuals,
        n_observations=len(observations),
        n_points=sum(len(o.lidar_points) for o in observations),
        converged=bool(result.success),
    )


def estimate_floor_plane(
    depth: np.ndarray,
    K: np.ndarray,
    *,
    max_range_m: float = 12.0,
    inlier_m: float = 0.03,
    iterations: int = 400,
    min_inlier_fraction: float = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float, float]:
    """RANSAC a plane through a depth map.

    Returns ``(normal, offset, inlier_fraction)`` in the camera frame, with the
    normal signed so that it points away from the camera's optical axis and
    the plane defined by ``normal . p = offset``.
    """
    rng = rng or np.random.default_rng(0)
    h, w = depth.shape[:2]
    vv, uu = np.mgrid[0:h, 0:w]
    d = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(d) & (d > 0.1) & (d < max_range_m)
    if valid.sum() < 100:
        raise ValueError(f"only {int(valid.sum())} usable depth pixels; cannot fit a floor")
    pts = backproject(
        K,
        np.stack([uu[valid].ravel(), vv[valid].ravel()], axis=1),
        d[valid].ravel(),
    )
    # Subsample: a 640x480 map is 300k points and RANSAC does not need them.
    if len(pts) > 20000:
        pts = pts[rng.choice(len(pts), 20000, replace=False)]

    best_inliers = np.zeros(len(pts), dtype=bool)
    best_count = 0
    for _ in range(iterations):
        idx = rng.choice(len(pts), 3, replace=False)
        a, b, c = pts[idx]
        n = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(n))
        if norm < 1e-9:
            continue
        n = n / norm
        offset = float(n @ a)
        inliers = np.abs(pts @ n - offset) < inlier_m
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    fraction = best_count / len(pts)
    if fraction < min_inlier_fraction:
        raise ValueError(
            f"no dominant plane in this depth map: best consensus {fraction:.1%} of "
            f"{len(pts)} points, threshold {min_inlier_fraction:.0%}.  Point the camera "
            "at an empty patch of floor."
        )
    # Total least squares on the consensus set.
    inlier_pts = pts[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    normal = Vt[-1]
    if normal @ centroid < 0:
        normal = -normal
    return normal, float(normal @ centroid), fraction


def world_from_floor(
    normal_cam: np.ndarray,
    offset_cam: float,
    *,
    origin_cam: Optional[np.ndarray] = None,
    forward_cam: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build ``T_world_camera`` from a floor plane in the camera frame.

    The floor fixes roll, pitch and height.  Yaw and the origin within the
    floor plane are not observable from a plane, so they come from arguments:
    ``origin_cam`` is a surveyed point on the floor that becomes the world
    origin and ``forward_cam`` a direction that becomes world +x.  Both
    default to something reasonable -- the camera's own footprint and its
    optical axis projected onto the floor -- and the caller is told as much in
    ``docs/calibration.md`` rather than left to discover it.
    """
    n = np.asarray(normal_cam, dtype=np.float64).reshape(3)
    n = n / np.linalg.norm(n)
    # World +z is up, i.e. away from the floor and towards the camera.
    up_cam = -n if (n @ np.array([0.0, -1.0, 0.0])) < 0 else n
    up_cam = up_cam / np.linalg.norm(up_cam)

    if origin_cam is None:
        # Foot of the perpendicular from the camera centre to the floor.
        origin_cam = n * offset_cam
    origin_cam = np.asarray(origin_cam, dtype=np.float64).reshape(3)

    if forward_cam is None:
        forward_cam = np.array([0.0, 0.0, 1.0])
    f = np.asarray(forward_cam, dtype=np.float64).reshape(3)
    f = f - up_cam * (f @ up_cam)
    if np.linalg.norm(f) < 1e-6:
        raise ValueError("forward direction is parallel to the floor normal")
    x_cam = f / np.linalg.norm(f)
    y_cam = np.cross(up_cam, x_cam)

    # Columns are the world axes expressed in the camera frame, so this is
    # R_camera_world; invert for the transform the caller asked for.
    R_camera_world = np.stack([x_cam, y_cam, up_cam], axis=1)
    U, _, Vt = np.linalg.svd(R_camera_world)
    R_camera_world = U @ Vt
    return se3_inverse(se3(R_camera_world, origin_cam))
