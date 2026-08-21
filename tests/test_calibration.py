"""Camera-to-lidar and camera-to-floor calibration.

Both routines are checked by giving them synthetic observations generated from
a transform we chose and asking whether they recover it.  That is the only
honest way to test a calibration: comparing against a second calibration only
tells you the two agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from szp.fusion.calibration import (
    PlaneObservation,
    calibrate_camera_lidar,
    estimate_floor_plane,
    world_from_floor,
)
from szp.geometry import (
    backproject,
    rotation_from_rpy,
    se3,
    se3_inverse,
    so3_log,
    transform_points,
)


def make_observations(
    T_camera_lidar: np.ndarray,
    n_poses: int = 7,
    points_per_pose: int = 25,
    noise_m: float = 0.0,
    seed: int = 0,
):
    """Board poses with lidar returns that genuinely lie on the board."""
    rng = np.random.default_rng(seed)
    observations = []
    tried = 0
    while len(observations) < n_poses and tried < 200:
        tried += 1
        normal = rng.normal(size=3)
        normal[2] = abs(normal[2]) + 0.4
        normal = normal / np.linalg.norm(normal)
        if normal[2] < 0.2:
            continue
        distance = float(rng.uniform(1.5, 4.0))

        # Lidar points on the board: pick points in the lidar's scan plane
        # whose camera-frame image satisfies the plane equation.
        angles = rng.uniform(-0.5, 0.5, points_per_pose)
        directions = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
        origin_cam = T_camera_lidar[:3, 3]
        dir_cam = directions @ T_camera_lidar[:3, :3].T
        denominator = dir_cam @ normal
        if np.abs(denominator).min() < 0.05:
            continue
        ranges = (distance - origin_cam @ normal) / denominator
        if ranges.min() < 0.3 or ranges.max() > 12.0:
            continue
        points = directions[:, :2] * ranges[:, None]
        if noise_m:
            points = points + rng.normal(0.0, noise_m, points.shape)
        observations.append(
            PlaneObservation(normal=normal, distance=distance, lidar_points=points)
        )
    assert len(observations) == n_poses, "could not build enough board poses"
    return observations


TRUE_TRANSFORM = se3(rotation_from_rpy(-1.55, 0.03, -1.60), [0.02, -0.31, 0.12])


def transform_error(estimated: np.ndarray, truth: np.ndarray):
    delta = se3_inverse(truth) @ estimated
    return (
        float(np.linalg.norm(delta[:3, 3])),
        float(np.degrees(np.linalg.norm(so3_log(delta[:3, :3])))),
    )


def test_noiseless_calibration_recovers_the_transform():
    result = calibrate_camera_lidar(make_observations(TRUE_TRANSFORM))
    translation_error, rotation_error = transform_error(result.T_camera_lidar, TRUE_TRANSFORM)
    assert translation_error < 1e-6
    assert rotation_error < 1e-4
    assert result.rms_m < 1e-8
    assert result.converged


def test_calibration_degrades_gracefully_with_noise():
    result = calibrate_camera_lidar(make_observations(TRUE_TRANSFORM, noise_m=0.01, seed=4))
    translation_error, rotation_error = transform_error(result.T_camera_lidar, TRUE_TRANSFORM)
    assert translation_error < 0.05
    assert rotation_error < 2.0
    assert result.rms_m < 0.02


def test_refinement_improves_on_the_closed_form():
    result = calibrate_camera_lidar(make_observations(TRUE_TRANSFORM, noise_m=0.01, seed=9))
    linear_rms = float(np.sqrt(np.mean(result.linear_residuals_m**2)))
    assert result.rms_m <= linear_rms + 1e-12


def test_too_few_poses_is_refused():
    observations = make_observations(TRUE_TRANSFORM, n_poses=7)[:2]
    with pytest.raises(ValueError, match="at least 3 board poses"):
        calibrate_camera_lidar(observations)


def test_degenerate_poses_are_refused_rather_than_fitted():
    """Every board at the same orientation leaves the problem rank deficient.
    Returning a confident-looking transform there is how a calibration comes
    out plausible and wrong."""
    one = make_observations(TRUE_TRANSFORM, n_poses=1, points_per_pose=40)[0]
    with pytest.raises(ValueError, match="rank"):
        calibrate_camera_lidar([one, one, one, one])


def test_a_summary_is_json_friendly():
    result = calibrate_camera_lidar(make_observations(TRUE_TRANSFORM))
    summary = result.summary()
    assert set(summary) >= {"rms_m", "max_abs_m", "n_observations", "n_points", "converged"}
    assert isinstance(summary["translation_m"], list)


# -- floor -------------------------------------------------------------------


def synthetic_floor_depth(height_m=2.5, tilt_deg=15.0, width=160, height=120, fx=200.0):
    """Depth map of a plane, seen by a camera at ``height_m`` tilted down."""
    K = np.array([[fx, 0, (width - 1) / 2], [0, fx, (height - 1) / 2], [0, 0, 1]])
    R = rotation_from_rpy(np.deg2rad(90.0 - tilt_deg), 0.0, 0.0)
    # Camera looking down at the floor: floor normal in the camera frame.
    normal_cam = R[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    normal_cam = normal_cam / np.linalg.norm(normal_cam)
    offset = -height_m * float(normal_cam @ np.array([0.0, 0.0, 1.0]))
    offset = abs(offset) if offset != 0 else height_m

    vv, uu = np.mgrid[0:height, 0:width]
    rays = backproject(K, np.stack([uu.ravel(), vv.ravel()], axis=1), np.ones(uu.size))
    denominator = rays @ normal_cam
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = offset / denominator
    depth = (scale * rays[:, 2]).reshape(height, width)
    depth = np.where(np.isfinite(depth) & (depth > 0.2) & (depth < 20.0), depth, 0.0)
    return depth.astype(np.float32), K, normal_cam, offset


def test_floor_plane_is_recovered_from_a_depth_map():
    depth, K, normal_true, offset_true = synthetic_floor_depth()
    normal, offset, fraction = estimate_floor_plane(depth, K, rng=np.random.default_rng(0))
    assert fraction > 0.9
    assert abs(abs(float(normal @ normal_true)) - 1.0) < 1e-3
    assert offset == pytest.approx(offset_true, abs=0.02)


def test_a_scene_with_no_dominant_plane_is_refused():
    rng = np.random.default_rng(0)
    depth = rng.uniform(1.0, 6.0, (120, 160)).astype(np.float32)
    with pytest.raises(ValueError, match="no dominant plane"):
        estimate_floor_plane(depth, np.eye(3), rng=np.random.default_rng(1))


def test_an_empty_depth_map_is_refused():
    with pytest.raises(ValueError, match="usable depth pixels"):
        estimate_floor_plane(np.zeros((40, 40), dtype=np.float32), np.eye(3))


def test_world_from_floor_puts_the_floor_at_z_zero():
    depth, K, _, _ = synthetic_floor_depth(height_m=2.5)
    normal, offset, _ = estimate_floor_plane(depth, K, rng=np.random.default_rng(0))
    T_world_camera = world_from_floor(normal, offset)

    # Every valid depth pixel is a point on the floor, so all of them must land
    # on z = 0 in the world frame.
    vv, uu = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    valid = depth > 0
    points_cam = backproject(
        K, np.stack([uu[valid], vv[valid]], axis=1), depth[valid]
    )
    points_world = transform_points(T_world_camera, points_cam)
    assert np.abs(points_world[:, 2]).max() < 0.05


def test_world_from_floor_refuses_a_forward_direction_along_the_normal():
    depth, K, _, _ = synthetic_floor_depth()
    normal, offset, _ = estimate_floor_plane(depth, K, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="parallel to the floor normal"):
        world_from_floor(normal, offset, forward_cam=normal)
