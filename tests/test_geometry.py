"""SE(3) primitives and the frame graph."""

from __future__ import annotations

import numpy as np
import pytest

from szp.fusion.extrinsics import (
    ExtrinsicSet,
    FrameGraphError,
    extrinsics_from_dict,
    required_frames,
)
from szp.geometry import (
    backproject,
    is_rotation,
    project_points,
    quaternion_from_rotation,
    rotation_from_quaternion,
    rotation_from_rpy,
    rpy_from_rotation,
    se3,
    se3_inverse,
    so3_exp,
    so3_log,
    transform_points,
)


def random_rotations(n: int, seed: int = 3) -> list:
    rng = np.random.default_rng(seed)
    return [so3_exp(rng.normal(size=3) * scale) for scale, _ in zip(
        rng.uniform(0.0, 3.0, n), range(n)
    )]


def test_rpy_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        roll = rng.uniform(-np.pi, np.pi)
        pitch = rng.uniform(-1.4, 1.4)   # away from gimbal lock
        yaw = rng.uniform(-np.pi, np.pi)
        R = rotation_from_rpy(roll, pitch, yaw)
        assert is_rotation(R)
        back = rotation_from_rpy(*rpy_from_rotation(R))
        assert np.allclose(R, back, atol=1e-9)


def test_quaternion_round_trip():
    for R in random_rotations(100):
        q = quaternion_from_rotation(R)
        assert q[0] >= 0
        assert np.allclose(rotation_from_quaternion(q), R, atol=1e-9)


def test_so3_log_is_the_inverse_of_exp():
    rng = np.random.default_rng(11)
    for _ in range(200):
        w = rng.normal(size=3)
        w = w / np.linalg.norm(w) * rng.uniform(0.0, np.pi - 1e-3)
        assert np.allclose(so3_log(so3_exp(w)), w, atol=1e-8)


def test_se3_rejects_a_denormalised_rotation():
    R = rotation_from_rpy(0.1, 0.2, 0.3)
    R[0, 0] += 1e-3
    with pytest.raises(ValueError, match="orthonormal"):
        se3(R, [0, 0, 0])


def test_se3_inverse_matches_the_matrix_inverse():
    for R in random_rotations(20, seed=5):
        T = se3(R, [1.0, -2.0, 0.5])
        assert np.allclose(se3_inverse(T), np.linalg.inv(T), atol=1e-12)


def test_projection_round_trips_through_backprojection():
    K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])
    rng = np.random.default_rng(2)
    pixels = rng.uniform([0, 0], [640, 480], size=(50, 2))
    depths = rng.uniform(0.5, 10.0, 50)
    points = backproject(K, pixels, depths)
    assert np.allclose(project_points(K, points), pixels, atol=1e-9)


def test_points_behind_the_camera_project_to_nan():
    K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])
    behind = np.array([[0.1, 0.1, -2.0], [0.0, 0.0, 0.0]])
    out = project_points(K, behind)
    assert np.isnan(out).all(), (
        "a point behind the camera must not produce a plausible pixel; that is "
        "how a detection silently disappears"
    )


def _example_extrinsics() -> ExtrinsicSet:
    return extrinsics_from_dict(
        {
            "transforms": [
                {
                    "parent": "world",
                    "child": "base",
                    "translation": [1.0, 2.0, 0.0],
                    "rpy": [0.0, 0.0, 0.4],
                },
                {
                    "parent": "base",
                    "child": "camera",
                    "translation": [0.2, 0.0, 1.5],
                    "rpy": [0.1, -0.2, 0.0],
                },
                {
                    "parent": "base",
                    "child": "lidar",
                    "translation": [0.4, 0.0, 0.3],
                    "rpy": [0.0, 0.0, 0.0],
                },
            ]
        }
    )


def test_frame_graph_composes_and_inverts():
    extrinsics = _example_extrinsics()
    T_world_camera = extrinsics.T("world", "camera")
    expected = extrinsics.T("world", "base") @ extrinsics.T("base", "camera")
    assert np.allclose(T_world_camera, expected)
    assert np.allclose(
        extrinsics.T("camera", "world"), se3_inverse(T_world_camera), atol=1e-12
    )


def test_frame_graph_traverses_backwards_through_a_shared_parent():
    extrinsics = _example_extrinsics()
    T = extrinsics.T("camera", "lidar")
    point_lidar = np.array([[1.0, 0.5, 0.0]])
    via_world = transform_points(
        extrinsics.T("camera", "world"),
        transform_points(extrinsics.T("world", "lidar"), point_lidar),
    )
    assert np.allclose(transform_points(T, point_lidar), via_world, atol=1e-12)


def test_missing_edge_raises_rather_than_defaulting_to_identity():
    extrinsics = _example_extrinsics()
    with pytest.raises(FrameGraphError, match="no chain of extrinsics"):
        extrinsics.T("world", "radar")
    with pytest.raises(FrameGraphError, match="do not provide"):
        required_frames(extrinsics, [("world", "radar")])


def test_extrinsics_round_trip_through_a_dictionary():
    extrinsics = _example_extrinsics()
    again = extrinsics_from_dict(extrinsics.to_dict())
    for key, T in extrinsics.edges.items():
        assert np.allclose(again.edges[key], T, atol=1e-9)


def test_duplicate_transform_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        extrinsics_from_dict(
            {
                "transforms": [
                    {"parent": "a", "child": "b", "translation": [0, 0, 0], "rpy": [0, 0, 0]},
                    {"parent": "a", "child": "b", "translation": [1, 0, 0], "rpy": [0, 0, 0]},
                ]
            }
        )


def test_repository_extrinsics_provide_every_frame_the_pipeline_asks_for(repo_root):
    from szp.fusion.extrinsics import load_extrinsics

    extrinsics = load_extrinsics(repo_root / "config" / "extrinsics.yaml")
    required_frames(
        extrinsics,
        [("world", "camera"), ("world", "lidar"), ("camera", "lidar"), ("camera", "depth")],
    )
    assert "camera" in extrinsics.intrinsics
