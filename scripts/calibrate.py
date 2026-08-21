#!/usr/bin/env python3
"""Solve for the camera-lidar extrinsic and the floor plane and write the YAML.

Two modes.

**Self-check** (no hardware and what CI runs):

    python scripts/calibrate.py --self-check

generates board observations from a transform it chose, hands them to the
solver and reports how well the solver got it back.  That is the honest
demonstration that the routine works: it is measured against a transform we
know, not against a second calibration.

**Real data**:

    python scripts/calibrate.py --observations calib/board_poses.json \\
        --floor-depth calib/floor_depth.npy --out config/extrinsics.yaml

``--observations`` is a JSON list of board poses:

    [
      {"normal": [nx, ny, nz], "distance": d, "lidar_points": [[x, y], ...]},
      ...
    ]

where ``normal`` and ``distance`` are the board plane in the *camera* frame,
from the usual checkerboard pose estimate and ``lidar_points`` are the scan
returns that landed on the board, in the lidar frame, in metres.  Segmenting
those returns from the rest of the scan is the one manual step; it is a
handful of clicks per pose and automating it badly is worse than doing it by
hand five times.  ``docs/calibration.md`` walks through capture.

The residual is written into the extrinsics file.  A calibration without its
residual is a number with no error bar and an extrinsic with no error bar is
how a systematic ranging error becomes invisible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from szp.fusion.calibration import (  # noqa: E402
    PlaneObservation,
    calibrate_camera_lidar,
    estimate_floor_plane,
    world_from_floor,
)
from szp.fusion.extrinsics import ExtrinsicSet, load_extrinsics, save_extrinsics  # noqa: E402
from szp.geometry import rotation_from_rpy, se3, se3_inverse, so3_log  # noqa: E402


def synthetic_observations(
    T_camera_lidar: np.ndarray, n_poses: int, points_per_pose: int, noise_m: float, seed: int
) -> List[PlaneObservation]:
    """Board poses consistent with ``T_camera_lidar``, for the self-check."""
    rng = np.random.default_rng(seed)
    observations: List[PlaneObservation] = []
    attempts = 0
    while len(observations) < n_poses and attempts < 500:
        attempts += 1
        normal = rng.normal(size=3)
        normal[2] = abs(normal[2]) + 0.4
        normal /= np.linalg.norm(normal)
        distance = float(rng.uniform(1.5, 4.0))
        angles = rng.uniform(-0.5, 0.5, points_per_pose)
        directions = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
        dir_cam = directions @ T_camera_lidar[:3, :3].T
        denominator = dir_cam @ normal
        if np.abs(denominator).min() < 0.05:
            continue
        ranges = (distance - T_camera_lidar[:3, 3] @ normal) / denominator
        if ranges.min() < 0.3 or ranges.max() > 12.0:
            continue
        points = directions[:, :2] * ranges[:, None]
        if noise_m:
            points = points + rng.normal(0.0, noise_m, points.shape)
        observations.append(PlaneObservation(normal=normal, distance=distance,
                                             lidar_points=points))
    if len(observations) < n_poses:
        raise SystemExit("could not generate enough board poses; try another seed")
    return observations


def load_observations(path: Path) -> List[PlaneObservation]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, list) or not doc:
        raise SystemExit(f"{path}: expected a non-empty JSON list of board poses")
    return [
        PlaneObservation(
            normal=np.asarray(entry["normal"], dtype=np.float64),
            distance=float(entry["distance"]),
            lidar_points=np.asarray(entry["lidar_points"], dtype=np.float64),
        )
        for entry in doc
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--observations", default=None)
    parser.add_argument("--floor-depth", default=None, help=".npy depth map of empty floor")
    parser.add_argument("--base-extrinsics", default=str(REPO_ROOT / "config" / "extrinsics.yaml"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--poses", type=int, default=7)
    parser.add_argument("--points-per-pose", type=int, default=25)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None, help="write the residual report here")
    args = parser.parse_args(argv)

    if not args.self_check and not args.observations:
        parser.error("pass --self-check or --observations")

    truth = None
    if args.self_check:
        truth = se3(rotation_from_rpy(-1.55, 0.03, -1.60), [0.02, -0.31, 0.12])
        observations = synthetic_observations(
            truth, args.poses, args.points_per_pose, args.noise, args.seed
        )
        print(
            f"self-check: {args.poses} board poses, {args.points_per_pose} lidar points each, "
            f"{args.noise * 1000:.0f} mm point noise"
        )
    else:
        observations = load_observations(Path(args.observations))
        print(f"{len(observations)} board poses from {args.observations}")

    result = calibrate_camera_lidar(observations)
    report = {"camera_lidar": result.summary()}
    print(json.dumps(result.summary(), indent=2))

    if truth is not None:
        delta = se3_inverse(truth) @ result.T_camera_lidar
        translation_error = float(np.linalg.norm(delta[:3, 3]))
        rotation_error = float(np.degrees(np.linalg.norm(so3_log(delta[:3, :3]))))
        report["self_check"] = {
            "translation_error_m": translation_error,
            "rotation_error_deg": rotation_error,
        }
        print(
            f"\nrecovered the true transform to {translation_error * 1000:.2f} mm "
            f"and {rotation_error:.3f} degrees"
        )
        linear_rms = float(np.sqrt(np.mean(result.linear_residuals_m**2)))
        print(
            f"closed form residual {linear_rms * 1000:.2f} mm, "
            f"after refinement {result.rms_m * 1000:.2f} mm"
        )

    if args.floor_depth:
        base = load_extrinsics(args.base_extrinsics)
        intrinsics = base.intrinsics.get("camera")
        if intrinsics is None:
            raise SystemExit(f"{args.base_extrinsics} has no 'camera' intrinsics")
        depth = np.load(args.floor_depth)
        normal, offset, fraction = estimate_floor_plane(depth, intrinsics.K)
        T_world_camera = world_from_floor(normal, offset)
        report["floor"] = {
            "inlier_fraction": fraction,
            "height_m": float(T_world_camera[2, 3]),
            "normal_camera": [float(v) for v in normal],
        }
        print(json.dumps(report["floor"], indent=2))
    else:
        T_world_camera = None

    if args.out:
        base = load_extrinsics(args.base_extrinsics)
        edges = dict(base.edges)
        edges[("camera", "lidar")] = result.T_camera_lidar
        # A camera-to-lidar edge plus a world-to-camera edge is enough; drop
        # any stale world-to-lidar so the graph cannot hold two answers.
        edges.pop(("world", "lidar"), None)
        if T_world_camera is not None:
            edges[("world", "camera")] = T_world_camera
        updated = ExtrinsicSet(
            edges=edges,
            intrinsics=dict(base.intrinsics),
            meta=dict(
                base.meta,
                source="scripts/calibrate.py",
                residual_rms_m=result.rms_m,
                residual_max_abs_m=result.max_abs_m,
                board_poses=result.n_observations,
                lidar_points=result.n_points,
            ),
        )
        save_extrinsics(updated, args.out)
        print(f"\nwrote {args.out}")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
