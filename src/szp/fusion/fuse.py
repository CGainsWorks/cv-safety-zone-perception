"""Detections plus depth plus lidar to 3D boxes in the world frame.

Three rules run through this stage and all three come from the same place: a
missed person is the failure that matters.

1.  **A detection is never dropped for want of a range.**  If the depth map has
    no return and no lidar cluster lines up, the detection still becomes a
    ``Box3D``, tagged ``UNRANGED``, whose footprint is the whole wedge of floor
    the person could be standing on.  That wedge usually overlaps a zone, which
    trips a stop.  A pipeline that instead discarded the detection would report
    an empty cell.

2.  **When the two sensors disagree, believe the nearer one.**  Depth and lidar
    are fused by inverse-variance weighting only while they agree inside a
    Mahalanobis gate.  Outside the gate one of them is wrong and there is no
    way to know which, so the estimate goes to whichever puts the person closer
    to the machine, with the uncertainty inflated to span the disagreement.

3.  **Uncertainty is carried, not rounded away.**  ``sigma_xy`` travels with the
    box and the zone stage dilates the footprint by a multiple of it, so a
    badly ranged person occupies more floor, not less.

The one geometric assumption worth stating plainly: people are taken to be
standing on the floor plane, z = 0 in the world frame.  Somebody on a step or a
mezzanine has their height mis-estimated and their range with it.  That is
entry FM-07 in the FMEA and it is in ``docs/limitations.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from szp.fusion.depth import DepthConfig, DepthEstimate, estimate_depth
from szp.fusion.extrinsics import ExtrinsicSet
from szp.fusion.lidar import LidarConfig, ScanCluster, fit_circle_centre, segment_scan
from szp.geometry import backproject, project_points, se3_inverse, transform_points
from szp.types import Box3D, CameraIntrinsics, Detection2D, RangeSource

__all__ = ["FusionConfig", "Fuser", "FusionDebug"]


@dataclass(frozen=True)
class FusionConfig:
    depth: DepthConfig = field(default_factory=DepthConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)

    camera_frame: str = "camera"
    lidar_frame: str = "lidar"
    world_frame: str = "world"

    #: Plan-view footprint assigned to one person, metres.
    person_width_m: float = 0.60
    person_depth_m: float = 0.45
    #: Height prior, used when the box is truncated by the image border and
    #: apparent height means nothing.
    person_height_prior_m: float = 1.75
    person_height_min_m: float = 1.40
    person_height_max_m: float = 2.05

    #: Horizontal gate on lidar association: a cluster must project within this
    #: many pixels of the detection box, as a fraction of the box width.
    associate_pixel_gate: float = 0.6
    #: Floor on that gate, in pixels.  A person halfway out of frame gives a
    #: box a few pixels wide and a gate proportional to it would be a few
    #: pixels too -- so the lidar cluster that is plainly theirs falls outside
    #: it and the person ends up ranged by a depth window full of background.
    associate_min_gate_px: float = 24.0
    #: On a side where the box is clipped by the image border the person
    #: continues outside the frame, so the gate on that side opens up to this
    #: fraction of the image width.  The lidar sees past the camera's field of
    #: view and there is no reason to throw that away.
    associate_truncated_gate_fraction: float = 0.35
    #: How strongly a lidar cluster is penalised for disagreeing with the
    #: depth estimate, per standard deviation of disagreement.  It is a cost
    #: and deliberately not a gate: a depth camera that is systematically two
    #: metres out would otherwise gate away every lidar cluster that
    #: contradicts it, the pipeline would believe the depth alone and the
    #: person would be placed outside the zone they are standing in.  A soft
    #: cost lets the disagreement reach rule 2 above, which takes the nearer of
    #: the two readings.
    associate_range_mismatch_weight: float = 0.15
    #: Cap on that penalty, so a wildly inconsistent cluster is still preferred
    #: to no cluster at all.
    associate_range_mismatch_cap: float = 20.0
    lidar_sigma_m: float = 0.035
    #: Mahalanobis gate for fusing the two horizontal estimates.
    fuse_gate_sigma: float = 3.0
    #: Extra sigma added when depth and lidar disagree, as a fraction of the
    #: disagreement.
    disagreement_inflation: float = 0.5

    #: Wedge parameters for unranged detections.
    unranged_max_range_m: float = 12.0
    unranged_min_range_m: float = 0.4
    #: Angular padding on the wedge, radians, covering box jitter.
    unranged_angular_pad_rad: float = np.deg2rad(2.0)
    #: Sigma reported for an unranged box.  Deliberately large.
    unranged_sigma_m: float = 1.0

    #: Correct the two systematic biases, both of which put people closer to
    #: the sensor than they are.  Turn either off only to reproduce the
    #: "uncorrected" rows in analysis/geometry_accuracy.py.
    correct_depth_surface_bias: bool = True
    correct_lidar_arc_bias: bool = True


#: Distance from the visible front surface of a person to their centre, as a
#: multiple of body radius, for a depth estimate taken as the median over the
#: box.  Model the torso as a circle of radius r seen head on: the depth at
#: image offset x is z_c - sqrt(r^2 - x^2) and pixels are uniform in x, so the
#: median over the silhouette is at x = r/2 and the median depth is
#: z_c - r*sqrt(3)/2.  Adding that back is a bias correction, not a fudge; the
#: derivation is here so it can be argued with.
DEPTH_SURFACE_TO_CENTRE = float(np.sqrt(3.0) / 2.0)


@dataclass
class FusionDebug:
    """Per-frame counters, for the analysis scripts and nothing else."""

    n_detections: int = 0
    n_depth_ok: int = 0
    n_lidar_associated: int = 0
    n_fused: int = 0
    n_disagreement: int = 0
    n_unranged: int = 0
    n_clusters: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _rotate2(vector: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


def _wedge_polygon(
    origin_xy: np.ndarray,
    direction_a: np.ndarray,
    direction_b: np.ndarray,
    near_m: float,
    far_m: float,
    lateral_pad_m: float,
) -> np.ndarray:
    """Quadrilateral covering a depth interval between two viewing rays.

    ``direction_a`` and ``direction_b`` are world-frame *plan view* displacements
    per unit of camera-frame depth, so a person at depth ``t`` along one of them
    stands at ``origin + t * direction``.  Passing bearings instead and treating
    the depth as a horizontal radius is wrong for any camera that is tilted and
    it was wrong here first: the wedge came out too long and, for a person off
    to one side, missed them.
    """
    pad_angle = float(np.arctan2(lateral_pad_m, max(far_m, 1e-3)))
    cross = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    sign = 1.0 if cross >= 0 else -1.0
    a = _rotate2(direction_a, -sign * pad_angle)
    b = _rotate2(direction_b, sign * pad_angle)
    return np.asarray(
        [
            origin_xy + near_m * a,
            origin_xy + near_m * b,
            origin_xy + far_m * b,
            origin_xy + far_m * a,
        ],
        dtype=np.float64,
    )


def _rectangle(center_xy: np.ndarray, length: float, width: float) -> np.ndarray:
    hx, hy = 0.5 * length, 0.5 * width
    return center_xy + np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64)


class Fuser:
    """Stateless per-frame fusion.  Construct once, call :meth:`fuse` per frame."""

    def __init__(self, extrinsics: ExtrinsicSet, config: Optional[FusionConfig] = None):
        self.extrinsics = extrinsics
        self.config = config or FusionConfig()
        self.T_world_camera = extrinsics.T(self.config.world_frame, self.config.camera_frame)
        self.T_world_lidar = extrinsics.T(self.config.world_frame, self.config.lidar_frame)
        self.T_camera_lidar = extrinsics.T(self.config.camera_frame, self.config.lidar_frame)
        self.camera_origin_world = self.T_world_camera[:3, 3]

    # -- lidar -------------------------------------------------------------

    def _cluster_pixels(self, clusters: Sequence[ScanCluster], K: np.ndarray) -> np.ndarray:
        """Project cluster centroids into the image.  NaN where behind the camera."""
        if not clusters:
            return np.zeros((0, 2))
        centroids = np.stack([c.centroid for c in clusters], axis=0)
        pts_lidar = np.concatenate([centroids, np.zeros((len(centroids), 1))], axis=1)
        pts_cam = transform_points(self.T_camera_lidar, pts_lidar)
        return project_points(K, pts_cam)

    def _associate(
        self,
        detection: Detection2D,
        clusters: Sequence[ScanCluster],
        pixels: np.ndarray,
        depth_estimate: Optional[DepthEstimate],
        intrinsics: CameraIntrinsics,
    ) -> Optional[int]:
        if len(clusters) == 0:
            return None
        cfg = self.config
        x0, _, x1, _ = detection.xyxy
        gate = max(
            cfg.associate_pixel_gate * detection.width, cfg.associate_min_gate_px
        )
        gate_low = gate_high = gate
        wide = cfg.associate_truncated_gate_fraction * intrinsics.width
        if x0 <= 2.0:
            gate_low = max(gate_low, wide)
        if x1 >= intrinsics.width - 2.0:
            gate_high = max(gate_high, wide)
        best: Optional[Tuple[float, int]] = None
        for i, cluster in enumerate(clusters):
            u = pixels[i, 0]
            if not np.isfinite(u):
                continue
            if u < x0 - gate_low or u > x1 + gate_high:
                continue
            # Camera-frame z of the cluster, for the range consistency check.
            p_cam = transform_points(
                self.T_camera_lidar, np.append(cluster.centroid, 0.0).reshape(1, 3)
            )[0]
            if p_cam[2] <= 0:
                continue
            centre_u = 0.5 * (x0 + x1)
            # Prefer the cluster nearest the box centre, breaking ties towards
            # the nearer cluster: if two people overlap in the image the closer
            # one is the one that matters.
            # Normalise the pixel error by the gate, not by the box width: a
            # sliver of a truncated person would otherwise make every cluster
            # look impossibly far away and the tie-break meaningless.
            cost = abs(u - centre_u) / max(gate_low, gate_high) + 0.05 * cluster.mean_range_m
            if not cluster.person_like:
                cost += 0.25
            if depth_estimate is not None:
                spread = float(np.hypot(depth_estimate.sigma_m, cfg.lidar_sigma_m))
                mismatch = abs(float(p_cam[2]) - depth_estimate.depth_m) / max(spread, 1e-6)
                cost += cfg.associate_range_mismatch_weight * min(
                    mismatch, cfg.associate_range_mismatch_cap
                )
            if best is None or cost < best[0]:
                best = (cost, i)
        return None if best is None else best[1]

    # -- height ------------------------------------------------------------

    def _height_from_box(
        self, detection: Detection2D, range_m: float, intr: CameraIntrinsics
    ) -> float:
        if detection.touches_border(intr.width, intr.height):
            return self.config.person_height_prior_m
        apparent = detection.height * range_m / max(intr.fy, 1e-6)
        return float(
            np.clip(apparent, self.config.person_height_min_m, self.config.person_height_max_m)
        )

    # -- main --------------------------------------------------------------

    def fuse(
        self,
        detections: Sequence[Detection2D],
        depth: Optional[np.ndarray],
        scan_ranges: Optional[np.ndarray],
        scan_angles: Optional[np.ndarray],
        intrinsics: CameraIntrinsics,
        *,
        debug: Optional[FusionDebug] = None,
    ) -> List[Box3D]:
        cfg = self.config
        K = intrinsics.K
        clusters: List[ScanCluster] = []
        if scan_ranges is not None and scan_angles is not None and len(scan_ranges):
            clusters = segment_scan(scan_ranges, scan_angles, cfg.lidar)
        pixels = self._cluster_pixels(clusters, K)

        if debug is not None:
            debug.n_detections = len(detections)
            debug.n_clusters = len(clusters)

        boxes: List[Box3D] = []
        for detection in detections:
            depth_estimate = estimate_depth(detection, depth, cfg.depth, intrinsics.fx)
            if depth_estimate is not None and debug is not None:
                debug.n_depth_ok += 1
            cluster_index = self._associate(
                detection, clusters, pixels, depth_estimate, intrinsics
            )
            if cluster_index is not None and debug is not None:
                debug.n_lidar_associated += 1

            box = self._build_box(
                detection,
                depth_estimate,
                clusters[cluster_index] if cluster_index is not None else None,
                intrinsics,
                debug,
            )
            boxes.append(box)
        return boxes

    def _build_box(
        self,
        detection: Detection2D,
        depth_estimate: Optional[DepthEstimate],
        cluster: Optional[ScanCluster],
        intrinsics: CameraIntrinsics,
        debug: Optional[FusionDebug],
    ) -> Box3D:
        cfg = self.config
        K = intrinsics.K
        u, v = detection.center

        depth_xy: Optional[np.ndarray] = None
        depth_sigma = float("inf")
        range_for_height = cfg.unranged_max_range_m
        if depth_estimate is not None:
            centre_depth = depth_estimate.depth_m
            if cfg.correct_depth_surface_bias:
                centre_depth += DEPTH_SURFACE_TO_CENTRE * 0.5 * cfg.person_depth_m
            p_cam = backproject(K, np.array([[u, v]]), np.array([centre_depth]))[0]
            depth_xy = transform_points(self.T_world_camera, p_cam.reshape(1, 3))[0][:2]
            depth_sigma = float(depth_estimate.sigma_m)
            range_for_height = depth_estimate.depth_m

        lidar_xy: Optional[np.ndarray] = None
        lidar_sigma = float("inf")
        if cluster is not None:
            if cfg.correct_lidar_arc_bias:
                centre_lidar = fit_circle_centre(
                    cluster.points, 0.5 * cfg.person_depth_m, np.zeros(2)
                )
            else:
                centre_lidar = cluster.centroid
            p_world = transform_points(
                self.T_world_lidar, np.append(centre_lidar, 0.0).reshape(1, 3)
            )[0]
            lidar_xy = p_world[:2]
            lidar_sigma = cfg.lidar_sigma_m
            p_cam = transform_points(
                self.T_camera_lidar, np.append(cluster.centroid, 0.0).reshape(1, 3)
            )[0]
            if depth_estimate is None:
                range_for_height = float(p_cam[2])

        if depth_xy is None and lidar_xy is None:
            if debug is not None:
                debug.n_unranged += 1
            return self._unranged_box(detection, intrinsics)

        if depth_xy is not None and lidar_xy is not None:
            separation = float(np.linalg.norm(depth_xy - lidar_xy))
            spread = float(np.hypot(depth_sigma, lidar_sigma))
            if separation <= cfg.fuse_gate_sigma * spread:
                w_depth = 1.0 / depth_sigma**2
                w_lidar = 1.0 / lidar_sigma**2
                center_xy = (w_depth * depth_xy + w_lidar * lidar_xy) / (w_depth + w_lidar)
                sigma = float(1.0 / np.sqrt(w_depth + w_lidar))
                source = RangeSource.FUSED
                if debug is not None:
                    debug.n_fused += 1
            else:
                # They disagree.  Take whichever puts the person closer to the
                # camera and widen the uncertainty to cover the argument.
                d_depth = float(np.linalg.norm(depth_xy - self.camera_origin_world[:2]))
                d_lidar = float(np.linalg.norm(lidar_xy - self.camera_origin_world[:2]))
                if d_lidar <= d_depth:
                    center_xy, sigma, source = lidar_xy, lidar_sigma, RangeSource.LIDAR
                    range_for_height = min(range_for_height, d_lidar)
                else:
                    center_xy, sigma, source = depth_xy, depth_sigma, RangeSource.DEPTH
                sigma = float(sigma + cfg.disagreement_inflation * separation)
                if debug is not None:
                    debug.n_disagreement += 1
        elif depth_xy is not None:
            center_xy, sigma, source = depth_xy, depth_sigma, RangeSource.DEPTH
        else:
            center_xy, sigma, source = lidar_xy, lidar_sigma, RangeSource.LIDAR

        height = self._height_from_box(detection, float(range_for_height), intrinsics)
        footprint = _rectangle(np.asarray(center_xy), cfg.person_depth_m, cfg.person_width_m)
        return Box3D(
            center=(float(center_xy[0]), float(center_xy[1]), float(height / 2.0)),
            size=(cfg.person_depth_m, cfg.person_width_m, float(height)),
            score=detection.score,
            source=source,
            sigma_xy=float(sigma),
            frame=cfg.world_frame,
            detection=detection,
            footprint=footprint,
        )

    def _unranged_box(self, detection: Detection2D, intrinsics: CameraIntrinsics) -> Box3D:
        """A detection with no range becomes the wedge it could be standing in.

        The near and far bounds come from the apparent height: a box that is
        ``h`` pixels tall is a person between ``person_height_min_m`` and
        ``person_height_max_m`` tall at ``H fy / h`` metres.  A box clipped by
        the image border carries no height information at all, so the wedge runs
        the full working range.
        """
        cfg = self.config
        K = intrinsics.K
        x0, _, x1, _ = detection.xyxy
        u, v = detection.center

        if detection.touches_border(intrinsics.width, intrinsics.height) or detection.height < 2:
            near, far = cfg.unranged_min_range_m, cfg.unranged_max_range_m
        else:
            near = cfg.person_height_min_m * intrinsics.fy / detection.height
            far = cfg.person_height_max_m * intrinsics.fy / detection.height
            near = max(near, cfg.unranged_min_range_m)
            far = min(far, cfg.unranged_max_range_m)
            if far <= near:
                near, far = cfg.unranged_min_range_m, cfg.unranged_max_range_m

        # Plan-view displacement per metre of depth, for each vertical box
        # edge.  These are not unit vectors and must not be normalised: the
        # near and far bounds are depths and multiplying a depth by one of
        # these gives the right point on the floor for a tilted camera.
        directions = []
        for edge_u in (x0, x1):
            ray_cam = backproject(K, np.array([[edge_u, v]]), np.array([1.0]))[0]
            ray_world = (self.T_world_camera[:3, :3] @ ray_cam)[:2]
            if np.linalg.norm(ray_world) < 1e-9:
                ray_world = np.array([1.0, 0.0])
            directions.append(ray_world)

        origin = self.camera_origin_world[:2]
        footprint = _wedge_polygon(
            origin,
            directions[0],
            directions[1],
            near,
            far,
            0.5 * cfg.person_width_m,
        )
        mid = 0.5 * (near + far)
        centre_ray_cam = backproject(K, np.array([[u, v]]), np.array([1.0]))[0]
        centre_dir = (self.T_world_camera[:3, :3] @ centre_ray_cam)[:2]
        if np.linalg.norm(centre_dir) < 1e-9:
            centre_dir = np.array([1.0, 0.0])
        center_xy = origin + mid * centre_dir
        return Box3D(
            center=(
                float(center_xy[0]),
                float(center_xy[1]),
                float(cfg.person_height_prior_m / 2.0),
            ),
            size=(cfg.person_depth_m, cfg.person_width_m, float(cfg.person_height_prior_m)),
            score=detection.score,
            source=RangeSource.UNRANGED,
            sigma_xy=float(max(cfg.unranged_sigma_m, (far - near) / 4.0)),
            frame=cfg.world_frame,
            detection=detection,
            range_interval=(float(near), float(far)),
            footprint=footprint,
        )


def camera_position_in_world(extrinsics: ExtrinsicSet, camera_frame: str = "camera") -> np.ndarray:
    """Convenience for the plots: where the camera is, in world metres."""
    return extrinsics.T("world", camera_frame)[:3, 3]


def world_to_camera(extrinsics: ExtrinsicSet, camera_frame: str = "camera") -> np.ndarray:
    return se3_inverse(extrinsics.T("world", camera_frame))
