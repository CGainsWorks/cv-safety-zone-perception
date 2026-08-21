"""A synthetic cell, rendered analytically, with ground truth.

Why this exists
---------------
The detection numbers in this repository come from real photographs, because
recall on rendered people would mean nothing.  The *geometry* numbers cannot
come from the same place: COCO has no depth, no lidar and no metric ground
truth and there is no public dataset that pairs a registered RGB-D camera, a
2D lidar at knee height and surveyed 3D person positions in one cell.  So the
fusion error, the zone confusion matrix, the state machine behaviour and the
end-to-end latency are measured here instead, on a scene where the true
position of every person is known exactly.

The split is deliberate and it is stated everywhere the numbers appear.  What
this rig can prove: that the fusion arithmetic, the extrinsics, the zone tests
and the fail-safe logic are right and that they are right under sensor
dropout, depth holes and lidar occlusion.  What it cannot prove: that a
detector will find a person in a real factory.  Only the COCO analysis speaks
to that and it is the weaker of the two guarantees.

Rendering
---------
No graphics library.  People are vertical cylinders, the room is planes and
both are intersected with camera rays in closed form, which makes the depth map
exact to floating point and the ground truth free.  A rasteriser would
introduce its own error into the thing being measured.

Sensor imperfections are modelled explicitly rather than left out:

* depth noise growing with the square of range, plus dropout holes with a
  spatial extent, plus a hard maximum range beyond which the sensor returns
  nothing at all
* lidar range noise, random beam dropout and a limited field of view
* detector jitter on box corners, missed detections and occasional false
  positives

Every one of those is a knob and ``analysis/`` sweeps them.  A pipeline whose
numbers only hold with perfect sensors is not worth reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from szp.detection.postprocess import iou_matrix
from szp.fusion.extrinsics import ExtrinsicSet
from szp.geometry import project_points, rotation_from_rpy, se3, se3_inverse, transform_points
from szp.types import CameraIntrinsics, Detection2D, NS_PER_S, FrameBundle

__all__ = [
    "SceneConfig",
    "PersonTrack",
    "GtPerson",
    "SyntheticScene",
    "SimulatedDetector",
    "TimedRealDetector",
    "build_scene",
]


@dataclass(frozen=True)
class PersonTrack:
    """One person walking a straight line, wrapping at the ends.

    Straight lines rather than something more interesting because the point of
    the rig is metric accuracy under occlusion and dropout, not tracking and a
    predictable path makes a failure easy to read off a plot.
    """

    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    speed_mps: float = 1.2
    height_m: float = 1.75
    radius_m: float = 0.22
    phase: float = 0.0
    #: The person is only in the cell between these times.  A scene where
    #: everybody is present from the first frame never exercises the CLEAR ->
    #: WARN -> STOP -> recovery path, which is most of what the state machine
    #: does, so the demo scene has people walking in and out.
    active_from_s: float = 0.0
    active_until_s: float = float("inf")

    def active_at(self, t_s: float) -> bool:
        return self.active_from_s <= t_s < self.active_until_s

    def position(self, t_s: float) -> np.ndarray:
        start = np.asarray(self.start_xy, dtype=np.float64)
        end = np.asarray(self.end_xy, dtype=np.float64)
        length = float(np.linalg.norm(end - start))
        if length < 1e-9 or self.speed_mps <= 0:
            return start
        # Time runs from the moment this person entered the cell, so they walk
        # in from ``start_xy`` rather than materialising part-way along.
        elapsed = max(0.0, t_s - self.active_from_s)
        period = 2.0 * length / self.speed_mps
        phase = (elapsed + self.phase * period) % period
        travelled = phase * self.speed_mps
        if travelled > length:
            travelled = 2.0 * length - travelled
        return start + (end - start) * (travelled / length)


@dataclass(frozen=True)
class SceneConfig:
    duration_s: float = 24.0
    fps: float = 15.0
    width: int = 640
    height: int = 480
    fx: float = 525.0
    fy: float = 525.0

    #: Camera at 2.6 m looking along world +x, tilted down.
    camera_xyz: Tuple[float, float, float] = (-1.0, 0.0, 2.6)
    camera_tilt_down_deg: float = 18.0
    camera_yaw_deg: float = 0.0

    #: Lidar on the machine guard at knee height, aligned with the world axes.
    lidar_xyz: Tuple[float, float, float] = (-0.6, 0.0, 0.45)
    lidar_fov_deg: float = 240.0
    lidar_angle_step_deg: float = 0.25
    lidar_range_sigma_m: float = 0.015
    lidar_dropout: float = 0.01
    lidar_max_range_m: float = 15.0

    #: Room: floor at z = 0, a back wall and two side walls.
    room_x_max: float = 9.0
    room_y: float = 4.0
    room_x_min: float = -3.0

    depth_max_range_m: float = 8.0
    depth_disparity_noise_px: float = 0.08
    depth_baseline_m: float = 0.05
    #: Fraction of pixels lost to holes and the radius of a hole in pixels.
    depth_dropout: float = 0.02
    depth_hole_radius_px: int = 6

    #: The default cast walks through the cell one at a time so that the state
    #: machine is driven through CLEAR, WARN, STOP and recovery rather than
    #: sitting in STOP from the first frame.  ``analysis/`` overrides this for
    #: the crowded and occluded cases.
    people: Tuple[PersonTrack, ...] = field(
        default_factory=lambda: (
            PersonTrack(
                start_xy=(3.0, -3.4), end_xy=(3.0, 3.4), speed_mps=1.0,
                active_from_s=2.0, active_until_s=10.0,
            ),
            PersonTrack(
                start_xy=(7.2, 2.8), end_xy=(1.8, -0.6), speed_mps=0.95,
                active_from_s=13.0, active_until_s=22.0,
            ),
            PersonTrack(
                start_xy=(4.6, -2.5), end_xy=(2.4, 1.8), speed_mps=1.3,
                active_from_s=16.5, active_until_s=22.0,
            ),
        )
    )
    seed: int = 20260821


@dataclass(frozen=True)
class GtPerson:
    """Ground truth for one person on one frame."""

    index: int
    center_xy: np.ndarray
    height_m: float
    radius_m: float
    #: Amodal 2D box in image pixels: where the person would be if nothing were
    #: in front of them.  Clipped to the image.
    box_xyxy: Optional[Tuple[float, float, float, float]]
    #: Visible pixels divided by pixels that would be visible with no other
    #: person in the way.  1.0 for an unoccluded person, 0.0 for a hidden one.
    visibility: float
    #: True when the camera can see any part of them at all.
    in_view: bool
    #: True when at least one lidar beam returned from them.
    lidar_hits: int
    range_m: float


def _camera_rotation(tilt_down_deg: float, yaw_deg: float) -> np.ndarray:
    """R_world_camera for an OpenCV camera looking along world +x."""
    base = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
    )
    tilt = np.deg2rad(tilt_down_deg)
    # Rotate about the camera's own +x axis so the optical axis dips towards
    # the floor.  The sign is checked by test_synthetic_camera_looks_down.
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(-tilt), -np.sin(-tilt)],
            [0.0, np.sin(-tilt), np.cos(-tilt)],
        ]
    )
    yaw = rotation_from_rpy(0.0, 0.0, np.deg2rad(yaw_deg))
    return yaw @ base @ rx


class SyntheticScene:
    """Render frames and hand back the truth alongside them."""

    def __init__(self, config: Optional[SceneConfig] = None):
        self.config = config or SceneConfig()
        cfg = self.config
        self.intrinsics = CameraIntrinsics(
            fx=cfg.fx,
            fy=cfg.fy,
            cx=(cfg.width - 1) / 2.0,
            cy=(cfg.height - 1) / 2.0,
            width=cfg.width,
            height=cfg.height,
        )
        R_wc = _camera_rotation(cfg.camera_tilt_down_deg, cfg.camera_yaw_deg)
        self.T_world_camera = se3(R_wc, cfg.camera_xyz)
        self.T_world_lidar = se3(np.eye(3), cfg.lidar_xyz)
        self.extrinsics = ExtrinsicSet(
            edges={
                ("world", "camera"): self.T_world_camera,
                ("world", "lidar"): self.T_world_lidar,
            },
            intrinsics={"camera": self.intrinsics},
            meta={"source": "szp.data.synthetic", "exact": True},
        )
        self.T_camera_world = se3_inverse(self.T_world_camera)

        self._rays = self._build_rays()
        # The room does not move, so its depth map is computed once.  It is by
        # far the most expensive part of a frame and recomputing it per frame
        # made the demo three times slower for no reason.
        self._static_cache: Optional[np.ndarray] = None
        self.lidar_angles = np.deg2rad(
            np.arange(
                -cfg.lidar_fov_deg / 2.0,
                cfg.lidar_fov_deg / 2.0 + 1e-9,
                cfg.lidar_angle_step_deg,
            )
        )
        self._rng = np.random.default_rng(cfg.seed)

    # -- geometry ----------------------------------------------------------

    def _build_rays(self) -> np.ndarray:
        cfg = self.config
        vv, uu = np.mgrid[0 : cfg.height, 0 : cfg.width]
        k = self.intrinsics
        rays_cam = np.stack(
            [(uu - k.cx) / k.fx, (vv - k.cy) / k.fy, np.ones_like(uu, dtype=np.float64)],
            axis=-1,
        )
        return rays_cam @ self.T_world_camera[:3, :3].T  # (H, W, 3) in world

    @property
    def n_frames(self) -> int:
        return int(round(self.config.duration_s * self.config.fps))

    def people_at(self, t_s: float) -> List[np.ndarray]:
        """Positions of the people who are in the cell at ``t_s``."""
        return [track.position(t_s) for track in self.config.people if track.active_at(t_s)]

    def active_tracks(self, t_s: float) -> List[Tuple[int, PersonTrack]]:
        return [(i, tr) for i, tr in enumerate(self.config.people) if tr.active_at(t_s)]

    def _static_depth(self) -> np.ndarray:
        """Depth to the floor and walls, in camera z, for every pixel."""
        if self._static_cache is not None:
            return self._static_cache
        cfg = self.config
        origin = self.T_world_camera[:3, 3]
        d = self._rays
        best = np.full(d.shape[:2], np.inf)
        planes = (
            (np.array([0.0, 0.0, 1.0]), 0.0),          # floor
            (np.array([1.0, 0.0, 0.0]), cfg.room_x_max),   # back wall
            (np.array([1.0, 0.0, 0.0]), cfg.room_x_min),   # wall behind the camera
            (np.array([0.0, 1.0, 0.0]), cfg.room_y),
            (np.array([0.0, 1.0, 0.0]), -cfg.room_y),
        )
        for normal, offset in planes:
            denom = d @ normal
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (offset - origin @ normal) / denom
            t = np.where((denom != 0) & (t > 0), t, np.inf)
            # Stay inside the room: reject hits outside the other bounds.
            point = origin + t[..., None] * d
            valid = (
                (point[..., 0] >= cfg.room_x_min - 1e-6)
                & (point[..., 0] <= cfg.room_x_max + 1e-6)
                & (np.abs(point[..., 1]) <= cfg.room_y + 1e-6)
                & (point[..., 2] >= -1e-6)
            )
            t = np.where(valid, t, np.inf)
            best = np.minimum(best, t)
        self._static_cache = best
        return best

    def _person_depth(self, center_xy: np.ndarray, track: PersonTrack) -> np.ndarray:
        """Depth to one cylinder, inf where the ray misses."""
        origin = self.T_world_camera[:3, 3]
        d = self._rays
        oc = origin[:2] - center_xy
        a = d[..., 0] ** 2 + d[..., 1] ** 2
        b = 2.0 * (oc[0] * d[..., 0] + oc[1] * d[..., 1])
        c = oc @ oc - track.radius_m**2
        disc = b * b - 4 * a * c
        out = np.full(d.shape[:2], np.inf)
        hit = (disc > 0) & (a > 0)
        if not hit.any():
            return out
        sqrt_disc = np.sqrt(np.where(hit, disc, 0.0))
        for sign in (-1.0, 1.0):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (-b + sign * sqrt_disc) / (2.0 * a)
            z = origin[2] + t * d[..., 2]
            ok = hit & (t > 0) & (z >= 0.0) & (z <= track.height_m) & (t < out)
            out = np.where(ok, t, out)
        return out

    def render(self, frame_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[GtPerson]]:
        """(depth, instance, rgb, ground truth) for one frame.

        ``instance`` is -1 for room surfaces and the person index otherwise.
        ``depth`` is exact -- noise is added later, by :meth:`frame`, so that
        the truth and the measurement stay separable.
        """
        cfg = self.config
        t_s = frame_index / cfg.fps
        active = self.active_tracks(t_s)
        indices = [i for i, _ in active]
        tracks = [tr for _, tr in active]
        centers = [tr.position(t_s) for tr in tracks]
        static = self._static_depth()
        person_depths = [
            self._person_depth(center, track) for center, track in zip(centers, tracks)
        ]

        depth = static.copy()
        instance = np.full(static.shape, -1, dtype=np.int16)
        for slot, person in enumerate(person_depths):
            closer = person < depth
            depth = np.where(closer, person, depth)
            instance = np.where(closer, np.int16(indices[slot]), instance)

        truth: List[GtPerson] = []
        camera_xy = self.T_world_camera[:3, 3][:2]
        for slot, (center, track, person) in enumerate(zip(centers, tracks, person_depths)):
            i = indices[slot]
            alone = person < static
            visible = instance == i
            alone_count = int(alone.sum())
            visible_count = int(visible.sum())
            box = self._amodal_box(center, track)
            truth.append(
                GtPerson(
                    index=i,
                    center_xy=center,
                    height_m=track.height_m,
                    radius_m=track.radius_m,
                    box_xyxy=box,
                    visibility=(visible_count / alone_count) if alone_count else 0.0,
                    in_view=alone_count > 0,
                    lidar_hits=0,
                    range_m=float(np.linalg.norm(center - camera_xy)),
                )
            )
        rgb = self._shade(depth, instance)
        return depth, instance, rgb, truth

    def _amodal_box(
        self, center_xy: np.ndarray, track: PersonTrack
    ) -> Optional[Tuple[float, float, float, float]]:
        """Project the cylinder's silhouette, ignoring occlusion."""
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring = np.stack(
            [
                center_xy[0] + track.radius_m * np.cos(angles),
                center_xy[1] + track.radius_m * np.sin(angles),
            ],
            axis=1,
        )
        points = np.concatenate(
            [
                np.concatenate([ring, np.zeros((len(ring), 1))], axis=1),
                np.concatenate([ring, np.full((len(ring), 1), track.height_m)], axis=1),
            ]
        )
        cam = transform_points(self.T_camera_world, points)
        cam = cam[cam[:, 2] > 1e-3]
        if len(cam) == 0:
            return None
        pixels = project_points(self.intrinsics.K, cam)
        pixels = pixels[np.isfinite(pixels).all(axis=1)]
        if len(pixels) == 0:
            return None
        x0 = float(np.clip(pixels[:, 0].min(), 0, self.config.width))
        x1 = float(np.clip(pixels[:, 0].max(), 0, self.config.width))
        y0 = float(np.clip(pixels[:, 1].min(), 0, self.config.height))
        y1 = float(np.clip(pixels[:, 1].max(), 0, self.config.height))
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            return None
        return (x0, y0, x1, y1)

    def _shade(self, depth: np.ndarray, instance: np.ndarray) -> np.ndarray:
        """A plausible-looking BGR image.

        Used for latency measurement and for the demo overlay.  It is *not*
        used for any detection accuracy claim: rendered cylinders are not
        people and a recall number measured on them would be meaningless.
        """
        origin = self.T_world_camera[:3, 3]
        finite = np.isfinite(depth)
        world = origin + np.where(finite, depth, 0.0)[..., None] * self._rays
        checker = (
            (np.floor(world[..., 0] / 0.5).astype(int) + np.floor(world[..., 1] / 0.5).astype(int))
            % 2
        ).astype(np.float64)
        base = np.where(checker > 0, 165.0, 135.0)
        base = np.where(world[..., 2] > 0.05, 105.0, base)  # walls darker than floor
        person = instance >= 0
        gradient = np.clip(1.0 - world[..., 2] / 2.2, 0.2, 1.0)
        base = np.where(person, 70.0 + 60.0 * gradient, base)
        shade = np.clip(1.0 - np.where(finite, depth, 20.0) / 18.0, 0.25, 1.0)
        grey = np.clip(base * shade, 0, 255)
        image = np.stack([grey * 0.95, grey, grey * 1.02], axis=-1)
        image = np.where(person[..., None], image * np.array([1.15, 0.85, 0.8]), image)
        image = image + self._rng.normal(0.0, 3.0, image.shape)
        return np.clip(image, 0, 255).astype(np.uint8)

    # -- sensors -----------------------------------------------------------

    def _measure_depth(self, depth: np.ndarray) -> np.ndarray:
        cfg = self.config
        out = depth.astype(np.float32).copy()
        out[~np.isfinite(out)] = 0.0
        out[out > cfg.depth_max_range_m] = 0.0
        valid = out > 0
        sigma = (
            out**2 * cfg.depth_disparity_noise_px / (self.intrinsics.fx * cfg.depth_baseline_m)
        )
        out = np.where(valid, out + self._rng.normal(0.0, 1.0, out.shape) * sigma, 0.0)

        if cfg.depth_dropout > 0:
            # Holes, not salt and pepper.  A structured-light sensor loses
            # patches -- a dark jumper, a specular floor -- and a patch is much
            # harder for the estimator to survive than scattered bad pixels.
            n_holes = int(
                cfg.depth_dropout * out.size / (np.pi * cfg.depth_hole_radius_px**2 + 1)
            )
            if n_holes > 0:
                ys = self._rng.integers(0, out.shape[0], n_holes)
                xs = self._rng.integers(0, out.shape[1], n_holes)
                r = cfg.depth_hole_radius_px
                for y, x in zip(ys, xs):
                    out[max(0, y - r) : y + r, max(0, x - r) : x + r] = 0.0
        return out.astype(np.float32)

    def _measure_lidar(
        self, frame_index: int, truth: List[GtPerson]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        cfg = self.config
        origin = np.asarray(cfg.lidar_xyz, dtype=np.float64)
        angles = self.lidar_angles
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        ranges = np.full(len(angles), np.inf)
        source = np.full(len(angles), -1, dtype=np.int32)

        for wall_normal, offset in (
            (np.array([1.0, 0.0]), cfg.room_x_max),
            (np.array([1.0, 0.0]), cfg.room_x_min),
            (np.array([0.0, 1.0]), cfg.room_y),
            (np.array([0.0, 1.0]), -cfg.room_y),
        ):
            denom = directions @ wall_normal
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (offset - origin[:2] @ wall_normal) / denom
            t = np.where((denom != 0) & (t > 0), t, np.inf)
            with np.errstate(invalid="ignore"):
                point = origin[:2] + np.where(np.isfinite(t), t, 0.0)[:, None] * directions
            inside = (
                (point[:, 0] >= cfg.room_x_min - 1e-6)
                & (point[:, 0] <= cfg.room_x_max + 1e-6)
                & (np.abs(point[:, 1]) <= cfg.room_y + 1e-6)
            )
            t = np.where(inside, t, np.inf)
            ranges = np.minimum(ranges, t)

        hits: Dict[int, int] = {p.index: 0 for p in truth}
        for person in truth:
            track = cfg.people[person.index]
            if not (0.0 <= origin[2] <= track.height_m):
                continue
            oc = origin[:2] - person.center_xy
            b = 2.0 * (directions @ oc)
            c = oc @ oc - track.radius_m**2
            disc = b * b - 4.0 * c
            with np.errstate(invalid="ignore"):
                t = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0.0))) / 2.0, np.inf)
            t = np.where(t > 0, t, np.inf)
            closer = t < ranges
            ranges = np.where(closer, t, ranges)
            source = np.where(closer, np.int32(person.index), source)
            hits[person.index] = int(closer.sum())

        ranges = np.where(ranges > cfg.lidar_max_range_m, np.nan, ranges)
        noise = self._rng.normal(0.0, cfg.lidar_range_sigma_m, ranges.shape)
        ranges = ranges + noise
        if cfg.lidar_dropout > 0:
            dropped = self._rng.random(ranges.shape) < cfg.lidar_dropout
            ranges = np.where(dropped, np.nan, ranges)
        del frame_index
        return ranges.astype(np.float64), angles.copy(), hits

    # -- public ------------------------------------------------------------

    def frame(self, frame_index: int, t_arrival_ns: Optional[int] = None) -> Tuple[
        FrameBundle, List[GtPerson]
    ]:
        depth_true, instance, rgb, truth = self.render(frame_index)
        depth = self._measure_depth(depth_true)
        ranges, angles, hits = self._measure_lidar(frame_index, truth)
        truth = [
            GtPerson(
                index=p.index,
                center_xy=p.center_xy,
                height_m=p.height_m,
                radius_m=p.radius_m,
                box_xyxy=p.box_xyxy,
                visibility=p.visibility,
                in_view=p.in_view,
                lidar_hits=hits.get(p.index, 0),
                range_m=p.range_m,
            )
            for p in truth
        ]
        t_ns = (
            t_arrival_ns
            if t_arrival_ns is not None
            else int(frame_index / self.config.fps * NS_PER_S)
        )
        bundle = FrameBundle(
            index=frame_index,
            t_arrival_ns=t_ns,
            rgb=rgb,
            depth=depth,
            scan_ranges=ranges,
            scan_angles=angles,
            intrinsics=self.intrinsics,
            t_rgb_ns=t_ns,
            t_depth_ns=t_ns,
            t_scan_ns=t_ns,
            meta={"instance_map": instance, "depth_true": depth_true},
        )
        return bundle, truth

    def frames(self, limit: Optional[int] = None):
        total = self.n_frames if limit is None else min(limit, self.n_frames)
        for i in range(total):
            yield self.frame(i)

    def summary(self) -> Dict[str, object]:
        cfg = self.config
        return {
            "duration_s": cfg.duration_s,
            "fps": cfg.fps,
            "frames": self.n_frames,
            "people": len(cfg.people),
            "image": [cfg.width, cfg.height],
            "camera_xyz": list(cfg.camera_xyz),
            "camera_tilt_down_deg": cfg.camera_tilt_down_deg,
            "lidar_xyz": list(cfg.lidar_xyz),
            "lidar_beams": len(self.lidar_angles),
            "depth_max_range_m": cfg.depth_max_range_m,
            "depth_dropout": cfg.depth_dropout,
            "lidar_dropout": cfg.lidar_dropout,
            "seed": cfg.seed,
        }


class SimulatedDetector:
    """Ground-truth boxes with detector-like faults injected.

    **This is a test harness, not a detector.**  It never looks at the image.
    It exists so that the fusion, zone and safety stages can be evaluated
    against known 3D truth with a *controllable* detection quality: at
    ``miss_probability = 0`` the geometry error is the fusion error alone and
    turning the misses and false positives up shows how the safety layer
    behaves when the detector is poor.

    Detection quality on real images is measured in
    ``analysis/detection_metrics.py`` with the real ONNX model and the two
    numbers are never mixed.
    """

    name = "simulated"

    def __init__(
        self,
        scene: SyntheticScene,
        *,
        jitter_px: float = 2.5,
        miss_probability: float = 0.05,
        false_positive_rate: float = 0.005,
        min_visibility: float = 0.25,
        min_height_px: float = 18.0,
        score_mean: float = 0.72,
        seed: int = 7,
    ):
        self.scene = scene
        self.jitter_px = jitter_px
        self.miss_probability = miss_probability
        self.false_positive_rate = false_positive_rate
        self.min_visibility = min_visibility
        self.min_height_px = min_height_px
        self.score_mean = score_mean
        self._rng = np.random.default_rng(seed)
        self._truth: List[GtPerson] = []
        self.misses = 0
        self.emitted = 0
        self.false_positives = 0

    def set_truth(self, truth: Sequence[GtPerson]) -> None:
        self._truth = list(truth)

    def detect(self, bgr: np.ndarray) -> List[Detection2D]:
        del bgr
        out: List[Detection2D] = []
        for person in self._truth:
            if person.box_xyxy is None or not person.in_view:
                continue
            x0, y0, x1, y1 = person.box_xyxy
            if (y1 - y0) < self.min_height_px:
                continue
            if person.visibility < self.min_visibility:
                self.misses += 1
                continue
            if self._rng.random() < self.miss_probability:
                self.misses += 1
                continue
            jitter = self._rng.normal(0.0, self.jitter_px, 4)
            score = float(np.clip(self._rng.normal(self.score_mean, 0.12), 0.05, 0.99))
            out.append(
                Detection2D(
                    xyxy=(
                        float(x0 + jitter[0]),
                        float(y0 + jitter[1]),
                        float(x1 + jitter[2]),
                        float(y1 + jitter[3]),
                    ),
                    score=score,
                )
            )
            self.emitted += 1
        if self._rng.random() < self.false_positive_rate:
            w = self.scene.config.width
            h = self.scene.config.height
            x = float(self._rng.uniform(0, w - 60))
            y = float(self._rng.uniform(h * 0.3, h - 90))
            out.append(
                Detection2D(
                    xyxy=(x, y, x + 40.0, y + 85.0),
                    score=float(self._rng.uniform(0.2, 0.4)),
                )
            )
            self.false_positives += 1
        return out

    def warmup(self, iterations: int = 0) -> None:
        del iterations

    def close(self) -> None:
        return None

    @property
    def info(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "note": "ground truth with injected faults; not a detector",
            "jitter_px": self.jitter_px,
            "miss_probability": self.miss_probability,
            "false_positive_rate": self.false_positive_rate,
            "min_visibility": self.min_visibility,
        }


class TimedRealDetector:
    """Runs the real model for its *cost* and returns ground truth for its *answers*.

    A timing harness and it exists because of an awkward fact: the real
    detector does not see the rig's people.  They are shaded cylinders, not
    photographs, so a YOLO model trained on COCO finds almost nothing in a
    rendered frame -- which is exactly what ``data/DATASET_CARD.md`` says and
    is the reason no accuracy claim comes from this rig.

    That leaves the reaction-time measurement stuck.  Reaction time is
    "arrival to the PLC withdrawing the permit" and there is no permit
    withdrawal if nothing is ever detected.  Running the whole thing with the
    simulated detector instead measures a pipeline with no model in it, which
    is not the pipeline anybody would deploy.

    So this wrapper does both: it calls the real ONNX session on every frame,
    so the inference cost, its variance and its tail are genuinely in the
    measurement and then throws the model's output away and returns the
    detections the rig knows to be correct.  The result is an honest
    measurement of *timing* and no measurement of accuracy at all.

    It is named after what it is, it lives beside the other harness rather than
    under ``detection/`` and every result file that used it records
    ``detector_kind: onnx_timing``.
    """

    name = "onnx_timing"

    def __init__(self, real_detector, truth_detector: "SimulatedDetector"):
        self.real = real_detector
        self.truth = truth_detector
        self.last_inference_ms = 0.0

    def set_truth(self, truth: Sequence[GtPerson]) -> None:
        self.truth.set_truth(truth)

    def detect(self, bgr: np.ndarray) -> List[Detection2D]:
        self.real.detect(bgr)
        self.last_inference_ms = getattr(self.real, "last_inference_ms", 0.0)
        return self.truth.detect(bgr)

    def warmup(self, iterations: int = 5) -> None:
        self.real.warmup(iterations)

    def close(self) -> None:
        self.real.close()

    @property
    def info(self) -> Dict[str, object]:
        return dict(
            self.real.info,
            backend=self.name,
            note=(
                "the real model runs on every frame so the timing is real; the "
                "detections come from the synthetic rig's ground truth, so no "
                "accuracy claim can be made from this run"
            ),
        )


def build_scene(**overrides) -> SyntheticScene:
    return SyntheticScene(SceneConfig(**overrides))


def match_boxes(
    predicted: Sequence[Detection2D],
    truth: Sequence[GtPerson],
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy IoU matching, highest score first.  (matches, unmatched_gt, false_positives)."""
    gt_boxes = [t.box_xyxy for t in truth if t.box_xyxy is not None]
    gt_index = [i for i, t in enumerate(truth) if t.box_xyxy is not None]
    if not predicted or not gt_boxes:
        return [], list(range(len(gt_boxes))), list(range(len(predicted)))
    order = sorted(range(len(predicted)), key=lambda i: -predicted[i].score)
    ious = iou_matrix([p.xyxy for p in predicted], gt_boxes)
    taken = set()
    matches: List[Tuple[int, int]] = []
    unmatched_pred: List[int] = []
    for p in order:
        best, best_iou = -1, iou_threshold
        for g in range(len(gt_boxes)):
            if g in taken:
                continue
            if ious[p, g] >= best_iou:
                best, best_iou = g, ious[p, g]
        if best >= 0:
            taken.add(best)
            matches.append((p, gt_index[best]))
        else:
            unmatched_pred.append(p)
    missed = [gt_index[g] for g in range(len(gt_boxes)) if g not in taken]
    return matches, missed, unmatched_pred
