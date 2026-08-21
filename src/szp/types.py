"""The data that moves between pipeline stages.

Everything here is a frozen dataclass holding plain floats or numpy arrays.  No
stage mutates another stage's output, which is what lets the latency
instrumentation timestamp a stage boundary and trust that the object it saw is
the object the next stage received.

Coordinate frames, fixed once here and never re-argued elsewhere:

    camera   OpenCV.  +x right, +y down, +z forward along the optical axis.
    depth    the depth sensor.  Assumed registered to ``camera`` unless
             ``extrinsics.yaml`` says otherwise.
    lidar    REP-103.  +x forward, +y left, +z up.  The scan is a horizontal
             ring in the lidar's z = 0 plane.
    base     the machine.  REP-103, +z up, origin at the centre of the machine
             footprint on the floor.
    world    the fixed cell frame.  +z up, z = 0 is the floor.  Zone polygons
             are defined in this frame and in no other.

A transform named ``T_a_b`` maps a point expressed in ``b`` into ``a``, so that
``p_a = T_a_b @ p_b``.  This is the only convention used in the repository and
``fusion.extrinsics`` enforces it on load.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

# Nanoseconds from a monotonic clock.  Wall-clock time appears nowhere in the
# hot path: a step in the system clock must not be able to trip the watchdog.
Nanos = int

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000


class RangeSource(enum.Enum):
    """Where a 3D box got its range from.

    ``FUSED`` is the best case and ``UNRANGED`` means the detector saw a person
    that neither ranging sensor confirmed.  An ``UNRANGED`` box is never
    discarded; see ``Box3D.range_ambiguous``.
    """

    FUSED = "fused"          # depth and lidar agreed
    DEPTH = "depth"          # depth only
    LIDAR = "lidar"          # lidar only
    UNRANGED = "unranged"    # neither; range inferred from apparent height


class SafetyState(enum.IntEnum):
    """The states the machine-facing FSM can be in.

    The integer values are part of the CAN wire format and must not be
    renumbered without bumping ``safety.can_frames.PROTOCOL_VERSION``.
    """

    INIT = 0        # starting up, nothing trusted yet
    CLEAR = 1       # zones empty, sensors healthy, motion permitted
    WARN = 2        # someone is in a warning zone; motion permitted, derated
    STOP = 3        # someone is in a stop zone; motion not permitted
    FAULT = 4       # the pipeline cannot vouch for the scene; motion not permitted
    RECOVERING = 5  # cause cleared and operator has reset, waiting out the debounce

    @property
    def motion_permitted(self) -> bool:
        return self in (SafetyState.CLEAR, SafetyState.WARN)


class Cause(enum.IntFlag):
    """Why the FSM is not in CLEAR.  Packed into one CAN byte, so eight at most."""

    NONE = 0
    ZONE_OCCUPIED = 1 << 0
    SENSOR_STALE = 1 << 1
    SENSOR_FAILED = 1 << 2
    DEADLINE_MISS = 1 << 3
    HEARTBEAT_LOST = 1 << 4
    UNRANGED_DETECTION = 1 << 5
    INTERNAL_ERROR = 1 << 6
    OPERATOR_LATCH = 1 << 7


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics together with the image size they were calibrated at."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: Tuple[float, ...] = ()

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def scaled(self, width: int, height: int) -> "CameraIntrinsics":
        """Intrinsics for the same camera at a different image size."""
        sx, sy = width / self.width, height / self.height
        return replace(
            self,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=width,
            height=height,
        )


@dataclass(frozen=True)
class Detection2D:
    """One person in image coordinates.

    ``xyxy`` is in pixels on the original frame, not on the letterboxed tensor.
    ``score`` is the detector's class confidence for ``person``.
    """

    xyxy: Tuple[float, float, float, float]
    score: float
    label: str = "person"

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.xyxy
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    def touches_border(self, width: int, height: int, margin: float = 2.0) -> bool:
        x0, y0, x1, y1 = self.xyxy
        return x0 <= margin or y0 <= margin or x1 >= width - margin or y1 >= height - margin


@dataclass(frozen=True)
class Box3D:
    """A person as an upright box in the world frame.

    ``center`` is (x, y, z) with z = 0 on the floor, so ``center[2]`` is half
    the box height for anyone standing on the ground.  ``size`` is
    (length, width, height) in metres.  Yaw is not estimated: a 2D lidar plus a
    monocular depth map does not constrain torso orientation and for zone
    occupancy it does not matter, so a person is modelled as axis-aligned in
    plan view.

    ``sigma_xy`` is the standard deviation of the horizontal position estimate
    in metres.  It is not decoration.  ``zones.occupancy`` dilates the
    footprint by a multiple of it before testing, so a poorly ranged person
    occupies a larger area and is more likely to trip a zone, not less.
    """

    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    score: float
    source: RangeSource
    sigma_xy: float
    frame: str = "world"
    detection: Optional[Detection2D] = None
    #: For UNRANGED boxes, the (near, far) metres along the viewing ray that the
    #: person could plausibly occupy.  None otherwise.
    range_interval: Optional[Tuple[float, float]] = None
    #: Plan-view outline in the world frame, (N, 2), counter-clockwise.  The
    #: fusion stage sets this because it is the only stage that knows where the
    #: camera was: an unranged person is a swept wedge along the viewing ray,
    #: not a rectangle around a guessed centre.  When None the zone stage falls
    #: back to an axis-aligned rectangle from ``center`` and ``size``.
    footprint: Optional[np.ndarray] = None

    @property
    def range_ambiguous(self) -> bool:
        return self.source is RangeSource.UNRANGED

    @property
    def footprint_radius(self) -> float:
        """Half-diagonal of the plan-view footprint, metres."""
        return 0.5 * float(np.hypot(self.size[0], self.size[1]))


@dataclass(frozen=True)
class ZoneOccupancy:
    """The result of testing one zone against the current box set."""

    zone_id: str
    kind: str                 # "stop" or "warn"
    occupied: bool
    #: Signed distance in metres from the zone boundary to the nearest
    #: footprint, negative when inside.  Infinite when there are no boxes.
    nearest_m: float
    box_indices: Tuple[int, ...] = ()
    #: True when occupancy was decided by a range-ambiguous box alone.
    from_unranged: bool = False


@dataclass(frozen=True)
class Health:
    """What the watchdog thinks of each input at one instant."""

    sources_ok: Mapping[str, bool]
    stale: Tuple[str, ...] = ()
    failed: Tuple[str, ...] = ()
    heartbeat_ok: bool = True
    deadline_miss: bool = False
    period_ms: float = 0.0

    @property
    def all_ok(self) -> bool:
        return (
            all(self.sources_ok.values())
            and not self.stale
            and not self.failed
            and self.heartbeat_ok
            and not self.deadline_miss
        )


@dataclass(frozen=True)
class SafetyDecision:
    """What the FSM decided for one frame and why."""

    state: SafetyState
    cause: Cause
    zone_mask: int
    min_range_m: float
    latched: bool
    detail: str = ""


@dataclass(frozen=True)
class FrameBundle:
    """One synchronised set of sensor readings.

    ``rgb`` is BGR uint8, in OpenCV channel order.  ``depth`` is float32 metres
    where non-finite or non-positive entries mean "no return".  ``scan_ranges``
    is metres and ``scan_angles`` radians in the lidar frame, both 1D and the
    same length; a non-finite range means "no return", which is not the same as
    a maximum-range return.
    """

    index: int
    t_arrival_ns: Nanos
    rgb: np.ndarray
    depth: Optional[np.ndarray] = None
    scan_ranges: Optional[np.ndarray] = None
    scan_angles: Optional[np.ndarray] = None
    intrinsics: Optional[CameraIntrinsics] = None
    t_rgb_ns: Optional[Nanos] = None
    t_depth_ns: Optional[Nanos] = None
    t_scan_ns: Optional[Nanos] = None
    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    """Everything one frame produced, for logging and for the analysis scripts."""

    index: int
    t_arrival_ns: Nanos
    detections: Sequence[Detection2D]
    boxes: Sequence[Box3D]
    occupancy: Sequence[ZoneOccupancy]
    health: Health
    decision: SafetyDecision
    stage_ns: Mapping[str, int]
    t_published_ns: Optional[Nanos] = None

    @property
    def e2e_ms(self) -> float:
        """Frame arrival to the CAN frame hitting the bus, in milliseconds."""
        if self.t_published_ns is None:
            return float("nan")
        return (self.t_published_ns - self.t_arrival_ns) / NS_PER_MS
