"""One frame in, one CAN frame out.

The whole pipeline is single-threaded and synchronous on purpose.  A worker
pool would raise throughput and would also make the end-to-end latency a
distribution over queue depths rather than a property of the code and the
number that matters here is the worst case from frame arrival to e-stop frame.
When the target cannot keep up the right answer is a smaller model or a lower
resolution, not a deeper queue: a queue converts a throughput problem into a
latency problem and hides it.

Order of work and why it is that order:

1. **detect** first, because everything downstream depends on it and because
   it is by far the longest stage; anything that can be done while it runs is
   done in the frame source instead.
2. **fuse**, then **zones**, then **safety**.  Each stage narrows: pixels to
   metres, metres to a boolean per zone, booleans to one state.
3. **publish** last and unconditionally.  The e-stop frame goes out on every
   frame including frames where nothing changed, because the receiver treats
   silence as a stop and a pipeline that only speaks when something changes
   would be indistinguishable from a dead one.

An exception anywhere in stages 1 to 3 does not propagate.  It is caught, the
frame is marked as an internal error and the state machine is driven to FAULT
with that cause -- which stops the machine.  A traceback that escapes to the
caller would stop the loop, stop the publishing and stop the machine too but
only after the permit timeout and with no diagnostic on the bus.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from szp.detection.base import Detector, DetectorConfig
from szp.fusion.extrinsics import ExtrinsicSet
from szp.fusion.fuse import FusionConfig, FusionDebug, Fuser
from szp.latency.instrument import Instrument
from szp.safety.can_frames import decode_heartbeat, CanFrame, ID_HEARTBEAT
from szp.safety.fsm import FsmConfig, SafetyFsm
from szp.safety.publisher import CanPublisher
from szp.safety.watchdog import Watchdog, WatchdogConfig
from szp.types import (
    Box3D,
    Cause,
    Detection2D,
    FrameBundle,
    Health,
    NS_PER_MS,
    PipelineResult,
    SafetyDecision,
    SafetyState,
    ZoneOccupancy,
)
from szp.zones.config import ZoneSet
from szp.zones.occupancy import OccupancyEvaluator, nearest_range_m, zone_mask

__all__ = ["Pipeline", "PipelineConfig"]


@dataclass
class PipelineConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    fsm: FsmConfig = field(default_factory=FsmConfig)
    #: Zone and health frames are lower priority and are sent less often.
    zones_every: int = 3
    health_every: int = 6
    #: Poll the bus for the PLC heartbeat once a frame.  Zero timeout: the
    #: pipeline never blocks on the bus.
    poll_heartbeat: bool = True


class Pipeline:
    def __init__(
        self,
        detector: Detector,
        extrinsics: ExtrinsicSet,
        zone_set: ZoneSet,
        publisher: Optional[CanPublisher] = None,
        config: Optional[PipelineConfig] = None,
        instrument: Optional[Instrument] = None,
    ):
        self.config = config or PipelineConfig()
        self.detector = detector
        self.extrinsics = extrinsics
        self.zone_set = zone_set
        self.publisher = publisher
        self.fuser = Fuser(extrinsics, self.config.fusion)
        self.occupancy = OccupancyEvaluator(zone_set)
        self.watchdog = Watchdog(self.config.watchdog)
        self.fsm = SafetyFsm(self.config.fsm)
        self.instrument = instrument or Instrument()
        self.fusion_debug = FusionDebug()
        self._frame_count = 0
        self._errors: List[str] = []
        # Snapshot the detector's description now.  ``report()`` is usually
        # called after ``close()`` and a closed ONNX session reports an empty
        # provider list, which is how the first run of this pipeline came to
        # claim it had no execution provider at all.
        self._detector_info = dict(detector.info)

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.detector.close()
        finally:
            if self.publisher is not None:
                self.publisher.close()

    # -- per frame ---------------------------------------------------------

    def process(self, bundle: FrameBundle) -> PipelineResult:
        instrument = self.instrument
        now_ns = time.perf_counter_ns()
        instrument.begin_frame(now_ns)

        detections: Sequence[Detection2D] = ()
        boxes: Sequence[Box3D] = ()
        occupancy: List[ZoneOccupancy] = []
        internal_error = ""

        try:
            with instrument.stage("detect_pre_post"):
                detections = self.detector.detect(bundle.rgb)
            inference_ns = int(getattr(self.detector, "last_inference_ms", 0.0) * NS_PER_MS)
            if inference_ns:
                instrument.record_stage("inference", inference_ns)
                # detect_pre_post was timed around the whole call; take the
                # model out of it so the two stages sum to the detector cost
                # instead of double counting it.
                instrument.adjust_stage("detect_pre_post", -inference_ns)

            intrinsics = bundle.intrinsics or self.extrinsics.intrinsics.get("camera")
            if intrinsics is None:
                raise ValueError(
                    "no camera intrinsics: neither the frame nor config/extrinsics.yaml "
                    "provides them"
                )
            if (intrinsics.width, intrinsics.height) != (
                bundle.rgb.shape[1],
                bundle.rgb.shape[0],
            ):
                intrinsics = intrinsics.scaled(bundle.rgb.shape[1], bundle.rgb.shape[0])

            with instrument.stage("fusion"):
                self.fusion_debug = FusionDebug()
                boxes = self.fuser.fuse(
                    detections,
                    bundle.depth,
                    bundle.scan_ranges,
                    bundle.scan_angles,
                    intrinsics,
                    debug=self.fusion_debug,
                )

            with instrument.stage("zones"):
                occupancy = self.occupancy.evaluate(boxes)
        except Exception:  # the whole point is that nothing escapes
            internal_error = traceback.format_exc(limit=3)
            self._errors.append(internal_error)
            instrument.count("internal_errors")

        with instrument.stage("safety"):
            health = self._health(bundle, now_ns, degraded=bool(internal_error))
            mask = zone_mask(occupancy, self.zone_set)
            min_range = nearest_range_m(occupancy, boxes)
            if internal_error:
                decision = self._error_decision(mask, min_range)
            else:
                decision = self.fsm.step(
                    occupancy, health, min_range_m=min_range, zone_mask=mask
                )

        with instrument.stage("publish"):
            published_ns = self._publish_estop(decision)
        # The zone and health frames are diagnostics.  They go out after the
        # e-stop frame so that nothing shares the critical path with it and
        # they are timed under a name that the latency budget excludes.
        with instrument.stage("publish_aux"):
            self._publish_aux(occupancy, health)

        stage_ns = instrument.end_frame(
            index=bundle.index,
            arrival_ns=now_ns,
            published_ns=published_ns,
            extra={
                "state": int(decision.state),
                "detections": len(detections),
                "boxes": len(boxes),
                "unranged": self.fusion_debug.n_unranged,
                "arrival_ns": bundle.t_arrival_ns,
            },
        )
        self._frame_count += 1
        return PipelineResult(
            index=bundle.index,
            t_arrival_ns=now_ns,
            detections=list(detections),
            boxes=list(boxes),
            occupancy=occupancy,
            health=health,
            decision=decision,
            stage_ns=stage_ns,
            t_published_ns=published_ns,
        )

    # -- helpers -----------------------------------------------------------

    def _health(self, bundle: FrameBundle, now_ns: int, *, degraded: bool) -> Health:
        observed: Dict[str, Optional[int]] = {
            "rgb": now_ns if bundle.rgb is not None else None,
            "depth": now_ns if bundle.depth is not None and bundle.depth.size else None,
            "lidar": (
                now_ns
                if bundle.scan_ranges is not None and len(bundle.scan_ranges)
                else None
            ),
            "detector": None if degraded else now_ns,
        }
        if self.config.poll_heartbeat and self.publisher is not None:
            message = self.publisher.poll_heartbeat(timeout=0.0)
            if message is not None:
                decoded = decode_heartbeat(
                    CanFrame(ID_HEARTBEAT, bytes(message.data), bool(message.is_extended_id))
                )
                if decoded is not None:
                    self.watchdog.note_heartbeat(now_ns)
                    if decoded["reset_requested"]:
                        self.fsm.request_reset()
        else:
            # No bus, so no heartbeat to lose.  Feeding one keeps offline runs
            # out of a permanent HEARTBEAT_LOST fault and the results record
            # that the heartbeat was simulated.
            self.watchdog.note_heartbeat(now_ns)
        return self.watchdog.update(now_ns, observed=observed)

    def _error_decision(self, mask: int, min_range: float) -> SafetyDecision:
        self.fsm.state = SafetyState.FAULT
        return SafetyDecision(
            state=SafetyState.FAULT,
            cause=Cause.INTERNAL_ERROR,
            zone_mask=mask,
            min_range_m=min_range,
            latched=True,
            detail="internal error in the perception stages; see pipeline.errors",
        )

    def _publish_estop(self, decision: SafetyDecision) -> Optional[int]:
        if self.publisher is None:
            return None
        return self.publisher.publish_estop(decision).t_bus_ns

    def _publish_aux(self, occupancy: Sequence[ZoneOccupancy], health: Health) -> None:
        if self.publisher is None:
            return
        index = self._frame_count
        if self.config.zones_every and index % self.config.zones_every == 0:
            stop_mask = zone_mask(
                [o for o in occupancy if o.kind == "stop"], self.zone_set
            )
            warn_mask = zone_mask(
                [o for o in occupancy if o.kind == "warn"], self.zone_set
            )
            nearest = min((o.nearest_m for o in occupancy), default=float("inf"))
            self.publisher.publish_zones(stop_mask, warn_mask, nearest)
        if self.config.health_every and index % self.config.health_every == 0:
            self.publisher.publish_health(
                health, self.watchdog.period_ms, self.instrument.p99_e2e_ms()
            )

    # -- reporting ---------------------------------------------------------

    @property
    def errors(self) -> Sequence[str]:
        return tuple(self._errors)

    def report(self) -> Dict[str, object]:
        report = self.instrument.report()
        report["detector"] = dict(self._detector_info)
        report["zones"] = [z.id for z in self.zone_set.zones]
        report["internal_errors"] = len(self._errors)
        if self.publisher is not None:
            report["can"] = dict(self.publisher.stats)
        return report


def boxes_to_dicts(boxes: Sequence[Box3D]) -> List[Dict[str, object]]:
    """JSON-friendly boxes, for the result files."""
    out = []
    for box in boxes:
        out.append(
            {
                "center": [float(v) for v in box.center],
                "size": [float(v) for v in box.size],
                "score": float(box.score),
                "source": box.source.value,
                "sigma_xy": float(box.sigma_xy),
                "range_interval": (
                    [float(v) for v in box.range_interval] if box.range_interval else None
                ),
                "footprint": (
                    [[float(x), float(y)] for x, y in np.asarray(box.footprint)]
                    if box.footprint is not None
                    else None
                ),
            }
        )
    return out
