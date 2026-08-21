"""The fail-safe state machine.

    INIT -> CLEAR                once the inputs have been healthy for
                                 ``healthy_frames_to_start`` consecutive frames
    CLEAR <-> WARN               a warning zone is occupied and back again
    any -> STOP                  a stop zone is occupied
    any -> FAULT                 the inputs cannot be vouched for
    STOP/FAULT -> RECOVERING     the cause has gone and the operator has reset
    RECOVERING -> CLEAR/WARN     after ``clear_frames_to_release`` clear frames

Four properties this machine is written to have, each of which has a test in
``tests/test_fsm.py`` named after it:

*latching*
    STOP and FAULT do not clear themselves.  Once entered, the machine stays
    there until the cause has gone **and** the operator has pressed reset.  A
    person who steps out of a zone does not restart the machine; a person is
    what stops it, an operator is what starts it.

*fault dominance*
    When the inputs are not healthy the state is FAULT regardless of what the
    zones say, because an empty-looking zone from a dead camera is
    indistinguishable from an empty zone.

*no self-clearing on reset alone*
    Pressing reset while somebody is still standing in a stop zone does
    nothing.  The cause is re-evaluated every frame and reset is only consumed
    when the cause has already gone.

*monotone entry*
    Entering STOP takes effect on the frame the cause appears.  There is no
    debounce on the way in; ``exit_frames`` in the zone config debounces the
    way out only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from szp.types import Cause, Health, Nanos, SafetyDecision, SafetyState, ZoneOccupancy

__all__ = ["FsmConfig", "SafetyFsm"]


@dataclass(frozen=True)
class FsmConfig:
    #: Consecutive healthy frames required before leaving INIT.  The pipeline
    #: does not permit motion the instant it starts: the watchdog needs a few
    #: frames before staleness means anything.
    healthy_frames_to_start: int = 5
    #: Consecutive clear frames in RECOVERING before motion is permitted again.
    clear_frames_to_release: int = 5
    #: When False the machine leaves STOP on its own once the zone clears.
    #: Left in because a demo without an operator needs it; the default is the
    #: latching behaviour and ``scripts/run_demo.py`` says which it used.
    require_operator_reset: bool = True
    #: A stop zone tripped only by a range-ambiguous detection still stops the
    #: machine.  Turning this off is what a system optimised for uptime rather
    #: than for people would do and the flag exists so the analysis can
    #: quantify what that trade costs.
    unranged_causes_stop: bool = True


class SafetyFsm:
    def __init__(self, config: Optional[FsmConfig] = None):
        self.config = config or FsmConfig()
        self.state = SafetyState.INIT
        self._latched = False
        self._latched_cause = Cause.NONE
        self._healthy_run = 0
        self._clear_run = 0
        self._reset_pending = False
        self._history: List[SafetyState] = []

    # -- operator ----------------------------------------------------------

    def request_reset(self) -> None:
        """Record an operator reset.  Consumed only when the cause has gone."""
        self._reset_pending = True

    def reset_state(self) -> None:
        """Full re-initialisation.  For tests and for a clean restart."""
        self.state = SafetyState.INIT
        self._latched = False
        self._latched_cause = Cause.NONE
        self._healthy_run = 0
        self._clear_run = 0
        self._reset_pending = False
        self._history.clear()

    @property
    def latched(self) -> bool:
        return self._latched

    @property
    def history(self) -> Sequence[SafetyState]:
        return tuple(self._history)

    # -- per frame ---------------------------------------------------------

    def step(
        self,
        occupancy: Sequence[ZoneOccupancy],
        health: Health,
        *,
        min_range_m: float = float("inf"),
        zone_mask: int = 0,
        now_ns: Optional[Nanos] = None,
    ) -> SafetyDecision:
        del now_ns  # the FSM is driven by frames, the watchdog by the clock
        cfg = self.config

        cause = Cause.NONE
        if health.failed:
            cause |= Cause.SENSOR_FAILED
        if health.stale:
            cause |= Cause.SENSOR_STALE
        if health.deadline_miss:
            cause |= Cause.DEADLINE_MISS
        if not health.heartbeat_ok:
            cause |= Cause.HEARTBEAT_LOST
        unhealthy = bool(cause)

        stop_hits = [
            o
            for o in occupancy
            if o.occupied and o.kind == "stop" and (cfg.unranged_causes_stop or not o.from_unranged)
        ]
        warn_hits = [o for o in occupancy if o.occupied and o.kind == "warn"]
        if stop_hits:
            cause |= Cause.ZONE_OCCUPIED
            if any(o.from_unranged for o in stop_hits):
                cause |= Cause.UNRANGED_DETECTION

        blocking = unhealthy or bool(stop_hits)

        if blocking:
            self._clear_run = 0
            self._latched = True
            self._latched_cause = cause
            self._reset_pending = False
            self.state = SafetyState.FAULT if unhealthy else SafetyState.STOP
            detail = self._describe(stop_hits, health)
            return self._emit(cause, zone_mask, min_range_m, detail)

        self._clear_run += 1

        if self.state is SafetyState.INIT:
            self._healthy_run += 1
            if self._healthy_run < cfg.healthy_frames_to_start:
                return self._emit(
                    Cause.NONE,
                    zone_mask,
                    min_range_m,
                    f"starting up, {self._healthy_run}/"
                    f"{cfg.healthy_frames_to_start} healthy frames",
                )
            self.state = SafetyState.CLEAR

        if self._latched:
            if cfg.require_operator_reset and not self._reset_pending:
                self.state = SafetyState.STOP
                return self._emit(
                    self._latched_cause | Cause.OPERATOR_LATCH,
                    zone_mask,
                    min_range_m,
                    "cause cleared, waiting for operator reset",
                )
            self._reset_pending = False
            self._latched = False
            self._latched_cause = Cause.NONE
            self._clear_run = 1
            self.state = SafetyState.RECOVERING

        if self.state is SafetyState.RECOVERING:
            if self._clear_run < cfg.clear_frames_to_release:
                return self._emit(
                    Cause.NONE,
                    zone_mask,
                    min_range_m,
                    f"recovering, {self._clear_run}/{cfg.clear_frames_to_release} clear frames",
                )
            self.state = SafetyState.CLEAR

        self.state = SafetyState.WARN if warn_hits else SafetyState.CLEAR
        detail = (
            "warning zone occupied: " + ", ".join(o.zone_id for o in warn_hits)
            if warn_hits
            else "clear"
        )
        return self._emit(Cause.NONE, zone_mask, min_range_m, detail)

    # -- helpers -----------------------------------------------------------

    def _describe(self, stop_hits: Sequence[ZoneOccupancy], health: Health) -> str:
        parts: List[str] = []
        if stop_hits:
            parts.append("stop zone occupied: " + ", ".join(o.zone_id for o in stop_hits))
        if health.failed:
            parts.append("failed: " + ", ".join(health.failed))
        if health.stale:
            parts.append("stale: " + ", ".join(health.stale))
        if health.deadline_miss:
            parts.append(f"deadline miss, period {health.period_ms:.0f} ms")
        if not health.heartbeat_ok:
            parts.append("no PLC heartbeat")
        return "; ".join(parts) or "blocked"

    def _emit(
        self, cause: Cause, zone_mask: int, min_range_m: float, detail: str
    ) -> SafetyDecision:
        self._history.append(self.state)
        return SafetyDecision(
            state=self.state,
            cause=cause,
            zone_mask=zone_mask,
            min_range_m=min_range_m,
            latched=self._latched,
            detail=detail,
        )
