"""Deadline monitoring for every input the pipeline depends on.

The watchdog answers one question per frame: is there anything about the state
of the inputs that means the pipeline cannot vouch for the scene?  Three ways
that happens:

* **Staleness.**  A source has not produced anything within its deadline.  A
  depth camera that has stopped publishing looks exactly like a depth camera
  looking at an empty room and the fusion stage cannot tell the difference, so
  the age of each input is tracked here instead.
* **Failure.**  A source has been stale for several consecutive frames or the
  driver raised.  Staleness that persists is a different fault from a single
  dropped frame and the FSM treats it as one.
* **Deadline miss.**  The pipeline itself has overrun its period.  A pipeline
  running at 4 Hz is not a pipeline running slowly, it is a pipeline whose
  reaction time no longer matches the stopping distance the zones were sized
  for.

All ages are computed from a monotonic clock.  The system clock is not used
anywhere in this file: an NTP step must not be able to age every sensor out at
once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple

from szp.types import Health, Nanos, NS_PER_MS

__all__ = ["WatchdogConfig", "Watchdog"]

DEFAULT_DEADLINES_MS: Dict[str, float] = {
    "rgb": 100.0,
    "depth": 150.0,
    "lidar": 150.0,
    "detector": 200.0,
}


@dataclass(frozen=True)
class WatchdogConfig:
    #: Maximum age, milliseconds, before a source counts as stale.  Sized at
    #: roughly three nominal periods for each source: one dropped frame is
    #: normal on a USB camera, three in a row is not.
    deadlines_ms: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DEADLINES_MS)
    )
    #: A source stale for this many consecutive pipeline frames is called failed.
    failed_after_frames: int = 3
    #: No heartbeat from the PLC for this long and the link counts as lost.
    heartbeat_timeout_ms: float = 300.0
    #: Longest permitted interval between pipeline frames.
    period_deadline_ms: float = 150.0
    #: Set False for offline replay, where wall-clock gaps between frames are
    #: an artefact of reading files and not of the pipeline being slow.
    enforce_period: bool = True


class Watchdog:
    """Stateful across frames.  One instance per pipeline."""

    def __init__(self, config: Optional[WatchdogConfig] = None):
        self.config = config or WatchdogConfig()
        self._last_seen: Dict[str, Nanos] = {}
        self._stale_run: Dict[str, int] = {}
        self._last_frame_ns: Optional[Nanos] = None
        self._last_heartbeat_ns: Optional[Nanos] = None
        self._period_ms: float = 0.0

    def reset(self) -> None:
        self._last_seen.clear()
        self._stale_run.clear()
        self._last_frame_ns = None
        self._last_heartbeat_ns = None
        self._period_ms = 0.0

    def note_heartbeat(self, now_ns: Nanos) -> None:
        self._last_heartbeat_ns = now_ns

    def note_source(self, name: str, when_ns: Nanos) -> None:
        self._last_seen[name] = when_ns

    @property
    def period_ms(self) -> float:
        return self._period_ms

    def update(
        self,
        now_ns: Nanos,
        *,
        observed: Optional[Mapping[str, Optional[Nanos]]] = None,
        expected: Optional[Iterable[str]] = None,
    ) -> Health:
        """Advance one pipeline frame and report health.

        ``observed`` maps a source name to the timestamp of the reading that
        arrived this frame or ``None`` when nothing arrived.  ``expected``
        lists the sources that are supposed to be present at all; a source that
        is configured but has never once reported is failed, not merely stale,
        because it has probably never been connected.
        """
        cfg = self.config
        for name, when in (observed or {}).items():
            if when is not None:
                self._last_seen[name] = when

        names = list(expected) if expected is not None else list(cfg.deadlines_ms)
        stale: list = []
        failed: list = []
        sources_ok: Dict[str, bool] = {}

        for name in names:
            deadline = float(cfg.deadlines_ms.get(name, 150.0))
            last = self._last_seen.get(name)
            if last is None:
                failed.append(name)
                sources_ok[name] = False
                self._stale_run[name] = self._stale_run.get(name, 0) + 1
                continue
            age_ms = (now_ns - last) / NS_PER_MS
            if age_ms > deadline:
                self._stale_run[name] = self._stale_run.get(name, 0) + 1
                sources_ok[name] = False
                if self._stale_run[name] >= cfg.failed_after_frames:
                    failed.append(name)
                else:
                    stale.append(name)
            else:
                self._stale_run[name] = 0
                sources_ok[name] = True

        heartbeat_ok = True
        if self._last_heartbeat_ns is None:
            # Nothing has been heard yet.  That is only a fault once the
            # timeout has elapsed since the first frame, so a pipeline that
            # starts before the PLC does not fault immediately.
            if self._last_frame_ns is not None:
                heartbeat_ok = False
        else:
            age_ms = (now_ns - self._last_heartbeat_ns) / NS_PER_MS
            heartbeat_ok = age_ms <= cfg.heartbeat_timeout_ms

        deadline_miss = False
        if self._last_frame_ns is not None:
            self._period_ms = (now_ns - self._last_frame_ns) / NS_PER_MS
            if cfg.enforce_period and self._period_ms > cfg.period_deadline_ms:
                deadline_miss = True
        self._last_frame_ns = now_ns

        return Health(
            sources_ok=sources_ok,
            stale=tuple(stale),
            failed=tuple(failed),
            heartbeat_ok=heartbeat_ok,
            deadline_miss=deadline_miss,
            period_ms=self._period_ms,
        )

    def ages_ms(self, now_ns: Nanos) -> Dict[str, float]:
        """Per-source age, for the health log and the plots."""
        return {
            name: (now_ns - when) / NS_PER_MS for name, when in sorted(self._last_seen.items())
        }

    def deadlines(self) -> Tuple[Tuple[str, float], ...]:
        return tuple(sorted((k, float(v)) for k, v in self.config.deadlines_ms.items()))
