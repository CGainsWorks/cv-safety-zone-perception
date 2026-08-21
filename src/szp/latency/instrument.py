"""Stage timing and the percentiles that come out of it.

Three decisions worth explaining.

**Percentiles, not means.**  The mean latency of a safety pipeline is close to
useless.  What sizes the stop distance is the worst case and what shows up as
an intermittent fault in the field is the tail.  Everything reported here is
p50, p95, p99 and max; the mean appears only so that a reader can see how far
the tail is from it.

**perf_counter_ns everywhere.**  A monotonic counter, in integers.  Floating
point seconds lose resolution as the process ages and the system clock can
step.

**Stage boundaries are recorded, not summed.**  Each stage stores its own
start and end, so the difference between the sum of the stages and the measured
end to end is visible.  That gap is real -- it is queueing, allocation and
Python overhead between stages -- and a budget that hides it inside the stages
is a budget that will not add up on the target.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from szp.types import NS_PER_MS, Nanos

__all__ = ["Instrument", "StageStats", "percentiles", "STAGES"]

#: The canonical stage order.  ``analysis/latency_budget.py`` reads it so the
#: budget table and the code cannot drift apart.
STAGES = (
    "acquire",
    "detect_pre_post",   # letterbox in, decode and NMS out
    "inference",         # the model itself, timed inside the backend
    "fusion",
    "zones",
    "safety",
    "publish",
)


def percentiles(values_ms: Sequence[float]) -> Dict[str, float]:
    if len(values_ms) == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    array = np.asarray(values_ms, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


@dataclass
class StageStats:
    name: str
    samples_ms: List[float] = field(default_factory=list)

    def add(self, duration_ns: int) -> None:
        self.samples_ms.append(duration_ns / NS_PER_MS)

    def summary(self) -> Dict[str, float]:
        return percentiles(self.samples_ms)


class Instrument:
    """One per pipeline.  Not thread-safe; the pipeline is single-threaded."""

    def __init__(self, window: int = 4096):
        self._stages: Dict[str, StageStats] = {}
        self._frame: Dict[str, int] = {}
        self._e2e_ms: Deque[float] = deque(maxlen=window)
        self._period_ms: Deque[float] = deque(maxlen=window)
        self._frames: List[Dict[str, object]] = []
        self._last_arrival_ns: Optional[Nanos] = None
        self._counters: Dict[str, int] = defaultdict(int)

    # -- timing ------------------------------------------------------------

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            duration = time.perf_counter_ns() - start
            self._frame[name] = self._frame.get(name, 0) + duration
            self._stages.setdefault(name, StageStats(name)).add(duration)

    def record_stage(self, name: str, duration_ns: int) -> None:
        """For a stage timed elsewhere, such as inference inside a backend."""
        self._frame[name] = duration_ns
        self._stages.setdefault(name, StageStats(name)).add(duration_ns)

    def adjust_stage(self, name: str, delta_ns: int) -> None:
        """Correct a stage that was timed around a nested one.

        The detector reports its own inference time and the pipeline times the
        whole ``detect`` call around it.  Without this the two would overlap and
        the stage sum would exceed the measured end to end, which makes the
        budget table nonsense.
        """
        if name not in self._frame:
            return
        corrected = max(0, self._frame[name] + delta_ns)
        self._frame[name] = corrected
        stats = self._stages.get(name)
        if stats is not None and stats.samples_ms:
            stats.samples_ms[-1] = corrected / NS_PER_MS

    def count(self, name: str, amount: int = 1) -> None:
        self._counters[name] += amount

    # -- frames ------------------------------------------------------------

    def begin_frame(self, arrival_ns: Nanos) -> None:
        self._frame = {}
        if self._last_arrival_ns is not None:
            self._period_ms.append((arrival_ns - self._last_arrival_ns) / NS_PER_MS)
        self._last_arrival_ns = arrival_ns

    def end_frame(
        self,
        index: int,
        arrival_ns: Nanos,
        published_ns: Optional[Nanos],
        extra: Optional[Mapping[str, object]] = None,
    ) -> Dict[str, int]:
        stage_ns = dict(self._frame)
        record: Dict[str, object] = {
            "index": index,
            "stage_ms": {k: v / NS_PER_MS for k, v in stage_ns.items()},
        }
        if published_ns is not None:
            e2e = (published_ns - arrival_ns) / NS_PER_MS
            self._e2e_ms.append(e2e)
            record["e2e_ms"] = e2e
            # Only the critical-path stages count towards the budget.
            # Anything else -- the zone and health frames, which go out after
            # the e-stop frame is already on the bus -- is measured and
            # reported but is not part of the reaction time.
            accounted = sum(v for k, v in stage_ns.items() if k in STAGES) / NS_PER_MS
            record["unaccounted_ms"] = e2e - accounted
        if extra:
            record.update(dict(extra))
        self._frames.append(record)
        return stage_ns

    # -- reporting ---------------------------------------------------------

    @property
    def e2e_ms(self) -> List[float]:
        return list(self._e2e_ms)

    @property
    def frames(self) -> Sequence[Mapping[str, object]]:
        return tuple(self._frames)

    def p99_e2e_ms(self) -> float:
        return percentiles(self._e2e_ms)["p99"]

    def stage_summary(self) -> Dict[str, Dict[str, float]]:
        ordered = [name for name in STAGES if name in self._stages]
        ordered += [name for name in sorted(self._stages) if name not in STAGES]
        return {name: self._stages[name].summary() for name in ordered}

    def report(self) -> Dict[str, object]:
        stages = self.stage_summary()
        e2e = percentiles(self._e2e_ms)
        unaccounted = [
            float(f["unaccounted_ms"]) for f in self._frames if "unaccounted_ms" in f
        ]
        return {
            "frames": len(self._frames),
            "stages_ms": stages,
            "end_to_end_ms": e2e,
            "period_ms": percentiles(self._period_ms),
            "unaccounted_ms": percentiles(unaccounted),
            "stage_sum_p50_ms": float(
                sum(s["p50"] for name, s in stages.items() if name in STAGES)
            ),
            "counters": dict(self._counters),
        }

    def reset(self) -> None:
        self._stages.clear()
        self._frame.clear()
        self._e2e_ms.clear()
        self._period_ms.clear()
        self._frames.clear()
        self._last_arrival_ns = None
        self._counters.clear()
