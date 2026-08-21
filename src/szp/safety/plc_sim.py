"""A simulated PLC on the other end of the bus.

It exists for two reasons.  The first is that a publisher with nothing
listening proves very little: the end-to-end latency that matters is frame
arrival to *the receiver acting on the stop* and only a receiver can timestamp
that.  The second is that writing the receive rule out in code is the clearest
way to specify it -- ``permit_granted`` in ``can_frames`` is the rule and this
class is a machine that obeys it.

The PLC runs in its own thread with its own bus handle, so the timings include
the transport.  It stops itself on ``close()`` and on leaving the context
manager and the thread is a daemon so an abandoned instance cannot keep the
process alive.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import can

from szp.safety.can_frames import (
    HEARTBEAT_MACHINE_MOVING,
    HEARTBEAT_RESET_REQUEST,
    ID_ESTOP,
    PERMIT_TIMEOUT_MS,
    CanFrame,
    decode_estop,
    encode_heartbeat,
    permit_granted,
)
from szp.safety.publisher import default_interface
from szp.types import Nanos, SafetyState

__all__ = ["SimulatedPlc", "PermitEvent"]


@dataclass(frozen=True)
class PermitEvent:
    """One change in what the PLC would let the machine do."""

    t_ns: Nanos
    permitted: bool
    reason: str
    state: Optional[SafetyState]
    zone_mask: int
    alive: int


@dataclass
class PlcStats:
    frames_seen: int = 0
    crc_failures: int = 0
    frozen_counter_events: int = 0
    timeouts: int = 0
    heartbeats_sent: int = 0
    events: List[PermitEvent] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "frames_seen": self.frames_seen,
            "crc_failures": self.crc_failures,
            "frozen_counter_events": self.frozen_counter_events,
            "timeouts": self.timeouts,
            "heartbeats_sent": self.heartbeats_sent,
            "permit_changes": len(self.events),
        }


class SimulatedPlc:
    """Consumes the permit and emits heartbeats.  Thread-safe to read."""

    def __init__(
        self,
        channel: Optional[str] = None,
        interface: Optional[str] = None,
        *,
        heartbeat_period_s: float = 0.1,
        poll_timeout_s: float = 0.005,
        bus: Optional[can.BusABC] = None,
    ):
        if bus is not None:
            self.bus = bus
            self._owns_bus = False
            self.interface, self.channel = "provided", getattr(bus, "channel_info", "provided")
        else:
            resolved_interface, resolved_channel = default_interface(channel)
            self.interface = interface or resolved_interface
            self.channel = resolved_channel
            kwargs: Dict[str, object] = {
                "interface": self.interface,
                "channel": self.channel,
                "receive_own_messages": False,
            }
            if self.interface != "virtual":
                kwargs["bitrate"] = 500_000
            self.bus = can.Bus(**kwargs)  # type: ignore[arg-type]
            self._owns_bus = True

        self.heartbeat_period_s = heartbeat_period_s
        self.poll_timeout_s = poll_timeout_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._permitted = False
        self._previous_alive: Optional[int] = None
        self._last_frame_perf: Optional[float] = None
        self._heartbeat_alive = 0
        self._reset_requested = False
        self.stats = PlcStats()

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "SimulatedPlc":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="szp-plc-sim", daemon=True)
        self._thread.start()

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._owns_bus and self.bus is not None:
            try:
                self.bus.shutdown()
            finally:
                self.bus = None  # type: ignore[assignment]

    # -- operator ----------------------------------------------------------

    def request_reset(self) -> None:
        """Latch a reset request into the next heartbeat."""
        with self._lock:
            self._reset_requested = True

    @property
    def permitted(self) -> bool:
        with self._lock:
            return self._permitted

    @property
    def events(self) -> Tuple[PermitEvent, ...]:
        with self._lock:
            return tuple(self.stats.events)

    def first_denial_after(self, t_ns: Nanos) -> Optional[PermitEvent]:
        """The first loss of permit at or after ``t_ns``.

        This is the receiver-side half of the end-to-end latency measurement:
        subtract the frame arrival time from ``event.t_ns``.
        """
        for event in self.events:
            if not event.permitted and event.t_ns >= t_ns:
                return event
        return None

    # -- thread ------------------------------------------------------------

    def _run(self) -> None:
        next_heartbeat = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            if now >= next_heartbeat:
                self._send_heartbeat()
                next_heartbeat = now + self.heartbeat_period_s
            timeout = min(self.poll_timeout_s, max(0.0, next_heartbeat - now))
            message = self.bus.recv(timeout=timeout)
            if message is None:
                self._check_timeout()
                continue
            if message.arbitration_id != ID_ESTOP:
                continue
            self._consume(message)

    def _send_heartbeat(self) -> None:
        with self._lock:
            flags = HEARTBEAT_MACHINE_MOVING if self._permitted else 0
            if self._reset_requested:
                flags |= HEARTBEAT_RESET_REQUEST
                self._reset_requested = False
            self._heartbeat_alive = (self._heartbeat_alive + 1) & 0xFF
            alive = self._heartbeat_alive
        frame = encode_heartbeat(flags, alive)
        try:
            self.bus.send(
                can.Message(arbitration_id=frame.arbitration_id, data=frame.data,
                            is_extended_id=False)
            )
        except can.CanError:
            return
        with self._lock:
            self.stats.heartbeats_sent += 1

    def _check_timeout(self) -> None:
        """Drop the permit when nothing has arrived inside the window.

        The age check comes first.  Calling ``permit_granted(None, ...)`` on
        every empty ``recv`` would deny on the first quiet poll, which is not a
        timeout, it is the receiver being faster than the sender.
        """
        with self._lock:
            if not self._permitted or self._last_frame_perf is None:
                return
            age_ms = (time.perf_counter() - self._last_frame_perf) * 1000.0
        if age_ms <= PERMIT_TIMEOUT_MS:
            return
        with self._lock:
            self.stats.timeouts += 1
        self._record(False, f"permit lapsed: stale by {age_ms:.1f} ms", None, 0, 0)

    def _consume(self, message: can.Message) -> None:
        frame = CanFrame(
            arbitration_id=message.arbitration_id,
            data=bytes(message.data),
            is_extended_id=bool(message.is_extended_id),
        )
        decoded = decode_estop(frame)
        with self._lock:
            self.stats.frames_seen += 1
            if decoded is None:
                self.stats.crc_failures += 1
            previous_alive = self._previous_alive
            self._last_frame_perf = time.perf_counter()
        age_ms = 0.0
        permitted, reason = permit_granted(decoded, age_ms, previous_alive)
        if decoded is not None:
            if previous_alive is not None and int(decoded["alive"]) == previous_alive:
                with self._lock:
                    self.stats.frozen_counter_events += 1
            with self._lock:
                self._previous_alive = int(decoded["alive"])  # type: ignore[arg-type]
        if permitted != self.permitted:
            self._record(
                permitted,
                reason,
                decoded["state"] if decoded else None,  # type: ignore[index]
                int(decoded["zone_mask"]) if decoded else 0,  # type: ignore[arg-type]
                int(decoded["alive"]) if decoded else 0,  # type: ignore[arg-type]
            )

    def _record(
        self,
        permitted: bool,
        reason: str,
        state: Optional[SafetyState],
        zone_mask: int,
        alive: int,
    ) -> None:
        event = PermitEvent(
            t_ns=time.perf_counter_ns(),
            permitted=permitted,
            reason=reason,
            state=state,
            zone_mask=zone_mask,
            alive=alive,
        )
        with self._lock:
            self._permitted = permitted
            self.stats.events.append(event)
