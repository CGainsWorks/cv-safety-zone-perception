"""Putting the frames on a bus.

Two backends, chosen automatically:

``socketcan``
    Linux, on a ``vcan`` interface created by ``scripts/setup_vcan.sh``.  The
    frame goes through the kernel's CAN stack, which is the same path a real
    interface takes minus the transceiver, so the timings measured over vcan
    include the syscall and the queueing and exclude only the wire.

``virtual``
    python-can's in-process bus.  Used on Windows and macOS, where vcan does
    not exist.  It is a queue between two objects in one address space and it
    is *not* representative of transport latency; anything measured on it is
    labelled as such in the results.

Transmit timestamping
---------------------
``bus.send()`` returning tells you the kernel accepted the frame, not that the
frame is on the bus.  Where the backend supports it this class turns on
``receive_own_messages`` and waits for the loopback echo and takes
``perf_counter_ns`` at the moment the echo arrives.  That is the number the
end-to-end figure uses: it covers the send syscall, the CAN stack and the
loopback receive.

python-can's own ``Message.timestamp`` is deliberately *not* used for this.  It
is wall-clock seconds from ``time.time()`` or from the kernel's realtime clock,
and converting it into the ``perf_counter`` base the rest of the pipeline works
in introduces an offset that drifts over a run.  On the first version of this
file that drift showed up as four milliseconds of "unaccounted" latency that
was not real.  Two clocks, one of which can step, are not worth a slightly
earlier timestamp.
"""

from __future__ import annotations

import platform
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import can

from szp.safety.can_frames import (
    CanFrame,
    ID_ESTOP,
    ID_HEALTH,
    ID_HEARTBEAT,
    ID_ZONES,
    encode_estop,
    encode_health,
    encode_zones,
)
from szp.types import Health, Nanos, SafetyDecision

__all__ = ["CanPublisher", "PublishResult", "default_interface", "TX_IDS"]

TX_IDS = (ID_ESTOP, ID_ZONES, ID_HEALTH)


def default_interface(channel: Optional[str] = None) -> Tuple[str, str]:
    """(interface, channel) for this machine.

    Overridable with ``SZP_CAN_INTERFACE`` and ``SZP_CAN_CHANNEL`` so the same
    code runs against a real ``can0`` without an edit.
    """
    import os

    interface = os.environ.get("SZP_CAN_INTERFACE")
    channel = channel or os.environ.get("SZP_CAN_CHANNEL")
    if interface:
        return interface, channel or "vcan0"
    if platform.system() == "Linux":
        return "socketcan", channel or "vcan0"
    return "virtual", channel or "szp"


@dataclass(frozen=True)
class PublishResult:
    """When one frame reached the bus."""

    t_send_return_ns: Nanos
    #: Monotonic nanoseconds at the moment the loopback echo was seen or the
    #: send-return time when the backend does not echo.
    t_bus_ns: Nanos
    echoed: bool
    alive: int


class CanPublisher:
    """Owns the bus.  One per pipeline; use as a context manager."""

    def __init__(
        self,
        channel: Optional[str] = None,
        interface: Optional[str] = None,
        *,
        receive_own_messages: bool = True,
        echo_timeout_s: float = 0.01,
        bus: Optional[can.BusABC] = None,
    ):
        if bus is not None:
            self.interface, self.channel = "provided", getattr(bus, "channel_info", "provided")
            self.bus = bus
            self._owns_bus = False
        else:
            resolved_interface, resolved_channel = default_interface(channel)
            self.interface = interface or resolved_interface
            self.channel = resolved_channel
            kwargs: Dict[str, object] = {
                "interface": self.interface,
                "channel": self.channel,
                "receive_own_messages": receive_own_messages,
            }
            if self.interface == "virtual":
                # The virtual bus needs no bitrate and rejects one.
                pass
            else:
                kwargs["bitrate"] = 500_000
            self.bus = can.Bus(**kwargs)  # type: ignore[arg-type]
            self._owns_bus = True
        self.receive_own_messages = receive_own_messages
        self.echo_timeout_s = echo_timeout_s
        self._alive = 0
        self._inbox: Deque[can.Message] = deque(maxlen=256)
        self._sent = 0
        self._echo_misses = 0

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "CanPublisher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_bus and self.bus is not None:
            try:
                self.bus.shutdown()
            finally:
                self.bus = None  # type: ignore[assignment]

    # -- sending -----------------------------------------------------------

    def _next_alive(self) -> int:
        self._alive = (self._alive + 1) & 0xFF
        return self._alive

    def _send(self, frame: CanFrame, alive: int) -> PublishResult:
        message = can.Message(
            arbitration_id=frame.arbitration_id,
            data=frame.data,
            is_extended_id=frame.is_extended_id,
        )
        self.bus.send(message)
        t_return = time.perf_counter_ns()
        self._sent += 1
        t_bus, echoed = t_return, False
        if self.receive_own_messages:
            echo = self._await_echo(frame)
            if echo is not None:
                echoed = True
                t_bus = time.perf_counter_ns()
            else:
                self._echo_misses += 1
        return PublishResult(
            t_send_return_ns=t_return, t_bus_ns=t_bus, echoed=echoed, alive=alive
        )

    def _await_echo(self, frame: CanFrame) -> Optional[can.Message]:
        deadline = time.perf_counter() + self.echo_timeout_s
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return None
            message = self.bus.recv(timeout=remaining)
            if message is None:
                return None
            if message.arbitration_id == frame.arbitration_id and bytes(message.data) == frame.data:
                return message
            self._inbox.append(message)

    def publish_estop(self, decision: SafetyDecision) -> PublishResult:
        alive = self._next_alive()
        return self._send(encode_estop(decision, alive), alive)

    def publish_zones(self, stop_mask: int, warn_mask: int, nearest_gap_m: float) -> PublishResult:
        alive = self._next_alive()
        return self._send(encode_zones(stop_mask, warn_mask, nearest_gap_m, alive), alive)

    def publish_health(
        self, health: Health, period_ms: float, p99_latency_ms: float
    ) -> PublishResult:
        alive = self._next_alive()
        return self._send(encode_health(health, period_ms, p99_latency_ms, alive), alive)

    # -- receiving ---------------------------------------------------------

    def poll(self, timeout: float = 0.0) -> List[can.Message]:
        """Every frame from somebody else that has arrived since the last call."""
        out: List[can.Message] = []
        while self._inbox:
            out.append(self._inbox.popleft())
        deadline = time.perf_counter() + max(0.0, timeout)
        while True:
            remaining = max(0.0, deadline - time.perf_counter())
            message = self.bus.recv(timeout=remaining)
            if message is None:
                break
            if message.arbitration_id in TX_IDS:
                continue  # our own echo, already accounted for
            out.append(message)
            if remaining <= 0:
                break
        return [m for m in out if m.arbitration_id not in TX_IDS]

    def poll_heartbeat(self, timeout: float = 0.0) -> Optional[can.Message]:
        latest = None
        for message in self.poll(timeout):
            if message.arbitration_id == ID_HEARTBEAT:
                latest = message
        return latest

    # -- diagnostics -------------------------------------------------------

    @property
    def stats(self) -> Dict[str, object]:
        return {
            "interface": self.interface,
            "channel": str(self.channel),
            "frames_sent": self._sent,
            "echo_misses": self._echo_misses,
            "receive_own_messages": self.receive_own_messages,
        }
