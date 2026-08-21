"""Publisher and simulated PLC over a real bus.

These run on python-can's ``virtual`` backend everywhere and on ``socketcan``
over ``vcan0`` where one exists -- which on CI means the workflow has run
``scripts/setup_vcan.sh`` first.  The same assertions apply to both; only the
timings differ and the timings are not what these tests are about.
"""

from __future__ import annotations

import os
import platform
import time

import can
import pytest

from szp.safety.can_frames import (
    ID_ESTOP,
    PERMIT_TIMEOUT_MS,
    CanFrame,
    decode_estop,
    decode_heartbeat,
)
from szp.safety.plc_sim import SimulatedPlc
from szp.safety.publisher import CanPublisher, default_interface
from szp.types import Cause, Health, SafetyDecision, SafetyState

BACKENDS = ["virtual"]
if platform.system() == "Linux" and os.path.exists("/sys/class/net/vcan0"):
    BACKENDS.append("socketcan")


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


def channel_for(interface: str) -> str:
    return "vcan0" if interface == "socketcan" else f"szp-test-{time.perf_counter_ns()}"


def decision(state=SafetyState.CLEAR, mask=0, range_m=float("inf")):
    return SafetyDecision(
        state=state,
        cause=Cause.NONE if state is SafetyState.CLEAR else Cause.ZONE_OCCUPIED,
        zone_mask=mask,
        min_range_m=range_m,
        latched=state is SafetyState.STOP,
    )


def test_default_interface_matches_the_platform():
    interface, channel = default_interface()
    if platform.system() == "Linux":
        assert interface == "socketcan"
        assert channel == "vcan0"
    else:
        assert interface == "virtual"


def test_a_published_frame_can_be_read_back_and_decoded(backend):
    channel = channel_for(backend)
    with CanPublisher(channel=channel, interface=backend) as publisher:
        listener = can.Bus(interface=backend, channel=channel, receive_own_messages=False)
        try:
            result = publisher.publish_estop(decision(SafetyState.STOP, mask=0b11, range_m=1.5))
            assert result.t_bus_ns >= result.t_send_return_ns
            message = listener.recv(timeout=1.0)
            assert message is not None
            assert message.arbitration_id == ID_ESTOP
            decoded = decode_estop(CanFrame(message.arbitration_id, bytes(message.data)))
            assert decoded["state"] is SafetyState.STOP
            assert decoded["zone_mask"] == 0b11
            assert decoded["min_range_m"] == pytest.approx(1.5, abs=0.01)
        finally:
            listener.shutdown()


def test_the_alive_counter_advances_on_every_frame(backend):
    channel = channel_for(backend)
    with CanPublisher(channel=channel, interface=backend) as publisher:
        alives = [publisher.publish_estop(decision()).alive for _ in range(5)]
    assert alives == [(alives[0] + i) % 256 for i in range(5)]


def test_the_alive_counter_wraps_without_repeating_consecutively(backend):
    channel = channel_for(backend)
    with CanPublisher(channel=channel, interface=backend) as publisher:
        alives = [publisher.publish_estop(decision()).alive for _ in range(300)]
    assert all(a != b for a, b in zip(alives, alives[1:]))


def test_the_publisher_closes_its_bus():
    publisher = CanPublisher(channel=channel_for("virtual"), interface="virtual")
    publisher.publish_estop(decision())
    publisher.close()
    assert publisher.bus is None
    publisher.close()   # idempotent


@pytest.mark.slow
def test_the_plc_grants_and_withdraws_the_permit(backend):
    channel = channel_for(backend)
    with CanPublisher(channel=channel, interface=backend) as publisher:
        with SimulatedPlc(channel=channel, interface=backend) as plc:
            for _ in range(5):
                publisher.publish_estop(decision(SafetyState.CLEAR))
                time.sleep(0.005)
            deadline = time.perf_counter() + 2.0
            while not plc.permitted and time.perf_counter() < deadline:
                publisher.publish_estop(decision(SafetyState.CLEAR))
                time.sleep(0.005)
            assert plc.permitted, "a healthy CLEAR stream must grant the permit"

            publisher.publish_estop(decision(SafetyState.STOP, mask=0b1))
            deadline = time.perf_counter() + 2.0
            while plc.permitted and time.perf_counter() < deadline:
                time.sleep(0.002)
            assert not plc.permitted, "a STOP frame must withdraw it"
    assert plc.stats.crc_failures == 0


@pytest.mark.slow
def test_the_permit_lapses_when_the_pipeline_goes_quiet(backend):
    """The whole fail-safe argument in one test: silence stops the machine."""
    channel = channel_for(backend)
    with CanPublisher(channel=channel, interface=backend) as publisher:
        with SimulatedPlc(channel=channel, interface=backend) as plc:
            deadline = time.perf_counter() + 2.0
            while not plc.permitted and time.perf_counter() < deadline:
                publisher.publish_estop(decision(SafetyState.CLEAR))
                time.sleep(0.005)
            assert plc.permitted

            # Now say nothing at all.
            quiet_started = time.perf_counter()
            deadline = quiet_started + 2.0
            while plc.permitted and time.perf_counter() < deadline:
                time.sleep(0.005)
            assert not plc.permitted
            elapsed_ms = (time.perf_counter() - quiet_started) * 1000.0
            assert elapsed_ms >= PERMIT_TIMEOUT_MS * 0.8
            assert plc.stats.timeouts >= 1


@pytest.mark.slow
def test_the_plc_sends_heartbeats_and_relays_a_reset_request(backend):
    channel = channel_for(backend)
    with SimulatedPlc(channel=channel, interface=backend, heartbeat_period_s=0.02) as plc:
        listener = can.Bus(interface=backend, channel=channel, receive_own_messages=False)
        try:
            plc.request_reset()
            deadline = time.perf_counter() + 2.0
            seen_reset = False
            while time.perf_counter() < deadline and not seen_reset:
                message = listener.recv(timeout=0.2)
                if message is None:
                    continue
                decoded = decode_heartbeat(
                    CanFrame(message.arbitration_id, bytes(message.data))
                )
                if decoded and decoded["reset_requested"]:
                    seen_reset = True
            assert seen_reset
        finally:
            listener.shutdown()
    assert plc.stats.heartbeats_sent > 0


def test_the_plc_thread_stops_itself():
    plc = SimulatedPlc(channel=channel_for("virtual"), interface="virtual")
    plc.start()
    assert plc._thread is not None and plc._thread.is_alive()
    plc.close()
    assert plc._thread is None
    plc.close()   # idempotent


def test_the_publisher_hands_back_frames_from_other_senders(backend):
    channel = channel_for(backend)
    with CanPublisher(channel=channel, interface=backend) as publisher:
        other = can.Bus(interface=backend, channel=channel, receive_own_messages=False)
        try:
            other.send(can.Message(arbitration_id=0x123, data=bytes(8), is_extended_id=False))
            deadline = time.perf_counter() + 1.0
            received = []
            while time.perf_counter() < deadline and not received:
                received = [m for m in publisher.poll(timeout=0.05) if m.arbitration_id == 0x123]
            assert received, "a frame from another node must reach poll()"
        finally:
            other.shutdown()


def test_publisher_health_frame_carries_the_latency(backend):
    channel = channel_for(backend)
    with CanPublisher(channel=channel, interface=backend) as publisher:
        listener = can.Bus(interface=backend, channel=channel, receive_own_messages=False)
        try:
            publisher.publish_health(Health(sources_ok={"rgb": True}), 66.0, 12.3)
            message = listener.recv(timeout=1.0)
            assert message is not None
            from szp.safety.can_frames import decode_health

            decoded = decode_health(CanFrame(message.arbitration_id, bytes(message.data)))
            assert decoded["p99_latency_ms"] == pytest.approx(12.3, abs=0.1)
        finally:
            listener.shutdown()
