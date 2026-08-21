"""The wire format and the receiving rule that makes it fail safe."""

from __future__ import annotations

import itertools

import pytest

from szp.safety.can_frames import (
    CanFrame,
    DATA_ID,
    HEARTBEAT_RESET_REQUEST,
    ID_ESTOP,
    ID_HEALTH,
    ID_ZONES,
    PERMIT_TIMEOUT_MS,
    RANGE_UNKNOWN_CM,
    crc8_autosar,
    frame_crc,
    decode_estop,
    decode_health,
    decode_heartbeat,
    decode_zones,
    encode_estop,
    encode_health,
    encode_heartbeat,
    encode_zones,
    permit_granted,
)
from szp.types import Cause, Health, SafetyDecision, SafetyState


def decision(state=SafetyState.STOP, cause=Cause.ZONE_OCCUPIED, mask=0b101, range_m=1.23):
    return SafetyDecision(
        state=state, cause=cause, zone_mask=mask, min_range_m=range_m, latched=True
    )


def test_every_frame_is_eight_bytes():
    frames = [
        encode_estop(decision(), 7),
        encode_zones(0b1, 0b10, 0.4, 8),
        encode_health(Health(sources_ok={"rgb": True}), 66.0, 12.5, 9),
        encode_heartbeat(HEARTBEAT_RESET_REQUEST, 10),
    ]
    assert all(len(f.data) == 8 for f in frames)
    with pytest.raises(ValueError, match="8 bytes"):
        CanFrame(ID_ESTOP, b"\x00" * 7)


def test_estop_round_trip():
    for state in SafetyState:
        frame = encode_estop(decision(state=state), 42)
        decoded = decode_estop(frame)
        assert decoded is not None
        assert decoded["state"] is state
        assert decoded["zone_mask"] == 0b101
        assert decoded["min_range_m"] == pytest.approx(1.23, abs=0.01)
        assert decoded["alive"] == 42


def test_an_unknown_range_survives_the_round_trip_as_infinity():
    frame = encode_estop(decision(range_m=float("inf")), 1)
    assert decode_estop(frame)["min_range_m"] == float("inf")
    assert frame.data[4] | (frame.data[5] << 8) == RANGE_UNKNOWN_CM


def test_a_huge_range_is_clamped_below_the_sentinel():
    """A bad fusion result must not alias onto 'nobody there'."""
    frame = encode_estop(decision(range_m=1e6), 1)
    raw = frame.data[4] | (frame.data[5] << 8)
    assert raw != RANGE_UNKNOWN_CM
    assert decode_estop(frame)["min_range_m"] != float("inf")


def test_every_single_bit_flip_in_the_payload_is_caught():
    frame = encode_estop(decision(), 42)
    for byte_index, bit in itertools.product(range(8), range(8)):
        corrupted = bytearray(frame.data)
        corrupted[byte_index] ^= 1 << bit
        broken = CanFrame(frame.arbitration_id, bytes(corrupted))
        assert decode_estop(broken) is None, f"byte {byte_index} bit {bit} slipped through"


def test_the_data_id_stops_a_frame_being_decoded_as_the_wrong_message():
    zones = encode_zones(0b1, 0b10, 0.4, 5)
    assert decode_estop(CanFrame(ID_ESTOP, zones.data)) is None
    assert decode_zones(CanFrame(ID_ZONES, encode_estop(decision(), 5).data)) is None
    assert len(set(DATA_ID.values())) == len(DATA_ID)


def test_a_frame_on_the_wrong_arbitration_id_is_rejected():
    frame = encode_estop(decision(), 1)
    assert decode_estop(CanFrame(ID_HEALTH, frame.data)) is None


def test_crc_is_the_documented_autosar_variant():
    # The published check value for CRC-8/AUTOSAR.  Pinned so a PLC-side
    # implementation has a known-good vector rather than this file's word.
    assert crc8_autosar(b"123456789") == 0xDF
    assert frame_crc(bytes(7), 0x11) != frame_crc(bytes(7), 0x22)


# -- the receive rule --------------------------------------------------------


def test_permit_is_granted_only_for_a_fresh_valid_moving_state():
    decoded = decode_estop(encode_estop(decision(state=SafetyState.CLEAR), 5))
    granted, reason = permit_granted(decoded, age_ms=10.0, previous_alive=4)
    assert granted, reason


@pytest.mark.parametrize(
    "state", [SafetyState.STOP, SafetyState.FAULT, SafetyState.INIT, SafetyState.RECOVERING]
)
def test_permit_is_refused_for_every_non_moving_state(state):
    decoded = decode_estop(encode_estop(decision(state=state), 5))
    assert not permit_granted(decoded, age_ms=1.0, previous_alive=4)[0]


def test_no_frame_means_no_permit():
    assert not permit_granted(None, age_ms=0.0, previous_alive=None)[0]


def test_a_stale_frame_means_no_permit():
    decoded = decode_estop(encode_estop(decision(state=SafetyState.CLEAR), 5))
    assert permit_granted(decoded, PERMIT_TIMEOUT_MS - 1, 4)[0]
    assert not permit_granted(decoded, PERMIT_TIMEOUT_MS + 1, 4)[0]


def test_a_frozen_alive_counter_means_no_permit():
    """A crashed sender whose last frame is repeating out of a buffer looks
    exactly like a healthy one without this check."""
    decoded = decode_estop(encode_estop(decision(state=SafetyState.CLEAR), 5))
    assert not permit_granted(decoded, 1.0, previous_alive=5)[0]


def test_there_is_no_path_that_grants_a_permit_by_default():
    """Exhaustive over the decision inputs that matter."""
    for state in SafetyState:
        for age in (0.0, PERMIT_TIMEOUT_MS + 1):
            for previous in (None, 5, 6):
                decoded = decode_estop(encode_estop(decision(state=state), 5))
                granted, _ = permit_granted(decoded, age, previous)
                expected = (
                    state.motion_permitted
                    and age <= PERMIT_TIMEOUT_MS
                    and previous != 5
                )
                assert granted is expected, (state, age, previous)


# -- the other messages ------------------------------------------------------


def test_zones_frame_round_trip_including_a_negative_gap():
    frame = encode_zones(0b1011, 0b0100, -0.42, 3)
    decoded = decode_zones(frame)
    assert decoded["stop_mask"] == 0b1011
    assert decoded["warn_mask"] == 0b0100
    assert decoded["nearest_gap_m"] == pytest.approx(-0.42, abs=0.01)


def test_health_frame_reports_stale_sources_as_not_ok():
    health = Health(
        sources_ok={"rgb": True, "depth": True, "lidar": True},
        stale=("depth",),
        heartbeat_ok=False,
    )
    decoded = decode_health(encode_health(health, 66.7, 12.34, 4))
    assert decoded["sensors_ok"]["rgb"]
    assert not decoded["sensors_ok"]["depth"]
    assert not decoded["sensors_ok"]["heartbeat"]
    assert decoded["fault_code"] & Cause.SENSOR_STALE
    assert decoded["fault_code"] & Cause.HEARTBEAT_LOST
    assert decoded["period_ms"] == pytest.approx(66.7, abs=0.1)
    assert decoded["p99_latency_ms"] == pytest.approx(12.3, abs=0.1)


def test_heartbeat_round_trip():
    decoded = decode_heartbeat(encode_heartbeat(HEARTBEAT_RESET_REQUEST, 77))
    assert decoded["reset_requested"]
    assert not decoded["machine_moving"]
    assert decoded["alive"] == 77
