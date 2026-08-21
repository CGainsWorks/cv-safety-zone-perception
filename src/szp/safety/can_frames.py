"""The CAN wire format and nothing else.

This module has no dependency on python-can.  It turns pipeline state into
eight bytes and back, so the encoding can be tested exhaustively without a bus
and so a PLC-side implementation has one file to read.

Protection
----------
Every frame the pipeline sends carries a rolling ``ALIVE`` counter and a CRC-8
over the payload with a per-message data identifier mixed in.  This is the
scheme AUTOSAR E2E Profile 1 uses and it exists because CAN's own 15-bit CRC
protects the wire and nothing else.  It does not tell the receiver that the
sender is still computing new values.  The three failures worth naming:

* a corrupted payload that survives the CAN CRC       -> caught by the CRC-8
* a sender that has crashed with its last frame       -> caught by ALIVE not
  latched in a hardware buffer and repeating             advancing
* the wrong message decoded as this one after an       -> caught by the data
  arbitration-ID mix-up or a mis-generated database       ID in the CRC

The fail-safe direction
-----------------------
``SZP_ESTOP`` is a **permit**, not a command.  The PLC is specified to allow
motion only while all of the following hold:

* a ``SZP_ESTOP`` frame arrived within the last ``PERMIT_TIMEOUT_MS``
* its CRC checks
* its ALIVE counter differs from the previous frame's
* its ``state`` is CLEAR or WARN

A pipeline that dies, hangs, loses power or gets unplugged from the bus sends
nothing, the permit lapses and the machine stops.  There is no message the
pipeline can fail to send that leaves the machine running.  Getting this the
other way round -- a "stop now" command that has to arrive to be obeyed -- is
the classic way to build a safety function that fails silently and it is
entry FM-11 in the FMEA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from szp.types import Cause, Health, SafetyDecision, SafetyState

__all__ = [
    "PROTOCOL_VERSION",
    "CanFrame",
    "crc8_autosar",
    "frame_crc",
    "encode_estop",
    "decode_estop",
    "encode_zones",
    "decode_zones",
    "encode_health",
    "decode_health",
    "encode_heartbeat",
    "decode_heartbeat",
    "ID_ESTOP",
    "ID_ZONES",
    "ID_HEALTH",
    "ID_HEARTBEAT",
    "PERMIT_TIMEOUT_MS",
    "CYCLE_MS",
    "permit_granted",
]

PROTOCOL_VERSION = 1

# Standard 11-bit identifiers.  Low numbers win arbitration and the e-stop
# permit is the most time-critical frame on this bus, so it takes the lowest.
ID_ESTOP = 0x080
ID_ZONES = 0x081
ID_HEALTH = 0x082
ID_HEARTBEAT = 0x700   # PLC -> pipeline

# Data identifiers mixed into each CRC.  Distinct per message so a frame
# decoded under the wrong identifier fails its CRC instead of producing a
# plausible-looking state.
DATA_ID: Dict[int, int] = {
    ID_ESTOP: 0x11,
    ID_ZONES: 0x22,
    ID_HEALTH: 0x33,
    ID_HEARTBEAT: 0x44,
}

#: The PLC drops the permit if no valid frame arrives inside this window.
#: Five times the 20 ms nominal cycle, so four consecutive losses are tolerated
#: and the fifth stops the machine.
PERMIT_TIMEOUT_MS = 100

#: Nominal transmit periods, milliseconds.  The e-stop frame speeds up while
#: the state is not CLEAR so that the PLC sees a change sooner and so that the
#: permit lapses sooner if the pipeline dies mid-stop.
CYCLE_MS = {
    "estop_clear": 20,
    "estop_active": 10,
    "zones": 50,
    "health": 100,
}

#: Sentinel for "no person in view" in the 16-bit centimetre range field.
RANGE_UNKNOWN_CM = 0xFFFF
#: Largest real range the field can carry, 655.34 m, far beyond any working
#: range; the clamp exists so a bad fusion result cannot alias onto the
#: sentinel and read as "nobody there".
RANGE_MAX_CM = 0xFFFE

SENSOR_BITS = {"rgb": 0, "depth": 1, "lidar": 2, "detector": 3, "heartbeat": 4}


@dataclass(frozen=True)
class CanFrame:
    """One frame, independent of any CAN library."""

    arbitration_id: int
    data: bytes
    is_extended_id: bool = False

    def __post_init__(self) -> None:
        if len(self.data) != 8:
            raise ValueError(f"every frame in this protocol is 8 bytes, got {len(self.data)}")


def crc8_autosar(data: Sequence[int]) -> int:
    """CRC-8/AUTOSAR: polynomial 0x2F, init 0xFF, no reflection, final xor 0xFF.

    The check value for the ASCII string "123456789" is 0xDF, which is what
    ``tests/test_can_protocol.py`` pins so that a PLC-side implementation has a
    known-good vector rather than having to trust this one.
    """
    crc = 0xFF
    for byte in data:
        crc ^= byte & 0xFF
        for _ in range(8):
            crc = ((crc << 1) ^ 0x2F) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc ^ 0xFF


def frame_crc(payload: Sequence[int], data_id: int) -> int:
    """The CRC that goes in byte 7.

    The message's data identifier is appended to the payload before the CRC is
    taken, which is what makes the checksum message-specific: the same seven
    bytes under a different identifier produce a different byte 7, so a frame
    decoded as the wrong message fails its CRC instead of yielding a
    plausible-looking state.
    """
    return crc8_autosar(list(payload) + [data_id & 0xFF])


def _finish(arbitration_id: int, payload: bytearray, alive: int) -> CanFrame:
    payload[6] = alive & 0xFF
    payload[7] = frame_crc(payload[:7], DATA_ID[arbitration_id])
    return CanFrame(arbitration_id=arbitration_id, data=bytes(payload))


def _check(frame: CanFrame, expected_id: int) -> bool:
    if frame.arbitration_id != expected_id:
        return False
    return frame.data[7] == frame_crc(frame.data[:7], DATA_ID[expected_id])


def _range_to_cm(range_m: float) -> int:
    if range_m is None or range_m != range_m or range_m == float("inf"):
        return RANGE_UNKNOWN_CM
    return int(min(max(0.0, range_m) * 100.0, RANGE_MAX_CM))


def _cm_to_range(value: int) -> float:
    return float("inf") if value == RANGE_UNKNOWN_CM else value / 100.0


# -- SZP_ESTOP ---------------------------------------------------------------


def encode_estop(decision: SafetyDecision, alive: int) -> CanFrame:
    """
    byte 0     state          uint8, SafetyState
    byte 1     cause          uint8, Cause bitfield
    bytes 2-3  zone_mask      uint16 little endian, bit i = zone i occupied
    bytes 4-5  min_range_cm   uint16 little endian, 0xFFFF = nobody in view
    byte 6     alive          uint8, rolling
    byte 7     crc8           over bytes 0..6 plus the data id
    """
    payload = bytearray(8)
    payload[0] = int(decision.state) & 0xFF
    payload[1] = int(decision.cause) & 0xFF
    mask = decision.zone_mask & 0xFFFF
    payload[2] = mask & 0xFF
    payload[3] = (mask >> 8) & 0xFF
    cm = _range_to_cm(decision.min_range_m)
    payload[4] = cm & 0xFF
    payload[5] = (cm >> 8) & 0xFF
    return _finish(ID_ESTOP, payload, alive)


def decode_estop(frame: CanFrame) -> Optional[Dict[str, object]]:
    """Decode or ``None`` when the identifier or the CRC is wrong."""
    if not _check(frame, ID_ESTOP):
        return None
    data = frame.data
    try:
        state = SafetyState(data[0])
    except ValueError:
        return None
    return {
        "state": state,
        "cause": Cause(data[1]),
        "zone_mask": data[2] | (data[3] << 8),
        "min_range_m": _cm_to_range(data[4] | (data[5] << 8)),
        "alive": data[6],
    }


def permit_granted(
    decoded: Optional[Dict[str, object]],
    age_ms: float,
    previous_alive: Optional[int],
) -> Tuple[bool, str]:
    """The receiving PLC's decision rule, written out so it can be tested.

    Returns ``(permitted, reason)``.  Every path that is not an explicit grant
    returns ``False``; there is no default-permit branch.
    """
    if decoded is None:
        return False, "no valid frame"
    if age_ms > PERMIT_TIMEOUT_MS:
        return False, f"stale by {age_ms:.1f} ms"
    alive = int(decoded["alive"])  # type: ignore[arg-type]
    if previous_alive is not None and alive == previous_alive:
        return False, "alive counter frozen"
    state = decoded["state"]
    if not isinstance(state, SafetyState) or not state.motion_permitted:
        return False, f"state {state}"
    return True, "permitted"


# -- SZP_ZONES ---------------------------------------------------------------


def encode_zones(stop_mask: int, warn_mask: int, nearest_gap_m: float, alive: int) -> CanFrame:
    """
    bytes 0-1  stop_mask      uint16 LE
    bytes 2-3  warn_mask      uint16 LE
    bytes 4-5  nearest_gap_cm int16 LE, negative when somebody is inside a zone
    byte 6     alive
    byte 7     crc8
    """
    payload = bytearray(8)
    payload[0] = stop_mask & 0xFF
    payload[1] = (stop_mask >> 8) & 0xFF
    payload[2] = warn_mask & 0xFF
    payload[3] = (warn_mask >> 8) & 0xFF
    if nearest_gap_m != nearest_gap_m or nearest_gap_m == float("inf"):
        gap = 0x7FFF
    else:
        gap = int(max(-32767.0, min(32766.0, nearest_gap_m * 100.0)))
    payload[4] = gap & 0xFF
    payload[5] = (gap >> 8) & 0xFF
    return _finish(ID_ZONES, payload, alive)


def decode_zones(frame: CanFrame) -> Optional[Dict[str, object]]:
    if not _check(frame, ID_ZONES):
        return None
    data = frame.data
    raw = data[4] | (data[5] << 8)
    gap = raw - 0x10000 if raw & 0x8000 else raw
    return {
        "stop_mask": data[0] | (data[1] << 8),
        "warn_mask": data[2] | (data[3] << 8),
        "nearest_gap_m": float("inf") if raw == 0x7FFF else gap / 100.0,
        "alive": data[6],
    }


# -- SZP_HEALTH --------------------------------------------------------------


def encode_health(health: Health, period_ms: float, p99_latency_ms: float, alive: int) -> CanFrame:
    """
    byte 0     sensors_ok     uint8 bitfield, see SENSOR_BITS
    byte 1     fault_code     uint8, Cause bits that are currently set
    bytes 2-3  period         uint16 LE, tenths of a millisecond
    bytes 4-5  p99_latency    uint16 LE, tenths of a millisecond
    byte 6     alive
    byte 7     crc8
    """
    payload = bytearray(8)
    bits = 0
    for name, bit in SENSOR_BITS.items():
        if name == "heartbeat":
            ok = health.heartbeat_ok
        else:
            ok = bool(health.sources_ok.get(name, False)) and name not in health.stale
        if ok:
            bits |= 1 << bit
    payload[0] = bits
    fault = 0
    if health.stale:
        fault |= int(Cause.SENSOR_STALE)
    if health.failed:
        fault |= int(Cause.SENSOR_FAILED)
    if health.deadline_miss:
        fault |= int(Cause.DEADLINE_MISS)
    if not health.heartbeat_ok:
        fault |= int(Cause.HEARTBEAT_LOST)
    payload[1] = fault & 0xFF
    for offset, value in ((2, period_ms), (4, p99_latency_ms)):
        ticks = 0 if value != value else int(min(max(0.0, value) * 10.0, 0xFFFF))
        payload[offset] = ticks & 0xFF
        payload[offset + 1] = (ticks >> 8) & 0xFF
    return _finish(ID_HEALTH, payload, alive)


def decode_health(frame: CanFrame) -> Optional[Dict[str, object]]:
    if not _check(frame, ID_HEALTH):
        return None
    data = frame.data
    return {
        "sensors_ok": {name: bool(data[0] & (1 << bit)) for name, bit in SENSOR_BITS.items()},
        "fault_code": Cause(data[1]),
        "period_ms": (data[2] | (data[3] << 8)) / 10.0,
        "p99_latency_ms": (data[4] | (data[5] << 8)) / 10.0,
        "alive": data[6],
    }


# -- SZP_HEARTBEAT (PLC -> pipeline) -----------------------------------------

HEARTBEAT_RESET_REQUEST = 1 << 0
HEARTBEAT_MACHINE_MOVING = 1 << 1


def encode_heartbeat(flags: int, alive: int) -> CanFrame:
    """
    byte 0     flags          uint8, HEARTBEAT_* bits
    bytes 1-5  reserved, zero
    byte 6     alive
    byte 7     crc8
    """
    payload = bytearray(8)
    payload[0] = flags & 0xFF
    return _finish(ID_HEARTBEAT, payload, alive)


def decode_heartbeat(frame: CanFrame) -> Optional[Dict[str, object]]:
    if not _check(frame, ID_HEARTBEAT):
        return None
    data = frame.data
    return {
        "flags": data[0],
        "reset_requested": bool(data[0] & HEARTBEAT_RESET_REQUEST),
        "machine_moving": bool(data[0] & HEARTBEAT_MACHINE_MOVING),
        "alive": data[6],
    }
