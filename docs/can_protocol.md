# CAN protocol

Read [SAFETY_DISCLAIMER.md](SAFETY_DISCLAIMER.md) first.

Protocol version 1. The reference implementation is
`src/szp/safety/can_frames.py`, which has no dependency on python-can so that a
PLC-side implementation has exactly one file to read and the encoding can be
tested exhaustively without a bus.

## Bus

| | |
|---|---|
| Physical layer | Classic CAN, 500 kbit/s |
| Identifiers | 11-bit standard |
| All frames | 8 data bytes, always |
| Byte order | little endian for multi-byte fields |
| Test interface | Linux `vcan` (`scripts/setup_vcan.sh`) or python-can's in-process `virtual` bus where vcan does not exist |

## Messages

| ID | Name | Direction | Cycle | Purpose |
|---|---|---|---|---|
| `0x080` | `SZP_ESTOP` | pipeline to PLC | 20 ms, 10 ms while not CLEAR | the motion permit |
| `0x081` | `SZP_ZONES` | pipeline to PLC | 50 ms | which zones are occupied and by how much |
| `0x082` | `SZP_HEALTH` | pipeline to PLC | 100 ms | sensor health, period, p99 latency |
| `0x700` | `SZP_HEARTBEAT` | PLC to pipeline | 100 ms | liveness and the operator reset request |

`SZP_ESTOP` takes the lowest identifier deliberately: low identifiers win CAN
arbitration and it is the most time-critical frame on the bus.

## The rule that makes it fail safe

`SZP_ESTOP` is a **permit**, not a command.

> The PLC shall enable motion only while **all** of the following hold:
>
> 1. a `SZP_ESTOP` frame has been received within the last **100 ms**;
> 2. its CRC checks;
> 3. its `alive` counter differs from the previous accepted frame's;
> 4. its `state` is `CLEAR` (1) or `WARN` (2).
>
> If any condition fails, motion shall not be enabled.

There is no default-permit path. `permit_granted()` in `can_frames.py` is that
rule written out and `tests/test_can_protocol.py::test_there_is_no_path_that_grants_a_permit_by_default`
enumerates every combination of state, age and counter to prove it.

The consequence is the point of the whole design: a pipeline that crashes,
hangs, loses power or is unplugged sends nothing, the permit lapses inside
100 ms and the machine stops. **There is no message the pipeline can fail to
send that leaves the machine running.** Getting this the other way round -- a
"stop now" command that has to arrive to be obeyed -- is the classic way to
build a safety function that fails silently and it is FM-11 in the FMEA.

100 ms is five nominal cycles, so four consecutive frame losses are tolerated
and the fifth stops the machine.

## End-to-end protection

CAN's own 15-bit CRC protects the wire. It does not tell the receiver that the
sender is still computing new values and it does not help if the wrong message
is decoded as this one. So each frame carries:

- **byte 6, `alive`**: a counter that advances on every transmitted frame and
  wraps at 256. A sender hung with its last frame repeating out of a hardware
  buffer is caught here (FM-13).
- **byte 7, `crc`**: CRC-8/AUTOSAR (polynomial `0x2F`, init `0xFF`, no
  reflection, final xor `0xFF`) computed over bytes 0 to 6 **followed by the
  message's data identifier**. The published check value for the ASCII string
  `123456789` is `0xDF`, pinned in the tests so a PLC-side implementation has a
  known-good vector.

Data identifiers: `SZP_ESTOP` `0x11`, `SZP_ZONES` `0x22`, `SZP_HEALTH` `0x33`,
`SZP_HEARTBEAT` `0x44`. Mixing them into the CRC means the same seven payload
bytes produce a different byte 7 under a different message, so a frame decoded
under the wrong identifier fails its checksum rather than yielding a
plausible-looking state (FM-14).

This is AUTOSAR E2E Profile 1 in shape. The implementation is written from the
description; no AUTOSAR code is included.

## `SZP_ESTOP` (0x080)

| Byte | Field | Type | Notes |
|---|---|---|---|
| 0 | `state` | uint8 | 0 INIT, 1 CLEAR, 2 WARN, 3 STOP, 4 FAULT, 5 RECOVERING |
| 1 | `cause` | uint8 | bitfield, see below |
| 2-3 | `zone_mask` | uint16 LE | bit *i* set means zone *i* is occupied; zone order is file order in `config/zones.yaml` |
| 4-5 | `min_range_cm` | uint16 LE | distance to the nearest person, centimetres. `0xFFFF` means nobody in view |
| 6 | `alive` | uint8 | rolling |
| 7 | `crc` | uint8 | CRC-8/AUTOSAR over bytes 0-6 plus data id `0x11` |

`min_range_cm` is clamped to `0xFFFE`, so a bad fusion result cannot alias onto
the "nobody there" sentinel. For a range-ambiguous detection it carries the
**near** end of the possible interval: the pessimistic reading.

`cause` bits:

| Bit | Meaning |
|---|---|
| 0 | zone occupied |
| 1 | a sensor is stale |
| 2 | a sensor has failed |
| 3 | pipeline deadline missed |
| 4 | PLC heartbeat lost |
| 5 | the stop was tripped by a detection with no confirmed range |
| 6 | internal error in the pipeline |
| 7 | cause has cleared, waiting for an operator reset |

Bit 5 is worth wiring to an indicator. It means the pipeline saw somebody it
could not range and stopped the machine anyway, which is intended behaviour and
is also the most likely source of a stop the operator will not understand.

## `SZP_ZONES` (0x081)

| Byte | Field | Type | Notes |
|---|---|---|---|
| 0-1 | `stop_mask` | uint16 LE | occupied stop zones |
| 2-3 | `warn_mask` | uint16 LE | occupied warning zones |
| 4-5 | `nearest_gap_cm` | int16 LE | signed distance from the nearest footprint to the nearest zone boundary; negative means inside. `0x7FFF` means no detections |
| 6 | `alive` | uint8 | |
| 7 | `crc` | uint8 | data id `0x22` |

Diagnostic only. It is sent *after* the e-stop frame every cycle, so it never
shares the critical path with it.

## `SZP_HEALTH` (0x082)

| Byte | Field | Type | Notes |
|---|---|---|---|
| 0 | `sensors_ok` | uint8 | bit 0 rgb, 1 depth, 2 lidar, 3 detector, 4 heartbeat |
| 1 | `fault_code` | uint8 | the `cause` bits currently set by health |
| 2-3 | `period` | uint16 LE | pipeline frame period, tenths of a millisecond |
| 4-5 | `p99_latency` | uint16 LE | p99 end-to-end latency, tenths of a millisecond |
| 6 | `alive` | uint8 | |
| 7 | `crc` | uint8 | data id `0x33` |

`p99_latency` rather than the mean, so a PLC or an HMI can watch the tail grow
before it becomes a deadline miss.

## `SZP_HEARTBEAT` (0x700, PLC to pipeline)

| Byte | Field | Type | Notes |
|---|---|---|---|
| 0 | `flags` | uint8 | bit 0 reset requested, bit 1 machine moving |
| 1-5 | reserved | | zero |
| 6 | `alive` | uint8 | |
| 7 | `crc` | uint8 | data id `0x44` |

Losing this for 300 ms is a `HEARTBEAT_LOST` fault, which stops the machine.
The reset request is what lets an operator clear a latched stop; it is only
acted on once the cause has already gone and a reset asserted *during* the
hazard is discarded rather than queued.

## Trying it

```bash
sudo scripts/setup_vcan.sh          # Linux
python scripts/run_demo.py --can socketcan --channel vcan0
candump -tz vcan0                   # from can-utils
```

On macOS or Windows there is no vcan; the pipeline falls back to python-can's
in-process virtual bus automatically and every result file records which one it
used. That fallback is fine for correctness and useless for timing and
`docs/latency.md` says so.

## Implementing the PLC side

1. Copy the rule in "The rule that makes it fail safe" verbatim into the safety
   logic. Every condition, including the counter.
2. Take the CRC-8 implementation and the check value from
   `src/szp/safety/can_frames.py` and verify it against `0xDF` before trusting
   anything else.
3. Treat a stale, absent or invalid frame identically to `STOP`. There is no
   third case.
4. Do not use `SZP_ZONES` or `SZP_HEALTH` for any interlock decision. They are
   diagnostics, they are sent less often and they may lag the permit by
   several cycles.
