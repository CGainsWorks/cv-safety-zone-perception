# FMEA

**This is an engineering FMEA for a demonstrator, not a safety case.** It has
not been reviewed by a functional safety assessor, no PL or SIL is claimed and
several of the mitigations below are "the code does this" rather than "an
independent channel does this". Read [SAFETY_DISCLAIMER.md](../docs/SAFETY_DISCLAIMER.md)
first.

## How to read it

Severity, occurrence and detection are scored 1 to 10 in the usual FMEA
convention and RPN is their product. The scores are **judgements, not
measurements**, with two exceptions that are marked: where a column says
*measured*, the number comes from a script in `analysis/` and the run that
produced it is named.

Severity is scored against what happens to a person, not to the machine.
Anything that can leave a person unprotected while the machine is running is a
9 or a 10 whether or not it is likely.

The column that matters most here is **detection**: a low score means the
failure announces itself, a high score means the system carries on looking
healthy. Most of the design effort in this repository went into pushing
detection scores down, because an undetected failure of a protective function
is the thing that actually hurts somebody.

## Failure modes

| ID | Item | Failure mode | Effect | S | O | D | RPN | Mitigation in this repository | Residual |
|----|------|--------------|---|---|---|---|-----|-------------------------------|----------|
| FM-01 | Detector | Person present, no detection (false negative) | Zone reads clear, machine keeps moving | 10 | 5 | 8 | 400 | Operating point chosen from the measured miss-rate curve, not a default (`analysis/detection_metrics.py`); low confidence threshold accepted along with the resulting nuisance stops; zone exit debounced by 5 frames so a short dropout does not clear a stop | **High.** A monocular detector is a single channel with no independent confirmation. In a real installation this is why a light curtain or scanner stays in the safety circuit and this pipeline does not replace it |
| FM-02 | Detector | Person too small in frame to detect | As FM-01, at long range | 8 | 6 | 8 | 384 | Miss rate reported separately for people above and below the working-range pixel height; zones sized so that the stop zone lies inside the range where people subtend enough pixels | Medium. Depends on the camera being mounted at the height `config/extrinsics.yaml` assumes |
| FM-03 | Depth camera | No return from a dark or specular surface | No range for a detected person | 7 | 7 | 3 | 147 | Detection is never discarded for want of a range: it becomes an `UNRANGED` box whose footprint is the whole wedge it could occupy, which usually trips the zone (`fusion/fuse.py`, measured: `no_ranging` case in `analysis/geometry_accuracy.py`) | Low. Costs nuisance stops, not protection |
| FM-04 | Lidar | Beam blocked or absorbed | No lidar confirmation | 6 | 6 | 3 | 108 | Depth alone still ranges; both missing falls through to FM-03's wedge | Low |
| FM-05 | Lidar mounting | Person crouching or lying below the 0.45 m scan plane | Lidar sees nothing; only the camera protects | 8 | 4 | 6 | 192 | Scan height documented in `config/extrinsics.yaml`; the camera is the primary channel and the lidar is confirmation, never the other way round | **Medium.** A 2D lidar at any single height has this problem. Named rather than solved |
| FM-06 | Extrinsics | Calibration drifts after a knock | Everybody placed systematically wrong; a person at the zone edge reads as outside | 9 | 4 | 8 | 288 | Rotations validated as orthonormal on load; a missing frame chain raises instead of defaulting to identity; residual recorded in the extrinsics file | **High.** Nothing here detects drift *at run time*. A production system needs a periodic reference-target check |
| FM-07 | Fusion | Person standing on a step or platform | Apparent height wrong, so range wrong | 7 | 3 | 7 | 147 | Documented assumption (people stand on z = 0); truncated boxes fall back to a height prior instead of trusting apparent size | Medium. Not solved |
| FM-08 | Fusion | Depth and lidar disagree | One of them is wrong and there is no third opinion | 8 | 4 | 4 | 128 | Outside the Mahalanobis gate the estimate takes the **nearer** of the two and inflates `sigma_xy` by half the disagreement; association is a soft cost, not a hard gate, so a miscalibrated depth camera cannot exclude the lidar that contradicts it | Low |
| FM-09 | Zones | Zone file edited wrongly: self-intersecting, negative margin, no stop zone | Gap in the guarding or none at all | 10 | 4 | 5 | 200 | Every one of those is a start-up failure, not a warning (`zones/config.py`, tested in `tests/test_zones.py`) | Low for the checked cases. A geometrically valid but **wrong** polygon is not detectable by software |
| FM-10 | Pipeline | Unhandled exception in a perception stage | Loop dies, no more frames | 10 | 3 | 2 | 60 | Exceptions are caught per frame and drive the FSM to FAULT with `INTERNAL_ERROR`, which withdraws the permit within one frame; a crash of the whole process withdraws it within `PERMIT_TIMEOUT_MS` | Low |
| FM-11 | CAN | Pipeline stops transmitting: crash, power loss, cable pulled | Machine would keep running if the protocol were command-based | 10 | 3 | 1 | 30 | The e-stop frame is a **permit**, not a command. The PLC enables motion only while a valid, fresh, counter-advancing frame keeps arriving; silence is a stop (`safety/can_frames.py`, tested in `tests/test_can_link.py::test_the_permit_lapses_when_the_pipeline_goes_quiet`) | Low |
| FM-12 | CAN | Payload corrupted in a way the CAN CRC misses | PLC acts on a wrong state | 9 | 2 | 2 | 36 | CRC-8/AUTOSAR over the payload with a per-message data identifier; every single-bit flip in the frame is caught (measured: `tests/test_can_protocol.py::test_every_single_bit_flip_in_the_payload_is_caught`) | Low |
| FM-13 | CAN | Sender hangs with its last frame repeating from a buffer | Stale CLEAR looks healthy | 10 | 3 | 2 | 60 | Rolling ALIVE counter; the PLC refuses the permit when it does not advance | Low |
| FM-14 | CAN | Frame decoded as the wrong message after an ID mix-up | Nonsense state accepted | 8 | 2 | 2 | 32 | The data identifier is mixed into the CRC, so a frame read under the wrong identifier fails its checksum | Low |
| FM-15 | Timing | Pipeline overruns its period under load | Reaction time no longer matches the stopping distance the zones were sized for | 9 | 5 | 3 | 135 | Period watchdog faults on overrun; end-to-end latency measured per frame and its p99 published on the health frame | Medium. The Python implementation has no hard real-time guarantee and does not claim one |
| FM-16 | Timing | Sensor stops publishing but the last frame is still in memory | Empty-looking zone that is not empty | 10 | 4 | 2 | 80 | Per-source staleness deadlines on a monotonic clock; persistent staleness escalates to failed; either faults | Low |
| FM-17 | Clock | System clock steps (NTP) | Every sensor ages out at once or none does | 6 | 3 | 2 | 36 | Nothing in the hot path reads the wall clock; all ages come from `perf_counter_ns` | Low |
| FM-18 | Recovery | Stop clears itself when the person steps out | Machine restarts with somebody still in the cell | 10 | 3 | 3 | 90 | STOP and FAULT latch; recovery needs the cause gone **and** an operator reset **and** a debounce; a reset pressed during the hazard is discarded (`tests/test_safety_fsm.py`) | Low |
| FM-19 | Detector | False positive stops the machine for nobody | Lost production; operators disable the guard | 3 | 6 | 1 | 18 | False-positive rate measured against confidence so the trade is explicit; the nuisance-stop rate is reported alongside the miss rate | Medium. **Worth stating plainly: a guard that nuisance-trips gets bypassed and a bypassed guard protects nobody. The uptime cost of a low threshold is a safety issue, not only a commercial one** |
| FM-20 | Model | Weights swapped for a different or corrupted file | Unknown detection behaviour | 9 | 2 | 6 | 108 | Weight SHA-256 recorded in `models/model_card.json` and in every result file | Medium. Nothing verifies the hash at start-up; that would be a sensible addition |
| FM-21 | TensorRT | Engine built for a different GPU or TRT version | Detector fails to start or silently differs | 8 | 3 | 3 | 72 | Engines are never committed; deserialisation failure is a hard error with an explanatory message; `tests/test_backends.py` cross-checks TensorRT against ONNX Runtime on a machine with a GPU | Medium. The cross-check does not run in CI because CI has no GPU; it passed on a desktop RTX 4060 (`results/published/tensorrt.json`) and CI checks the TensorRT API usage against the pinned wheel |
| FM-22 | Zones | Machine's stopping distance longer than assumed | Zone too small; person reached before motion stops | 10 | 3 | 7 | 210 | `config/zones.yaml` records the assumed reaction and stop times and derives the margin from them; `docs/zones.md` shows the ISO 13855 calculation | **High.** The stop time is an assumption in this repository. On real plant it must be measured |

## What the table says

Sorted by RPN, the top of the list is the detector: FM-01, FM-02 and FM-22.
Two of those are properties of monocular detection and the third is a number
that has to come from the machine rather than from this code. That is the
honest summary of where the risk in this design sits and it is why the
disclaimer says what it says.

The bottom of the list -- the CAN protocol failures -- is where the design is
genuinely strong, because those failures have cheap, testable mitigations and
every one of them has a test. That asymmetry is worth noticing: the parts that
were easy to make safe are safe and the part that matters most is the part
that a second, independent protective device is for.

## Reproducing the measured entries

```bash
make experiments      # detection metrics, geometry accuracy, latency budget
python -m pytest tests/test_can_protocol.py tests/test_can_link.py -q
```

`analysis/make_fmea_summary.py` regenerates the RPN ordering and cross-checks
that every test named in this table exists.
