# Latency

Read [SAFETY_DISCLAIMER.md](SAFETY_DISCLAIMER.md) first. Nothing here is a
worst-case bound. These are measurements of what happened on one machine.

## Two numbers and the difference between them

**Pipeline latency** starts when the frame arrives at `Pipeline.process` and
stops when the e-stop frame's loopback echo comes back off the CAN stack. It is
what you optimise against and it breaks down per stage.

**Reaction latency** starts on the same frame arrival and stops when the
simulated PLC decides the machine may no longer move. It is the number that
belongs in a stopping-distance calculation. It is always larger and the gap is
a property of the bus and of the receiver's polling, not of this code.

Both come from `analysis/latency_budget.py`, which measures the reaction time
over *entry events* -- somebody steps into a stop zone -- and not over every
frame, because averaging over frames where nothing changed measures the cost of
publishing rather than the cost of reacting.

## What is not measured and what to add

The pipeline's clock starts when a `FrameBundle` reaches it. On real hardware
there is a good deal of latency before that and it is often larger than
everything in the table:

| Term | Typical | How to get yours |
|---|---|---|
| Exposure and sensor readout | 5-30 ms | Half the exposure time plus the readout time from the sensor datasheet |
| Driver and USB or GigE transfer | 5-40 ms | Timestamp at the driver callback against the frame's own hardware timestamp |
| Frame queueing before the pipeline | 0 to one frame period | Whatever sits between the driver and `process()` |
| **This repository's measurement** | see below | `analysis/latency_budget.py` |
| CAN arbitration and transmission | ~0.25 ms at 500 kbit/s | 130 bits for a standard 8-byte frame, plus arbitration delay under load |
| PLC scan cycle | 1-20 ms | The PLC's own cycle time; worst case is one full scan |
| Contactor or valve response | 10-50 ms | Component datasheet or measure |

Add them up before sizing a zone. `docs/zones.md` works the sum through.

## The measurement

`make latency` runs it. On the reference machine (recorded in the result file:
processor, OS and Python version), with the real ONNX Runtime detector on the
CPU and python-can's in-process virtual bus:

```bash
python analysis/latency_budget.py --detector onnx_timing --repeats 30 \
    --out results/published/latency_onnx.json
```

and with no model at all, which isolates everything else:

```bash
python analysis/latency_budget.py --detector simulated --repeats 30 \
    --out results/published/latency_simulated.json
```

### Why the published run says `onnx_timing` and not `onnx`

The real detector does not see the rig's people. They are shaded cylinders, not
photographs and a COCO model finds almost nothing in a rendered frame -- which
is what `data/DATASET_CARD.md` says and is exactly why no accuracy claim comes
from the rig.

That leaves reaction time unmeasurable in a plain `--detector onnx` run: with
nothing ever detected the machine never stops and there is no permit
withdrawal to time. `--detector onnx_timing` runs the real ONNX session on
every frame, so its cost, its variance and its tail are genuinely in the
measurement and then takes the detections from the rig's ground truth. It
measures *timing* honestly and supports no accuracy claim at all.
`TimedRealDetector` in `src/szp/data/synthetic.py` says so at the point of use,
and the result file records `detector_kind: onnx_timing`.

`--detector onnx` is still there. It gives an honest per-stage budget on
rendered frames and, for the reason above, usually reports zero entry events.

### Rendering is kept out of the timed loop

The entry scene's people stand still, so one cycle of frames is rendered up
front and replayed. Leaving the renderer in the loop put tens of milliseconds
between publishes -- which is not something a camera does -- and that was
enough to make the simulated PLC's permit lapse repeatedly, showing up as
"timeouts" that were an artefact of the harness rather than a property of the
pipeline.

The published numbers are in `results/published/` and the README quotes them.
Every one is reproducible with the command above.

### Reading the budget table

Stage shares are computed from the **means**, not the medians. Percentiles do
not add: the median of a sum is not the sum of the medians and a budget built
from p50 columns reports a large fictional "unaccounted" remainder that is an
artefact of the arithmetic rather than a real gap. The p50 and p99 columns are
still shown, because a deadline is sized against the tail and not against the
mean.

The `unaccounted` row is the real remainder: allocation, dataclass
construction, the Python interpreter between stages. It is small and it is
shown rather than folded into the stages, because a budget that hides its
overheads inside the stages is a budget that will not add up on the target.

`publish_aux` -- the zone and health frames -- is measured but excluded from
the budget. Those frames go out *after* the e-stop frame is already on the bus,
so they are not on the critical path.

## Where the time goes

At 640x640 on a CPU, the detector is close to the whole budget. Everything
after it -- fusion, zones, the state machine, the CAN encode -- is a couple of
milliseconds put together. That has two consequences worth stating.

**Optimising the pipeline is optimising the detector.** Rewriting the zone
tests in C would save microseconds against tens of milliseconds of inference.
The measurements say so; the alternative is guessing.

**The detector's tail is the pipeline's tail.** The p99 of a CPU inference is
noticeably above its p50 and no amount of care downstream compensates. This is
the argument for TensorRT and for a smaller input resolution and it is why the
health frame carries the p99 rather than the mean: the PLC can see the tail
growing before it becomes a deadline miss.

## TensorRT

`docs/tensorrt.md` covers the export and the numbers. The short version: on a
desktop RTX 4060, inside NVIDIA's TensorRT container, the detector falls to a
few milliseconds and frame to bus to under 5 ms at p50, with the same recall.
That is a desktop measurement, not a Jetson one and it is not repeated in CI
because CI has no GPU.

## Timing hygiene

Everything is `time.perf_counter_ns`: a monotonic integer counter. The wall
clock is not read anywhere in the hot path, so an NTP step cannot age every
sensor out at once or produce a negative latency.

The publisher deliberately does not use python-can's `Message.timestamp` for
the end-to-end figure. It is wall-clock seconds and converting it into the
`perf_counter` base drifts over a run -- on the first version of this
repository that drift showed up as four milliseconds of "unaccounted" latency
that was not real. `src/szp/safety/publisher.py` says so at the top.

The detector is warmed up before anything is timed. The first inference in an
ONNX Runtime session is several times slower than the steady state and
`scripts/benchmark.py` prints the warm-up curve so the choice of how many
iterations can be checked rather than taken on trust.
