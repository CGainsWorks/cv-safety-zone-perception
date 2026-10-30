# Limitations

Read [SAFETY_DISCLAIMER.md](SAFETY_DISCLAIMER.md) first. That file covers why
this is not a safety function. This one covers what the engineering does not do
even taken on its own terms.

## The detection numbers are on photographs, not on a factory

Recall is measured on COCO val2017. COCO is photographs of everyday scenes.
An industrial cell has fixed overhead viewpoints, high-visibility clothing,
hard hats, crouching and kneeling postures, occlusion by machinery, mixed
artificial lighting and motion blur. None of that is in COCO and the direction
of the transfer error is unknown -- a fixed viewpoint might help, unusual
postures certainly hurt.

**The single largest gap in this repository is that there is no evaluation on
industrial imagery**, because no suitable public dataset exists that could be
used clean-room. Anyone deploying this would have to collect and label their
own and the miss rate on it is the number that would matter.

## The geometry numbers are on a synthetic rig

Fusion accuracy, zone confusion, state-machine behaviour and latency are all
measured on the rig in `src/szp/data/synthetic.py`, because it is the only
place a metric ground truth exists. What that proves is that the arithmetic,
the extrinsics, the zone tests and the fail-safe logic are correct, including
under sensor dropout and occlusion. What it does not prove is anything about
real sensors.

Two specific ways the rig flatters the pipeline:

- **People are cylinders of exactly the radius the fusion stage assumes.** Both
  bias corrections -- front surface to centre for depth, arc to centre for
  lidar -- are therefore exactly right and the best-case residual error is
  unrealistically small. The `wrong_body_size` case quantifies the sensitivity
  and the README quotes it alongside.
- **Depth noise is Gaussian and range dependent, which is the model the
  estimator assumes.** Real depth cameras have structured failures -- multipath,
  flying pixels at depth edges, interference between units -- that this rig does
  not reproduce.

## No tracking

There is no filter and no data association across frames. A person's position
is estimated independently every frame. This is deliberate: a track would smooth
the estimates and would also coast through a detection dropout, reporting a
person where they used to be rather than where they are. The five-frame zone
exit debounce is the only temporal memory and it is on the side that keeps a
zone tripped rather than the side that clears it.

The cost is real. Position estimates are noisier frame to frame than they need
to be and there is no velocity estimate, so the system cannot anticipate
somebody about to enter a zone. Speed-and-separation monitoring in the ISO/TS
15066 sense needs that velocity and this does not have it.

## One camera

The frame graph supports several. Nothing else does. A single viewpoint means:

- a person fully occluded by machinery is invisible and the pipeline has no
  way to know the difference between occluded and absent;
- there is no independent channel, so a detector failure is a failure of the
  whole function;
- the working volume is one frustum and `scripts/show_zones.py` will warn
  about a zone outside it but cannot fix it.

## The 2D lidar sees one height

0.45 m in the reference configuration: knee height, high enough to clear most
floor clutter and low enough to see somebody who has crouched. It cannot see
somebody lying down and it cannot see an arm reaching in over the top of it.
This is a property of any single-plane scanner and it is FM-05 in the FMEA.

## People are assumed to stand on the floor

`z = 0` in the world frame. Somebody on a step, a pallet or a mezzanine has
their height mis-estimated and through the height prior their range with it.
The height prior for a truncated box makes this worse rather than better: it
assumes 1.75 m of person regardless. FM-07.

## No run-time detection of extrinsic drift

Extrinsics are validated at load -- orthonormality, frame connectivity -- and
never again. A bracket knocked out of alignment produces a systematic position
error that looks exactly like sensor noise. A production system wants a
reference target permanently in view and a periodic check against it. FM-06 is
one of the three highest-RPN entries in the FMEA for exactly this reason.

## No worst-case execution time

Python, on a general-purpose operating system, with a garbage collector and an
allocator. The latency figures are measurements, not bounds. The period
watchdog catches an overrun after the fact, which is the right thing to do and
is not the same as guaranteeing it will not happen.

## The stopping time is assumed

`config/zones.yaml` records a machine stopping time of 250 ms and derives the
zone margin from it. That number is an invention for a fictional cell. It is
usually the largest single term in the sizing calculation and it must be
measured on real plant. FM-22 is the highest-severity entry with no software
mitigation at all.

## TensorRT is not exercised by CI

There is no GPU on GitHub's runners, so `scripts/export_tensorrt.py` never runs
there and `tests/test_backends.py::test_tensorrt_matches_onnx` skips. A skipped
test is not a passing test. CI does check every TensorRT symbol the code uses
against the pinned wheel. The equivalence check passed on a desktop RTX 4060
against FP32 and FP16 engines, as `results/published/tensorrt.json` records;
between runs of `make tensorrt-desktop` the claim that the TensorRT path
produces the same detections as the ONNX path is untested. Nothing has run on
a Jetson.

## The false-positive cost is a safety issue, not only a commercial one

The operating point is chosen for recall, which produces more nuisance stops
than a conventional threshold would. That is the right trade for a protective
function considered in isolation. In a real installation a guard that stops
production for nobody several times a shift gets bypassed and a bypassed guard
protects nobody. FM-19 scores low on RPN and is one of the more likely ways for
a system like this to fail in practice.

## What would be next

In rough order of how much each would change the answer:

1. Collect and label industrial imagery and measure the miss rate on it.
2. A second camera, for occlusion coverage and an independent channel.
3. A run-time extrinsic check against a fixed reference target.
4. Velocity estimation, for speed-and-separation monitoring rather than a
   binary zone test.
5. A C++ or Rust implementation of the post-detector stages, to get a defensible
   worst-case execution time. The measurements say this would save single-digit
   milliseconds against tens of milliseconds of inference, so it is fifth and
   not first.
