# TensorRT

Read [SAFETY_DISCLAIMER.md](SAFETY_DISCLAIMER.md) first.

## Status, stated plainly

**The TensorRT path is implemented and is not measured by CI.** GitHub's
runners have no NVIDIA GPU, so `scripts/export_tensorrt.py` never runs there
and `tests/test_backends.py::test_tensorrt_matches_onnx` skips. A skipped test
is not a passing test.

What CI does check is that `src/szp/detection/tensorrt_backend.py` imports
cleanly without the runtime present and that both backends use the *same*
letterbox and the *same* decoder -- so a divergence between them can only come
from the engine, not from a second copy of the pre-processing that drifted.
That check is `tests/test_backends.py::test_both_backends_share_the_preprocessing_and_the_decoder`
and it runs everywhere.

If you have a GPU, the numbers below are yours to reproduce. If you do not,
treat every TensorRT figure in this repository as unverified on your hardware.

## Building an engine

```bash
pip install -r requirements.txt
# plus tensorrt and pycuda for your CUDA version; on a Jetson both come with JetPack

python scripts/fetch_model.py            # produces models/yolov8n.onnx
python scripts/export_tensorrt.py --fp16 # produces models/yolov8n.engine
python -m pytest tests/test_backends.py -q
python scripts/benchmark.py --out results/benchmark_trt.json
```

`export_tensorrt.py` writes `models/yolov8n.engine.json` next to the plan,
recording the TensorRT version, the platform and the build time.

## Engines are never committed

A serialised TensorRT plan is tied to the GPU architecture, the driver and the
TensorRT version that built it. An engine from somewhere else either refuses to
deserialise -- which is the good case and the error message says so -- or
deserialises and behaves differently. `.gitignore` and `.gitattributes` both
cover `*.engine` and `models/` is ignored entirely.

## FP16

On by default. It roughly halves inference time on Jetson-class hardware and,
at this operating point, changes detections by less than the box jitter. That
last part is a claim, not an assumption:
`tests/test_backends.py::test_tensorrt_matches_onnx` runs both backends over
the same images and requires every ONNX detection to be matched by a TensorRT
detection at IoU above 0.95 with scores agreeing to 0.05. If FP16 broke
something, that test is where it would show.

Build with `--no-fp16` to compare.

## Why the backends are interchangeable

`src/szp/detection/base.py` defines a `Detector` protocol and
`build_detector()` returns whichever backend is available. The pipeline never
imports a backend directly. `build_detector(backend="auto")` prefers TensorRT
when an `.engine` file sits next to the ONNX file and the runtime imports and
falls back to ONNX Runtime otherwise -- but it never substitutes one for the
other silently: the chosen backend is in `detector.info` and every result file
records it.

Both backends call `preprocess.letterbox` and `postprocess.decode_yolov8`.
Neither is allowed to define its own and a test asserts that.

## Implementation notes

Written against the **TensorRT 10** API, which replaced TensorRT 8's binding
indices with named tensors (`get_tensor_name`, `set_tensor_address`,
`execute_async_v3`).

Device and pinned host memory is allocated once at construction and reused. A
per-frame `cudaMalloc` costs a few hundred microseconds and, worse, is a
per-frame opportunity for the allocator to block, which turns a tail-latency
problem into an intermittent deadline miss.

Warm-up is 10 iterations rather than the ONNX path's 3. A TensorRT engine's
first few executions include lazy kernel loads and on a Jetson the clocks have
usually not ramped. `scripts/benchmark.py` prints the warm-up curve so that
choice can be checked rather than taken on trust.

## The machine the published TensorRT numbers came from

**None.** No TensorRT figure appears in the README or in `results/published/`,
because none has been measured. When one is, this section will name the GPU,
the driver, the TensorRT version and the JetPack release and
`results/published/` will hold the file the number came from.

The optional physical target named in the brief is a Jetson Xavier NX. Nothing
in this repository has run on one.

## What to expect if you do measure it

The CPU ONNX Runtime path spends the overwhelming majority of its per-frame
budget inside the model -- see `docs/latency.md` and
`results/published/latency_onnx.json`. Everything after the detector costs a
couple of milliseconds together. So the speed-up from TensorRT is very nearly
the speed-up on the model alone and the rest of the pipeline will not become
the bottleneck until inference is down in the low single-digit milliseconds.

That is a prediction from the measured budget, not a measurement. It is written
here so it can be checked and, if wrong, corrected.
