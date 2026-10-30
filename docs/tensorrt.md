# TensorRT

Read [SAFETY_DISCLAIMER.md](SAFETY_DISCLAIMER.md) first.

## Status, stated plainly

**The TensorRT path has run, on a desktop RTX 4060 and not on a Jetson.**
`make tensorrt-desktop` builds FP32 and FP16 engines with
`scripts/export_tensorrt.py` inside NVIDIA's `nvcr.io/nvidia/tensorrt:25.06-py3`
container under WSL2, scores them on COCO val2017 with the same recall and MR-2
code as the published ONNX numbers, times the whole pipeline from frame arrival
to the frame at the simulated PLC and runs
`tests/test_backends.py::test_tensorrt_matches_onnx` against each engine on
real photographs.  `results/published/tensorrt.json` holds the result and the
table below is generated from it by `analysis/tensorrt_table.py`;
`scripts/check_docs.py` fails if the two disagree.

**CI cannot build or run an engine.**  GitHub's runners have no NVIDIA GPU.
The pinned TensorRT wheel installs and imports there but `trt.Builder` needs a
CUDA driver and aborts the process without one, so creating a network
definition, parsing the ONNX graph and building an engine are a manual step on
a machine with a GPU.  What CI does check:

* the `tensorrt-api` job installs `tensorrt-cu12==10.11.0.33`
  (`requirements-tensorrt.txt`, the TensorRT in the container above), imports
  the backend and resolves every TensorRT symbol that
  `src/szp/detection/tensorrt_backend.py` and `scripts/export_tensorrt.py` use
  (`tests/test_tensorrt_api.py`);
* every job checks that both backends use the *same* letterbox and the *same*
  decoder, so a divergence between them can only come from the engine
  (`tests/test_backends.py::test_both_backends_share_the_preprocessing_and_the_decoder`).

`test_tensorrt_matches_onnx` still skips in CI.  A skipped test is not a
passing test; it passed on the desktop run against both engines and the result
file records that.

## The desktop measurement

<!-- BEGIN generated:tensorrt -->
Measured on a desktop RTX 4060, not a Jetson.  Driver `591.86`, CUDA `12.9.1.010`, TensorRT `10.11.0.33`, image `nvcr.io/nvidia/tensorrt:25.06-py3` under WSL2 Docker.  GPU: no other compute process listed but utilisation reached 12 % in the idle windows before or after the run, above the 10 % threshold; this GPU also drives the desktop display (host nvidia-smi, sampled every 5 s through the run, listed no other compute process).  CPU not busy, load average 0.54 before and 1.08 after on 16 logical cores shared with other containers.  Recall on COCO val2017, 5000 images; latency over 29 entry events on python-can's `virtual` bus.

| Detector | Recall, safety subset, 0.15 | Recall, safety subset, 0.25 | MR-2, safety subset | Inference p50 / p99 | Pipeline p50 / p99 | Reaction at the PLC p50 / p99 | Same detections as ONNX |
|---|---|---|---|---|---|---|---|
| TensorRT FP32 | 88.1 % | 84.0 % | 0.227 | 2.86 / 3.57 ms | 4.60 / 8.92 ms | 7.61 / 8.42 ms | yes |
| TensorRT FP16 | 88.2 % | 83.9 % | 0.227 | 2.24 / 4.95 ms | 3.99 / 8.29 ms | 7.27 / 9.56 ms | yes |
| ONNX Runtime, CPU (published) | 88.2 % | see above | 0.227 | 26.32 ms p50 | 31.11 ms p50 | see above | reference |
<!-- END generated:tensorrt -->

What it shows:

* **The same detections.**  Recall on the safety subset at the 0.15 operating
  point moves by 0.00042 for FP32 and 0.00014 for FP16 and MR-2 by 0.0002 and
  0.00007, against ONNX Runtime on the CPU.  FP16 costs nothing measurable in
  recall at this operating point.  The equivalence test, which needs every ONNX
  detection matched by a TensorRT one at IoU above 0.95 with scores within
  0.05, passed on 16 COCO images with people in them for both engines.
* **The prediction at the end of this page was right.**  Detector inference
  falls from 26.32 ms on the CPU to 2.86 ms (FP32) and 2.24 ms (FP16) at p50
  and frame arrival to the frame on the bus from 31.11 ms to 4.60 ms and
  3.99 ms.  With inference this small, the couple of milliseconds the rest of
  the pipeline costs is now a large share of the budget rather than a rounding
  error.
* **No INT8 row.**  This repository has no INT8 path and no calibration set:
  the detector is Ultralytics' COCO checkpoint used as shipped and calibrating
  on val2017 would calibrate on the images it is then scored on.

Everything in that table is a desktop number.  An Xavier NX has a fraction of
the throughput and shares its memory with the CPU, so only the recall column
is expected to carry over and only for the same TensorRT version.

## Building an engine by hand

```bash
pip install -r requirements.txt
# plus tensorrt and pycuda for your CUDA version; on a Jetson both come with JetPack

python scripts/fetch_model.py                                   # models/yolov8n.onnx
python scripts/export_tensorrt.py --fp16                        # models/yolov8n.engine
python -m pytest tests/test_backends.py -q
python analysis/detection_metrics.py --backend tensorrt --out results/detection_trt.json
python analysis/latency_budget.py --backend tensorrt --engine models/yolov8n.engine
```

`export_tensorrt.py` writes `models/yolov8n.engine.json` next to the plan,
recording the TensorRT version, the platform and the build time.

## Engines are never committed

A serialised TensorRT plan is tied to the GPU architecture, the driver and the
TensorRT version that built it.  An engine from somewhere else either refuses to
deserialise, which is the good case and the error message says so, or
deserialises and behaves differently.  `.gitignore` and `.gitattributes` both
cover `*.engine` and `models/` is ignored entirely.

## FP16

On by default.  That it costs nothing in recall at this operating point is now
a measurement rather than a claim: the table above and
`tests/test_backends.py::test_tensorrt_matches_onnx`, which runs both backends
over the same images.  Build with `--no-fp16` to compare.

## Why the backends are interchangeable

`src/szp/detection/base.py` defines a `Detector` protocol and
`build_detector()` returns whichever backend is asked for.  The pipeline never
imports a backend directly.  `build_detector(backend="auto")` prefers TensorRT
when an `.engine` file sits next to the ONNX file and the runtime imports and
falls back to ONNX Runtime otherwise but it never substitutes one for the
other silently: the chosen backend is in `detector.info` and every result file
records it.  `engine_path` names a specific engine, which is how the FP32 and
FP16 engines are scored side by side.

Both backends call `preprocess.letterbox` and `postprocess.decode_yolov8`.
Neither is allowed to define its own and a test asserts that.

## Implementation notes

Written against the **TensorRT 10** API, which replaced TensorRT 8's binding
indices with named tensors (`get_tensor_name`, `set_tensor_address`,
`execute_async_v3`).

Device and pinned host memory is allocated once at construction and reused.  A
per-frame `cudaMalloc` costs a few hundred microseconds and, worse, is a
per-frame opportunity for the allocator to block, which turns a tail-latency
problem into an intermittent deadline miss.

Warm-up is 10 iterations rather than the ONNX path's 3.  A TensorRT engine's
first few executions include lazy kernel loads and on a Jetson the clocks have
usually not ramped.

## Jetson: a hardware TODO

The optional physical target named in the original plan is a Jetson Xavier NX.
Nothing in this repository has run on one and no Jetson number exists.  When
one does, it belongs in its own result file next to the desktop one, with the
JetPack release named.

## The prediction, as written before the measurement

The CPU ONNX Runtime path spends the overwhelming majority of its per-frame
budget inside the model (see `docs/latency.md` and
`results/published/latency_onnx.json`).  Everything after the detector costs a
couple of milliseconds together.  So the speed-up from TensorRT is very nearly
the speed-up on the model alone and the rest of the pipeline will not become
the bottleneck until inference is down in the low single-digit milliseconds.

The desktop measurement above put inference in the low single-digit
milliseconds and the rest of the pipeline is now comparable to it.
