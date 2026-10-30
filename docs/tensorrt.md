# TensorRT

Read [SAFETY_DISCLAIMER.md](SAFETY_DISCLAIMER.md) first.

## Status, stated plainly

**The TensorRT path has run, on a desktop RTX 4060 and not on a Jetson.**
`make tensorrt-desktop` builds FP32, FP16 and INT8 engines with
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
passing test; the desktop run records, per engine, whether it passed and, if
not, the first difference it found.

## The desktop measurement

<!-- BEGIN generated:tensorrt -->
Measured on a desktop RTX 4060, not a Jetson.  Driver `591.86`, CUDA `12.9.1.010`, TensorRT `10.11.0.33`, image `nvcr.io/nvidia/tensorrt:25.06-py3` under WSL2 Docker.  GPU: no other compute process, at most 5 % utilisation before and after (host nvidia-smi, sampled every 5 s through the run, listed no other compute process).  CPU not busy, load average 1.24 before and 1.21 after on 16 logical cores shared with other containers.  Recall on COCO val2017, 5000 images; latency over 29 entry events on python-can's `virtual` bus.

| Detector | Recall, safety subset, 0.15 | Recall, safety subset, 0.25 | MR-2, safety subset | Inference p50 / p99 | Pipeline p50 / p99 | Reaction at the PLC p50 / p99 | Same detections as ONNX |
|---|---|---|---|---|---|---|---|
| TensorRT FP32 | 89.8 % | 87.6 % | 0.205 | 3.16 / 3.98 ms | 4.97 / 8.82 ms | 8.06 / 11.09 ms | yes |
| TensorRT FP16 | 89.8 % | 87.5 % | 0.206 | 2.43 / 5.17 ms | 4.17 / 7.96 ms | 7.27 / 7.96 ms | no, see below |
| TensorRT INT8 | 87.3 % | 84.1 % | 0.239 | 2.28 / 4.83 ms | 3.98 / 8.23 ms | 6.93 / 7.59 ms | no, see below |
| ONNX Runtime, CPU (published) | 89.8 % | see above | 0.205 | 31.26 ms p50 | 35.98 ms p50 | see above | reference |

The box-for-box equivalence test on 16 COCO photographs with people in them is required of FP32 only; where an engine did not pass it, the first difference it found:

* TensorRT FP16: AssertionError: image 7: ONNX found 6 people, TensorRT found 7
* TensorRT INT8: AssertionError: image 0: boxes disagree
<!-- END generated:tensorrt -->

What it shows:

* **FP32 is the same model.**  Recall on the safety subset at the 0.15
  operating point moves by 0.0001 and MR-2 by 0.0001 against ONNX Runtime on
  the CPU and the equivalence test, which needs every ONNX detection matched
  by a TensorRT one at IoU above 0.95 with scores within 0.05, passed on 16 COCO
  images with people in them.
* **FP16 costs nothing measurable in recall but is not box for box.**  Recall
  is identical to four decimal places over all 5000 images and MR-2 moves by
  0.0003 but on one of the 16 equivalence photographs a person whose score sits
  at the threshold is found by the FP16 engine and not by ONNX Runtime.  With
  the previous detector both engines passed; the test is required of FP32 only
  now and the failure is printed under the table rather than tested away.
* **INT8 is calibrated on train2017 and costs 2.5 points.**  TensorRT's entropy
  calibrator on 512 COCO train2017 images, disjoint from the val2017 images it
  is scored on, loses 2.5 points of safety-subset recall and raises MR-2 from
  0.205 to 0.239, less than the ONNX Runtime INT8 graph's 3.7 points.  It is
  barely faster than FP16 at batch 1, so FP16 remains the engine to deploy.
* **The prediction at the end of this page was right.**  Detector inference
  falls from 31.26 ms on the CPU to 3.16 ms (FP32) and 2.43 ms (FP16) at p50
  and frame arrival to the frame on the bus from 35.98 ms to 4.97 ms and
  4.17 ms.  With inference this small, the couple of milliseconds the rest of
  the pipeline costs is now a large share of the budget rather than a rounding
  error.
* **Against the previous detector.**  YOLOX-Tiny's engines are a little slower
  than YOLOv8n's were on the same GPU, 3.16 ms against 2.86 ms at FP32 and 2.43
  ms against 2.24 ms at FP16, because it is the larger network.  Recall is
  higher by the same margin as on ONNX Runtime.
  `results/published/detector_comparison.json` has the side by side.

Everything in that table is a desktop number.  An Xavier NX has a fraction of
the throughput and shares its memory with the CPU, so only the recall column
is expected to carry over and only for the same TensorRT version.

## Building an engine by hand

```bash
pip install -r requirements.txt
# plus tensorrt and pycuda for your CUDA version; on a Jetson both come with JetPack

python scripts/fetch_model.py                                   # models/yolox_tiny.onnx
python scripts/export_tensorrt.py --precision fp16              # models/yolox_tiny.engine
python -m pytest tests/test_backends.py -q
python analysis/detection_metrics.py --backend tensorrt --out results/detection_trt.json
python analysis/latency_budget.py --backend tensorrt --engine models/yolox_tiny.engine

# INT8: the calibration images first, from COCO train2017, never val2017
python scripts/fetch_calibration.py
python scripts/export_tensorrt.py --precision int8 --engine models/yolox_tiny_int8.engine
```

`export_tensorrt.py` writes `<engine>.json` next to the plan, recording the
precision, the TensorRT version, the platform, the build time and, for INT8,
how many calibration images it used; the calibration cache is written next to
the engine so a rebuild skips calibration.

## Engines are never committed

A serialised TensorRT plan is tied to the GPU architecture, the driver and the
TensorRT version that built it.  An engine from somewhere else either refuses to
deserialise, which is the good case and the error message says so, or
deserialises and behaves differently.  `.gitignore` and `.gitattributes` both
cover `*.engine` and `models/` is ignored entirely.

## Precision

`--precision fp16` is the default.  That it costs nothing in recall at this
operating point is a measurement rather than a claim: the recall column in the
table above, on all 5000 val2017 images.  `--precision fp32` and
`--precision int8` build the other two.  INT8 keeps the FP16 flag on as well,
so layers TensorRT will not quantise fall back to half rather than to full
precision and is calibrated with TensorRT's entropy calibrator on the
train2017 list in `data/calibration_train2017.txt`.

## Why the backends are interchangeable

`src/szp/detection/base.py` defines a `Detector` protocol and
`build_detector()` returns whichever backend is asked for.  The pipeline never
imports a backend directly.  `build_detector(backend="auto")` prefers TensorRT
when an `.engine` file sits next to the ONNX file and the runtime imports and
falls back to ONNX Runtime otherwise but it never substitutes one for the
other silently: the chosen backend is in `detector.info` and every result file
records it.  `engine_path` names a specific engine, which is how the FP32 and
FP16 engines are scored side by side.

Both backends call `preprocess.letterbox` and `postprocess.decode_head`.
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
