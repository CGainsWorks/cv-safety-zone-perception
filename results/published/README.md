# Published results

Every number quoted in the README comes from one of these files.
`scripts/check_docs.py` enforces that and CI runs it, so the README cannot
drift from the measurements without the build going red.

| File | Produced by | What it measures |
|---|---|---|
| `detection_val2017.json` | `analysis/detection_metrics.py` | person-detection recall and miss rate on COCO val2017, real photographs, FP32 ONNX graph |
| `detection_val2017_fp16.json`, `detection_val2017_int8.json` | `analysis/detection_metrics.py --model models/yolox_tiny_{fp16,int8}.onnx` | the same for the FP16 and INT8 ONNX graphs |
| `quantisation.json` | `scripts/quantize_onnx.py` | how the INT8 ONNX graph was made: calibration set, excluded head nodes, size |
| `detector_comparison.json` | `analysis/compare_detectors.py` | YOLOX-Tiny beside the YOLOv8n numbers it replaced, read from git history |
| `geometry_accuracy.json` | `analysis/geometry_accuracy.py` | fusion position error and zone confusion on the synthetic cell, under sensor degradation |
| `latency_onnx.json` | `analysis/latency_budget.py --detector onnx` | end-to-end latency with the real model in the loop |
| `latency_simulated.json` | `analysis/latency_budget.py --detector simulated` | the same, with the model taken out, so everything else is visible |
| `demo.json`, `demo_timeline.csv` | `scripts/run_demo.py` | the whole pipeline over the reference scene, with the state machine and the simulated PLC |
| `calibration_self_check.json` | `scripts/calibrate.py --self-check` | how well the camera-lidar solver recovers a transform it was not told |
| `benchmark.json` | `scripts/benchmark.py` | per-stage micro-benchmarks and the detector's warm-up curve |
| `tensorrt.json` | `scripts/measure_tensorrt.py` (`make tensorrt-desktop`) | FP32, FP16 and INT8 TensorRT engines on a desktop RTX 4060: COCO recall and MR-2, pipeline latency, agreement with ONNX Runtime, versions and contention |

## Regenerating them

```bash
make model          # download and export the detector, once
make data           # download COCO val2017, once, about 1.1 GB
make experiments    # regenerates every file above except the INT8 and TensorRT ones
make int8           # the FP16 and INT8 graphs and their COCO scores
```

`make experiments` takes a few minutes, most of it inference on the CPU.

## The machine these came from

Recorded inside each file rather than here, so a file cannot be separated from
the machine that produced it. `benchmark.json` has the fullest record:
platform, processor, Python and numpy versions and the detector's execution
provider.

None of these numbers came from a Jetson.  Only `tensorrt.json` came from a
GPU: a desktop RTX 4060 running TensorRT engines in NVIDIA's container, written
by `scripts/measure_tensorrt.py` and recording the driver, CUDA and TensorRT
versions and the contention state.  Everything else is CPU.  See
`docs/tensorrt.md`.

## Two kinds of number and the difference matters

**Real data.** `detection_val2017.json` is the only file measured on real
images. It is the weaker guarantee and the more important one: it says how
often a general-purpose detector finds a person in a photograph, at a chosen
operating point and nothing about how often it would find one in a factory.

**Synthetic rig.** Everything else is measured on `src/szp/data/synthetic.py`,
where the true position of every person is known exactly. That proves the
geometry, the zone logic and the fail-safe behaviour are correct, including
under dropout and occlusion. It proves nothing about detection.

`data/DATASET_CARD.md` sets out what each one can and cannot show.

## Reading the CAN interface field

Result files record `can_interface`. On Linux it is `socketcan` over a `vcan`
interface, so the frames go through the kernel's CAN stack. On macOS and
Windows there is no vcan and it is `virtual`, python-can's in-process bus,
which is a queue between two objects in one address space. Correctness is
identical; timings are not comparable and `docs/latency.md` says so.
