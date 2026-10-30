# Status

Last updated: 2026-10-30

This is an engineering demonstrator and **not a certified safety function**;
read [docs/SAFETY_DISCLAIMER.md](docs/SAFETY_DISCLAIMER.md) first.

## Complete

- **Detector swapped to YOLOX-Tiny, Apache-2.0 code and weights.**  Megvii's
  COCO checkpoint used as shipped, exported by `scripts/fetch_model.py` through
  `scripts/yolox_export.py` with the input normalisation folded into the stem
  and the box decode in the graph, so the head layout, both backends and the
  equivalence test are unchanged.  The AGPL-3.0 obligation the YOLOv8n weights
  carried is gone; `docs/licences.md` says why YOLOX rather than RTMDet or
  D-FINE and `docs/MODEL_CARD.md` describes the model.
- Person detection through ONNX Runtime at FP32, FP16 and INT8, scored on COCO
  val2017 for recall, MR-2 and where the misses are
  (`results/published/detection_val2017*.json`).  INT8 is calibrated on a
  fixed, seeded list of COCO train2017 images, disjoint from the evaluation
  set (`data/calibration_train2017.txt`).
- Every detector-dependent published number re-measured with the new detector
  and the YOLOv8n numbers kept beside them in
  `results/published/detector_comparison.json`, read from the last commit
  before the swap by `analysis/compare_detectors.py`.
- Depth and 2D lidar fusion with unranged-wedge fallback, polygon zones, a
  latching fail-safe state machine and a CAN permit frame with CRC-8/AUTOSAR
  and an alive counter, published to a simulated PLC; CI runs it over a real
  Linux vcan interface.
- Geometry, latency and calibration measured on a synthetic cell with exact
  ground truth; FMEA kept consistent by `analysis/make_fmea_summary.py`.
- TensorRT engines at FP32, FP16 and INT8 built by `scripts/export_tensorrt.py`
  in NVIDIA's `nvcr.io/nvidia/tensorrt:25.06-py3` container on a desktop RTX
  4060 under WSL2, scored on COCO val2017, timed through the whole pipeline and
  compared with ONNX Runtime on real photographs (`make tensorrt-desktop`,
  `results/published/tensorrt.json`, the generated tables in the README and
  `docs/tensorrt.md`).
- CI `tensorrt-api` job: imports the backend, checks every TensorRT symbol the
  backend and the engine builder use against the pinned `tensorrt-cu12` wheel
  and builds the INT8 calibrator on TensorRT's real base class, on a runner
  with no GPU.
- Tests: 204 in total.  191 pass locally on Windows with 3 skipped (the 11
  TensorRT API tests as one module; the TensorRT equivalence test, which needs
  a GPU; one fusion test whose scene has too few depth-only boxes).  The
  TensorRT API tests pass in the CI job that installs the wheel and the
  equivalence test passed against the FP32 engine on the desktop GPU.

## Where the new detector is worse

Measured, not assumed; `results/published/detector_comparison.json` has the
full set.

- **More false positives at the operating point**: 0.67 per image on the
  safety subset at confidence 0.15 against 0.42, about 60 per cent more
  nuisance stops at the same threshold.  Recall is higher (89.8 % against
  88.2 %) and MR-2 lower (0.205 against 0.227), so the detector is better on
  the miss-rate curve but the threshold has not been re-chosen for it.
- Slower: 31.3 ms against 26.3 ms inference on the CPU, 3.16 ms against 2.86 ms
  on TensorRT FP32.
- The FP16 engine is not box-for-box identical to ONNX Runtime (one extra
  borderline detection on one of 16 photographs), where YOLOv8n's was; its
  recall over 5000 images is the same to four decimal places.
- People taller than 200 px are missed slightly more often, 5.3 % against
  4.6 %.

## In progress

Nothing.

## Not started

- Re-choose the confidence operating point for YOLOX-Tiny, trading the higher
  false-positive rate against recall and re-score FM-19 in the FMEA with it.
- Tracking (ByteTrack from cv-yolo-detect-track-jetson) to fix the stated "no
  tracking" limitation.
- A real-sensor input path: camera and lidar drivers or bag ingestion.
- Repositioning as non-safety-rated near-miss and zone-intrusion analytics,
  with events published as telemetry.

## Limitations

- **No Jetson (hardware TODO).**  The TensorRT numbers come from a desktop RTX
  4060; nothing has run on a Jetson Xavier NX.
- **CI cannot build a TensorRT engine.**  Hosted runners have no GPU and
  `trt.Builder` aborts without a CUDA driver, so engine builds and the
  equivalence test are a manual step (`make tensorrt-desktop`).
- INT8 costs recall: 3.7 points through ONNX Runtime and 2.5 through TensorRT
  on the safety subset, for little speed on this GPU, so FP16 remains the
  engine to deploy.
- The YOLOX checkpoint is read as Apache-2.0 because Megvii publishes it in an
  Apache-2.0 repository without separate terms; `docs/licences.md` flags that
  as a point to confirm before commercial use.
- Detection is measured on photographs, not on a factory floor; geometry on a
  synthetic rig; one camera, one lidar height, no extrinsic drift detection,
  no worst-case execution time.  `docs/limitations.md` has the full list.
