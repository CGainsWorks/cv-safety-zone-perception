# Status

Last updated: 2026-10-16

This is an engineering demonstrator and **not a certified safety function**;
read [docs/SAFETY_DISCLAIMER.md](docs/SAFETY_DISCLAIMER.md) first.

## Complete

- YOLOv8n person detection through ONNX Runtime, scored on COCO val2017 for
  recall, MR-2 and where the misses are (`results/published/detection_val2017.json`).
- Depth and 2D lidar fusion with unranged-wedge fallback, polygon zones, a
  latching fail-safe state machine and a CAN permit frame with CRC-8/AUTOSAR
  and an alive counter, published to a simulated PLC; CI runs it over a real
  Linux vcan interface.
- Geometry, latency and calibration measured on a synthetic cell with exact
  ground truth; FMEA kept consistent by `analysis/make_fmea_summary.py`.
- TensorRT engines at FP32 and FP16 built by `scripts/export_tensorrt.py` in
  NVIDIA's `nvcr.io/nvidia/tensorrt:25.06-py3` container on a desktop RTX 4060
  under WSL2, scored on COCO val2017, timed through the whole pipeline and
  compared with ONNX Runtime on real photographs (`make tensorrt-desktop`,
  `results/published/tensorrt.json`, the generated tables in the README and
  `docs/tensorrt.md`).
- CI `tensorrt-api` job: imports the backend and checks every TensorRT symbol
  the backend and the engine builder use against the pinned `tensorrt-cu12`
  wheel, on a runner with no GPU.
- `docs/licences.md` states the AGPL-3.0 obligation on the detector weights.
- Tests: 203 in total.  191 pass locally on Windows with 3 skipped (the 10
  TensorRT API tests as one module; the TensorRT equivalence test, which needs
  a GPU; one fusion test whose scene has too few depth-only boxes); the
  TensorRT API tests pass in the CI job that installs the wheel and the
  equivalence test passed on the desktop GPU against both engines.

## In progress

Nothing.

## Not started

- **Swap the detector for an Apache-2.0 one (YOLOX, RTMDet or D-FINE).**  The
  weights are Ultralytics' AGPL-3.0 COCO checkpoint used as shipped, so closed
  commercial use needs an Ultralytics Enterprise Licence.  A head that is not
  YOLOv8-shaped needs a second decoder in `src/szp/detection/postprocess.py`
  and the COCO numbers need re-measuring.
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
- No INT8: there is no INT8 path and no calibration set independent of the
  evaluation images.
- The detector weights carry an AGPL-3.0 obligation; see `docs/licences.md`.
- Detection is measured on photographs, not on a factory floor; geometry on a
  synthetic rig; one camera, one lidar height, no extrinsic drift detection,
  no worst-case execution time.  `docs/limitations.md` has the full list.
