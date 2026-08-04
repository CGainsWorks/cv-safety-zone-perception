# cv-safety-zone-perception

> **Status: planned. This repository is a brief, not an implementation.**
> Nothing here is built yet. The sections below define what goes in it.

> ### Clean-room reimplementation
> This repository is an **independent reimplementation of a problem class**, built from
> scratch on personal time using public or synthetic data and open-source simulation.
> It is **not** derived from, and contains no material belonging to, any employer or
> client: no employer source code, schematics, calibration data, client data or internal
> documentation. It exists to demonstrate the architecture and engineering approach only.

A perception pipeline that detects people, fuses depth-camera and 2D LiDAR data into 3D bounding boxes in world coordinates, evaluates occupancy against configured safety zones and publishes emergency-stop frames over a virtual CAN interface. Built with public pedestrian datasets and a simulated PLC. This is a demonstration of architecture and latency engineering, not a certified safety function.

---

## What this covers

CV entries this repository is the evidence for:

- Real-time human-detection safety system with CAN-bus e-stop (Printed Electronics)

---

## Requirements

### Hardware

- None required for the clean-room build
- Optional physical target: NVIDIA Jetson Xavier NX, depth camera, 2D LiDAR, CAN interface
- Simulation substitute: public pedestrian datasets plus Linux vcan (virtual CAN)

### Software

- YOLOv8 or a person-detection model
- CUDA and TensorRT
- Depth and LiDAR fusion into 3D boxes
- Zone definition and occupancy logic
- python-can over vcan
- Fail-safe and error-recovery state machine
- Docker

### Report and documentation deliverables

- Detection accuracy with explicit false-negative analysis
- End-to-end latency budget from frame arrival to e-stop frame
- Sensor extrinsic calibration
- FMEA table
- Zone configuration guide
- Prominent statement that this is not a certified safety function

---

## Inputs

- RGB frames
- Depth maps
- 2D LiDAR scans
- Sensor extrinsics
- Safety zone polygon definitions
- Heartbeat signals

## Outputs

- 2D detections
- Fused 3D bounding boxes in world coordinates
- Zone occupancy state
- CAN e-stop frames
- Health and watchdog status
- Latency telemetry

---

## Intended structure

```
src/
  detection/              person detector, TensorRT export
  fusion/                 depth + LiDAR to 3D boxes
  zones/                  polygon config, occupancy evaluation
  safety/                 FSM, watchdog, CAN publisher
  latency/                instrumentation
config/                   zones.yaml, extrinsics.yaml
data/                     public dataset loaders, DATASET_CARD.md
analysis/                 latency budget, FMEA, detection metrics
docker/
docs/
  SAFETY_DISCLAIMER.md
```

Plus the standard baseline: `README.md`, `LICENSE`, `.gitignore`, `.gitattributes`
(Git LFS rules for weights, bags and media), `.github/workflows/ci.yml`, `docs/`,
`scripts/setup.sh`, `scripts/run_demo.sh` and `tests/`.

---

## Build checklist

- [ ] Person detection on a public dataset with reported recall
- [ ] TensorRT export and latency measurement
- [ ] Depth and LiDAR extrinsic calibration
- [ ] Fusion producing 3D boxes in world coordinates
- [ ] Zone polygon config format and occupancy evaluation
- [ ] vcan e-stop publisher with a defined frame format
- [ ] Watchdog and fail-safe state machine
- [ ] End-to-end latency instrumentation and budget
- [ ] FMEA table and false-negative analysis

---

## Definition of done

This repository is finished when:

- [ ] The whole thing runs from one command on a clean machine.
- [ ] The README opens with a diagram or a GIF, not a wall of text.
- [ ] Every number quoted in the README is reproducible from a script in the repo.
- [ ] Limitations are stated honestly in their own section.
- [ ] A licence is declared.
- [ ] Large binaries live in Releases or LFS, not in the Git history.
- [ ] The repository URL is added to the matching CV bullet point.

---

## Starter prompt

Paste this into your coding agent of choice to begin.

<details>
<summary>Expand</summary>

```
You are helping me build a portfolio project from scratch. Read this brief and then
propose a concrete implementation plan before writing any code.

PROJECT: cv-safety-zone-perception
A perception pipeline that detects people, fuses depth-camera and 2D LiDAR data into 3D bounding boxes in world coordinates, evaluates occupancy against configured safety zones and publishes emergency-stop frames over a virtual CAN interface. Built with public pedestrian datasets and a simulated PLC. This is a demonstration of architecture and latency engineering, not a certified safety function.

TARGET STACK
- YOLOv8 or a person-detection model
- CUDA and TensorRT
- Depth and LiDAR fusion into 3D boxes
- Zone definition and occupancy logic
- python-can over vcan
- Fail-safe and error-recovery state machine
- Docker

HARDWARE CONTEXT
- None required for the clean-room build
- Optional physical target: NVIDIA Jetson Xavier NX, depth camera, 2D LiDAR, CAN interface
- Simulation substitute: public pedestrian datasets plus Linux vcan (virtual CAN)

THE SYSTEM MUST TAKE THESE INPUTS
- RGB frames
- Depth maps
- 2D LiDAR scans
- Sensor extrinsics
- Safety zone polygon definitions
- Heartbeat signals

AND PRODUCE THESE OUTPUTS
- 2D detections
- Fused 3D bounding boxes in world coordinates
- Zone occupancy state
- CAN e-stop frames
- Health and watchdog status
- Latency telemetry

INTENDED REPOSITORY LAYOUT
src/
  detection/              person detector, TensorRT export
  fusion/                 depth + LiDAR to 3D boxes
  zones/                  polygon config, occupancy evaluation
  safety/                 FSM, watchdog, CAN publisher
  latency/                instrumentation
config/                   zones.yaml, extrinsics.yaml
data/                     public dataset loaders, DATASET_CARD.md
analysis/                 latency budget, FMEA, detection metrics
docker/
docs/
  SAFETY_DISCLAIMER.md

CONSTRAINTS
- Everything must run from a single command: `docker compose up` or `make demo`.
- No placeholder or stub code presented as working. If something is not implemented, say so.
- Every quantitative claim in the README must be reproducible from a script in this repo.
- Use only public or synthetic data. Record the licence of every dataset used.
- This is a clean-room reimplementation. Do not reproduce any proprietary design, protocol detail or dataset. Build the equivalent capability independently.

START BY
1. Confirming or challenging the proposed repository layout.
2. Listing the specific libraries and versions you intend to pin.
3. Identifying the single riskiest technical unknown and how to de-risk it first.
Then work through the milestones in the README build checklist, one at a time.

```

</details>

---

## Notes

- Do not backdate commits to match CV dates. Publishing older work today is normal and expected.
- Record dataset licences before training on anything. Several common research datasets are
  non-commercial only.
- Flip this repository to public once there is something in it worth showing.
