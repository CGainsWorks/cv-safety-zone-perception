# cv-safety-zone-perception

Detect people, fuse a depth camera and a 2D lidar into 3D boxes on the factory
floor, test those boxes against configured safety zones and publish an
emergency-stop permit over CAN, with the whole path from frame arrival to the
frame on the bus measured.

![pipeline](docs/figures/pipeline.png)

> ### This is **not a certified safety function**
> It is an engineering demonstrator. Nothing here has been assessed against
> IEC 61508, ISO 13849 or ISO 3691-4, no Performance Level is claimed and it
> must never be the sole protective measure for any machine.
> **Read [docs/SAFETY_DISCLAIMER.md](docs/SAFETY_DISCLAIMER.md) before anything
> else.**

> ### Clean-room reimplementation
> An independent reimplementation of a problem class, built from scratch on
> personal time using public and synthetic data and open-source simulation. It
> is not derived from and contains no material belonging to, any employer or
> client: no employer source code, schematics, calibration data, client data or
> internal documentation.

---

## Run it

```bash
scripts/setup.sh --export        # venv, dependencies and the ONNX detector
sudo scripts/setup_vcan.sh       # a real Linux virtual CAN interface
make demo                        # the whole pipeline, end to end
```

or, without touching the host:

```bash
docker compose -f docker/docker-compose.yml up --build
```

which brings up a `vcan0` interface, runs the pipeline on it and dumps the
e-stop frames with `candump` so they can be seen rather than taken on trust.

`make check` runs exactly what CI runs: lint, the test suite, the pipeline end
to end over vcan, the calibration self-check, the FMEA consistency check and
the check that every number below comes from a file in `results/published/`.

---

## The one idea

**A missed person is the only failure that matters.** Precision is close to
irrelevant beside it. Every design decision in this repository that traded
recall against something else went to recall and the cost is written down.

| Decision | The usual choice | Why this one |
|---|---|---|
| Confidence threshold **0.15** | 0.25 | Measured: on the safety subset the miss rate at 0.25 is about a third higher than at 0.15. The cost is twice the nuisance stops, reported next to it |
| **Nearest dense depth cluster** | Median depth in the box | The median of a person-shaped box is the wall behind them, which puts people metres too far away and too far away means outside the zone |
| Lidar association is a **soft cost** | A hard range gate | A depth camera that is systematically wrong would gate away every lidar cluster that contradicts it |
| On disagreement, **take the nearer** | Take the more confident | There is no way to tell which sensor is wrong and the nearer reading is the one that stops the machine |
| Unranged detections become **wedges** | Discard them | A detection with no range is still a person. The wedge is the floor they could be standing on |
| Footprints dilated by **k·σ** | Use the point estimate | Uncertainty must make a person occupy more floor, not less |
| Enter on **1** frame, exit after **5** | Symmetric debounce | Debouncing entry adds latency to the one transition where latency costs something |
| STOP **latches** | Clears when the zone clears | A person is what stops the machine; an operator is what restarts it |
| The CAN frame is a **permit** | A stop command | A command that fails to arrive leaves the machine running. A permit that fails to arrive stops it |

[docs/design.md](docs/design.md) works through each of these.

---

## What was measured

Two separate bodies of evidence, kept apart on purpose.

### 1. Detection, on real photographs

YOLOv8n exported to ONNX, run over **COCO val2017**: 5,000 images,
**10,777** annotated people. Crowd regions are ignore regions, not misses.

![detection](docs/figures/detection.png)

The **safety subset** is the **7,098** people at least 50 px tall. At the
working distance of the reference cell, anyone smaller is beyond the range the
sensors cover anyway. Both are reported because the difference is the point.

| At confidence | Recall, safety subset | Recall, whole set | False positives per image |
|---|---|---|---|
| 0.15 (the operating point) | **88.2 %** | 72.7 % | 0.42 / 0.45 |
| 0.25 (the usual default) | 83.9 % | 66.9 % | 0.21 / 0.22 |

Log-average miss rate (MR-2, the standard pedestrian metric): **0.227** on the
safety subset, 0.416 on the whole set.

Where the misses are, at 0.15: **71 % of every missed person is under 50 px
tall**. Above 200 px the miss rate is 5 %; between 25 and 50 px it is 42 %.
That is the shape a guarding application needs to know and it is the reason
the operating point is where it is.

Reproduce: `make data && make detection`.

### 2. Geometry and safety logic, on a synthetic cell

COCO has no depth, no lidar and no metric ground truth and no public dataset
pairs a registered RGB-D camera, a knee-height 2D lidar and surveyed 3D person
positions in one cell. So the geometry is measured on a rig where the true
position of every person is known exactly, with depth noise, depth dropout,
lidar dropout and detector faults all modelled and swept.

![cell](docs/figures/cell.png)

| Case | p50 position error | Stop-zone frames caught |
|---|---|---|
| Depth and lidar, healthy | **0.7 cm** | **111/111** |
| **Neither ranging sensor** | 18.2 cm | **111/111** |
| Bodies a third wider than assumed | 8.7 cm | 115/115 |

The middle row is the one worth pausing on. With the depth camera and the
lidar both gone, every detection becomes an unranged wedge covering the floor
the person could be standing on and **not one occupied frame was missed**.
A pipeline that discarded unrangeable detections would have reported an empty
cell.

The 0.7 cm is flattering and says so: the rig's people are cylinders of exactly
the radius the bias corrections assume. The third row breaks that assumption on
purpose and it is the more honest number.

Reproduce: `make geometry`.

### 3. Latency, frame arrival to e-stop frame

![timeline](docs/figures/timeline.png)

With the real ONNX model in the loop on a CPU:

| | p50 | p99 |
|---|---|---|
| Pipeline: frame arrival to the frame on the bus | **31.1 ms** | 36.8 ms |
| Reaction: the same, measured at the receiving PLC | 34.5 ms | **36.8 ms** |
| Detector inference alone | 26.3 ms | 29.0 ms |

The detector is **83 %** of the budget. Everything after it (fusion, zones,
the state machine, the CAN encode) is around two milliseconds together: with
the model taken out entirely the p99 is 6.5 ms. Optimising this pipeline means
optimising the detector and the measurements say so rather than intuition.

What is **not** in those numbers: camera exposure, driver latency and the PLC
scan cycle, which on real hardware are often larger than everything above.
[docs/latency.md](docs/latency.md) says which numbers to substitute where and
[docs/zones.md](docs/zones.md) works the ISO 13855 sum through.

Reproduce: `make latency`.

### 4. Calibration

The camera-to-lidar extrinsic is solved by the planar-target method of Zhang
and Pless, from a checkerboard alone. `scripts/calibrate.py --self-check`
generates board observations from a transform it chose and asks the solver to
recover it: with seven poses at 10 mm of point noise it comes back to
**8.9 mm** and the nonlinear refinement takes the residual from 46.7 mm to
**8.5 mm**.

Degenerate pose sets are **refused**, with an explanation, rather than fitted.
A confident-looking transform from a rank-deficient system is how a systematic
half-metre ranging error becomes invisible.

Reproduce: `make calibrate`.

---

## The machine interface

The e-stop frame is a permit, not a command:

> The PLC enables motion only while it is receiving a `SZP_ESTOP` frame that
> arrived within 100 ms, passes its CRC, has an `alive` counter that has
> advanced and carries `CLEAR` or `WARN`.

So a pipeline that crashes, hangs, loses power or is unplugged sends nothing,
the permit lapses and the machine stops. **There is no message this pipeline
can fail to send that leaves the machine running.**

Each frame carries a rolling counter and a CRC-8/AUTOSAR over the payload with
a per-message data identifier mixed in, which catches three distinct failures:
corruption past the CAN CRC, a sender frozen with its last frame repeating out
of a buffer and a frame decoded under the wrong arbitration ID. The test suite
flips **every single bit of every byte** and requires all of them to be caught.

Full specification: [docs/can_protocol.md](docs/can_protocol.md).

---

## FMEA

[analysis/FMEA.md](analysis/FMEA.md): 22 failure modes, scored, with the
mitigation in this repository and the residual risk named for each. Entries
marked *measured* cite the script or the test that produced the number and
`analysis/make_fmea_summary.py` fails CI if an RPN stops matching S×O×D or if a
cited test no longer exists.

The top of the table by RPN is the detector (FM-01, FM-02) and the machine's
stopping time (FM-22). Two of those are properties of monocular detection and
the third is a number that has to come from real plant. The bottom of the table
is the CAN protocol, where every failure has a cheap mitigation and a test. That
asymmetry is the honest summary of where the risk in this design sits.

---

## Limitations

The short version: the detection numbers are on photographs and not on a
factory floor; the geometry numbers are on a synthetic rig; there is no
tracking, one camera and a lidar that sees one height; people are assumed to
stand on the floor; nothing detects extrinsic drift at run time; there is no
worst-case execution time; and the machine stopping time is an assumption.

The long version, including what would be next and in what order:
[docs/limitations.md](docs/limitations.md).

---

## Layout

```
src/szp/
  types.py          the data between stages; coordinate frames defined once
  geometry.py       SE(3), strict about orthonormality
  detection/        letterbox, YOLO decode, ONNX and TensorRT backends
  fusion/           depth ranging, scan segmentation, calibration, 3D boxes
  zones/            polygon geometry, the zone file format, occupancy
  safety/           CAN frame format, watchdog, fail-safe FSM, simulated PLC
  latency/          per-stage instrumentation
  data/             COCO loader and the synthetic cell with ground truth
  pipeline.py       one frame in, one CAN frame out
config/             zones.yaml, extrinsics.yaml
analysis/           the scripts every published number comes from and the FMEA
scripts/            setup, fetch, demo, calibrate, export, benchmark, figures
results/published/  the result files the README quotes
docs/               disclaimer, design, zones, latency, calibration, CAN, licences
```

`src/szp/…` rather than the flat `src/detection/…` the brief proposed: a flat
`src/` is not importable as a package without path games. That is the only
deviation.

---

## Licence

MIT, in [LICENSE](LICENSE).

The detector weights come from ultralytics, which is **AGPL-3.0**. It is used
exactly once, by `scripts/fetch_model.py`, to produce the ONNX file and it is
in `requirements-export.txt` and not in `requirements.txt`. Nothing under
`src/` imports it or torch. CI **fails the build** if either turns up in the
runtime Docker image. [docs/licences.md](docs/licences.md) sets out what that
does and does not buy you and what to do if you want to use this commercially.

Datasets and their terms: [data/DATASET_CARD.md](data/DATASET_CARD.md). Nothing
is redistributed by this repository and nothing here is trained.
