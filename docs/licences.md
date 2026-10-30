# Licences

## This repository

MIT, in [LICENSE](../LICENSE).

## The detector: YOLOX-Tiny, Apache-2.0

The person detector is YOLOX-Tiny (Ge et al., 2021), Megvii's COCO checkpoint
used as shipped.  The code that defines and exports it and the checkpoint
itself are **Apache-2.0**, so the deployed pipeline carries no copyleft
obligation and closed commercial use is permitted.

| Part | Licence | Where it comes from |
|---|---|---|
| `yolox` 0.3.0, the network definition | Apache-2.0, Copyright 2021-2022 Megvii Inc. | PyPI, `requirements-yolox.txt`, export time only |
| `yolox_tiny.pth`, the COCO checkpoint | Apache-2.0, as a release asset of the YOLOX repository | GitHub release `0.1.1rc0` of `Megvii-BaseDetection/YOLOX`; URL and SHA-256 in `scripts/fetch_model.py` |
| `models/yolox_tiny*.onnx` and any engine built from them | Apache-2.0, as a modification of the above | `scripts/fetch_model.py`, `scripts/quantize_onnx.py`, `scripts/export_tensorrt.py` |

Nothing is retrained.  The ONNX graph is Megvii's weights with two exact
rearrangements, both in `scripts/yolox_export.py`: the input normalisation
folded into the stem convolution and the box decode built into the graph.
Apache-2.0 asks anyone who redistributes the weights, or a product containing
them, to include a copy of the licence and the attribution and to say that the
files were modified.  The licence text as Megvii ships it, with its copyright
line, is [LICENSES/YOLOX-Apache-2.0.txt](../LICENSES/YOLOX-Apache-2.0.txt); the
YOLOX repository has no NOTICE file.  This repository does not redistribute
the weights at all: `scripts/fetch_model.py` downloads them from Megvii's
release and checks the digest.

**One thing to check rather than assume.**  Megvii publishes the checkpoints as
release assets of an Apache-2.0 repository and states no separate terms for
them, so this repository reads them as Apache-2.0.  That is the plain reading
but it is a reading of what Megvii published rather than a statement Megvii
made about the weights specifically and anyone shipping a product on them
should have it confirmed by their own counsel.

**The training data behind the weights.**  Megvii trained the checkpoint on
COCO train2017.  That is a provenance question every COCO-trained detector
shares, the previous one included and it is not a licence on the weights.

### Why YOLOX, rather than RTMDet or D-FINE

All three are Apache-2.0.  YOLOX fits this pipeline without changing what it
measures: it is anchor-free on the same stride 8, 16 and 32 grids as the
previous detector, so with the grid decode and `obj * cls` scoring done in the
exported graph it emits the `(1, 4 + 80, 8400)` head that
`src/szp/detection/postprocess.py` already decoded and both backends, the
equivalence test and the latency instrumentation are unchanged.  Its graph is
plain convolutions and elementwise arithmetic, which TensorRT's INT8
calibrator handles; D-FINE's deformable attention samples features with grid
sampling, the classic failure case for implicit INT8 and emits a fixed set of
queries rather than the dense, NMS-based output the recall sweep and the
confidence operating point are built around.  RTMDet lives in MMDetection,
whose compiled `mmcv` operators are pinned to specific torch and CUDA pairs.
The same detector is used in the sibling repository
[cv-yolo-detect-track-jetson](https://github.com/CGainsWorks/cv-yolo-detect-track-jetson),
where ByteTrack, which was designed around YOLOX, consumes it.

### History: Ultralytics YOLOv8, AGPL-3.0

Until October 2026 the detector was Ultralytics' YOLOv8n COCO checkpoint.
Ultralytics distributes the library and the checkpoint under AGPL-3.0 and
offers an Enterprise Licence for use outside it, so every deployment of the
pipeline with those weights, including one reached over a network, had to meet
AGPL-3.0 and closed commercial use was blocked.  Keeping `ultralytics` out of
the runtime image kept its code out but not the obligation, which travelled
with the weights inside the ONNX graph.  Replacing the detector was the fix.
`ultralytics` stays on the list the `docker` CI job refuses to find in the
runtime image, so it cannot come back by accident.

## Why torch and yolox are not runtime dependencies

`torch` and `yolox` are used once, by `scripts/fetch_model.py`, to turn the
checkpoint into ONNX graphs.  They are in `requirements-export.txt` and
`requirements-yolox.txt` and **not** in `requirements.txt`.  Nothing under
`src/` imports either.  With an Apache-2.0 detector this is no longer a licence
boundary; it is a size one, because torch is two gigabytes of wheel the
deployed image has no use for.

```
requirements.txt          numpy, scipy, opencv-python-headless, pyyaml,
                          python-can, matplotlib, tqdm, onnxruntime
                          -> what the pipeline needs to run

requirements-export.txt   the above, plus torch, onnx, onnxslim and yolox's
requirements-yolox.txt    own dependencies, then yolox itself
                          -> what you need once, to produce models/yolox_tiny.onnx
```

The Docker images follow the same split: `docker/Dockerfile` installs the
runtime only and `docker/Dockerfile.export` is a separate image for the export
step.  The `docker` CI job imports torch, yolox and ultralytics inside the
runtime image and fails if any succeeds.

## Datasets

Detection accuracy is measured on **COCO val2017**.  INT8 is calibrated on a
fixed list of COCO **train2017** images (`data/calibration_train2017.txt`), so
the calibration set and the evaluation set are disjoint.

| Part | Licence |
|---|---|
| Annotations | CC BY 4.0, COCO Consortium |
| Images | Flickr photographs under a mixture of Creative Commons terms, made available by the COCO Consortium under their terms of use |

Nothing from COCO is committed to this repository or redistributed by it; the
calibration list is file names only.  `scripts/fetch_data.py` and
`scripts/fetch_calibration.py` download to a directory outside the repository
and `data/DATASET_CARD.md` records what is used and how.  This repository
**trains nothing**; COCO is used for evaluation and INT8 calibration only.

The geometry, zone and latency measurements use the synthetic rig in
`src/szp/data/synthetic.py`, which is original code and generates its own data.

## Third-party runtime dependencies

| Package | Licence |
|---|---|
| numpy | BSD-3-Clause |
| scipy | BSD-3-Clause |
| opencv-python-headless | Apache-2.0 (OpenCV 4.5 and later) |
| onnxruntime | MIT |
| python-can | LGPL-3.0 |
| PyYAML | MIT |
| matplotlib | PSF-based, BSD-compatible |
| tqdm | MPL-2.0 and MIT |

python-can is LGPL-3.0. It is used as an unmodified library through its public
API, which is the case the LGPL is written for. If you static-link or vendor a
modified copy, read the licence.

## Export time only

| Package | Licence |
|---|---|
| torch, torchvision | BSD-3-Clause |
| yolox | Apache-2.0 |
| onnx | Apache-2.0 |
| onnxslim | MIT |
| loguru, tabulate, thop | MIT |
| psutil | BSD-3-Clause |
| pycocotools | BSD-2-Clause |

## Optional, GPU only

| Package | Licence |
|---|---|
| tensorrt | NVIDIA proprietary; see the SDK EULA |
| pycuda | MIT |
| onnxruntime-gpu | MIT, with CUDA under NVIDIA's terms |

None of these is installed by `requirements.txt` or needed to run the
pipeline.  The desktop TensorRT measurement takes TensorRT and pycuda from
NVIDIA's own `nvcr.io/nvidia/tensorrt` container (`docker/Dockerfile.tensorrt`)
and CI installs NVIDIA's `tensorrt-cu12` wheel at run time
(`requirements-tensorrt.txt`); nothing here redistributes them.

## Attribution

- Ge, Liu, Wang, Li and Sun, *YOLOX: Exceeding YOLO Series in 2021*, arXiv
  2107.08430 -- the detector, used through Megvii's `yolox` package and
  checkpoint under Apache-2.0.
- Zhang and Pless, *Extrinsic calibration of a camera and laser range finder*,
  IROS 2004 -- the camera-lidar calibration method in
  `src/szp/fusion/calibration.py`.
- Borges and Aldon, *Line extraction in 2D range images for mobile robotics*,
  JIRS 2004 -- the adaptive breakpoint detector in `src/szp/fusion/lidar.py`.
- Dollar and others, *Pedestrian detection: an evaluation of the state of the
  art*, PAMI 2012 -- the log-average miss rate (MR-2) used in
  `analysis/detection_metrics.py`.
- AUTOSAR E2E Profile 1 -- the counter-and-CRC scheme in
  `src/szp/safety/can_frames.py`. The implementation is written from the
  description; no AUTOSAR code is included.
