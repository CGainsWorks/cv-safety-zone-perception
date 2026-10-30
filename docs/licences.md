# Licences

## This repository

MIT, in [LICENSE](../LICENSE).

## The detector weights and why ultralytics is not a runtime dependency

The person detector is YOLOv8n, whose published checkpoint and the
`ultralytics` package that produced it are **AGPL-3.0**. That matters and the
repository is arranged around it.

`ultralytics` is used exactly once, by `scripts/fetch_model.py`, to turn the
published checkpoint into an ONNX graph. It is in `requirements-export.txt` and
**not** in `requirements.txt`. Nothing under `src/` imports it and nothing
under `src/` imports torch either. The runtime is ONNX Runtime, numpy, OpenCV,
python-can, scipy and PyYAML, all permissively licensed.

```
requirements.txt          numpy, scipy, opencv-python-headless, pyyaml,
                          python-can, matplotlib, tqdm, onnxruntime
                          -> what the pipeline needs to run

requirements-export.txt   the above, plus ultralytics (AGPL-3.0), onnx, onnxslim
                          -> what you need once, to produce models/yolov8n.onnx
```

The Docker images follow the same split: `docker/Dockerfile` installs the
runtime only and `docker/Dockerfile.export` is a separate image for the export
step.

### The obligation, stated plainly

**The detector weights are AGPL-3.0 and that blocks closed commercial use.**
`models/yolov8n.onnx` is Ultralytics' published COCO checkpoint
(`yolov8n.pt`, AGPL-3.0) converted to ONNX; it is not retrained here, so it is
the same weights in a different file format.  Ultralytics licenses its models
under AGPL-3.0 and offers an Enterprise Licence for use outside it.  So any
deployment of this pipeline with these weights, including one reached over a
network, must meet AGPL-3.0: the complete corresponding source of the deployed
system, under AGPL-3.0, to every user who interacts with it.  A closed or
commercial deployment needs an Ultralytics Enterprise Licence or a different
detector.  The MIT licence on this repository covers the code written here; it
does not relicense the weights and cannot.

An earlier version of this section said the keeping of ultralytics out of the
runtime "keeps the deployed image free of AGPL code" and left the weights as an
open question.  The first half is true of the code and the second half is not
open: the weights are Ultralytics' AGPL-3.0 model.

### What the runtime split does buy

It keeps Ultralytics' *code* and torch out of the deployed image, which is
worth having because it is the part a detector swap would not need to touch.
It does not remove the obligation attached to the weights, which travel inside
the ONNX graph whichever runtime loads it, TensorRT included.

**The fix is a detector that is permissively licensed end to end, weights
included:** YOLOX, RTMDet or D-FINE, all Apache-2.0.  `STATUS.md` carries it as
an open item; it has not been done in this pass, so the detector and its
numbers are unchanged.  The pipeline is ready for it:
`src/szp/detection/base.py` takes any ONNX graph and the decoder in
`postprocess.py` has a `class_index` argument for models whose person class is
not zero.  A detector whose head is not YOLOv8-shaped (YOLOX, RTMDet) needs a
second decoder there.

## Datasets

Detection accuracy is measured on **COCO val2017**.

| Part | Licence |
|---|---|
| Annotations | CC BY 4.0, COCO Consortium |
| Images | Flickr photographs under a mixture of Creative Commons terms, made available by the COCO Consortium under their terms of use |

Nothing from COCO is committed to this repository or redistributed by it.
`scripts/fetch_data.py` downloads it to a directory outside the repository and
`data/DATASET_CARD.md` records what is used and how. This repository **trains
nothing**; COCO is used for evaluation only, which is the least contentious use
of it.

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
