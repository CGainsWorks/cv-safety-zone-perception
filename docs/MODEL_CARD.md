# Model card: the person detector

Read [SAFETY_DISCLAIMER.md](SAFETY_DISCLAIMER.md) first.  Nothing on this page
makes the detector, or the pipeline around it, a safety function.

| | |
| --- | --- |
| Model | YOLOX-Tiny (Ge et al., 2021), depth 0.33, width 0.375, 5.06 million parameters, 80 COCO classes |
| Weights | Megvii's COCO checkpoint `yolox_tiny.pth`, GitHub release `0.1.1rc0` of `Megvii-BaseDetection/YOLOX`, used as shipped; SHA-256 in `scripts/fetch_model.py` and checked on download |
| Licence | Apache-2.0, code and weights; [licences.md](licences.md) |
| Trained by | Megvii, on COCO train2017.  **Nothing is trained or fine-tuned here.** |
| Class used | 0, person; the other 79 are ignored by the decoder |
| Input | 640x640 letterbox, RGB in 0..1 |
| Output | `(1, 84, 8400)`: centre, size and 80 class scores per anchor, `obj * cls` already applied |
| Operating point | confidence 0.15, NMS IoU 0.55 |
| Replaced | Ultralytics YOLOv8n, AGPL-3.0, in October 2026 |

## What the export changes

`scripts/yolox_export.py` makes two exact rearrangements of Megvii's network
before the ONNX export.  YOLOX takes BGR in 0..255; the pipeline feeds RGB in
0..1, so the stem convolution's weights are scaled by 255 and their colour
order reversed, which is the same arithmetic in a different order.  The grid
decode and the multiplication of objectness into the class scores are built
into the graph, so the decoder in `src/szp/detection/postprocess.py` reads one
score per class with no objectness term.  Nothing else about the weights
changes.

## The variants

| Variant | Made by | Input | Notes |
| --- | --- | --- | --- |
| `yolox_tiny.onnx` | `scripts/fetch_model.py`, CPU export at opset 13 | float32 | the reference every other variant is compared with |
| `yolox_tiny_fp16.onnx` | `scripts/fetch_model.py --fp16`, exported from a half-precision module | float16 | the ONNX backend casts the letterboxed frame |
| `yolox_tiny_int8.onnx` | `scripts/quantize_onnx.py`, static QDQ, convolutions only, the nine head output convolutions left in float | float32 | calibrated on 256 COCO train2017 images |
| TensorRT FP32, FP16 and INT8 engines | `scripts/export_tensorrt.py` | float32 | INT8 uses TensorRT's entropy calibrator on 512 train2017 images; engines are never committed |

The calibration images are listed in `data/calibration_train2017.txt`, drawn
with a fixed seed from train2017 so they are disjoint from val2017, which is
what every variant is scored on.

## How it was measured

Person recall, false positives per image and log-average miss rate (MR-2) on
all 5000 COCO val2017 images, 10,777 people, by `analysis/detection_metrics.py`,
with crowd regions as ignore regions.  The **safety subset** is the people at
least 50 px tall.  The numbers are in `results/published/detection_val2017*.json`
and the README quotes them; `results/published/detector_comparison.json` puts
them beside the YOLOv8n numbers they replaced.

What changed with the swap, at the 0.15 operating point on the safety subset:

* **Recall is higher**, 89.8 % against 88.2 % and MR-2 is lower, 0.205 against
  0.227.  Small people improved most: the miss rate between 25 and 50 px fell
  from 42 % to 28 %.  People taller than 200 px are missed slightly more often,
  5.3 % against 4.6 %.
* **False positives per image are higher**, 0.67 against 0.42.  At the same
  threshold the pipeline would raise about 60 per cent more nuisance stops,
  which `analysis/FMEA.md` FM-19 treats as a safety issue in its own right
  because a guard that nuisance-trips gets bypassed.  The confidence threshold
  was not re-chosen for the new detector; the README shows the curve.
* **It is slower on the CPU**, about a fifth, because it is the larger network.
* **INT8 costs recall**: the ONNX Runtime INT8 graph loses 3.7 points of
  safety-subset recall and TensorRT's INT8 engine 2.5 points.  The previous
  detector had no INT8 variant to compare with.

## Intended use

Finding people in a camera frame as one input to the zone logic of this
engineering demonstrator.

## Out of scope

Any certified or sole protective function.  Any use where a missed person is
not caught by an independent, rated safeguard.  Factory-floor accuracy: COCO
is photographs, not an industrial cell and no number here transfers to one
without measuring there.

## Known biases and gaps

COCO's people are mostly upright, outdoors or in homes, photographed by people
who chose the shot.  Workers who are crouching, partly behind machinery, in
high-visibility clothing or at the top of a camera mounted above a cell are
under-represented and nothing here measures performance on them.  The miss
breakdown in the README shows that most misses are small people; it does not
show demographic effects, which COCO's annotations cannot support.
