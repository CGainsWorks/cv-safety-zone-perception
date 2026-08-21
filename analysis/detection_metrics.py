#!/usr/bin/env python3
"""Person-detection recall on COCO val2017, with the misses broken down.

    python analysis/detection_metrics.py --out results/detection.json

Why recall and not mAP
----------------------
mAP is the number a detection paper reports and it is the wrong number here.
It averages over confidence thresholds the system will never run at, it rewards
precision that this application does not care about and a single figure cannot
say *which* people were missed.  What sizes the risk in a guarding application
is: at the confidence the machine actually runs at, what fraction of people did
the detector not see and were they the people who mattered?

So this script reports

* **recall against confidence**, so the operating point is a choice with a
  visible cost rather than a default;
* **log-average miss rate against false positives per image** (MR-2), the
  standard pedestrian-detection curve, which is the one metric that puts the
  two error types on the same axis;
* **misses broken down** by pixel height, by occlusion proxy, by truncation and
  by crowding, because "8 per cent of people were missed" and "8 per cent of
  people were missed and all of them were four pixels tall" are different
  findings.

The safety subset
-----------------
COCO is full of people who are twelve pixels tall in the background of a
photograph of a sandwich.  A guarding system does not need to see them: at the
working distance of the cell in ``config/zones.yaml`` a person subtends at
least ``--min-height`` pixels and anything smaller is beyond the range the
sensors cover anyway.  Both the whole set and the subset are reported and the
README quotes the subset because it is the honest one for the application --
with the full-set number next to it so the reader can see the difference.

Crowd regions are excluded rather than counted.  A COCO ``iscrowd`` polygon is
a mass of people with no individual boxes; counting it as one miss or as one
false positive would both be wrong, so detections landing inside one are
dropped before scoring and the region itself is never expected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402

from szp.data.coco import CocoImage, CocoPersonDataset  # noqa: E402
from szp.detection.base import DetectorConfig, build_detector  # noqa: E402
from szp.detection.postprocess import iou_matrix  # noqa: E402
from szp.types import Detection2D  # noqa: E402

#: Confidence values the sweep reports.  The pipeline default sits in here.
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70)

HEIGHT_BUCKETS = ((0, 25), (25, 50), (50, 100), (100, 200), (200, 10**6))


def match(
    detections: Sequence[Detection2D],
    gt_boxes: np.ndarray,
    ignore_boxes: np.ndarray,
    iou_threshold: float,
):
    """Greedy highest-score-first matching.

    Returns ``(matched_gt, false_positives, ignored)``.  ``ignored`` counts
    detections that landed inside an ignore region -- a crowd polygon or a
    person below the subset's height floor -- which are neither true positives
    nor false positives.
    """
    if len(detections) == 0:
        return np.zeros(len(gt_boxes), dtype=bool), 0, 0
    boxes = np.array([d.xyxy for d in detections], dtype=np.float64)
    order = np.argsort([-d.score for d in detections])
    matched = np.zeros(len(gt_boxes), dtype=bool)
    ious = iou_matrix(boxes, gt_boxes) if len(gt_boxes) else np.zeros((len(boxes), 0))
    crowd_ious = (
        _crowd_overlap(boxes, ignore_boxes) if len(ignore_boxes) else np.zeros(len(boxes))
    )
    false_positives = 0
    ignored = 0
    for index in order:
        best, best_iou = -1, iou_threshold
        for g in range(len(gt_boxes)):
            if matched[g]:
                continue
            if ious[index, g] >= best_iou:
                best, best_iou = g, ious[index, g]
        if best >= 0:
            matched[best] = True
        elif crowd_ious[index] > 0.5:
            ignored += 1
        else:
            false_positives += 1
    return matched, false_positives, ignored


def _crowd_overlap(boxes: np.ndarray, crowd_boxes: np.ndarray) -> np.ndarray:
    """Fraction of each detection's own area that lies inside a crowd region."""
    lt = np.maximum(boxes[:, None, :2], crowd_boxes[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], crowd_boxes[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = (wh[..., 0] * wh[..., 1]).max(axis=1)
    area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(area > 0, inter / area, 0.0)


def log_average_miss_rate(miss_rates: Sequence[float], fppi: Sequence[float]) -> float:
    """MR-2: the log-average miss rate over nine FPPI points from 1e-2 to 1e0.

    This is the Caltech/CityPersons convention.  Points beyond the measured
    range are filled with the miss rate at the nearest measured point, which is
    what the reference implementation does; a curve that never reaches 1 FPPI
    is therefore scored optimistically and the number of points that had to be
    extrapolated is reported alongside.
    """
    miss_rates = np.asarray(miss_rates, dtype=np.float64)
    fppi = np.asarray(fppi, dtype=np.float64)
    order = np.argsort(fppi)
    fppi, miss_rates = fppi[order], miss_rates[order]
    reference = np.logspace(-2.0, 0.0, 9)
    sampled = []
    extrapolated = 0
    for point in reference:
        candidates = np.flatnonzero(fppi <= point)
        if len(candidates) == 0:
            sampled.append(miss_rates[0])
            extrapolated += 1
        else:
            sampled.append(miss_rates[candidates[-1]])
            if point > fppi[-1]:
                extrapolated += 1
    sampled = np.clip(np.asarray(sampled), 1e-10, 1.0)
    return float(np.exp(np.mean(np.log(sampled)))), extrapolated


def bucket_of(height: float) -> str:
    for low, high in HEIGHT_BUCKETS:
        if low <= height < high:
            return f"{low}-{high if high < 10**5 else 'inf'}px"
    return "unknown"


def crowding(gt_boxes: np.ndarray, index: int, threshold: float = 0.1) -> int:
    """How many other annotated people overlap this one."""
    if len(gt_boxes) < 2:
        return 0
    ious = iou_matrix(gt_boxes[index: index + 1], gt_boxes)[0]
    return int(((ious > threshold).sum()) - 1)


def evaluate(
    dataset: CocoPersonDataset,
    detector,
    *,
    iou_threshold: float,
    thresholds: Sequence[float],
    min_height_px: float,
    limit: Optional[int],
    progress: bool,
) -> Dict[str, object]:
    per_image: List[Dict[str, object]] = []
    images = [image for image in dataset if image.n_people or True]
    if limit:
        images = images[:limit]

    started = time.perf_counter()
    for count, image in enumerate(images):
        frame = cv2.imread(str(image.path))
        if frame is None:
            continue
        detections = detector.detect(frame)
        per_image.append({"image": image, "detections": detections})
        if progress and count % 200 == 0:
            elapsed = time.perf_counter() - started
            rate = (count + 1) / max(elapsed, 1e-9)
            print(
                f"  {count + 1}/{len(images)} images, {rate:.1f}/s, "
                f"eta {(len(images) - count - 1) / max(rate, 1e-9):.0f}s",
                flush=True,
            )

    results: Dict[str, object] = {
        "images_scored": len(per_image),
        "iou_threshold": iou_threshold,
        "min_height_px": min_height_px,
        "detector": dict(detector.info),
        "sweep": [],
    }

    for subset_name, height_floor in (("all", 0.0), ("safety_subset", min_height_px)):
        sweep = []
        for threshold in thresholds:
            sweep.append(
                _score(per_image, threshold, iou_threshold, height_floor)
            )
        miss = [1.0 - row["recall"] for row in sweep]
        fppi = [row["false_positives_per_image"] for row in sweep]
        mr2, extrapolated = log_average_miss_rate(miss, fppi)
        results[subset_name] = {
            "height_floor_px": height_floor,
            "people": sweep[0]["people"],
            "sweep": sweep,
            "mr2": mr2,
            "mr2_extrapolated_points": extrapolated,
        }
    results["misses"] = _miss_breakdown(per_image, thresholds, iou_threshold, min_height_px)
    return results


def _score(per_image, threshold, iou_threshold, height_floor):
    people = 0
    found = 0
    false_positives = 0
    ignored = 0
    for row in per_image:
        image: CocoImage = row["image"]
        detections = [d for d in row["detections"] if d.score >= threshold]
        keep = image.heights >= height_floor if image.n_people else np.zeros(0, dtype=bool)
        gt = image.boxes[keep] if image.n_people else np.zeros((0, 4))
        # People below the height floor become ignore regions rather than
        # disappearing.  Dropping them outright would count a correct detection
        # of a small person as a false positive, which inflates the false
        # positive rate of the subset and makes its MR-2 worse than the full
        # set's -- an artefact of the scoring, not of the detector.
        ignore = image.crowd_boxes
        if image.n_people and not keep.all():
            ignore = np.concatenate([ignore, image.boxes[~keep]], axis=0)
        matched, fp, ig = match(detections, gt, ignore, iou_threshold)
        people += len(gt)
        found += int(matched.sum())
        false_positives += fp
        ignored += ig
    n_images = max(len(per_image), 1)
    return {
        "conf": threshold,
        "people": people,
        "found": found,
        "missed": people - found,
        "recall": found / people if people else 0.0,
        "false_positives": false_positives,
        "false_positives_per_image": false_positives / n_images,
        "detections_in_crowd_ignored": ignored,
        "precision": found / max(found + false_positives, 1),
    }


def _miss_breakdown(per_image, thresholds, iou_threshold, min_height_px):
    """Where the misses are, at each threshold."""
    out = {}
    for threshold in thresholds:
        by_height: Dict[str, Dict[str, int]] = {}
        truncated = [0, 0]
        crowded = [0, 0]
        small = 0
        total_missed = 0
        for row in per_image:
            image: CocoImage = row["image"]
            if image.n_people == 0:
                continue
            detections = [d for d in row["detections"] if d.score >= threshold]
            matched, _, _ = match(detections, image.boxes, image.crowd_boxes, iou_threshold)
            border = image.touches_border()
            for index, was_found in enumerate(matched):
                height = float(image.heights[index])
                bucket = by_height.setdefault(bucket_of(height), {"people": 0, "missed": 0})
                bucket["people"] += 1
                slot = 1 if border[index] else 0
                is_crowded = crowding(image.boxes, index) > 0
                if not was_found:
                    bucket["missed"] += 1
                    total_missed += 1
                    truncated[slot] += 1
                    if is_crowded:
                        crowded[1] += 1
                    else:
                        crowded[0] += 1
                    if height < min_height_px:
                        small += 1
        out[f"{threshold:.2f}"] = {
            "by_height": {
                key: dict(value, miss_rate=value["missed"] / max(value["people"], 1))
                for key, value in sorted(by_height.items())
            },
            "missed_total": total_missed,
            "missed_untruncated": truncated[0],
            "missed_truncated": truncated[1],
            "missed_uncrowded": crowded[0],
            "missed_crowded": crowded[1],
            "missed_below_min_height": small,
            "missed_below_min_height_fraction": small / max(total_missed, 1),
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--coco-root", default=None)
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N images")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--min-height",
        type=float,
        default=50.0,
        help="pixel height below which a person is outside the sensor's working range",
    )
    parser.add_argument("--conf-floor", type=float, default=0.05)
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    dataset = CocoPersonDataset(args.coco_root, args.split)
    if not args.quiet:
        print(json.dumps(dataset.summary(), indent=2))

    detector = build_detector(args.model, DetectorConfig(conf_threshold=args.conf_floor))
    detector.warmup(5)
    try:
        results = evaluate(
            dataset,
            detector,
            iou_threshold=args.iou,
            thresholds=THRESHOLDS,
            min_height_px=args.min_height,
            limit=args.limit,
            progress=not args.quiet,
        )
    finally:
        detector.close()
    results["dataset"] = dataset.summary()

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"wrote {path}")

    if not args.quiet:
        for subset in ("all", "safety_subset"):
            block = results[subset]
            print(f"\n{subset}: {block['people']} people, MR-2 {block['mr2']:.4f}")
            print("   conf  recall   missed   FP/image  precision")
            for row in block["sweep"]:
                print(
                    f"  {row['conf']:5.2f}  {row['recall']:6.3f}  {row['missed']:7d}"
                    f"  {row['false_positives_per_image']:9.3f}  {row['precision']:9.3f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
