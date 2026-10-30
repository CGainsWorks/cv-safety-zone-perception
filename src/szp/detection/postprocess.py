"""Decoding the detection head and non-maximum suppression.

The raw head is ``(1, 4 + n_classes, n_anchors)``: four box parameters in
letterboxed pixels as centre-x, centre-y, width, height, followed by one score
per class in 0..1.  The detector is YOLOX-Tiny and its export
(``scripts/yolox_export.py``) does the grid decode and multiplies objectness
into the class scores inside the graph, so there is no objectness term left
and no anchor decoding to do here.  The only real work is picking the person
column, thresholding and suppressing duplicates.

NMS is written out rather than taken from OpenCV so that the tie-breaking is
deterministic.  A detector whose output depends on the order equal-scoring
boxes came out of the model is a detector whose recall number cannot be
reproduced and reproducibility of the numbers is the whole point of the
analysis directory.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from szp.detection.preprocess import LetterboxTransform
from szp.types import Detection2D

__all__ = ["decode_head", "nms", "boxes_to_detections", "xywh_to_xyxy", "iou_matrix"]

PERSON_CLASS_INDEX = 0


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2.0
    out[:, 1] = b[:, 1] - b[:, 3] / 2.0
    out[:, 2] = b[:, 0] + b[:, 2] / 2.0
    out[:, 3] = b[:, 1] + b[:, 3] / 2.0
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, M) intersection over union between two sets of xyxy boxes."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    """Greedy NMS.  Ties broken by box index, so the result is deterministic."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) == 0:
        return []
    order = np.lexsort((np.arange(len(scores)), -scores))
    keep: List[int] = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        rest = order[(~suppressed[order]) & (order != i)]
        if len(rest) == 0:
            continue
        overlaps = iou_matrix(boxes[i : i + 1], boxes[rest])[0]
        suppressed[rest[overlaps > iou_threshold]] = True
    return keep


def decode_head(
    output: np.ndarray,
    transform: LetterboxTransform,
    *,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
    class_index: int = PERSON_CLASS_INDEX,
    min_box_px: float = 1.0,
    channels_first: Optional[bool] = None,
) -> List[Detection2D]:
    """Raw model output to person detections in original-image pixels.

    ``channels_first`` says whether the head is ``(4 + n_classes, n_anchors)``
    or the transpose.  Left as ``None`` it is inferred from the shape, on the
    rule that a detection head always has far more anchors than channels --
    8400 against 84 at a 640 input.  The argument exists so the tests can be
    explicit rather than construct artificially large fixtures to satisfy a
    heuristic.
    """
    array = np.asarray(output)
    if array.ndim == 3:
        array = array[0]
    if channels_first is None:
        channels_first = array.shape[0] < array.shape[1]
    if channels_first:
        array = array.T
    if array.shape[1] < 5:
        raise ValueError(f"unexpected detection head shape {np.asarray(output).shape}")

    scores = array[:, 4 + class_index].astype(np.float64)
    mask = scores >= conf_threshold
    if not mask.any():
        return []
    boxes = xywh_to_xyxy(array[mask, :4])
    scores = scores[mask]

    keep = nms(boxes, scores, iou_threshold)[:max_detections]
    boxes = transform.to_original(boxes[keep])
    scores = scores[keep]

    detections: List[Detection2D] = []
    for (x0, y0, x1, y1), score in zip(boxes, scores):
        if (x1 - x0) < min_box_px or (y1 - y0) < min_box_px:
            continue
        detections.append(
            Detection2D(xyxy=(float(x0), float(y0), float(x1), float(y1)), score=float(score))
        )
    return detections


def boxes_to_detections(
    xyxy: Sequence[Sequence[float]], scores: Optional[Sequence[float]] = None
) -> List[Detection2D]:
    """For the tests and for replaying recorded detections."""
    scores = scores if scores is not None else [1.0] * len(xyxy)
    return [
        Detection2D(xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])), score=float(s))
        for b, s in zip(xyxy, scores)
    ]
