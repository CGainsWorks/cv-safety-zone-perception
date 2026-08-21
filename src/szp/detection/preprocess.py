"""Letterboxing and the arithmetic to undo it.

Kept apart from every backend on purpose.  Pre- and post-processing is where
detector ports go wrong -- an off-by-one in the padding, a forgotten aspect
ratio, BGR fed to a model trained on RGB -- and those mistakes show up as a
quiet loss of recall on small or edge-of-frame people rather than as an error.
With the transform in its own module the round trip
``undo(letterbox(box)) == box`` is a unit test that needs no model at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

__all__ = ["LetterboxTransform", "letterbox", "undo_letterbox"]


@dataclass(frozen=True)
class LetterboxTransform:
    """Everything needed to map coordinates back to the original image."""

    scale: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int
    target_width: int
    target_height: int

    def to_original(self, xyxy: np.ndarray) -> np.ndarray:
        """(N, 4) boxes in letterboxed pixels to (N, 4) in original pixels."""
        boxes = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4).copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - self.pad_x) / self.scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - self.pad_y) / self.scale
        # Clip to the image.  A person half out of frame gives a box that runs
        # off the edge and the ``touches_border`` test downstream depends on
        # the clipped box actually reaching the border.
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.original_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.original_height)
        return boxes

    def to_letterbox(self, xyxy: np.ndarray) -> np.ndarray:
        boxes = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4).copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * self.scale + self.pad_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * self.scale + self.pad_y
        return boxes


def letterbox(
    image: np.ndarray,
    target: Tuple[int, int] = (640, 640),
    *,
    pad_value: int = 114,
    to_rgb: bool = True,
) -> Tuple[np.ndarray, LetterboxTransform]:
    """Resize preserving aspect ratio and pad to ``target``.

    Returns the NCHW float32 tensor in 0..1 and the transform that undoes it.
    ``to_rgb`` is on by default because the models here are trained on RGB and
    the pipeline carries BGR; getting that wrong costs a few points of recall
    and nothing else, which is exactly why it is easy to miss.
    """
    height, width = image.shape[:2]
    target_w, target_h = target
    scale = min(target_w / width, target_h / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_x = (target_w - new_w) / 2.0
    pad_y = (target_h - new_h) / 2.0
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    bottom, right = target_h - new_h - top, target_w - new_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3
    )
    if to_rgb and padded.ndim == 3 and padded.shape[2] == 3:
        padded = padded[:, :, ::-1]

    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, LetterboxTransform(
        scale=scale,
        pad_x=float(left),
        pad_y=float(top),
        original_width=width,
        original_height=height,
        target_width=target_w,
        target_height=target_h,
    )


def undo_letterbox(xyxy: np.ndarray, transform: LetterboxTransform) -> np.ndarray:
    return transform.to_original(xyxy)
