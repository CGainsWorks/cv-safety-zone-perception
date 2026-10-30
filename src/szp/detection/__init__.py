"""Person detection: pre-processing, backends and decoding."""

from szp.detection.base import Detector, DetectorConfig, build_detector, default_model_path
from szp.detection.postprocess import decode_head, nms
from szp.detection.preprocess import LetterboxTransform, letterbox

__all__ = [
    "Detector",
    "DetectorConfig",
    "build_detector",
    "default_model_path",
    "letterbox",
    "LetterboxTransform",
    "decode_head",
    "nms",
]
