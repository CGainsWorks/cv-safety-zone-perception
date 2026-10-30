"""What a detector is, from the pipeline's point of view.

The pipeline never imports a backend directly.  It asks
:func:`build_detector` for something that satisfies :class:`Detector` and the
backends -- ONNX Runtime, TensorRT -- are interchangeable behind that.  This is
not architecture for its own sake: the whole TensorRT story in this repository
is "the same pre- and post-processing, the same detections, a different
execution provider" and the only way to make that claim checkable is for the
two backends to differ in exactly one class.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from szp.types import Detection2D

__all__ = ["Detector", "DetectorConfig", "build_detector", "default_model_path"]


@dataclass(frozen=True)
class DetectorConfig:
    input_size: Tuple[int, int] = (640, 640)
    #: The operating point.  Deliberately below the 0.25 that most demos use.
    #: ``analysis/detection_metrics.py`` sweeps this and the README quotes the
    #: miss rate at several values; the choice is a recall-against-nuisance
    #: trade and the repository shows the curve rather than asserting a number.
    conf_threshold: float = 0.15
    iou_threshold: float = 0.55
    max_detections: int = 100
    class_index: int = 0
    #: Number of intra-op threads for ONNX Runtime.  Zero lets it choose.
    threads: int = 0
    #: Only meaningful for the TensorRT backend.
    fp16: bool = True


@runtime_checkable
class Detector(Protocol):
    """Detect people in one BGR frame."""

    name: str

    def detect(self, bgr: np.ndarray) -> List[Detection2D]:
        ...

    def warmup(self, iterations: int = 3) -> None:
        ...

    def close(self) -> None:
        ...

    @property
    def info(self) -> Dict[str, object]:
        ...


def default_model_path(explicit: Optional[str] = None) -> Path:
    """Where the ONNX file lives.

    ``models/`` is gitignored.  ``scripts/fetch_model.py`` puts the file there
    and prints its SHA-256 so the results can name the exact weights they came
    from.
    """
    if explicit:
        return Path(explicit)
    env = os.environ.get("SZP_MODEL")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "models" / "yolox_tiny.onnx"


def build_detector(
    model_path: Optional[str] = None,
    config: Optional[DetectorConfig] = None,
    *,
    backend: str = "auto",
    engine_path: Optional[str] = None,
) -> Detector:
    """Construct a detector.

    ``backend`` is ``onnx``, ``tensorrt`` or ``auto``.  ``auto`` prefers
    TensorRT when a ``.engine`` file sits next to the ONNX file and the runtime
    imports and otherwise falls back to ONNX Runtime.  It never silently
    substitutes one for the other without saying so: the chosen backend is in
    ``detector.info`` and every result file records it.

    ``engine_path`` names a specific engine instead of the one next to the
    ONNX file, which is how the FP32 and FP16 engines are scored side by side.
    """
    path = default_model_path(model_path)
    config = config or DetectorConfig()

    if backend not in ("auto", "onnx", "tensorrt"):
        raise ValueError(f"unknown backend '{backend}'")

    if backend in ("auto", "tensorrt"):
        engine = Path(engine_path) if engine_path else path.with_suffix(".engine")
        if backend == "tensorrt" or engine.exists():
            try:
                from szp.detection.tensorrt_backend import TensorRTDetector

                return TensorRTDetector(engine, config)
            except ImportError as exc:
                if backend == "tensorrt":
                    raise RuntimeError(
                        f"the TensorRT backend was asked for but is not available: {exc}.  "
                        "Install tensorrt and pycuda or run with --backend onnx."
                    ) from exc

    from szp.detection.onnx_backend import OnnxDetector

    return OnnxDetector(path, config)


def detections_summary(detections: Sequence[Detection2D]) -> Dict[str, object]:
    return {
        "count": len(detections),
        "max_score": max((d.score for d in detections), default=0.0),
        "min_score": min((d.score for d in detections), default=0.0),
    }
