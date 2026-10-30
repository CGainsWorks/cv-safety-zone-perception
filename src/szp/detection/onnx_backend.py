"""ONNX Runtime backend.

This is the portable path: it runs on the CPU everywhere, on CUDA where
``onnxruntime-gpu`` is installed and it is what CI uses.  It is also the
reference the TensorRT engine is checked against -- ``tests/test_backends.py``
runs both over the same images and requires the detections to agree, which is
the only way to be sure the export did not quietly change the model.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort

from szp.detection.base import DetectorConfig
from szp.detection.postprocess import decode_head
from szp.detection.preprocess import letterbox
from szp.types import Detection2D

__all__ = ["OnnxDetector"]


def _sha256(path: Path, limit: int = 1 << 22) -> str:
    """Hash of the first few megabytes.  Enough to identify a weights file
    without reading a hundred megabytes on every start-up."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()[:16]


class OnnxDetector:
    name = "onnx"

    def __init__(
        self,
        model_path: Path,
        config: Optional[DetectorConfig] = None,
        *,
        providers: Optional[List[str]] = None,
    ):
        self.config = config or DetectorConfig()
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"no detector weights at {self.model_path}.  Run "
                "'python scripts/fetch_model.py' to download and export them or point "
                "SZP_MODEL at an existing ONNX file."
            )
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.config.threads:
            options.intra_op_num_threads = self.config.threads
        available = ort.get_available_providers()
        chosen = providers or [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                               if p in available]
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=chosen
        )
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # The FP16 graph is exported from a half-precision module, so it takes
        # float16 in.  Casting here keeps the letterbox shared and the cast
        # inside the timed region is the cost a deployment would pay too.
        self.input_dtype = (
            np.float16 if self.session.get_inputs()[0].type == "tensor(float16)" else np.float32
        )
        # A fixed-shape export pins the letterbox size; a dynamic one does not
        # and using the configured size for a fixed model is a silent shape
        # error at the first inference.
        self.input_size = self.config.input_size
        if isinstance(shape[3], int) and isinstance(shape[2], int):
            self.input_size = (int(shape[3]), int(shape[2]))
        self._last_infer_ns = 0
        self._infer_ns: List[int] = []
        self._sha = _sha256(self.model_path)

    def detect(self, bgr: np.ndarray) -> List[Detection2D]:
        tensor, transform = letterbox(bgr, self.input_size)
        start = time.perf_counter_ns()
        feed = {self.input_name: tensor.astype(self.input_dtype, copy=False)}
        outputs = self.session.run(None, feed)
        self._last_infer_ns = time.perf_counter_ns() - start
        self._infer_ns.append(self._last_infer_ns)
        return decode_head(
            outputs[0],
            transform,
            conf_threshold=self.config.conf_threshold,
            iou_threshold=self.config.iou_threshold,
            max_detections=self.config.max_detections,
            class_index=self.config.class_index,
        )

    def warmup(self, iterations: int = 3) -> None:
        """Run a few dummy frames.

        The first inference in an ONNX Runtime session is several times slower
        than the steady state because of allocator and kernel setup.  Timing
        that frame and calling it the latency would be flattering and wrong, so
        the benchmark scripts always warm up first and say how many iterations
        they used.
        """
        width, height = self.input_size
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(max(0, iterations)):
            self.detect(blank)
        self._infer_ns.clear()

    def close(self) -> None:
        self.session = None  # type: ignore[assignment]

    def __enter__(self) -> "OnnxDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def last_inference_ms(self) -> float:
        return self._last_infer_ns / 1e6

    @property
    def info(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "model": self.model_path.name,
            "model_sha256_prefix": self._sha,
            "providers": list(self.session.get_providers()) if self.session else [],
            "input_size": list(self.input_size),
            "input_dtype": np.dtype(self.input_dtype).name,
            "conf_threshold": self.config.conf_threshold,
            "iou_threshold": self.config.iou_threshold,
            "onnxruntime": ort.__version__,
        }
