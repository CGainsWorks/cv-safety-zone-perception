"""TensorRT backend.

Not exercised by CI.  GitHub's runners have no NVIDIA GPU, so nothing in this
file runs there and it would be dishonest to let a green tick imply otherwise.
What CI does check is that the module imports cleanly and that its pre- and
post-processing are the shared ones from ``preprocess`` and ``postprocess``,
which is where a divergence between the two backends would show up.  The
equivalence test itself, ``tests/test_backends.py::test_tensorrt_matches_onnx``,
skips without a GPU and runs on a machine with one.

Written against the TensorRT 10 API, which replaced the binding-index calls of
TensorRT 8 with named tensors.  ``docs/tensorrt.md`` records the exact
versions and the machine the published latency numbers came from.

Memory is allocated once at construction and reused.  A per-frame
``cudaMalloc`` costs a few hundred microseconds and, worse, is a per-frame
opportunity for the allocator to block, which turns a tail latency problem into
an intermittent deadline miss.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from szp.detection.base import DetectorConfig
from szp.detection.postprocess import decode_yolov8
from szp.detection.preprocess import letterbox
from szp.types import Detection2D

__all__ = ["TensorRTDetector", "tensorrt_available"]


def tensorrt_available() -> bool:
    try:
        import pycuda.driver  # noqa: F401
        import tensorrt  # noqa: F401
    except Exception:
        return False
    return True


class TensorRTDetector:
    name = "tensorrt"

    def __init__(self, engine_path: Path, config: Optional[DetectorConfig] = None):
        try:
            import pycuda.autoinit  # noqa: F401  (creates the context)
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise ImportError(
                f"the TensorRT backend needs tensorrt and pycuda: {exc}"
            ) from exc

        self._cuda = cuda
        self._trt = trt
        self.config = config or DetectorConfig()
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"no engine at {self.engine_path}.  Build one with "
                "'python scripts/export_tensorrt.py'.  Engines are specific to the GPU, "
                "driver and TensorRT version that built them and are never committed."
            )

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(self.engine_path, "rb") as handle:
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(
                f"{self.engine_path} did not deserialise.  An engine built with a different "
                "TensorRT version or for a different GPU will fail exactly like this."
            )
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self._tensors = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self._inputs = [
            n
            for n in self._tensors
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        self._outputs = [n for n in self._tensors if n not in self._inputs]
        if len(self._inputs) != 1:
            raise RuntimeError(f"expected one input tensor, engine has {self._inputs}")

        input_shape = tuple(self.engine.get_tensor_shape(self._inputs[0]))
        self.input_size = (int(input_shape[3]), int(input_shape[2]))
        self._host: Dict[str, np.ndarray] = {}
        self._device: Dict[str, object] = {}
        for name in self._tensors:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype).reshape(shape)
            device = cuda.mem_alloc(host.nbytes)
            self._host[name] = host
            self._device[name] = device
            self.context.set_tensor_address(name, int(device))
        self._last_infer_ns = 0

    def detect(self, bgr: np.ndarray) -> List[Detection2D]:
        cuda = self._cuda
        tensor, transform = letterbox(bgr, self.input_size)
        name = self._inputs[0]
        np.copyto(self._host[name], tensor.astype(self._host[name].dtype, copy=False))

        start = time.perf_counter_ns()
        cuda.memcpy_htod_async(self._device[name], self._host[name], self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for output in self._outputs:
            cuda.memcpy_dtoh_async(self._host[output], self._device[output], self.stream)
        self.stream.synchronize()
        self._last_infer_ns = time.perf_counter_ns() - start

        return decode_yolov8(
            self._host[self._outputs[0]],
            transform,
            conf_threshold=self.config.conf_threshold,
            iou_threshold=self.config.iou_threshold,
            max_detections=self.config.max_detections,
            class_index=self.config.class_index,
        )

    def warmup(self, iterations: int = 10) -> None:
        """More iterations than the ONNX path needs.

        A TensorRT engine's first few executions include lazy kernel loads and,
        on a Jetson, the clocks have usually not ramped.  Ten is what it takes
        for the per-frame time to settle on the Xavier NX; the benchmark script
        prints the warm-up curve so the choice can be checked rather than
        taken on trust.
        """
        width, height = self.input_size
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(max(0, iterations)):
            self.detect(blank)

    def close(self) -> None:
        self._device.clear()
        self._host.clear()
        self.context = None
        self.engine = None

    def __enter__(self) -> "TensorRTDetector":
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
            "engine": self.engine_path.name,
            "tensorrt": self._trt.__version__,
            "input_size": list(self.input_size),
            "conf_threshold": self.config.conf_threshold,
            "iou_threshold": self.config.iou_threshold,
            "fp16": self.config.fp16,
        }
