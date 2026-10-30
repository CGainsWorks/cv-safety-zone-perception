#!/usr/bin/env python3
"""Build a TensorRT engine from the exported FP32 ONNX graph.

    python scripts/export_tensorrt.py --precision fp16
    python scripts/export_tensorrt.py --precision int8 --calib-dir <dir>

Needs a machine with an NVIDIA GPU and TensorRT installed.  It does not run in
CI, which has no GPU.  ``make tensorrt-desktop`` runs it inside NVIDIA's
TensorRT container on a desktop RTX 4060; ``docs/tensorrt.md`` and
``results/published/tensorrt.json`` say exactly which machine and versions and
what came out.  Running it somewhere else will give different numbers, which
is the whole point of writing them down.

Engines are **never committed**.  A serialised TensorRT plan is tied to the
GPU architecture, the driver and the TensorRT version that built it and an
engine from elsewhere either refuses to deserialise or, worse, deserialises and
behaves differently.  ``.gitignore`` and ``.gitattributes`` both cover
``*.engine``; this script writes next to the ONNX file and prints the build
environment so a result can be traced to it.

FP16 roughly halves the inference time and, for a detector at this operating
point, changes the detections by less than the box jitter but "less than the
jitter" is a claim, so ``tests/test_backends.py::test_tensorrt_matches_onnx``
checks it against ONNX Runtime rather than asserting it.

INT8 uses TensorRT's entropy calibrator on photographs that are not in the
evaluation set: a fixed, seeded list of COCO **train2017** images
(``data/calibration_train2017.txt``, fetched by ``scripts/fetch_calibration.py``).
Calibrating on val2017 would set the activation ranges on the very images the
engine is then scored on.  The calibrator streams them through the pipeline's
own letterbox, so it measures the tensors the deployed engine will see and the
calibration cache is written next to the engine so a rebuild costs seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from szp.detection.base import default_model_path  # noqa: E402

PRECISIONS = ("fp32", "fp16", "int8")


def _calibrator_class(trt, images, input_size, cache_path, batch_size):
    """Build the calibrator class here, where ``trt`` is known to exist.

    Subclassing ``trt.IInt8EntropyCalibrator2`` at module scope would make this
    file unimportable without TensorRT and the tests import it.
    """
    import pycuda.driver as cuda

    from szp.detection.preprocess import letterbox

    class ImageCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self.images = list(images)
            self.batch_size = int(batch_size)
            self.input_size = input_size
            self.cache_path = Path(cache_path)
            self.index = 0
            width, height = input_size
            self.device_input = cuda.mem_alloc(
                self.batch_size * 3 * height * width * np.dtype(np.float32).itemsize
            )

        def get_batch_size(self) -> int:
            return self.batch_size

        def get_batch(self, names, p_str=None):  # noqa: ARG002 - TensorRT signature
            import cv2

            if self.index + self.batch_size > len(self.images):
                return None
            batch = []
            for path in self.images[self.index : self.index + self.batch_size]:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"could not decode calibration image {path}")
                tensor, _ = letterbox(image, self.input_size)
                batch.append(tensor[0])
            self.index += self.batch_size
            array = np.ascontiguousarray(np.stack(batch), dtype=np.float32)
            cuda.memcpy_htod(self.device_input, array)
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if self.cache_path.exists():
                return self.cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache) -> None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(cache)

    return ImageCalibrator


def build(
    onnx_path: Path,
    engine_path: Path,
    *,
    precision: str,
    workspace_mb: int,
    verbose: bool,
    calibration_images: Optional[List[Path]] = None,
    input_size: tuple = (640, 640),
    calib_batch: int = 8,
) -> dict:
    try:
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise SystemExit(
            f"tensorrt is not installed: {exc}\n"
            "On a Jetson it comes with JetPack.  On a desktop, install the TensorRT "
            "wheel that matches your CUDA version.  See docs/tensorrt.md."
        ) from None

    if precision == "int8":
        # pycuda needs a current CUDA context to allocate the calibrator's
        # device buffer and creating it before the builder means TensorRT
        # reads the pointer in the same context.  Without it the first batch
        # fails with "invalid device context".
        import pycuda.autoinit  # noqa: F401

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as handle:
        if not parser.parse(handle.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise SystemExit("ONNX parse failed:\n  " + "\n  ".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    calibration_images = list(calibration_images or [])
    if precision in ("fp16", "int8"):
        if not builder.platform_has_fast_fp16:
            print("warning: this platform has no fast FP16; building anyway")
        # Kept on for INT8 too, so layers TensorRT will not quantise fall back
        # to half rather than to full precision.
        config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        if not builder.platform_has_fast_int8:
            raise SystemExit("this device has no fast INT8 path; build FP16 instead")
        if not calibration_images:
            raise SystemExit("INT8 needs calibration images; run scripts/fetch_calibration.py")
        config.set_flag(trt.BuilderFlag.INT8)
        cache = engine_path.with_suffix(".calib.cache")
        calibrator_class = _calibrator_class(
            trt, calibration_images, input_size, cache, calib_batch
        )
        config.int8_calibrator = calibrator_class()

    print(f"building {engine_path.name} from {onnx_path.name} ...", flush=True)
    started = time.perf_counter()
    serialised = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - started
    if serialised is None:
        raise SystemExit("TensorRT returned no engine; run again with --verbose")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialised)

    return {
        "engine": engine_path.name,
        "onnx": onnx_path.name,
        "precision": precision,
        "fp16": precision in ("fp16", "int8"),
        "int8": precision == "int8",
        "calibration_images": len(calibration_images) if precision == "int8" else 0,
        "calibration_split": "train2017" if precision == "int8" else None,
        "workspace_mb": workspace_mb,
        "build_seconds": round(elapsed, 1),
        "engine_bytes": engine_path.stat().st_size,
        "tensorrt": trt.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "note": "engines are machine-specific and are never committed",
    }


def calibration_images(calib_dir: Path, listing: Path, limit: int) -> List[Path]:
    """The listed train2017 images, in the listed order, all of them present."""
    names = [line.strip() for line in listing.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    paths = [calib_dir / name for name in names[:limit]]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} calibration images missing from {calib_dir}, e.g. {missing[:3]}; "
            "run scripts/fetch_calibration.py"
        )
    return paths


def default_calib_dir() -> Path:
    from fetch_calibration import default_calib_dir as resolve

    return resolve()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=None, help="path to the FP32 ONNX file")
    parser.add_argument("--engine", default=None, help="where to write the plan")
    parser.add_argument("--precision", default="fp16", choices=PRECISIONS)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--calib-dir", default=os.environ.get("SZP_CALIB_DIR"))
    parser.add_argument(
        "--calib-list", default=str(REPO_ROOT / "data" / "calibration_train2017.txt")
    )
    parser.add_argument("--calib-images", type=int, default=512)
    parser.add_argument("--calib-batch", type=int, default=8)
    parser.add_argument("--workspace-mb", type=int, default=2048)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    onnx_path = default_model_path(args.model)
    if not onnx_path.exists():
        raise SystemExit(
            f"no ONNX file at {onnx_path}.  Run 'python scripts/fetch_model.py' first."
        )
    # The default sits next to the ONNX file with the same stem, which is where
    # build_detector(backend="auto") looks for it.
    engine_path = Path(args.engine) if args.engine else onnx_path.with_suffix(".engine")
    images: List[Path] = []
    if args.precision == "int8":
        calib_dir = Path(args.calib_dir) if args.calib_dir else default_calib_dir()
        images = calibration_images(calib_dir, Path(args.calib_list), args.calib_images)

    info = build(
        onnx_path,
        engine_path,
        precision=args.precision,
        workspace_mb=args.workspace_mb,
        verbose=args.verbose,
        calibration_images=images,
        input_size=(args.input_size, args.input_size),
        calib_batch=args.calib_batch,
    )
    card = engine_path.with_suffix(".engine.json")
    with open(card, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    print(f"\nwrote {engine_path} and {card}")
    print("check it against ONNX Runtime:  python -m pytest tests/test_backends.py -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
