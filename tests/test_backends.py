"""ONNX Runtime and TensorRT must produce the same detections.

The whole TensorRT claim in this repository is "the same model, the same
pre- and post-processing, a faster execution provider".  That claim is only
worth anything if somebody checks it, because the ways an export can go quietly
wrong -- a fused activation that rounds differently, an FP16 overflow in a
detection head, a shape inference that pads instead of resizing -- all show up
as *slightly* different boxes rather than as an error.

CI cannot run this: GitHub's runners have no NVIDIA GPU, so the TensorRT test
skips there and says so.  It runs on a machine with a GPU and an engine built
by ``scripts/export_tensorrt.py`` and ``docs/tensorrt.md`` records the last
machine it was run on.  A skipped test is not a passing test and the README
does not pretend otherwise.

What *does* run everywhere is the shared-code check: both backends must call
the same letterbox and the same decoder, so a divergence has one place to hide
and this file names it.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np
import pytest

from szp.detection import onnx_backend, tensorrt_backend
from szp.detection.base import DetectorConfig, build_detector, default_model_path
from szp.detection.postprocess import iou_matrix


def test_both_backends_share_the_preprocessing_and_the_decoder():
    """A divergence between the two can only come from the model, not the code."""
    onnx_source = inspect.getsource(onnx_backend)
    trt_source = inspect.getsource(tensorrt_backend)
    for symbol in ("from szp.detection.preprocess import letterbox",
                   "from szp.detection.postprocess import decode_head"):
        assert symbol in onnx_source, symbol
        assert symbol in trt_source, symbol
    # Neither backend may define its own.
    for source, name in ((onnx_source, "onnx"), (trt_source, "tensorrt")):
        assert "def letterbox" not in source, f"{name} defines its own letterbox"
        assert "def decode" not in source, f"{name} defines its own decoder"


def test_the_tensorrt_backend_imports_without_a_gpu():
    """Importing must not require the runtime; only constructing does."""
    assert hasattr(tensorrt_backend, "TensorRTDetector")
    assert isinstance(tensorrt_backend.tensorrt_available(), bool)


def _equivalence_images(count: int = 16):
    """Real photographs of people when COCO is on disk, noise otherwise.

    Noise alone makes a weak test: at this threshold it produces no detections
    and every comparison passes by being empty.  So when val2017 is present
    the first images with a person at least 50 px tall are used and the test
    requires that something was actually compared.
    """
    try:
        import cv2

        from szp.data.coco import CocoPersonDataset

        dataset = CocoPersonDataset(split="val2017")
        chosen = dataset.with_people(min_height_px=50.0)[:count]
        images = [cv2.imread(str(item.path)) for item in chosen]
        if images and all(image is not None for image in images):
            return images, True
    except (FileNotFoundError, OSError, ImportError):
        pass
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(8)], False


@pytest.mark.needs_model
def test_tensorrt_matches_onnx(model_path):
    """The equivalence check itself.  Skips without a GPU and an engine.

    ``SZP_ENGINE`` names an engine other than the one next to the ONNX file,
    which is how ``scripts/measure_tensorrt.py`` runs it against the FP32 and
    the FP16 engine in turn.
    """
    if not tensorrt_backend.tensorrt_available():
        pytest.skip("no tensorrt/pycuda on this machine")
    engine = Path(os.environ.get("SZP_ENGINE") or model_path.with_suffix(".engine"))
    if not engine.exists():
        pytest.skip(f"no engine at {engine}; build one with scripts/export_tensorrt.py")

    images, real = _equivalence_images()

    config = DetectorConfig(conf_threshold=0.20)
    reference = build_detector(str(model_path), config, backend="onnx")
    candidate = build_detector(str(model_path), config, backend="tensorrt", engine_path=engine)
    compared = 0
    try:
        reference.warmup(2)
        candidate.warmup(5)
        for index, image in enumerate(images):
            a = reference.detect(image)
            b = candidate.detect(image)
            assert len(a) == len(b), (
                f"image {index}: ONNX found {len(a)} people, TensorRT found {len(b)}"
            )
            if not a:
                continue
            ious = iou_matrix([d.xyxy for d in a], [d.xyxy for d in b])
            assert ious.max(axis=1).min() > 0.95, f"image {index}: boxes disagree"
            scores_a = np.sort([d.score for d in a])
            scores_b = np.sort([d.score for d in b])
            assert np.allclose(scores_a, scores_b, atol=0.05), f"image {index}: scores disagree"
            compared += len(a)
    finally:
        reference.close()
        candidate.close()
    if real:
        assert compared > 0, "no detections on real photographs; the comparison checked nothing"


@pytest.mark.needs_model
def test_the_onnx_backend_reports_its_provider(model_path):
    detector = build_detector(str(model_path), DetectorConfig())
    try:
        providers = detector.info["providers"]
        assert providers, "a live session must name at least one execution provider"
        assert "CPUExecutionProvider" in providers or "CUDAExecutionProvider" in providers
    finally:
        detector.close()


def test_the_default_model_path_is_under_models(repo_root):
    assert default_model_path().parent == repo_root / "models"
