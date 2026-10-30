"""Letterboxing, head decoding and NMS, then the real model if it is present."""

from __future__ import annotations

import numpy as np
import pytest

from szp.detection.base import DetectorConfig, build_detector
from szp.detection.postprocess import decode_head, iou_matrix, nms, xywh_to_xyxy
from szp.detection.preprocess import letterbox


# -- pre-processing ----------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(480, 640), (640, 480), (720, 1280), (100, 100), (57, 331)]
)
def test_letterbox_produces_the_requested_tensor_shape(shape):
    image = np.zeros((*shape, 3), dtype=np.uint8)
    tensor, transform = letterbox(image, (640, 640))
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0
    assert transform.original_width == shape[1]
    assert transform.original_height == shape[0]


@pytest.mark.parametrize("shape", [(480, 640), (640, 480), (720, 1280), (57, 331)])
def test_box_coordinates_survive_the_letterbox_round_trip(shape):
    image = np.zeros((*shape, 3), dtype=np.uint8)
    _, transform = letterbox(image, (640, 640))
    rng = np.random.default_rng(1)
    boxes = np.sort(rng.uniform(0, min(shape), size=(20, 4)), axis=1)
    boxes = boxes[:, [0, 1, 2, 3]]
    back = transform.to_original(transform.to_letterbox(boxes))
    assert np.allclose(back, boxes, atol=1e-6)


def test_letterbox_preserves_aspect_ratio():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, transform = letterbox(image, (640, 640))
    assert transform.scale == pytest.approx(1.0)
    assert transform.pad_y == pytest.approx(80.0)
    assert transform.pad_x == pytest.approx(0.0)


def test_letterbox_swaps_to_rgb_by_default():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = 255            # blue channel, in OpenCV order
    tensor, _ = letterbox(image, (8, 8), to_rgb=True)
    assert tensor[0, 2].max() == pytest.approx(1.0), "blue must land in the last channel"
    assert tensor[0, 0].max() == pytest.approx(0.0)


def test_boxes_are_clipped_to_the_image_so_truncation_is_detectable():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, transform = letterbox(image, (640, 640))
    out = transform.to_original(np.array([[-50.0, -50.0, 900.0, 900.0]]))
    assert out[0, 0] == 0.0
    assert out[0, 2] == 640.0
    assert out[0, 3] == 480.0


# -- post-processing ---------------------------------------------------------


def test_xywh_to_xyxy():
    out = xywh_to_xyxy(np.array([[10.0, 20.0, 4.0, 6.0]]))
    assert np.allclose(out, [[8.0, 17.0, 12.0, 23.0]])


def test_iou_of_a_box_with_itself_is_one_and_with_a_disjoint_box_is_zero():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[20.0, 20.0, 30.0, 30.0]])
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)


def test_iou_of_a_half_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0]])
    assert iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150)


def test_nms_keeps_the_best_of_an_overlapping_group():
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=float
    )
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), iou_threshold=0.5)
    assert keep == [0, 2]


def test_nms_is_deterministic_when_scores_tie():
    """Equal scores are broken by index, so a recall number is reproducible."""
    # All three overlap each other above the threshold, so exactly one
    # survives and it must always be the same one.
    boxes = np.array(
        [[0, 0, 10, 10], [0.5, 0.5, 10.5, 10.5], [1, 1, 11, 11]], dtype=float
    )
    scores = np.array([0.5, 0.5, 0.5])
    for _ in range(20):
        assert nms(boxes, scores, iou_threshold=0.5) == [0]
    assert nms(boxes[::-1], scores, iou_threshold=0.5) == [0]


def test_nms_on_an_empty_input():
    assert nms(np.zeros((0, 4)), np.zeros(0)) == []


def fake_head(entries, n_anchors=32, n_classes=80):
    """(1, 4 + n_classes, n_anchors) with the given (box, score) entries."""
    out = np.zeros((1, 4 + n_classes, n_anchors), dtype=np.float32)
    for i, (box, score) in enumerate(entries):
        out[0, :4, i] = box
        out[0, 4, i] = score
    return out


def test_decode_filters_by_confidence_and_suppresses_duplicates():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, transform = letterbox(image, (640, 640))
    head = fake_head(
        [
            ((320.0, 320.0, 100.0, 200.0), 0.9),
            ((322.0, 321.0, 100.0, 200.0), 0.8),   # duplicate of the first
            ((100.0, 300.0, 60.0, 120.0), 0.7),
            ((500.0, 300.0, 60.0, 120.0), 0.05),   # below threshold
        ]
    )
    detections = decode_head(head, transform, conf_threshold=0.25, channels_first=True)
    assert len(detections) == 2
    assert detections[0].score == pytest.approx(0.9, abs=1e-6)
    assert detections[0].xyxy == pytest.approx((270.0, 140.0, 370.0, 340.0))


def test_decode_uses_the_person_column_only():
    head = np.zeros((1, 84, 16), dtype=np.float32)
    head[0, :4, 0] = (320.0, 320.0, 40.0, 80.0)
    head[0, 5, 0] = 0.99      # a bicycle, not a person
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, transform = letterbox(image, (640, 640))
    assert decode_head(head, transform, conf_threshold=0.25, channels_first=True) == []


def test_decode_rejects_an_unexpected_head_shape():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, transform = letterbox(image, (640, 640))
    with pytest.raises(ValueError, match="unexpected detection head"):
        decode_head(np.zeros((1, 3, 8)), transform, channels_first=True)


# -- the real model ----------------------------------------------------------


@pytest.mark.needs_model
def test_the_onnx_detector_loads_and_runs(model_path):
    detector = build_detector(str(model_path), DetectorConfig())
    try:
        detector.warmup(1)
        detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert isinstance(detections, list)
        assert detector.info["backend"] == "onnx"
        assert detector.info["input_size"] == [640, 640]
        assert detector.last_inference_ms > 0
    finally:
        detector.close()


@pytest.mark.needs_model
def test_the_detector_is_deterministic(model_path):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    detector = build_detector(str(model_path), DetectorConfig())
    try:
        first = detector.detect(image)
        second = detector.detect(image)
    finally:
        detector.close()
    assert [d.xyxy for d in first] == [d.xyxy for d in second]
    assert [d.score for d in first] == [d.score for d in second]


def test_a_missing_model_file_fails_with_an_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_model"):
        build_detector(str(tmp_path / "absent.onnx"), DetectorConfig(), backend="onnx")


def test_asking_for_tensorrt_without_it_installed_says_so(tmp_path):
    from szp.detection.tensorrt_backend import tensorrt_available

    if tensorrt_available():
        pytest.skip("TensorRT is installed on this machine")
    with pytest.raises(RuntimeError, match="TensorRT backend was asked for"):
        build_detector(str(tmp_path / "absent.onnx"), DetectorConfig(), backend="tensorrt")
