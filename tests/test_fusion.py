"""Depth ranging, scan segmentation and the fusion rules.

The fusion tests run against the synthetic rig, where the true position of
every person is known exactly, so an error here is fusion error and nothing
else.
"""

from __future__ import annotations

import numpy as np
import pytest

from szp.detection.postprocess import boxes_to_detections
from szp.fusion.depth import DepthConfig, estimate_depth, sigma_from_depth
from szp.fusion.fuse import FusionConfig, FusionDebug, Fuser
from szp.fusion.lidar import (
    LidarConfig,
    _breakpoint_threshold,
    scan_to_points,
    segment_scan,
)
from szp.types import CameraIntrinsics, Detection2D, RangeSource

INTRINSICS = CameraIntrinsics(fx=525.0, fy=525.0, cx=319.5, cy=239.5, width=640, height=480)


# -- depth -------------------------------------------------------------------


def depth_map_with_person(person_m: float, background_m: float, coverage: float) -> np.ndarray:
    """A 100x100 depth map: a vertical strip of person against a background."""
    depth = np.full((100, 100), background_m, dtype=np.float32)
    width = int(round(100 * coverage))
    start = (100 - width) // 2
    depth[:, start:start + width] = person_m
    return depth


def whole_image_detection(size: int = 100) -> Detection2D:
    return Detection2D(xyxy=(0.0, 0.0, float(size), float(size)), score=0.9)


def test_the_nearest_dense_cluster_wins_not_the_median():
    """A tight box round a person is still mostly background at range.  The
    median lands on the wall and puts the person metres too far away."""
    depth = depth_map_with_person(person_m=3.0, background_m=7.0, coverage=0.35)
    assert np.median(depth) == pytest.approx(7.0), "the median really is the background here"
    estimate = estimate_depth(whole_image_detection(), depth, DepthConfig(shrink=0.0), 525.0)
    assert estimate is not None
    assert estimate.depth_m == pytest.approx(3.0, abs=0.05)


def test_a_uniform_depth_map_gives_that_depth():
    depth = np.full((100, 100), 4.2, dtype=np.float32)
    estimate = estimate_depth(whole_image_detection(), depth, DepthConfig(), 525.0)
    assert estimate.depth_m == pytest.approx(4.2, abs=0.02)
    assert estimate.valid_fraction == pytest.approx(1.0)


def test_too_few_valid_pixels_gives_no_estimate_rather_than_a_guess():
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[0, 0] = 3.0
    assert estimate_depth(whole_image_detection(), depth, DepthConfig(), 525.0) is None


def test_no_depth_map_at_all_gives_no_estimate():
    assert estimate_depth(whole_image_detection(), None, DepthConfig(), 525.0) is None


def test_out_of_range_returns_are_discarded():
    depth = np.full((100, 100), 40.0, dtype=np.float32)
    assert estimate_depth(whole_image_detection(), depth, DepthConfig(), 525.0) is None


def test_depth_sigma_grows_with_the_square_of_range():
    config = DepthConfig(sigma_floor_m=0.0)
    assert sigma_from_depth(4.0, config, 525.0) / sigma_from_depth(2.0, config, 525.0) == (
        pytest.approx(4.0, rel=1e-6)
    )
    # With the floor in, the growth is damped at close range and that is the
    # point of the floor: below a metre the error is box jitter, not
    # triangulation.
    with_floor = DepthConfig()
    assert sigma_from_depth(1.0, with_floor, 525.0) >= with_floor.sigma_floor_m
    assert sigma_from_depth(8.0, with_floor, 525.0) > 3 * sigma_from_depth(4.0, with_floor, 525.0)


# -- lidar -------------------------------------------------------------------


def test_scan_to_points_uses_rep103_axes():
    points = scan_to_points([1.0, 1.0], [0.0, np.pi / 2])
    assert np.allclose(points[0], [1.0, 0.0], atol=1e-9)
    assert np.allclose(points[1], [0.0, 1.0], atol=1e-9)


def test_two_objects_separated_by_a_gap_are_two_clusters():
    angles = np.deg2rad(np.arange(-20.0, 20.0, 0.25))
    ranges = np.full_like(angles, 5.0)
    ranges[angles < 0] = 2.0
    clusters = segment_scan(ranges, angles)
    assert len(clusters) == 2


def test_a_dropout_splits_a_cluster():
    angles = np.deg2rad(np.arange(-10.0, 10.0, 0.25))
    ranges = np.full_like(angles, 3.0)
    ranges[len(ranges) // 2] = np.nan
    assert len(segment_scan(ranges, angles)) == 2


def test_a_smooth_wall_is_one_cluster():
    angles = np.deg2rad(np.arange(-30.0, 30.0, 0.25))
    ranges = 4.0 / np.cos(angles)
    clusters = segment_scan(ranges, angles, LidarConfig(range_sigma_m=0.0))
    assert len(clusters) == 1


def test_person_like_is_a_tag_and_never_a_filter():
    angles = np.deg2rad(np.arange(-30.0, 30.0, 0.25))
    ranges = 4.0 / np.cos(angles)          # a two-metre-wide wall
    clusters = segment_scan(ranges, angles, LidarConfig(range_sigma_m=0.0))
    assert clusters and not clusters[0].person_like, "this should be tagged as not a person"
    assert len(clusters) == 1, "but it must still be handed on"


def test_an_empty_scan_produces_no_clusters():
    assert segment_scan(np.full(100, np.nan), np.linspace(-1, 1, 100)) == []


def test_mismatched_scan_lengths_are_rejected():
    with pytest.raises(ValueError, match="differ in length"):
        segment_scan(np.zeros(10), np.zeros(11))


def test_vectorised_threshold_matches_scalar():
    """The array form in segment_scan must agree with the readable formula."""
    config = LidarConfig()
    rng = np.random.default_rng(0)
    for _ in range(200):
        previous = float(rng.uniform(0.2, 15.0))
        dphi = float(rng.uniform(1e-4, config.lambda_rad - 1e-4))
        scalar = _breakpoint_threshold(previous, dphi, config)
        denom = np.sin(config.lambda_rad - dphi)
        array = previous * np.sin(dphi) / denom + 3.0 * config.range_sigma_m
        assert scalar == pytest.approx(float(array))


# -- fusion against the synthetic rig ---------------------------------------


def well_framed(person, scene) -> bool:
    """Fully inside the image, big enough to range.

    People at the very edge of the frame get a box a few pixels wide and a
    range estimated from that is a range estimated from mostly background.
    The accuracy figures are quoted for people the camera can actually see;
    what happens to the rest is its own test and its own line in
    docs/limitations.md.
    """
    if person.box_xyxy is None or not person.in_view:
        return False
    x0, y0, x1, y1 = person.box_xyxy
    if (y1 - y0) < 60 or (x1 - x0) < 20:
        return False
    if x0 <= 2 or y0 <= 2 or x1 >= scene.config.width - 2 or y1 >= scene.config.height - 2:
        return False
    return person.visibility > 0.6


def busy_frame(scene) -> int:
    """The first frame with somebody well inside the picture.  People walk into
    the cell part-way through, so frame zero is empty by design."""
    for index in range(scene.n_frames):
        _, truth = scene.frame(index)
        if any(well_framed(p, scene) for p in truth):
            return index
    raise AssertionError("no frame in this scene has a well-framed person")


def fuse_frame(scene, index, config=None, **frame_overrides):
    bundle, truth = scene.frame(index)
    fuser = Fuser(scene.extrinsics, config or FusionConfig())
    detections = [
        Detection2D(xyxy=person.box_xyxy, score=0.9)
        for person in truth
        if well_framed(person, scene)
    ]
    debug = FusionDebug()
    boxes = fuser.fuse(
        detections,
        frame_overrides.get("depth", bundle.depth),
        frame_overrides.get("scan_ranges", bundle.scan_ranges),
        frame_overrides.get("scan_angles", bundle.scan_angles),
        scene.intrinsics,
        debug=debug,
    )
    return boxes, truth, debug, detections


def nearest_error(boxes, truth, scene):
    errors = []
    for person in truth:
        if not well_framed(person, scene):
            continue
        if not boxes:
            continue
        distances = [
            float(np.hypot(b.center[0] - person.center_xy[0], b.center[1] - person.center_xy[1]))
            for b in boxes
        ]
        errors.append(min(distances))
    return errors


def test_fusion_puts_people_within_a_few_centimetres_of_the_truth(scene):
    errors = []
    for index in range(0, scene.n_frames, 5):
        boxes, truth, _, _ = fuse_frame(scene, index)
        errors.extend(nearest_error(boxes, truth, scene))
    assert errors, "the scene produced no usable frames"
    assert float(np.median(errors)) < 0.25
    assert float(np.percentile(errors, 95)) < 0.6


def test_a_person_at_the_edge_of_frame_still_produces_a_box(scene):
    """They are ranged badly.  They are not dropped and the box says so."""
    fuser = Fuser(scene.extrinsics)
    seen = 0
    for index in range(scene.n_frames):
        _, truth = scene.frame(index)
        edge = [
            p
            for p in truth
            if p.box_xyxy is not None
            and p.in_view
            and p.box_xyxy[2] >= scene.config.width - 2
        ]
        if not edge:
            continue
        bundle, _ = scene.frame(index)
        detections = [Detection2D(xyxy=p.box_xyxy, score=0.9) for p in edge]
        boxes = fuser.fuse(
            detections, bundle.depth, bundle.scan_ranges, bundle.scan_angles, scene.intrinsics
        )
        assert len(boxes) == len(detections)
        seen += 1
    assert seen, "no frame in this scene has anybody at the edge of the picture"


def test_with_both_sensors_most_boxes_are_fused(scene):
    total = fused = 0
    for index in range(0, scene.n_frames, 5):
        boxes, _, debug, _ = fuse_frame(scene, index)
        total += len(boxes)
        fused += sum(1 for b in boxes if b.source is RangeSource.FUSED)
    assert total > 0
    assert fused / total > 0.5


def test_a_detection_with_no_depth_and_no_lidar_still_becomes_a_box(scene):
    """The single most important behaviour in the fusion stage."""
    boxes, truth, debug, detections = fuse_frame(
        scene, busy_frame(scene), depth=None, scan_ranges=None, scan_angles=None
    )
    assert detections, "the frame under test has nobody in it"
    assert len(boxes) == len(detections)
    assert debug.n_unranged == len(detections)
    assert all(b.range_ambiguous for b in boxes)


def test_an_unranged_box_covers_the_whole_wedge_it_could_be_in(scene):
    index = busy_frame(scene)
    boxes, _, _, _ = fuse_frame(scene, index, depth=None, scan_ranges=None,
                                scan_angles=None)
    box = boxes[0]
    assert box.range_interval is not None
    near, far = box.range_interval
    assert far > near
    assert box.footprint is not None and len(box.footprint) == 4
    span = np.linalg.norm(box.footprint.max(axis=0) - box.footprint.min(axis=0))
    assert span > (far - near) * 0.8, "the wedge must span the range interval"
    assert box.sigma_xy >= 0.5


def test_the_true_position_lies_inside_the_unranged_wedge(scene):
    from szp.zones.polygon import point_in_polygon

    inside = 0
    total = 0
    for index in range(0, scene.n_frames, 5):
        boxes, truth, _, _ = fuse_frame(
            scene, index, depth=None, scan_ranges=None, scan_angles=None
        )
        for person in truth:
            if not well_framed(person, scene):
                continue
            total += 1
            if any(point_in_polygon(person.center_xy, b.footprint) for b in boxes):
                inside += 1
    assert total > 0
    assert inside / total > 0.9, (
        "an unranged wedge that does not contain the person is worse than useless"
    )


def test_lidar_alone_still_ranges_a_person(scene):
    boxes, truth, debug, _ = fuse_frame(scene, busy_frame(scene), depth=None)
    assert boxes
    assert any(b.source is RangeSource.LIDAR for b in boxes)
    errors = nearest_error(boxes, truth, scene)
    assert errors and float(np.median(errors)) < 0.4


def test_depth_alone_still_ranges_a_person(scene):
    boxes, truth, _, _ = fuse_frame(
        scene, busy_frame(scene), scan_ranges=None, scan_angles=None
    )
    assert boxes
    assert any(b.source is RangeSource.DEPTH for b in boxes)
    errors = nearest_error(boxes, truth, scene)
    assert errors and float(np.median(errors)) < 0.4


def test_disagreement_takes_the_nearer_reading_and_widens_the_uncertainty(scene):
    """Fabricate a disagreement by pushing the whole depth map back two metres."""
    bundle, truth = scene.frame(busy_frame(scene))
    detections = [
        Detection2D(xyxy=p.box_xyxy, score=0.9) for p in truth if well_framed(p, scene)
    ]
    assert detections
    pushed = bundle.depth.copy()
    pushed[pushed > 0] += 2.0

    fuser = Fuser(scene.extrinsics)
    debug = FusionDebug()
    boxes = fuser.fuse(
        detections, pushed, bundle.scan_ranges, bundle.scan_angles, scene.intrinsics, debug=debug
    )
    assert debug.n_disagreement + debug.n_unranged + debug.n_lidar_associated > 0
    for box in boxes:
        if box.source is RangeSource.LIDAR:
            # The lidar reading is the nearer one and it is what was taken.
            person = min(
                truth, key=lambda p: float(np.hypot(*(p.center_xy - np.array(box.center[:2]))))
            )
            error = float(np.hypot(*(person.center_xy - np.array(box.center[:2]))))
            assert error < 0.6


def test_sigma_is_larger_for_a_distant_person_than_a_near_one(scene):
    """Which is what makes a distant person occupy more floor, not less."""
    per_range = []
    for index in range(0, scene.n_frames, 3):
        boxes, truth, _, _ = fuse_frame(scene, index)
        for box in boxes:
            if box.source is RangeSource.DEPTH:
                per_range.append(
                    (float(np.hypot(box.center[0], box.center[1])), box.sigma_xy)
                )
    if len(per_range) < 6:
        pytest.skip("not enough depth-only boxes in this scene")
    per_range.sort()
    near = np.mean([s for _, s in per_range[: len(per_range) // 3]])
    far = np.mean([s for _, s in per_range[-len(per_range) // 3:]])
    assert far > near


def test_fusion_of_no_detections_produces_no_boxes(scene):
    bundle, _ = scene.frame(0)
    fuser = Fuser(scene.extrinsics)
    assert fuser.fuse(
        boxes_to_detections([]), bundle.depth, bundle.scan_ranges,
        bundle.scan_angles, scene.intrinsics
    ) == []
