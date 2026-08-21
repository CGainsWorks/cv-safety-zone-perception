"""The synthetic rig itself.

The rig is the measuring instrument for every geometric claim in this
repository, so it needs checking as carefully as the code it measures.  A
camera that is secretly pointing the wrong way or a ground-truth box that does
not match the pixels, would make the fusion tests pass while the fusion was
wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from szp.data.synthetic import PersonTrack, SceneConfig, SimulatedDetector, SyntheticScene
from szp.geometry import project_points, se3_inverse, transform_points


@pytest.fixture(scope="module")
def rig():
    return SyntheticScene(SceneConfig(duration_s=4.0, fps=10.0))


def test_the_camera_looks_down_and_forward(rig):
    R = rig.T_world_camera[:3, :3]
    optical_axis = R @ np.array([0.0, 0.0, 1.0])
    assert optical_axis[0] > 0.9, "the camera must look along world +x"
    assert optical_axis[2] < -0.2, "and be tilted down towards the floor"
    up_in_image = R @ np.array([0.0, -1.0, 0.0])
    assert up_in_image[2] > 0.9, "image -y must point up in the world"


def test_the_extrinsics_the_rig_publishes_are_the_ones_it_rendered_with(rig):
    assert np.allclose(rig.extrinsics.T("world", "camera"), rig.T_world_camera)
    assert np.allclose(rig.extrinsics.T("world", "lidar"), rig.T_world_lidar)


def test_depth_is_metric_and_matches_the_geometry(rig):
    """A pixel's depth, back-projected, must land on the surface it came from."""
    depth, instance, _, _ = rig.render(20)
    floor = (instance == -1) & np.isfinite(depth)
    assert floor.any()
    vv, uu = np.mgrid[0:rig.config.height, 0:rig.config.width]
    sample = np.flatnonzero(floor.ravel())[::997][:200]
    from szp.geometry import backproject

    points_cam = backproject(
        rig.intrinsics.K,
        np.stack([uu.ravel()[sample], vv.ravel()[sample]], axis=1),
        depth.ravel()[sample],
    )
    world = transform_points(rig.T_world_camera, points_cam)
    on_a_surface = (
        (np.abs(world[:, 2]) < 1e-6)
        | (np.abs(world[:, 0] - rig.config.room_x_max) < 1e-6)
        | (np.abs(world[:, 0] - rig.config.room_x_min) < 1e-6)
        | (np.abs(np.abs(world[:, 1]) - rig.config.room_y) < 1e-6)
    )
    assert on_a_surface.mean() > 0.99


def test_ground_truth_boxes_contain_the_pixels_they_describe(rig):
    for index in range(rig.n_frames):
        _, instance, _, truth = rig.render(index)
        for person in truth:
            pixels = np.argwhere(instance == person.index)
            if len(pixels) == 0 or person.box_xyxy is None:
                continue
            x0, y0, x1, y1 = person.box_xyxy
            assert pixels[:, 1].min() >= x0 - 1.5
            assert pixels[:, 1].max() <= x1 + 1.5
            assert pixels[:, 0].min() >= y0 - 1.5
            assert pixels[:, 0].max() <= y1 + 1.5


def test_the_true_position_projects_inside_the_true_box(rig):
    T_camera_world = se3_inverse(rig.T_world_camera)
    checked = 0
    for index in range(0, rig.n_frames, 3):
        _, _, _, truth = rig.render(index)
        for person in truth:
            if person.box_xyxy is None:
                continue
            waist = np.array([[person.center_xy[0], person.center_xy[1], person.height_m / 2]])
            cam = transform_points(T_camera_world, waist)
            if cam[0, 2] <= 0:
                continue
            pixel = project_points(rig.intrinsics.K, cam)[0]
            x0, y0, x1, y1 = person.box_xyxy
            # The box is clipped to the image, so only check the axis that is
            # not against a border.
            if x0 > 2 and x1 < rig.config.width - 2:
                assert x0 - 2 <= pixel[0] <= x1 + 2
                checked += 1
            if y0 > 2 and y1 < rig.config.height - 2:
                assert y0 - 2 <= pixel[1] <= y1 + 2
    assert checked > 0


def test_occlusion_shows_up_as_reduced_visibility():
    """Put one person directly behind another and check the rig notices."""
    config = SceneConfig(
        duration_s=0.2,
        fps=10.0,
        people=(
            PersonTrack(start_xy=(3.0, 0.0), end_xy=(3.0, 0.0), speed_mps=0.0),
            PersonTrack(start_xy=(4.0, 0.0), end_xy=(4.0, 0.0), speed_mps=0.0),
        ),
    )
    scene = SyntheticScene(config)
    _, _, _, truth = scene.render(0)
    front, behind = truth[0], truth[1]
    assert front.visibility == pytest.approx(1.0, abs=1e-6)
    assert behind.visibility < 0.9


def test_people_enter_and_leave_the_cell(rig):
    counts = [len(rig.render(i)[3]) for i in range(rig.n_frames)]
    assert min(counts) == 0, "the scene must start empty so CLEAR is reachable"
    assert max(counts) >= 1


def test_a_person_outside_the_active_window_is_not_rendered():
    config = SceneConfig(
        duration_s=1.0,
        fps=10.0,
        people=(
            PersonTrack(
                start_xy=(3.0, 0.0), end_xy=(3.0, 0.0), speed_mps=0.0,
                active_from_s=0.5, active_until_s=0.8,
            ),
        ),
    )
    scene = SyntheticScene(config)
    present = [len(scene.render(i)[3]) for i in range(10)]
    assert present[:5] == [0] * 5
    assert present[5] == 1
    assert present[8] == 0


def test_lidar_returns_come_from_the_people_who_are_in_the_scan_plane(rig):
    for index in range(rig.n_frames):
        bundle, truth = rig.frame(index)
        for person in truth:
            if person.range_m < 7.0 and person.visibility > 0.8:
                assert person.lidar_hits > 0, "a nearby person must return some beams"


def test_depth_noise_and_dropout_are_present_but_bounded(rig):
    bundle, _ = rig.frame(10)
    truth = bundle.meta["depth_true"]
    measured = bundle.depth
    both = (measured > 0) & np.isfinite(truth)
    error = np.abs(measured[both] - truth[both])
    assert error.mean() > 0.0, "the rig is supposed to add noise"
    assert np.percentile(error, 99) < 1.0
    dropped = (measured == 0) & np.isfinite(truth) & (truth < rig.config.depth_max_range_m)
    assert 0 < dropped.mean() < 0.2


def test_the_scene_is_reproducible_from_its_seed():
    a = SyntheticScene(SceneConfig(duration_s=1.0, fps=10.0, seed=5)).frame(3)[0]
    b = SyntheticScene(SceneConfig(duration_s=1.0, fps=10.0, seed=5)).frame(3)[0]
    assert np.array_equal(a.depth, b.depth)
    assert np.array_equal(a.rgb, b.rgb)
    assert np.allclose(a.scan_ranges, b.scan_ranges, equal_nan=True)


def test_a_different_seed_gives_different_noise():
    a = SyntheticScene(SceneConfig(duration_s=1.0, fps=10.0, seed=5)).frame(3)[0]
    b = SyntheticScene(SceneConfig(duration_s=1.0, fps=10.0, seed=6)).frame(3)[0]
    assert not np.array_equal(a.depth, b.depth)


def test_the_simulated_detector_says_plainly_that_it_is_not_a_detector(rig):
    detector = SimulatedDetector(rig)
    assert "not a detector" in str(detector.info["note"])


def test_the_simulated_detector_misses_at_the_configured_rate(rig):
    detector = SimulatedDetector(rig, miss_probability=0.5, false_positive_rate=0.0, seed=1)
    emitted = 0
    offered = 0
    for index in range(rig.n_frames):
        _, truth = rig.frame(index)
        visible = [
            p
            for p in truth
            if p.box_xyxy is not None
            and p.in_view
            and p.visibility >= detector.min_visibility
            and (p.box_xyxy[3] - p.box_xyxy[1]) >= detector.min_height_px
        ]
        offered += len(visible)
        detector.set_truth(truth)
        emitted += len(detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)))
    assert offered > 0
    assert 0.2 < emitted / max(offered, 1) < 0.8
