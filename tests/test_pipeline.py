"""The whole thing, frame in to CAN frame out.

These are the tests that would catch a stage being wired to the wrong thing.
The unit tests above check that each stage is right; these check that they are
connected, that nothing escapes and that the failure modes reach the bus.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from szp.data.synthetic import SceneConfig, SimulatedDetector, SyntheticScene
from szp.pipeline import Pipeline, PipelineConfig, boxes_to_dicts
from szp.safety.can_frames import ID_ESTOP, CanFrame, decode_estop
from szp.safety.fsm import FsmConfig
from szp.safety.publisher import CanPublisher
from szp.safety.watchdog import WatchdogConfig
from szp.types import Cause, SafetyState
from szp.zones.config import load_zones


def make_pipeline(scene, zone_set, publisher=None, **config_overrides):
    detector = SimulatedDetector(scene, miss_probability=0.0, false_positive_rate=0.0)
    config = PipelineConfig(
        watchdog=WatchdogConfig(enforce_period=False),
        fsm=FsmConfig(require_operator_reset=True),
        **config_overrides,
    )
    return Pipeline(
        detector=detector,
        extrinsics=scene.extrinsics,
        zone_set=zone_set,
        publisher=publisher,
        config=config,
    ), detector


@pytest.fixture
def short_scene():
    return SyntheticScene(SceneConfig(duration_s=8.0, fps=10.0))


def run(pipeline, detector, scene, frames=None):
    results = []
    total = frames or scene.n_frames
    for index in range(total):
        bundle, truth = scene.frame(index)
        detector.set_truth(truth)
        results.append(pipeline.process(bundle))
    return results


def test_the_pipeline_runs_a_whole_scene_without_an_internal_error(short_scene, zone_set):
    pipeline, detector = make_pipeline(short_scene, zone_set)
    try:
        results = run(pipeline, detector, short_scene)
    finally:
        pipeline.close()
    assert len(results) == short_scene.n_frames
    assert pipeline.errors == ()
    assert all(r.decision.state is not SafetyState.FAULT for r in results)


def test_somebody_walking_into_a_stop_zone_produces_a_stop(short_scene, zone_set):
    pipeline, detector = make_pipeline(short_scene, zone_set)
    try:
        results = run(pipeline, detector, short_scene)
    finally:
        pipeline.close()
    states = [r.decision.state for r in results]
    assert SafetyState.CLEAR in states, "the scene should start empty"
    assert SafetyState.STOP in states, "and somebody should walk into the machine envelope"
    first_stop = states.index(SafetyState.STOP)
    assert results[first_stop].decision.cause & Cause.ZONE_OCCUPIED
    assert results[first_stop].decision.zone_mask != 0


def test_the_stop_latches_until_the_operator_resets(short_scene, zone_set):
    pipeline, detector = make_pipeline(short_scene, zone_set)
    try:
        results = run(pipeline, detector, short_scene)
        states = [r.decision.state for r in results]
        assert SafetyState.STOP in states
        # Everybody has gone by now and it is still stopped.
        tail = states[-1]
        assert tail is SafetyState.STOP
        pipeline.fsm.request_reset()
        bundle, truth = short_scene.frame(short_scene.n_frames - 1)
        detector.set_truth([])
        after = [pipeline.process(bundle).decision.state for _ in range(10)]
    finally:
        pipeline.close()
    assert SafetyState.RECOVERING in after
    assert after[-1] in (SafetyState.CLEAR, SafetyState.WARN)


def test_a_detector_that_raises_drives_the_machine_to_fault_rather_than_crashing(
    short_scene, zone_set
):
    class Broken(SimulatedDetector):
        def detect(self, bgr):
            raise RuntimeError("the model file was truncated")

    detector = Broken(short_scene)
    pipeline = Pipeline(
        detector=detector,
        extrinsics=short_scene.extrinsics,
        zone_set=zone_set,
        config=PipelineConfig(watchdog=WatchdogConfig(enforce_period=False)),
    )
    try:
        bundle, truth = short_scene.frame(0)
        detector.set_truth(truth)
        result = pipeline.process(bundle)
    finally:
        pipeline.close()
    assert result.decision.state is SafetyState.FAULT
    assert result.decision.cause & Cause.INTERNAL_ERROR
    assert not result.decision.state.motion_permitted
    assert len(pipeline.errors) == 1
    assert "truncated" in pipeline.errors[0]


def test_a_dead_depth_camera_faults_rather_than_reporting_an_empty_cell(short_scene, zone_set):
    pipeline, detector = make_pipeline(short_scene, zone_set)
    try:
        states = []
        for index in range(20):
            bundle, truth = short_scene.frame(index)
            detector.set_truth(truth)
            states.append(pipeline.process(replace(bundle, depth=None)).decision.state)
    finally:
        pipeline.close()
    assert SafetyState.FAULT in states
    assert not any(s.motion_permitted for s in states[-5:])


def test_a_dead_lidar_faults_too(short_scene, zone_set):
    pipeline, detector = make_pipeline(short_scene, zone_set)
    try:
        states = []
        for index in range(20):
            bundle, truth = short_scene.frame(index)
            detector.set_truth(truth)
            states.append(
                pipeline.process(
                    replace(bundle, scan_ranges=None, scan_angles=None)
                ).decision.state
            )
    finally:
        pipeline.close()
    assert SafetyState.FAULT in states


def test_losing_both_ranging_sensors_still_detects_the_person(short_scene, zone_set):
    """FAULT stops the machine but the boxes must still be there: a fault plus
    an empty scene reads very differently in a log from a fault plus a person
    standing in the envelope."""
    pipeline, detector = make_pipeline(short_scene, zone_set)
    saw_boxes = False
    try:
        for index in range(short_scene.n_frames):
            bundle, truth = short_scene.frame(index)
            detector.set_truth(truth)
            result = pipeline.process(
                replace(bundle, depth=None, scan_ranges=None, scan_angles=None)
            )
            if result.detections:
                assert len(result.boxes) == len(result.detections)
                assert all(b.range_ambiguous for b in result.boxes)
                saw_boxes = True
    finally:
        pipeline.close()
    assert saw_boxes


def test_every_frame_publishes_including_the_ones_where_nothing_changed(
    short_scene, zone_set
):
    import can

    channel = "szp-test-publish"
    publisher = CanPublisher(channel=channel, interface="virtual")
    listener = can.Bus(interface="virtual", channel=channel, receive_own_messages=False)
    pipeline, detector = make_pipeline(short_scene, zone_set, publisher=publisher)
    try:
        results = run(pipeline, detector, short_scene, frames=12)
        seen = []
        while True:
            message = listener.recv(timeout=0.05)
            if message is None:
                break
            if message.arbitration_id == ID_ESTOP:
                seen.append(message)
    finally:
        pipeline.close()
        listener.shutdown()
    assert len(seen) == 12, "one e-stop frame per pipeline frame, unconditionally"
    assert all(r.t_published_ns is not None for r in results)
    decoded = [decode_estop(CanFrame(m.arbitration_id, bytes(m.data))) for m in seen]
    assert all(d is not None for d in decoded)
    alives = [d["alive"] for d in decoded]
    assert all(a != b for a, b in zip(alives, alives[1:]))


def test_the_state_on_the_bus_matches_the_state_in_the_pipeline(short_scene, zone_set):
    import can

    channel = "szp-test-agreement"
    publisher = CanPublisher(channel=channel, interface="virtual")
    listener = can.Bus(interface="virtual", channel=channel, receive_own_messages=False)
    pipeline, detector = make_pipeline(short_scene, zone_set, publisher=publisher)
    try:
        results = run(pipeline, detector, short_scene, frames=30)
        on_bus = []
        while True:
            message = listener.recv(timeout=0.05)
            if message is None:
                break
            if message.arbitration_id == ID_ESTOP:
                on_bus.append(decode_estop(CanFrame(message.arbitration_id, bytes(message.data))))
    finally:
        pipeline.close()
        listener.shutdown()
    assert [d["state"] for d in on_bus] == [r.decision.state for r in results]
    assert [d["zone_mask"] for d in on_bus] == [r.decision.zone_mask for r in results]


def test_latency_accounting_adds_up(short_scene, zone_set):
    publisher = CanPublisher(channel="szp-test-latency", interface="virtual")
    pipeline, detector = make_pipeline(short_scene, zone_set, publisher=publisher)
    try:
        run(pipeline, detector, short_scene, frames=30)
        report = pipeline.report()
    finally:
        pipeline.close()
    assert report["frames"] == 30
    assert report["end_to_end_ms"]["p50"] > 0
    # The stages must account for nearly all of the measured end to end.  A
    # large unexplained remainder means the budget table is fiction.
    assert abs(report["unaccounted_ms"]["p50"]) < 0.5 * report["end_to_end_ms"]["p50"]
    assert report["internal_errors"] == 0
    assert report["can"]["frames_sent"] > 30


def test_the_report_names_the_detector_after_close(short_scene, zone_set):
    pipeline, detector = make_pipeline(short_scene, zone_set)
    run(pipeline, detector, short_scene, frames=3)
    pipeline.close()
    assert pipeline.report()["detector"]["backend"] == "simulated"


def test_boxes_serialise_to_json_friendly_dictionaries(short_scene, zone_set):
    import json

    pipeline, detector = make_pipeline(short_scene, zone_set)
    try:
        results = run(pipeline, detector, short_scene)
    finally:
        pipeline.close()
    with_boxes = [r for r in results if r.boxes]
    assert with_boxes
    payload = boxes_to_dicts(with_boxes[0].boxes)
    json.dumps(payload)
    assert payload[0]["source"] in {"fused", "depth", "lidar", "unranged"}
    assert len(payload[0]["center"]) == 3


def test_the_pipeline_matches_the_repository_zone_file(repo_root, short_scene):
    """Guards against the demo passing with zones that only exist in a test."""
    zone_set = load_zones(repo_root / "config" / "zones.yaml")
    pipeline, detector = make_pipeline(short_scene, zone_set)
    try:
        results = run(pipeline, detector, short_scene)
    finally:
        pipeline.close()
    assert any(r.decision.state is SafetyState.STOP for r in results)


def test_running_twice_from_the_same_seed_gives_the_same_states(zone_set):
    def once():
        scene = SyntheticScene(SceneConfig(duration_s=5.0, fps=10.0))
        pipeline, detector = make_pipeline(scene, zone_set)
        try:
            return [r.decision.state for r in run(pipeline, detector, scene)]
        finally:
            pipeline.close()

    assert once() == once()


def test_intrinsics_are_rescaled_when_the_frame_size_differs(short_scene, zone_set):
    import cv2

    pipeline, detector = make_pipeline(short_scene, zone_set)
    try:
        bundle, truth = short_scene.frame(0)
        detector.set_truth(truth)
        small = replace(
            bundle,
            rgb=cv2.resize(bundle.rgb, (320, 240)),
            intrinsics=short_scene.intrinsics,
        )
        result = pipeline.process(small)
    finally:
        pipeline.close()
    assert result.decision.state is not SafetyState.FAULT
    assert np.isfinite(result.decision.min_range_m) or result.boxes == []
