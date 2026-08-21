"""The fail-safe state machine and the watchdog that feeds it.

Test names match the properties listed in ``szp/safety/fsm.py``.
"""

from __future__ import annotations

import pytest

from szp.safety.fsm import FsmConfig, SafetyFsm
from szp.safety.watchdog import Watchdog, WatchdogConfig
from szp.types import Cause, Health, NS_PER_MS, SafetyState, ZoneOccupancy

HEALTHY = Health(sources_ok={"rgb": True, "depth": True, "lidar": True, "detector": True})


def occupancy(stop=False, warn=False, from_unranged=False):
    return [
        ZoneOccupancy("stop_zone", "stop", stop, 0.0 if stop else 1.0,
                      (0,) if stop else (), from_unranged),
        ZoneOccupancy("warn_zone", "warn", warn, 0.0 if warn else 1.0, (0,) if warn else ()),
    ]


def started(config: FsmConfig = None) -> SafetyFsm:
    fsm = SafetyFsm(config or FsmConfig())
    for _ in range(fsm.config.healthy_frames_to_start):
        fsm.step(occupancy(), HEALTHY)
    assert fsm.state is SafetyState.CLEAR
    return fsm


# -- start-up ----------------------------------------------------------------


def test_the_machine_does_not_permit_motion_before_the_watchdog_has_settled():
    fsm = SafetyFsm(FsmConfig(healthy_frames_to_start=5))
    for i in range(4):
        decision = fsm.step(occupancy(), HEALTHY)
        assert decision.state is SafetyState.INIT, i
        assert not decision.state.motion_permitted
    assert fsm.step(occupancy(), HEALTHY).state is SafetyState.CLEAR


# -- monotone entry ----------------------------------------------------------


def test_monotone_entry_stop_takes_effect_on_the_frame_the_cause_appears():
    fsm = started()
    decision = fsm.step(occupancy(stop=True), HEALTHY)
    assert decision.state is SafetyState.STOP
    assert decision.cause & Cause.ZONE_OCCUPIED
    assert not decision.state.motion_permitted


def test_a_warning_zone_permits_motion():
    fsm = started()
    decision = fsm.step(occupancy(warn=True), HEALTHY)
    assert decision.state is SafetyState.WARN
    assert decision.state.motion_permitted


def test_warn_returns_to_clear_on_its_own():
    fsm = started()
    fsm.step(occupancy(warn=True), HEALTHY)
    assert fsm.step(occupancy(), HEALTHY).state is SafetyState.CLEAR


# -- latching ----------------------------------------------------------------


def test_latching_stop_does_not_clear_itself_when_the_person_leaves():
    fsm = started()
    fsm.step(occupancy(stop=True), HEALTHY)
    for _ in range(50):
        decision = fsm.step(occupancy(), HEALTHY)
        assert decision.state is SafetyState.STOP
        assert decision.cause & Cause.OPERATOR_LATCH
    assert fsm.latched


def test_no_self_clearing_on_reset_alone_while_the_cause_is_still_there():
    fsm = started()
    fsm.step(occupancy(stop=True), HEALTHY)
    for _ in range(10):
        fsm.request_reset()
        assert fsm.step(occupancy(stop=True), HEALTHY).state is SafetyState.STOP


def test_recovery_needs_the_cause_gone_and_a_reset_and_a_debounce():
    config = FsmConfig(clear_frames_to_release=5)
    fsm = started(config)
    fsm.step(occupancy(stop=True), HEALTHY)
    fsm.step(occupancy(), HEALTHY)          # cause gone, still latched
    fsm.request_reset()
    states = [fsm.step(occupancy(), HEALTHY).state for _ in range(6)]
    assert states[0] is SafetyState.RECOVERING
    assert states[:4] == [SafetyState.RECOVERING] * 4
    assert states[4] is SafetyState.CLEAR
    assert not fsm.latched


def test_a_reset_pressed_during_the_stop_is_discarded():
    """Holding the button down through the hazard must not arm the recovery."""
    fsm = started()
    fsm.request_reset()
    fsm.step(occupancy(stop=True), HEALTHY)
    for _ in range(10):
        assert fsm.step(occupancy(), HEALTHY).state is SafetyState.STOP


def test_non_latching_mode_releases_on_its_own():
    fsm = started(FsmConfig(require_operator_reset=False, clear_frames_to_release=2))
    fsm.step(occupancy(stop=True), HEALTHY)
    assert fsm.step(occupancy(), HEALTHY).state is SafetyState.RECOVERING
    assert fsm.step(occupancy(), HEALTHY).state is SafetyState.CLEAR


# -- fault dominance ---------------------------------------------------------


@pytest.mark.parametrize(
    "health, expected_cause",
    [
        (Health(sources_ok={"depth": False}, failed=("depth",)), Cause.SENSOR_FAILED),
        (Health(sources_ok={"depth": True}, stale=("depth",)), Cause.SENSOR_STALE),
        (Health(sources_ok={}, deadline_miss=True), Cause.DEADLINE_MISS),
        (Health(sources_ok={}, heartbeat_ok=False), Cause.HEARTBEAT_LOST),
    ],
)
def test_any_unhealthy_input_produces_fault(health, expected_cause):
    fsm = started()
    decision = fsm.step(occupancy(), health)
    assert decision.state is SafetyState.FAULT
    assert decision.cause & expected_cause
    assert not decision.state.motion_permitted


def test_fault_dominates_a_clear_scene_and_an_occupied_one_alike():
    unhealthy = Health(sources_ok={"lidar": False}, failed=("lidar",))
    for occ in (occupancy(), occupancy(stop=True)):
        fsm = started()
        assert fsm.step(occ, unhealthy).state is SafetyState.FAULT


def test_an_unhealthy_pipeline_cannot_start():
    fsm = SafetyFsm()
    for _ in range(20):
        decision = fsm.step(occupancy(), Health(sources_ok={"rgb": False}, failed=("rgb",)))
        assert decision.state is SafetyState.FAULT


# -- unranged detections -----------------------------------------------------


def test_an_unranged_detection_alone_stops_the_machine_by_default():
    fsm = started()
    decision = fsm.step(occupancy(stop=True, from_unranged=True), HEALTHY)
    assert decision.state is SafetyState.STOP
    assert decision.cause & Cause.UNRANGED_DETECTION


def test_the_uptime_biased_setting_ignores_unranged_detections():
    """Kept only so the analysis can put a number on what that trade costs."""
    fsm = started(FsmConfig(unranged_causes_stop=False))
    assert fsm.step(occupancy(stop=True, from_unranged=True), HEALTHY).state is SafetyState.CLEAR


# -- watchdog ----------------------------------------------------------------


def watchdog(**overrides) -> Watchdog:
    config = WatchdogConfig(
        deadlines_ms={"rgb": 100.0, "depth": 150.0},
        failed_after_frames=3,
        heartbeat_timeout_ms=300.0,
        period_deadline_ms=150.0,
        **overrides,
    )
    return Watchdog(config)


def test_a_source_that_has_never_reported_is_failed_not_merely_stale():
    dog = watchdog()
    dog.note_heartbeat(0)
    health = dog.update(0, observed={"rgb": 0})
    assert "depth" in health.failed
    assert "depth" not in health.stale


def test_staleness_becomes_failure_after_the_configured_run():
    dog = watchdog()
    dog.note_heartbeat(0)
    dog.update(0, observed={"rgb": 0, "depth": 0})
    now = 0
    seen = []
    for _ in range(4):
        # 160 ms is past the 150 ms depth deadline, so every one of these
        # frames finds depth overdue.
        now += 160 * NS_PER_MS
        dog.note_heartbeat(now)
        health = dog.update(now, observed={"rgb": now})
        seen.append(("depth" in health.stale, "depth" in health.failed))
    assert seen == [(True, False), (True, False), (False, True), (False, True)]


def test_a_source_that_comes_back_clears_its_run():
    dog = watchdog()
    dog.note_heartbeat(0)
    dog.update(0, observed={"rgb": 0, "depth": 0})
    dog.update(200 * NS_PER_MS, observed={"rgb": 200 * NS_PER_MS})
    health = dog.update(
        260 * NS_PER_MS, observed={"rgb": 260 * NS_PER_MS, "depth": 260 * NS_PER_MS}
    )
    assert health.all_ok


def test_a_missing_heartbeat_is_reported_once_the_timeout_has_passed():
    dog = watchdog()
    dog.note_heartbeat(0)
    assert dog.update(100 * NS_PER_MS, observed={"rgb": 100 * NS_PER_MS,
                                                 "depth": 100 * NS_PER_MS}).heartbeat_ok
    health = dog.update(
        500 * NS_PER_MS,
        observed={"rgb": 500 * NS_PER_MS, "depth": 500 * NS_PER_MS},
    )
    assert not health.heartbeat_ok


def test_an_overrun_period_is_a_deadline_miss():
    dog = watchdog()
    dog.note_heartbeat(0)
    dog.update(0, observed={"rgb": 0, "depth": 0})
    late = 400 * NS_PER_MS
    dog.note_heartbeat(late)
    health = dog.update(late, observed={"rgb": late, "depth": late})
    assert health.deadline_miss
    assert health.period_ms == pytest.approx(400.0)


def test_the_period_check_can_be_turned_off_for_offline_replay():
    dog = watchdog(enforce_period=False)
    dog.note_heartbeat(0)
    dog.update(0, observed={"rgb": 0, "depth": 0})
    late = 400 * NS_PER_MS
    dog.note_heartbeat(late)
    health = dog.update(late, observed={"rgb": late, "depth": late})
    assert not health.deadline_miss


def test_the_watchdog_uses_no_wall_clock():
    """Every age is computed from the caller's monotonic stamps."""
    dog = watchdog()
    dog.note_heartbeat(10**18)
    health = dog.update(10**18, observed={"rgb": 10**18, "depth": 10**18})
    assert health.all_ok
