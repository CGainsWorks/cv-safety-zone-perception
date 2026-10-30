#!/usr/bin/env python3
"""End-to-end latency, from frame arrival to the e-stop frame on the bus.

    python analysis/latency_budget.py --out results/latency.json

Two measurements and the difference between them is the point.

**Pipeline latency** is taken inside the process: the clock starts when the
frame arrives and stops when the e-stop frame's loopback echo comes back off
the CAN stack.  It is a per-stage budget and it is what you optimise against.

**Reaction latency** is taken at the receiver: the clock starts on the same
frame arrival and stops when the simulated PLC decides the machine may no
longer move.  It includes everything the first measurement does plus the
transport and the receiver's own polling and it is the number that belongs in
a stopping-distance calculation.  It is always the larger of the two and by
how much is a property of the bus rather than of this code.

The reaction latency is measured over *entry events*, not over every frame: a
person is teleported into a stop zone, the frame that first sees them is
timestamped and the PLC's denial is matched to it.  Averaging over frames on
which nothing changed would report the cost of publishing, which is not the
same question.

Three detector modes and the middle one is the interesting one:

``simulated``
    No model at all.  Measures everything except inference.

``onnx_timing`` (the default and what the published numbers use)
    The real ONNX session runs on every frame, so its cost and its tail are in
    the measurement and the *detections* come from the rig's ground truth.
    This exists because a COCO detector does not see shaded cylinders, so a
    pure ``onnx`` run never detects anybody, never stops and has no reaction
    time to measure.  It is a timing harness and it supports no accuracy claim
    whatsoever; ``TimedRealDetector`` says so at more length.

``onnx``
    The real model, with the real detections.  Gives an honest per-stage budget
    on rendered frames and, for the reason above, usually no entry events.

Rendering is kept out of the timed loop.  The entry scene's people stand still,
so one cycle of frames is rendered up front and replayed; leaving the renderer
in the loop added tens of milliseconds between frames, which is not something a
camera does and which was enough to make the PLC's permit lapse.

What this cannot measure
------------------------
Camera exposure and driver latency, which on a real sensor is often larger than
everything here put together and the PLC's own scan cycle.  Both are named in
``docs/latency.md`` with the numbers to substitute; neither is invented here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from szp.data.synthetic import (  # noqa: E402
    PersonTrack,
    SceneConfig,
    SimulatedDetector,
    SyntheticScene,
    TimedRealDetector,
)
from szp.detection.base import DetectorConfig, build_detector  # noqa: E402
from szp.latency.instrument import STAGES, Instrument, percentiles  # noqa: E402
from szp.pipeline import Pipeline, PipelineConfig  # noqa: E402
from szp.safety.fsm import FsmConfig  # noqa: E402
from szp.safety.plc_sim import SimulatedPlc  # noqa: E402
from szp.safety.publisher import CanPublisher, default_interface  # noqa: E402
from szp.types import NS_PER_MS  # noqa: E402
from szp.zones.config import load_zones  # noqa: E402
from szp.safety.watchdog import WatchdogConfig  # noqa: E402


def entry_scene(dwell_s: float, gap_s: float, repeats: int, fps: float) -> SceneConfig:
    """A person who steps into the machine envelope, leaves and comes back.

    Each entry is one measurable event.  They appear at (3.4, 0.0), which is
    inside ``machine_envelope`` in ``config/zones.yaml`` and vanish again.
    """
    people = []
    cycle = dwell_s + gap_s
    for i in range(repeats):
        start = gap_s + i * cycle
        people.append(
            PersonTrack(
                start_xy=(3.4, 0.0),
                end_xy=(3.4, 0.0),
                speed_mps=0.0,
                active_from_s=start,
                active_until_s=start + dwell_s,
            )
        )
    return SceneConfig(duration_s=repeats * cycle + gap_s, fps=fps, people=tuple(people))


def measure(
    scene: SyntheticScene,
    zone_set,
    *,
    detector_kind: str,
    model: Optional[str],
    can_interface: Optional[str],
    channel: Optional[str],
    repeats: int,
    backend: str = "onnx",
    engine: Optional[str] = None,
) -> Dict[str, object]:
    truth_detector = SimulatedDetector(scene, miss_probability=0.0, false_positive_rate=0.0)
    if detector_kind == "simulated":
        detector = truth_detector
    elif detector_kind == "onnx_timing":
        detector = TimedRealDetector(
            build_detector(model, DetectorConfig(), backend=backend, engine_path=engine),
            truth_detector,
        )
        detector.warmup(10)
    else:
        detector = build_detector(model, DetectorConfig(), backend=backend, engine_path=engine)
        detector.warmup(10)
    feeds_truth = hasattr(detector, "set_truth")

    # Pre-render one full cycle.  The entry scene's people do not move, so the
    # frames repeat; rendering inside the timed loop would put tens of
    # milliseconds between publishes, which is not something a camera does.
    cycle = min(scene.n_frames, int(round((scene.config.duration_s / max(repeats, 1))
                                          * scene.config.fps)) + 1)
    cycle = max(cycle, 2)
    prerendered = [scene.frame(index) for index in range(min(scene.n_frames, cycle * 2))]

    interface, resolved_channel = default_interface(channel)
    if can_interface:
        interface = can_interface
    publisher = CanPublisher(channel=resolved_channel, interface=interface)
    plc = SimulatedPlc(channel=resolved_channel, interface=interface, poll_timeout_s=0.002)
    plc.start()

    instrument = Instrument()
    pipeline = Pipeline(
        detector=detector,
        extrinsics=scene.extrinsics,
        zone_set=zone_set,
        publisher=publisher,
        config=PipelineConfig(
            watchdog=WatchdogConfig(enforce_period=False),
            # Latching would make the second entry unmeasurable, because the
            # permit would already be withdrawn.  The operator resets between
            # events instead, which is what the auto-reset here stands in for.
            fsm=FsmConfig(require_operator_reset=False, clear_frames_to_release=2),
        ),
        instrument=instrument,
    )

    entry_arrivals: List[int] = []
    was_permitted = False
    states: Dict[str, int] = {}
    try:
        for index in range(scene.n_frames):
            bundle, truth = prerendered[index % len(prerendered)]
            if feeds_truth:
                detector.set_truth(truth)
            arrival_ns = time.perf_counter_ns()
            result = pipeline.process(
                replace(
                    bundle,
                    t_arrival_ns=arrival_ns,
                    t_rgb_ns=arrival_ns,
                    t_depth_ns=arrival_ns,
                    t_scan_ns=arrival_ns,
                    meta={},
                )
            )
            states[result.decision.state.name] = states.get(result.decision.state.name, 0) + 1
            if result.decision.state.motion_permitted:
                was_permitted = True
            elif was_permitted:
                entry_arrivals.append(arrival_ns)
                was_permitted = False
            # Give the PLC thread a moment; without it the publisher outruns
            # the receiver and the reaction time measures the queue, not the bus.
            time.sleep(0.002)
        time.sleep(0.2)
    finally:
        pipeline.close()
        events = list(plc.events)
        denials = [e for e in events if not e.permitted]
        plc_stats = plc.stats.as_dict()
        plc.close()

    # Pair each entry with the PLC's denial.  An entry is only measurable if
    # the PLC had actually granted the permit beforehand: if it was already
    # withholding it -- because the previous entry has not been recovered from
    # yet -- there is no transition to time and counting the next unrelated
    # denial would produce a latency of several hundred milliseconds that
    # measures the scene script rather than the pipeline.
    reaction_ms: List[float] = []
    unmeasurable = 0
    for arrival in entry_arrivals:
        before = [e for e in events if e.t_ns < arrival]
        if not before or not before[-1].permitted:
            unmeasurable += 1
            continue
        after = [e for e in denials if e.t_ns >= arrival]
        if not after:
            unmeasurable += 1
            continue
        reaction_ms.append((after[0].t_ns - arrival) / NS_PER_MS)

    report = pipeline.report()
    report["detector_kind"] = detector_kind
    report["can_interface"] = interface
    report["states"] = states
    report["entry_events"] = len(entry_arrivals)
    report["entry_events_unmeasurable"] = unmeasurable
    report["reaction_ms"] = percentiles(reaction_ms)
    report["plc"] = plc_stats
    report["budget"] = _budget(report)
    return report


def _budget(report: Dict[str, object]) -> List[Dict[str, object]]:
    """The table that goes in the README, built from the measured stages.

    Shares are computed from the *means*, not the medians.  Percentiles do not
    add: the median of a sum is not the sum of the medians and a budget table
    built from p50s reports a large fictional "unaccounted" remainder that is
    an artefact of the arithmetic.  Means do add, so the shares are meaningful
    and sum to one.  The p50 and p99 columns are still shown because they are
    what you size a deadline against.
    """
    stages = report["stages_ms"]
    total = report["end_to_end_ms"]["mean"] or 1.0
    rows = []
    for name in STAGES:
        if name not in stages:
            continue
        row = stages[name]
        rows.append(
            {
                "stage": name,
                "mean_ms": row["mean"],
                "p50_ms": row["p50"],
                "p99_ms": row["p99"],
                "share_of_mean": row["mean"] / total,
            }
        )
    accounted = sum(r["mean_ms"] for r in rows)
    unaccounted = report["unaccounted_ms"]
    rows.append(
        {
            "stage": "unaccounted",
            "mean_ms": total - accounted,
            "p50_ms": unaccounted["p50"],
            "p99_ms": unaccounted["p99"],
            "share_of_mean": (total - accounted) / total,
        }
    )
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--detector",
        choices=("simulated", "onnx_timing", "onnx"),
        default="onnx_timing",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--backend",
        default="onnx",
        choices=("onnx", "tensorrt"),
        help="runtime for the real detector; the detector kinds keep their names",
    )
    parser.add_argument("--engine", default=None, help="the TensorRT engine to load")
    parser.add_argument("--repeats", type=int, default=30, help="how many entry events")
    parser.add_argument("--dwell", type=float, default=0.8)
    parser.add_argument(
        "--gap",
        type=float,
        default=1.6,
        help="quiet time between entries; long enough for the permit to be "
        "granted again or the next entry is not a measurable transition",
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--can", default=None, choices=(None, "virtual", "socketcan"))
    parser.add_argument("--channel", default=None)
    parser.add_argument("--zones", default=str(REPO_ROOT / "config" / "zones.yaml"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    zone_set = load_zones(args.zones)
    scene = SyntheticScene(entry_scene(args.dwell, args.gap, args.repeats, args.fps))
    report = measure(
        scene,
        zone_set,
        detector_kind=args.detector,
        model=args.model,
        can_interface=args.can,
        channel=args.channel,
        repeats=args.repeats,
        backend=args.backend,
        engine=args.engine,
    )
    report["scene"] = scene.summary()

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"wrote {path}")

    if not args.quiet:
        print(f"\ndetector: {report['detector_kind']}   bus: {report['can_interface']}")
        print(f"{'stage':<18}{'mean ms':>9}{'p50 ms':>9}{'p99 ms':>9}{'share':>9}")
        for row in report["budget"]:
            print(
                f"{row['stage']:<18}{row['mean_ms']:9.3f}{row['p50_ms']:9.3f}"
                f"{row['p99_ms']:9.3f}{row['share_of_mean'] * 100:8.1f}%"
            )
        e2e = report["end_to_end_ms"]
        reaction = report["reaction_ms"]
        print(
            f"\npipeline end to end   p50 {e2e['p50']:.2f}  p95 {e2e['p95']:.2f}  "
            f"p99 {e2e['p99']:.2f}  max {e2e['max']:.2f} ms  (n={e2e['n']})"
        )
        print(
            f"reaction at the PLC   p50 {reaction['p50']:.2f}  p95 {reaction['p95']:.2f}  "
            f"p99 {reaction['p99']:.2f}  max {reaction['max']:.2f} ms  (n={reaction['n']})"
        )
        print(
            f"  ({report['entry_events']} entry events, "
            f"{report['entry_events_unmeasurable']} of them not measurable)"
        )
        if reaction["n"] < 5:
            print(
                "  (too few entry events to trust those percentiles; raise --repeats)",
            )
        print(f"\nstates: {json.dumps(report['states'])}")
        print(f"plc:    {json.dumps(report['plc'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
