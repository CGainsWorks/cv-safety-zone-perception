#!/usr/bin/env python3
"""Run the whole pipeline over the synthetic cell and write a result file.

    python scripts/run_demo.py --out results/demo.json

What it does, end to end: renders each frame of ``src/szp/data/synthetic.py``,
runs detection, fuses depth and lidar into world-frame boxes, tests them
against ``config/zones.yaml``, drives the fail-safe state machine, publishes
the e-stop frame on a CAN bus and has a simulated PLC on the other end of that
bus decide whether the machine may move.  Everything is timed.

Two detector choices and the difference matters:

``--detector simulated`` (default)
    Ground-truth boxes with jitter, misses and false positives injected.  Use
    this when the question is whether the *geometry and the safety logic* are
    right, because the 3D error is then the fusion error and nothing else.

``--detector onnx``
    The real YOLOv8n ONNX model, run on the rendered frames.  Use this when the
    question is *how long a frame takes*.  Do not read anything into the
    detections: rendered cylinders are not people and detection accuracy is
    measured on COCO by ``analysis/detection_metrics.py`` instead.

The operator reset is simulated.  Because STOP latches, a run with no operator
would sit in STOP from the first person to the end of the file; ``--auto-reset``
presses the button once the cause has been clear for a moment, which is what an
operator would do and what makes the recovery path visible in the output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from szp.data.synthetic import SceneConfig, SimulatedDetector, SyntheticScene  # noqa: E402
from szp.detection.base import DetectorConfig, build_detector  # noqa: E402
from szp.fusion.extrinsics import load_extrinsics  # noqa: E402
from szp.latency.instrument import Instrument, percentiles  # noqa: E402
from szp.pipeline import Pipeline, PipelineConfig  # noqa: E402
from szp.safety.fsm import FsmConfig  # noqa: E402
from szp.safety.plc_sim import SimulatedPlc  # noqa: E402
from szp.safety.publisher import CanPublisher, default_interface  # noqa: E402
from szp.safety.watchdog import WatchdogConfig  # noqa: E402
from szp.types import NS_PER_MS, SafetyState  # noqa: E402
from szp.zones.config import load_zones  # noqa: E402


def build_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=None, help="default: the whole scene")
    parser.add_argument("--duration", type=float, default=None, help="scene length in seconds")
    parser.add_argument("--detector", choices=("simulated", "onnx"), default="simulated")
    parser.add_argument("--model", default=None, help="path to the ONNX file")
    parser.add_argument(
        "--can",
        choices=("auto", "virtual", "socketcan", "none"),
        default="auto",
        help="'none' skips the bus entirely and reports no end-to-end latency",
    )
    parser.add_argument("--channel", default=None)
    parser.add_argument("--zones", default=str(REPO_ROOT / "config" / "zones.yaml"))
    parser.add_argument("--extrinsics", default=str(REPO_ROOT / "config" / "extrinsics.yaml"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--timeline", default=None, help="write a per-frame CSV here")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--miss-probability", type=float, default=0.05)
    parser.add_argument("--auto-reset", action="store_true", default=True)
    parser.add_argument("--no-auto-reset", dest="auto_reset", action="store_false")
    parser.add_argument(
        "--reset-delay-s",
        type=float,
        default=1.0,
        help="how long the cause must have been clear before the operator resets",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arguments().parse_args(argv)

    scene_kwargs: Dict[str, object] = {}
    if args.duration is not None:
        scene_kwargs["duration_s"] = args.duration
    if args.seed is not None:
        scene_kwargs["seed"] = args.seed
    scene = SyntheticScene(SceneConfig(**scene_kwargs))  # type: ignore[arg-type]
    n_frames = scene.n_frames if args.frames is None else min(args.frames, scene.n_frames)

    zone_set = load_zones(args.zones)
    # The scene knows its own extrinsics exactly.  The file is loaded anyway
    # and checked against them, so that a demo cannot quietly pass while
    # config/extrinsics.yaml says something different from the rig.
    file_extrinsics = load_extrinsics(args.extrinsics)
    drift = float(
        np.abs(file_extrinsics.T("world", "camera") - scene.T_world_camera).max()
    )
    if drift > 1e-4:
        raise SystemExit(
            f"{args.extrinsics} disagrees with the synthetic rig by {drift:.2e}.  "
            "One of them has been edited; fix the file rather than the check."
        )

    if args.detector == "simulated":
        detector = SimulatedDetector(
            scene,
            miss_probability=args.miss_probability,
            seed=args.seed if args.seed is not None else 7,
        )
    else:
        detector = build_detector(args.model, DetectorConfig(), backend="onnx")
        detector.warmup(5)

    publisher = None
    plc = None
    if args.can != "none":
        interface, channel = default_interface(args.channel)
        if args.can in ("virtual", "socketcan"):
            interface = args.can
        publisher = CanPublisher(channel=channel, interface=interface)
        plc = SimulatedPlc(channel=channel, interface=interface)
        plc.start()

    instrument = Instrument()
    # Frames are produced as fast as they render rather than in real time, so
    # the interval between them says nothing about whether the pipeline is
    # keeping up.  The period watchdog is turned off for offline replay and the
    # result file records that it was.
    pipeline = Pipeline(
        detector=detector,
        extrinsics=scene.extrinsics,
        zone_set=zone_set,
        publisher=publisher,
        config=PipelineConfig(
            fsm=FsmConfig(require_operator_reset=True),
            watchdog=WatchdogConfig(enforce_period=False),
        ),
        instrument=instrument,
    )

    timeline: List[Dict[str, object]] = []
    plc_latencies_ms: List[float] = []
    blocking_frames: List[int] = []
    # INIT is not a permitted state but the PLC has never granted the permit
    # at that point either, so the start-up frames are not a transition.
    previously_permitted = False
    clear_since: Optional[int] = None
    states: Dict[str, int] = {}
    t_start = time.perf_counter()

    try:
        for index in range(n_frames):
            bundle, truth = scene.frame(index)
            if isinstance(detector, SimulatedDetector):
                detector.set_truth(truth)

            # Stamp the frame as arriving now.  The scene's own timestamps
            # are scene time and the latency being measured is wall-clock.
            arrival_ns = time.perf_counter_ns()
            bundle_now = replace(
                bundle,
                t_arrival_ns=arrival_ns,
                t_rgb_ns=arrival_ns,
                t_depth_ns=arrival_ns,
                t_scan_ns=arrival_ns,
                meta={},
            )
            result = pipeline.process(bundle_now)
            states[result.decision.state.name] = states.get(result.decision.state.name, 0) + 1

            # Record the frames on which the machine went from permitted to
            # blocked.  The PLC's own event for that transition arrives a
            # moment later on its thread, so the two are paired after the run
            # rather than here: asking the PLC immediately would race it and
            # find nothing, which is exactly the bug the first version had.
            if result.decision.state.motion_permitted:
                previously_permitted = True
            elif previously_permitted:
                blocking_frames.append(arrival_ns)
                previously_permitted = False

            blocking = result.decision.state in (SafetyState.STOP, SafetyState.FAULT)
            cause_gone = not any(
                o.occupied and o.kind == "stop" for o in result.occupancy
            ) and result.health.all_ok
            if args.auto_reset and blocking and cause_gone:
                if clear_since is None:
                    clear_since = index
                elif (index - clear_since) / scene.config.fps >= args.reset_delay_s:
                    pipeline.fsm.request_reset()
                    if plc is not None:
                        plc.request_reset()
                    clear_since = None
            elif not blocking:
                clear_since = None

            timeline.append(
                {
                    "index": index,
                    "t_s": round(index / scene.config.fps, 3),
                    "state": result.decision.state.name,
                    "cause": int(result.decision.cause),
                    "detections": len(result.detections),
                    "boxes": len(result.boxes),
                    "unranged": pipeline.fusion_debug.n_unranged,
                    "fused": pipeline.fusion_debug.n_fused,
                    "zone_mask": result.decision.zone_mask,
                    "min_range_m": (
                        None if not np.isfinite(result.decision.min_range_m)
                        else round(result.decision.min_range_m, 3)
                    ),
                    "e2e_ms": None if result.t_published_ns is None else round(result.e2e_ms, 3),
                    "people_present": len(truth),
                }
            )
            if not args.quiet and index % 30 == 0:
                print(
                    f"[{index:4d}/{n_frames}] {result.decision.state.name:10s} "
                    f"det={len(result.detections)} boxes={len(result.boxes)} "
                    f"{result.decision.detail}",
                    flush=True,
                )
    finally:
        pipeline.close()
        if plc is not None:
            plc.close()

    if plc is not None:
        denials = [e for e in plc.events if not e.permitted]
        cursor = 0
        for arrival_ns in blocking_frames:
            while cursor < len(denials) and denials[cursor].t_ns < arrival_ns:
                cursor += 1
            if cursor >= len(denials):
                break
            delta_ms = (denials[cursor].t_ns - arrival_ns) / NS_PER_MS
            # A denial more than a second later belongs to a different event.
            if delta_ms < 1000.0:
                plc_latencies_ms.append(delta_ms)
            cursor += 1

    wall_s = time.perf_counter() - t_start
    report = pipeline.report()
    report["scene"] = scene.summary()
    report["detector_kind"] = args.detector
    report["states"] = states
    report["wall_clock_s"] = round(wall_s, 2)
    report["frames_run"] = n_frames
    report["can_interface"] = "none" if publisher is None else publisher.interface
    report["period_watchdog_enforced"] = False
    report["plc_end_to_end_ms"] = percentiles(plc_latencies_ms)
    if plc is not None:
        report["plc"] = plc.stats.as_dict()
        first_event_ns = plc.events[0].t_ns if plc.events else 0
        report["plc_permit_changes"] = [
            {
                "t_ms_since_first_change": round((e.t_ns - first_event_ns) / NS_PER_MS, 3),
                "permitted": e.permitted,
                "reason": e.reason,
                "state": None if e.state is None else e.state.name,
            }
            for e in plc.events
        ]

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
    if args.timeline:
        timeline_path = Path(args.timeline)
        timeline_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(timeline[0].keys()) if timeline else []
        with open(timeline_path, "w", encoding="utf-8", newline="") as handle:
            handle.write(",".join(keys) + "\n")
            for row in timeline:
                handle.write(",".join("" if row[k] is None else str(row[k]) for k in keys) + "\n")
        print(f"wrote {timeline_path}")

    if not args.quiet:
        print()
        print("states:", json.dumps(states))
        print("stages ms:", json.dumps(report["stages_ms"], indent=2, default=str))
        print("end to end ms (pipeline):", json.dumps(report["end_to_end_ms"], default=str))
        print("end to end ms (at the PLC):", json.dumps(report["plc_end_to_end_ms"], default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
