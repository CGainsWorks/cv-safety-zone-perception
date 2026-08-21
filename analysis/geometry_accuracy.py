#!/usr/bin/env python3
"""Fusion accuracy and zone-decision accuracy on the synthetic cell.

    python analysis/geometry_accuracy.py --out results/geometry.json

This is the other half of the evidence.  ``detection_metrics.py`` answers "does
the detector find people in a real photograph"; this answers "given that it
did, is the person put in the right place on the floor and is the zone
decision that follows correct".  It runs on the synthetic rig because that is
the only place a metric ground truth exists -- see the module docstring of
``szp/data/synthetic.py`` for why and for what that does and does not prove.

Three things are measured.

**Fusion error.**  Horizontal distance between the estimated and the true
position of each person, broken down by which sensors contributed.  Reported
against range, because the error is not uniform: depth noise grows with the
square of range and the whole uncertainty model in the pipeline depends on that
being true.

**Zone confusion.**  For every frame and every zone, whether the pipeline said
occupied and whether it really was, giving a confusion matrix per zone.  The
number that matters is the false-negative count -- frames where somebody was
inside a stop zone and the pipeline did not say so -- and it is reported both
per frame and as runs, because ten consecutive missed frames is one dangerous
event and ten scattered ones are ten harmless ones.

**Sensor degradation.**  The same measurements with the depth camera removed,
the lidar removed, both removed and with dropout turned up.  A safety argument
that only holds with every sensor working is not a safety argument.

A caveat about the headline number
----------------------------------
The rig models a person as a cylinder and the two bias corrections in
``fusion/fuse.py`` -- front-surface to centre for depth, arc to centre for
lidar -- assume a body of exactly the radius ``FusionConfig.person_depth_m``
gives.  On this rig those assumptions are exactly true, so the residual error
in the ``both_sensors`` case is flatteringly small and should not be read as
what the pipeline would do on a real person.

The ``wrong_body_size`` case exists to put a number on that.  It renders people
who are a third wider than the fusion stage believes and reports the error
that results.  That is the more honest figure for the sensitivity of the
correction and it is the one the README quotes next to the other.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from szp.data.synthetic import GtPerson, SceneConfig, SimulatedDetector, SyntheticScene  # noqa: E402,E501
from szp.fusion.fuse import FusionConfig, FusionDebug, Fuser  # noqa: E402
from szp.latency.instrument import percentiles  # noqa: E402
from szp.types import Detection2D  # noqa: E402
from szp.zones.config import ZoneSet, load_zones  # noqa: E402
from szp.zones.occupancy import OccupancyEvaluator  # noqa: E402
from szp.zones.polygon import polygon_distance  # noqa: E402

#: A person is in view well enough to be ranged when their box is fully inside
#: the picture and tall enough to measure.  People at the very edge of frame are
#: counted separately: they are never dropped but their range comes from a few
#: pixels and the figures would be misleading mixed in with the rest.
MIN_BOX_HEIGHT_PX = 60.0


def well_framed(person: GtPerson, scene: SyntheticScene) -> bool:
    if person.box_xyxy is None or not person.in_view:
        return False
    x0, y0, x1, y1 = person.box_xyxy
    if (y1 - y0) < MIN_BOX_HEIGHT_PX:
        return False
    return not (
        x0 <= 2 or y0 <= 2 or x1 >= scene.config.width - 2 or y1 >= scene.config.height - 2
    )


def truth_footprint(person: GtPerson) -> np.ndarray:
    r = person.radius_m
    x, y = person.center_xy
    return np.array([[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y + r]])


def true_occupancy(truth: Sequence[GtPerson], zone_set: ZoneSet) -> Dict[str, bool]:
    """Whether each zone really is occupied, from the ground truth.

    Uses the zone's own ``margin_m`` but not ``sigma_k``: the margin is part of
    the definition of the protected area, the sigma term is the pipeline's
    allowance for its own uncertainty and counting it here would be marking the
    pipeline's homework with the pipeline's own ruler.
    """
    out = {}
    for zone in zone_set.zones:
        occupied = False
        for person in truth:
            if polygon_distance(truth_footprint(person), zone.polygon) <= zone.margin_m:
                occupied = True
                break
        out[zone.id] = occupied and zone.enabled
    return out


def run_case(
    scene: SyntheticScene,
    zone_set: ZoneSet,
    *,
    use_depth: bool,
    use_lidar: bool,
    miss_probability: float,
    seed: int,
) -> Dict[str, object]:
    fuser = Fuser(scene.extrinsics, FusionConfig())
    evaluator = OccupancyEvaluator(zone_set)
    detector = SimulatedDetector(
        scene, miss_probability=miss_probability, false_positive_rate=0.0, seed=seed
    )

    errors: List[float] = []
    by_source: Dict[str, List[float]] = defaultdict(list)
    by_range: Dict[str, List[float]] = defaultdict(list)
    edge_errors: List[float] = []
    counts = FusionDebug()
    confusion: Dict[str, Dict[str, int]] = {
        zone.id: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for zone in zone_set.zones
    }
    fn_runs: Dict[str, List[int]] = {zone.id: [] for zone in zone_set.zones}
    current_run: Dict[str, int] = {zone.id: 0 for zone in zone_set.zones}

    for index in range(scene.n_frames):
        bundle, truth = scene.frame(index)
        detector.set_truth(truth)
        detections: List[Detection2D] = detector.detect(bundle.rgb)
        frame_debug = FusionDebug()
        boxes = fuser.fuse(
            detections,
            bundle.depth if use_depth else None,
            bundle.scan_ranges if use_lidar else None,
            bundle.scan_angles if use_lidar else None,
            scene.intrinsics,
            debug=frame_debug,
        )
        for field in counts.__dict__:
            setattr(counts, field, getattr(counts, field) + getattr(frame_debug, field))

        for person in truth:
            if person.box_xyxy is None or not person.in_view or not boxes:
                continue
            distances = [
                float(np.hypot(b.center[0] - person.center_xy[0],
                               b.center[1] - person.center_xy[1]))
                for b in boxes
            ]
            best = int(np.argmin(distances))
            if well_framed(person, scene):
                errors.append(distances[best])
                by_source[boxes[best].source.value].append(distances[best])
                bucket = f"{int(person.range_m)}-{int(person.range_m) + 1}m"
                by_range[bucket].append(distances[best])
            else:
                edge_errors.append(distances[best])

        predicted = {o.zone_id: o.occupied for o in evaluator.evaluate(boxes)}
        actual = true_occupancy(truth, zone_set)
        for zone_id, is_occupied in actual.items():
            said = predicted.get(zone_id, False)
            key = ("tp" if said else "fn") if is_occupied else ("fp" if said else "tn")
            confusion[zone_id][key] += 1
            if key == "fn":
                current_run[zone_id] += 1
            elif current_run[zone_id]:
                fn_runs[zone_id].append(current_run[zone_id])
                current_run[zone_id] = 0
    for zone_id, run in current_run.items():
        if run:
            fn_runs[zone_id].append(run)

    summary = {
        "sensors": {"depth": use_depth, "lidar": use_lidar},
        "miss_probability": miss_probability,
        "frames": scene.n_frames,
        "fusion_counts": counts.as_dict(),
        "position_error_m": percentiles(errors),
        "position_error_by_source_m": {
            source: percentiles(values) for source, values in sorted(by_source.items())
        },
        "position_error_by_range_m": {
            bucket: percentiles(values) for bucket, values in sorted(by_range.items())
        },
        "edge_of_frame_error_m": percentiles(edge_errors),
        "zones": {},
    }
    for zone_id, matrix in confusion.items():
        occupied = matrix["tp"] + matrix["fn"]
        clear = matrix["tn"] + matrix["fp"]
        runs = fn_runs[zone_id]
        summary["zones"][zone_id] = dict(
            matrix,
            frames_occupied=occupied,
            frames_clear=clear,
            miss_rate=matrix["fn"] / occupied if occupied else 0.0,
            false_stop_rate=matrix["fp"] / clear if clear else 0.0,
            longest_miss_run_frames=max(runs) if runs else 0,
            miss_runs=len(runs),
        )
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--zones", default=str(REPO_ROOT / "config" / "zones.yaml"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    zone_set = load_zones(args.zones)
    base = SceneConfig(duration_s=args.duration, fps=args.fps, seed=args.seed)

    cases = {
        "both_sensors": dict(use_depth=True, use_lidar=True, miss_probability=0.05),
        "depth_only": dict(use_depth=True, use_lidar=False, miss_probability=0.05),
        "lidar_only": dict(use_depth=False, use_lidar=True, miss_probability=0.05),
        "no_ranging": dict(use_depth=False, use_lidar=False, miss_probability=0.05),
        "perfect_detector": dict(use_depth=True, use_lidar=True, miss_probability=0.0),
        "poor_detector": dict(use_depth=True, use_lidar=True, miss_probability=0.25),
    }
    results: Dict[str, object] = {"scene": None, "cases": {}}
    for name, kwargs in cases.items():
        scene = SyntheticScene(base)
        results["scene"] = scene.summary()
        if not args.quiet:
            print(f"running {name} ...", flush=True)
        results["cases"][name] = run_case(scene, zone_set, seed=args.seed, **kwargs)

    heavy = SyntheticScene(replace(base, depth_dropout=0.15, lidar_dropout=0.10))
    if not args.quiet:
        print("running heavy_dropout ...", flush=True)
    results["cases"]["heavy_dropout"] = run_case(
        heavy, zone_set, use_depth=True, use_lidar=True, miss_probability=0.05, seed=args.seed
    )

    # People a third wider than the fusion stage assumes.  The bias corrections
    # are calibrated on FusionConfig.person_depth_m, so this is the sensitivity
    # of those corrections to being wrong about the body.
    wider = replace(
        base,
        people=tuple(replace(track, radius_m=0.30) for track in base.people),
    )
    if not args.quiet:
        print("running wrong_body_size ...", flush=True)
    results["cases"]["wrong_body_size"] = run_case(
        SyntheticScene(wider),
        zone_set,
        use_depth=True,
        use_lidar=True,
        miss_probability=0.05,
        seed=args.seed,
    )
    results["cases"]["wrong_body_size"]["note"] = (
        "people rendered at radius 0.30 m while fusion assumes "
        f"{FusionConfig().person_depth_m / 2:.3f} m"
    )

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"wrote {path}")

    if not args.quiet:
        print()
        header = f"{'case':<18}{'p50 err':>9}{'p95 err':>9}{'unranged':>10}"
        for zone in zone_set.stop_zones:
            header += f"{zone.id + ' FN':>24}"
        print(header)
        for name, case in results["cases"].items():
            row = (
                f"{name:<18}"
                f"{case['position_error_m']['p50']:9.3f}"
                f"{case['position_error_m']['p95']:9.3f}"
                f"{case['fusion_counts']['n_unranged']:10d}"
            )
            for zone in zone_set.stop_zones:
                z = case["zones"][zone.id]
                row += f"{z['fn']:>10d}/{z['frames_occupied']:<5d}({z['miss_rate']:.3f})  "
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
