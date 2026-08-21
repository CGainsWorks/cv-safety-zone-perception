#!/usr/bin/env python3
"""Time the individual pieces, so the budget can be argued with.

    python scripts/benchmark.py --out results/benchmark.json

``analysis/latency_budget.py`` measures the pipeline end to end.  This measures
the parts on their own, which is what tells you whether a stage is slow because
of the work it does or because of what it is waiting for.  It also times the
detector's warm-up curve, because the first inference in a session is several
times slower than the steady state and quoting it would be flattering.

Everything here is machine-specific.  The machine is printed at the top of the
output and written into the result file and no number from this script belongs
in the README without it.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from szp.data.synthetic import SceneConfig, SyntheticScene  # noqa: E402
from szp.detection.base import DetectorConfig, build_detector  # noqa: E402
from szp.detection.postprocess import decode_yolov8  # noqa: E402
from szp.detection.preprocess import letterbox  # noqa: E402
from szp.fusion.fuse import Fuser  # noqa: E402
from szp.fusion.lidar import LidarConfig, ScanCluster, scan_to_points, segment_scan  # noqa: E402
from szp.latency.instrument import percentiles  # noqa: E402
from szp.safety.can_frames import encode_estop  # noqa: E402
from szp.types import Cause, Detection2D, SafetyDecision, SafetyState  # noqa: E402
from szp.zones.config import load_zones  # noqa: E402
from szp.zones.occupancy import OccupancyEvaluator  # noqa: E402


def timeit(name: str, function: Callable[[], object], repeats: int, warmup: int = 3) -> Dict:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    summary = percentiles(samples)
    summary["name"] = name
    return summary


def scalar_segment(ranges, angles, config: LidarConfig) -> List[ScanCluster]:
    """A deliberately naive per-beam segmentation, to time the vectorised one against.

    This is the shape ``segment_scan`` had before it was rewritten as array
    operations.  It is kept here and nowhere else: it is not used by the
    pipeline and the comment in ``fusion/lidar.py`` about the array version
    being faster points at this script rather than asserting a number.
    """
    r = np.asarray(ranges, dtype=np.float64)
    a = np.asarray(angles, dtype=np.float64)
    valid = np.isfinite(r) & (r > config.min_range_m) & (r < config.max_range_m)
    clusters: List[List[int]] = []
    run: List[int] = []
    for i in range(len(r)):
        if not valid[i]:
            if len(run) >= config.min_points:
                clusters.append(run)
            run = []
            continue
        if run:
            previous = run[-1]
            dphi = abs(float(a[i] - a[previous]))
            denominator = np.sin(config.lambda_rad - dphi)
            threshold = (
                float("inf")
                if denominator <= 1e-6
                else float(r[previous]) * np.sin(dphi) / denominator
                + 3.0 * config.range_sigma_m
            )
            if abs(float(r[i] - r[previous])) > threshold or dphi >= config.lambda_rad:
                if len(run) >= config.min_points:
                    clusters.append(run)
                run = []
        run.append(i)
    if len(run) >= config.min_points:
        clusters.append(run)
    out = []
    for idx in clusters:
        points = scan_to_points(r[idx], a[idx])
        out.append(
            ScanCluster(
                points=points,
                centroid=points.mean(axis=0),
                width_m=float(np.linalg.norm(points[-1] - points[0])),
                min_range_m=float(r[idx].min()),
                mean_range_m=float(r[idx].mean()),
                angular_width_rad=float(abs(a[idx[-1]] - a[idx[0]])),
                person_like=False,
                index_range=(idx[0], idx[-1]),
            )
        )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--model", default=None)
    parser.add_argument("--skip-detector", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    machine = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    print(json.dumps(machine, indent=2))

    scene = SyntheticScene(SceneConfig(duration_s=4.0, fps=10.0))
    zone_set = load_zones(REPO_ROOT / "config" / "zones.yaml")
    index = next(
        i for i in range(scene.n_frames) if any(p.box_xyxy for p in scene.frame(i)[1])
    )
    bundle, truth = scene.frame(index)
    detections = [Detection2D(xyxy=p.box_xyxy, score=0.9) for p in truth if p.box_xyxy]
    fuser = Fuser(scene.extrinsics)
    evaluator = OccupancyEvaluator(zone_set)
    boxes = fuser.fuse(
        detections, bundle.depth, bundle.scan_ranges, bundle.scan_angles, scene.intrinsics
    )
    decision = SafetyDecision(SafetyState.STOP, Cause.ZONE_OCCUPIED, 0b1, 2.0, True)

    results = [
        timeit("scene.render", lambda: scene.render(index), max(5, args.repeats // 5)),
        timeit(
            "lidar.segment_scan (vectorised)",
            lambda: segment_scan(bundle.scan_ranges, bundle.scan_angles),
            args.repeats,
        ),
        timeit(
            "lidar.segment_scan (per beam, not used)",
            lambda: scalar_segment(bundle.scan_ranges, bundle.scan_angles, LidarConfig()),
            args.repeats,
        ),
        timeit(
            "fusion.fuse",
            lambda: fuser.fuse(
                detections,
                bundle.depth,
                bundle.scan_ranges,
                bundle.scan_angles,
                scene.intrinsics,
            ),
            args.repeats,
        ),
        timeit("zones.evaluate", lambda: evaluator.evaluate(boxes), args.repeats),
        timeit("can.encode_estop", lambda: encode_estop(decision, 7), args.repeats * 10),
        timeit("preprocess.letterbox", lambda: letterbox(bundle.rgb), args.repeats),
    ]

    if not args.skip_detector:
        try:
            detector = build_detector(args.model, DetectorConfig(), backend="onnx")
        except FileNotFoundError as exc:
            print(f"\nskipping the detector: {exc}")
        else:
            try:
                warmup_curve = []
                for i in range(12):
                    start = time.perf_counter_ns()
                    detector.detect(bundle.rgb)
                    warmup_curve.append((time.perf_counter_ns() - start) / 1e6)
                results.append(
                    timeit("detector.detect (warm)", lambda: detector.detect(bundle.rgb),
                           max(10, args.repeats // 2), warmup=0)
                )
                tensor, transform = letterbox(bundle.rgb)
                raw = detector.session.run(None, {detector.input_name: tensor})[0]
                results.append(
                    timeit(
                        "postprocess.decode_yolov8",
                        lambda: decode_yolov8(raw, transform, conf_threshold=0.15),
                        args.repeats,
                    )
                )
                info = dict(detector.info)
            finally:
                detector.close()
            print("\nwarm-up curve, ms per inference:")
            print("  " + "  ".join(f"{v:.1f}" for v in warmup_curve))
            machine["detector"] = info
            machine["warmup_curve_ms"] = warmup_curve

    print(f"\n{'what':<42}{'p50 ms':>9}{'p99 ms':>9}{'n':>6}")
    for row in results:
        print(f"{row['name']:<42}{row['p50']:9.4f}{row['p99']:9.4f}{row['n']:6d}")

    vectorised = next(r for r in results if r["name"].endswith("(vectorised)"))
    per_beam = next(r for r in results if r["name"].endswith("(per beam, not used)"))
    speedup = per_beam["p50"] / max(vectorised["p50"], 1e-9)
    print(f"\nscan segmentation: the array form is {speedup:.1f}x the per-beam loop")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"machine": machine, "results": results, "segmentation_speedup": speedup},
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
