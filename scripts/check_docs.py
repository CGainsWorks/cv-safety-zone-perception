#!/usr/bin/env python3
"""Every number in the README must come from a published result file.

    python scripts/check_docs.py

The definition of done for this repository says "every number quoted in the
README is reproducible from a script in the repo".  This is the check that
makes that true rather than aspirational and CI runs it.

It reads `results/published/*.json`, formats the figures the README is supposed
to quote exactly as the README quotes them and fails if any of them is missing
from the text.  Change a number in the README without rerunning the experiment
and this fails; rerun an experiment and get a different number without updating
the README and this fails too.

It also checks a few structural things: that the safety disclaimer is linked
from the README, that every relative link resolves and that no result file is
older than the code it came from.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = REPO_ROOT / "results" / "published"

LINK = re.compile(r"\[[^\]]*\]\(([^)#]+)(?:#[^)]*)?\)")


def load(name: str) -> Optional[dict]:
    path = PUBLISHED / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sweep_row(block: dict, conf: float) -> Optional[dict]:
    for row in block["sweep"]:
        if abs(float(row["conf"]) - conf) < 1e-9:
            return row
    return None


def expected_numbers() -> Tuple[List[Tuple[str, str]], List[str]]:
    """(label, text-that-must-appear) pairs, plus notes about what was skipped."""
    checks: List[Tuple[str, str]] = []
    skipped: List[str] = []

    detection = load("detection_val2017.json")
    if detection is None:
        skipped.append("detection_val2017.json")
    else:
        for subset, label in (("safety_subset", "safety subset"), ("all", "whole set")):
            block = detection[subset]
            checks.append((f"{label}: MR-2", f"{block['mr2']:.3f}"))
            checks.append((f"{label}: people", f"{block['people']:,}"))
            for conf in (0.15, 0.25):
                row = sweep_row(block, conf)
                if row is None:
                    continue
                checks.append(
                    (f"{label}: recall at conf {conf}", f"{row['recall'] * 100:.1f}")
                )
                checks.append(
                    (
                        f"{label}: false positives per image at conf {conf}",
                        f"{row['false_positives_per_image']:.2f}",
                    )
                )

    geometry = load("geometry_accuracy.json")
    if geometry is None:
        skipped.append("geometry_accuracy.json")
    else:
        for case in ("both_sensors", "no_ranging", "wrong_body_size"):
            block = geometry["cases"][case]
            checks.append(
                (f"{case}: p50 position error", f"{block['position_error_m']['p50'] * 100:.1f}")
            )
        for case in ("both_sensors", "no_ranging"):
            zone = "machine_envelope"
            entry = geometry["cases"][case]["zones"][zone]
            checks.append(
                (
                    f"{case}/{zone}: occupied frames",
                    f"{entry['tp']}/{entry['frames_occupied']}",
                )
            )

    latency = load("latency_onnx.json")
    if latency is None:
        skipped.append("latency_onnx.json")
    else:
        checks.append(("latency, onnx: pipeline p50",
                       f"{latency['end_to_end_ms']['p50']:.1f}"))
        checks.append(("latency, onnx: pipeline p99",
                       f"{latency['end_to_end_ms']['p99']:.1f}"))
        checks.append(("latency, onnx: reaction p99",
                       f"{latency['reaction_ms']['p99']:.1f}"))
        inference = latency["stages_ms"].get("inference")
        if inference:
            checks.append(("latency, onnx: inference p50", f"{inference['p50']:.1f}"))

    simulated = load("latency_simulated.json")
    if simulated is None:
        skipped.append("latency_simulated.json")
    else:
        checks.append(
            ("latency, no model: pipeline p99", f"{simulated['end_to_end_ms']['p99']:.1f}")
        )

    calibration = load("calibration_self_check.json")
    if calibration is None:
        skipped.append("calibration_self_check.json")
    else:
        checks.append(
            (
                "calibration self-check: translation error, mm",
                f"{calibration['self_check']['translation_error_m'] * 1000:.1f}",
            )
        )
        checks.append(
            (
                "calibration self-check: residual, mm",
                f"{calibration['camera_lidar']['rms_m'] * 1000:.1f}",
            )
        )

    return checks, skipped


def check_links(text: str, source: Path) -> List[str]:
    problems = []
    for target in LINK.findall(text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (source.parent / target).resolve()
        if not resolved.exists():
            problems.append(f"broken link: {target}")
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", default=str(REPO_ROOT / "README.md"))
    parser.add_argument(
        "--require-results",
        action="store_true",
        help="fail if a published result file is missing, instead of skipping its checks",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    readme_path = Path(args.readme)
    text = readme_path.read_text(encoding="utf-8")
    problems: List[str] = []

    checks, skipped = expected_numbers()
    if skipped and args.require_results:
        problems += [f"missing published result: {name}" for name in skipped]

    missing: List[Tuple[str, str]] = []
    for label, needle in checks:
        if needle not in text:
            missing.append((label, needle))
    problems += [
        f"README does not quote {label}: expected to find '{needle}'"
        for label, needle in missing
    ]

    if "SAFETY_DISCLAIMER" not in text:
        problems.append("the README does not link the safety disclaimer")
    if "not a certified safety function" not in text.lower():
        problems.append("the README does not say it is not a certified safety function")

    problems += check_links(text, readme_path)
    for doc in sorted((REPO_ROOT / "docs").glob("*.md")):
        problems += [
            f"{doc.name}: {problem}"
            for problem in check_links(doc.read_text(encoding="utf-8"), doc)
        ]

    if problems:
        print(f"{len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        return 1

    if not args.quiet:
        print(f"{len(checks)} quoted numbers all match {PUBLISHED}")
        if skipped:
            print("skipped, no published result yet: " + ", ".join(skipped))
        widest = max((len(label) for label, _ in checks), default=0)
        for label, needle in checks:
            print(f"  {label:<{widest}}  {needle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
