#!/usr/bin/env python3
"""Side by side: the YOLOv8n detector this repository used to ship and YOLOX-Tiny.

    python analysis/compare_detectors.py

Writes ``results/published/detector_comparison.json``.  The YOLOv8n numbers are
``results/published/`` as it stood at the last commit before the swap
(``--baseline-ref``), read through ``git show``, so they are the published
measurements rather than numbers retyped from memory.  The YOLOX-Tiny numbers
are the current published files.  Every delta is new minus old: a negative
recall delta, or a positive miss-rate, false-positive or latency delta, means
the new detector is worse.

Both were scored on all 5000 val2017 images by the same code.  Latency was
measured on the same desktop but not on the same day, so differences of a few
per cent are within run-to-run variation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = Path("results/published")


def at_ref(ref: str, name: str) -> dict:
    blob = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{ref}:{(PUBLISHED / name).as_posix()}"],
        check=True, capture_output=True,
    ).stdout
    return json.loads(blob.decode("utf-8"))


def current(name: str) -> dict:
    return json.loads((REPO_ROOT / PUBLISHED / name).read_text(encoding="utf-8"))


def sweep(block: dict, conf: float) -> dict:
    return next(row for row in block["sweep"] if abs(row["conf"] - conf) < 1e-9)


def summarise(load) -> dict:
    detection = load("detection_val2017.json")
    latency = load("latency_onnx.json")
    trt = load("tensorrt.json")
    out = {"model": detection["detector"]["model"]}
    for subset in ("safety_subset", "all"):
        block = detection[subset]
        out[subset] = {"mr2": block["mr2"]}
        for conf in (0.15, 0.25):
            row = sweep(block, conf)
            out[subset][f"recall_{conf:.2f}"] = row["recall"]
            out[subset][f"fp_per_image_{conf:.2f}"] = row["false_positives_per_image"]
    misses = detection["misses"]["0.15"]
    out["missed_below_50px_fraction_015"] = misses["missed_below_min_height_fraction"]
    out["miss_rate_by_height_015"] = {
        bucket: entry["miss_rate"] for bucket, entry in misses["by_height"].items()
    }
    out["onnx_cpu"] = {
        "inference_p50_ms": latency["stages_ms"]["inference"]["p50"],
        "pipeline_p50_ms": latency["end_to_end_ms"]["p50"],
        "pipeline_p99_ms": latency["end_to_end_ms"]["p99"],
    }
    out["tensorrt"] = {
        row["precision"]: {
            "recall_safety_015": row["recall_safety_015"],
            "mr2_safety": row["mr2_safety"],
            "inference_p50_ms": row["inference_p50_ms"],
            "pipeline_p50_ms": row["pipeline_p50_ms"],
        }
        for row in trt["rows"]
    }
    return out


def delta(new, old):
    if isinstance(new, dict):
        return {k: delta(new[k], old[k]) for k in new if k in old}
    if isinstance(new, (int, float)) and isinstance(old, (int, float)):
        return round(new - old, 5)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline-ref", default="8e5d88e",
                        help="the last commit whose results/published is YOLOv8n's")
    parser.add_argument("--out", default=str(REPO_ROOT / PUBLISHED / "detector_comparison.json"))
    args = parser.parse_args()

    old = summarise(lambda name: at_ref(args.baseline_ref, name))
    new = summarise(current)
    report = {
        "note": "YOLOv8n numbers are results/published at the baseline commit; YOLOX-Tiny "
                "numbers are the current results/published.  delta is YOLOX-Tiny minus "
                "YOLOv8n.",
        "baseline_ref": args.baseline_ref,
        "yolov8n": old,
        "yolox_tiny": new,
        "delta": delta(new, old),
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["delta"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
