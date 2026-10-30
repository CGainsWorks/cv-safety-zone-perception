#!/usr/bin/env python3
"""The TensorRT table in the README and docs/tensorrt.md, generated.

    python analysis/tensorrt_table.py           # rewrite the generated blocks
    python analysis/tensorrt_table.py --check   # fail if a block is stale

The table is rendered from ``results/published/tensorrt.json`` and nothing
else, including the caption that names the GPU, the driver, the CUDA and
TensorRT versions and the contention state, so none of those can drift from
the file the numbers came from.  ``scripts/check_docs.py`` runs the check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = REPO_ROOT / "results" / "published" / "tensorrt.json"
DOCUMENTS = (REPO_ROOT / "README.md", REPO_ROOT / "docs" / "tensorrt.md")
BLOCK = re.compile(
    r"(<!-- BEGIN generated:tensorrt -->\n)(.*?)(<!-- END generated:tensorrt -->)", re.DOTALL
)


def render(summary: dict) -> str:
    host = summary["host"]
    contention = summary["gpu_contention"]
    idle = max(contention["before"]["utilisation_percent_max"],
               contention["after"]["utilisation_percent_max"])
    processes = max(contention["before"]["compute_processes_max"],
                    contention["after"]["compute_processes_max"])
    if processes:
        gpu_state = "another compute process was listed, so treat the latency as contended"
    elif contention["contended"]:
        gpu_state = (
            f"no other compute process listed but utilisation reached {idle} % in the "
            "idle windows before or after the run, above the "
            f"{contention['idle_threshold_percent']} % threshold; this GPU also drives "
            "the desktop display"
        )
    else:
        gpu_state = f"no other compute process, at most {idle} % utilisation before and after"
    if contention.get("host_note"):
        gpu_state += f" ({contention['host_note']})"
    cpu_state = (
        f"CPU {'busy' if contention['cpu_busy'] else 'not busy'}, load average "
        f"{contention['before']['cpu_load_1min_max']:.2f} before and "
        f"{contention['after']['cpu_load_1min_max']:.2f} after on {host['cpu_count']} "
        "logical cores shared with other containers"
    )
    reference = summary["onnx_reference"]
    rows = summary["rows"]
    lines = [
        f"Measured on a {summary['device']}.  Driver `{host['driver']}`, CUDA "
        f"`{host['cuda_runtime']}`, TensorRT `{host['tensorrt']}`, image "
        f"`{host['container_image']}` under {host['under']}.  GPU: {gpu_state}.  "
        f"{cpu_state}.  Recall on {summary['dataset']}, {summary['images']} images; "
        f"latency over {rows[0]['reaction_events']} entry events on python-can's "
        f"`{rows[0]['can_interface']}` bus.",
        "",
        "| Detector | Recall, safety subset, 0.15 | Recall, safety subset, 0.25 | "
        "MR-2, safety subset | Inference p50 / p99 | Pipeline p50 / p99 | "
        "Reaction at the PLC p50 / p99 | Same detections as ONNX |",
        "|---|---|---|---|---|---|---|---|",
    ]
    failures = []
    for row in rows:
        if row["matches_onnx_on_coco"]:
            same = "yes"
        else:
            failures.append(row)
            same = "no, required" if row.get("equivalence_required", True) else "no, see below"
        lines.append(
            f"| TensorRT {row['precision'].upper()} | {row['recall_safety_015'] * 100:.1f} % | "
            f"{row['recall_safety_025'] * 100:.1f} % | {row['mr2_safety']:.3f} | "
            f"{row['inference_p50_ms']:.2f} / {row['inference_p99_ms']:.2f} ms | "
            f"{row['pipeline_p50_ms']:.2f} / {row['pipeline_p99_ms']:.2f} ms | "
            f"{row['reaction_p50_ms']:.2f} / {row['reaction_p99_ms']:.2f} ms | "
            f"{same} |"
        )
    lines.append(
        f"| ONNX Runtime, CPU (published) | {reference['recall_safety_015'] * 100:.1f} % | "
        f"see above | {reference['mr2_safety']:.3f} | "
        f"{reference['inference_p50_ms']:.2f} ms p50 | {reference['pipeline_p50_ms']:.2f} ms p50 "
        "| see above | reference |"
    )
    if failures:
        lines.append("")
        lines.append(
            "The box-for-box equivalence test on 16 COCO photographs with people in them "
            "is required of FP32 only; where an engine did not pass it, the first "
            "difference it found:"
        )
        lines.append("")
        for row in failures:
            lines.append(
                f"* TensorRT {row['precision'].upper()}: "
                f"{row.get('equivalence_failure') or 'no assertion message recorded'}"
            )
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if not SUMMARY.exists():
        print(f"no {SUMMARY.name}; nothing to render")
        return 0
    body = render(json.loads(SUMMARY.read_text(encoding="utf-8")))
    stale = []
    for document in DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        match = BLOCK.search(text)
        if match is None:
            stale.append(f"{document.name}: no generated:tensorrt block")
            continue
        if match.group(2) == body:
            continue
        if args.check:
            stale.append(f"{document.name}: the TensorRT table is stale")
            continue
        updated = text[: match.start(2)] + body + text[match.end(2):]
        with open(document, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(updated)
        print(f"rewrote the TensorRT table in {document.name}")
    for problem in stale:
        print(problem, file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
