#!/usr/bin/env python3
"""Build the TensorRT engines on this GPU, score them on COCO and time them.

    docker build -f docker/Dockerfile.tensorrt -t szp:tensorrt .
    docker run --rm --gpus all --network none -v "$PWD:/work" \\
        -v ~/datasets/coco:/coco:ro -e SZP_COCO_ROOT=/coco szp:tensorrt \\
        python3 scripts/measure_tensorrt.py

or ``make tensorrt-desktop``, which runs exactly that.

Nothing here is a second implementation.  ``scripts/export_tensorrt.py``
builds the engines, ``analysis/detection_metrics.py --backend tensorrt`` scores
them with the same decoder and the same recall and MR-2 code that produced
``results/published/detection_val2017.json``, ``analysis/latency_budget.py
--backend tensorrt`` times the whole pipeline from frame arrival to the frame
at the simulated PLC and ``tests/test_backends.py::test_tensorrt_matches_onnx``
compares each engine with ONNX Runtime on real photographs.  This script runs
them in order, watches the machine and writes one compact summary to
``results/published/tensorrt.json``.

Precisions.  FP32 and FP16.  There is no INT8 row because this repository has
no INT8 path and no calibration set: the detector is Ultralytics' COCO
checkpoint used as shipped and a calibration set carved out of val2017 would
calibrate on the images it is then scored on.

What the numbers are.  Desktop numbers from whichever NVIDIA GPU the container
was given, an RTX 4060 under WSL2 for the published file.  Not Jetson numbers
and the summary says so in its ``device`` field.  The CAN bus inside the
container is python-can's in-process virtual bus, as in
``results/published/latency_onnx.json``; it is not representative of a real
bus and the latency table says which interface it used.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

PRECISIONS = ("fp32", "fp16")
IDLE_UTILISATION_PERCENT = 10
# A one-minute load average above this fraction of the logical cores counts as
# a busy CPU.  Pre- and post-processing, fusion and the CAN path all run on the
# CPU, so a busy CPU moves the pipeline numbers even when the GPU is idle.
BUSY_CPU_FRACTION = 0.25


def _smi(*query: str) -> list:
    output = subprocess.run(
        ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    return [[cell.strip() for cell in line.split(",")] for line in output.splitlines() if line]


def gpu_sample() -> dict:
    name, driver, total, used, util = _smi(
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu"
    )[0]
    return {
        "name": name,
        "driver": driver,
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "utilisation_percent": int(util),
        "compute_processes": len(_smi("--query-compute-apps=pid,process_name,used_memory")),
    }


def load_average() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except OSError:  # pragma: no cover - not Linux
        return -1.0


def watch(seconds: int) -> dict:
    samples, loads = [], []
    for _ in range(seconds):
        samples.append(gpu_sample())
        loads.append(load_average())
        time.sleep(1.0)
    return {
        "samples": len(samples),
        "utilisation_percent_max": max(s["utilisation_percent"] for s in samples),
        "memory_used_mib_max": max(s["memory_used_mib"] for s in samples),
        "compute_processes_max": max(s["compute_processes"] for s in samples),
        "cpu_load_1min_max": max(loads),
    }


def run(command: list, env: dict | None = None) -> int:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, env={**os.environ, **(env or {})}).returncode


def sweep_row(block: dict, conf: float) -> dict:
    return next(row for row in block["sweep"] if abs(row["conf"] - conf) < 1e-9)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="models/yolov8n.onnx")
    parser.add_argument("--repeats", type=int, default=30, help="entry events for the latency run")
    parser.add_argument("--image", default=os.environ.get("TRT_IMAGE", "unknown"))
    parser.add_argument("--host-note", default="",
                        help="what nvidia-smi on the host showed during the run")
    parser.add_argument("--reuse-engines", action="store_true")
    parser.add_argument("--out", default="results/published/tensorrt.json")
    args = parser.parse_args(argv)

    import pycuda
    import tensorrt

    python = sys.executable
    model = Path(args.model)
    engines = {p: model.with_name(f"{model.stem}_{p}.engine") for p in PRECISIONS}

    before = watch(5)
    for precision, engine in engines.items():
        if args.reuse_engines and engine.exists():
            continue
        flag = "--fp16" if precision == "fp16" else "--no-fp16"
        if run([python, "scripts/export_tensorrt.py", "--model", str(model),
                "--engine", str(engine), flag]):
            raise SystemExit(f"the {precision} engine did not build")

    published = json.loads(
        (REPO_ROOT / "results/published/detection_val2017.json").read_text(encoding="utf-8")
    )
    published_latency = json.loads(
        (REPO_ROOT / "results/published/latency_onnx.json").read_text(encoding="utf-8")
    )
    rows = []
    for precision, engine in engines.items():
        detection_out = Path(f"results/tensorrt_detection_{precision}.json")
        latency_out = Path(f"results/tensorrt_latency_{precision}.json")
        if run([python, "analysis/detection_metrics.py", "--model", str(model),
                "--backend", "tensorrt", "--engine", str(engine),
                "--out", str(detection_out), "--quiet"]):
            raise SystemExit(f"scoring the {precision} engine failed")
        # python-can's in-process virtual bus, as in the published ONNX row, so
        # the two differ only in the detector.  Inside a container there is no
        # vcan0 for the Linux default to open.
        if run([python, "analysis/latency_budget.py", "--detector", "onnx_timing",
                "--backend", "tensorrt", "--engine", str(engine), "--model", str(model),
                "--repeats", str(args.repeats), "--out", str(latency_out), "--quiet"],
               env={"SZP_CAN_INTERFACE": "virtual", "SZP_CAN_CHANNEL": "szp"}):
            raise SystemExit(f"timing the {precision} engine failed")
        # A skipped test also exits 0, so the summary line is what counts:
        # exactly one test must have passed.
        equivalence = subprocess.run(
            [python, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", "-rs",
             "tests/test_backends.py::test_tensorrt_matches_onnx"],
            env={**os.environ, "SZP_ENGINE": str(engine), "SZP_MODEL": str(model)},
            capture_output=True, text=True,
        )
        print(equivalence.stdout.strip().splitlines()[-1], flush=True)
        matched = equivalence.returncode == 0 and " 1 passed" in equivalence.stdout

        detection = json.loads(detection_out.read_text(encoding="utf-8"))
        latency = json.loads(latency_out.read_text(encoding="utf-8"))
        card = json.loads(engine.with_suffix(".engine.json").read_text(encoding="utf-8"))
        safety, whole = detection["safety_subset"], detection["all"]
        stages = {row["stage"]: row for row in latency["budget"]}
        rows.append({
            "variant": f"tensorrt-{precision}",
            "precision": precision,
            "engine_bytes": card["engine_bytes"],
            "build_seconds": card["build_seconds"],
            "recall_safety_015": sweep_row(safety, 0.15)["recall"],
            "recall_safety_025": sweep_row(safety, 0.25)["recall"],
            "recall_all_015": sweep_row(whole, 0.15)["recall"],
            "fp_per_image_safety_015": sweep_row(safety, 0.15)["false_positives_per_image"],
            "mr2_safety": safety["mr2"],
            "mr2_all": whole["mr2"],
            "recall_safety_015_delta_vs_onnx": round(
                sweep_row(safety, 0.15)["recall"]
                - sweep_row(published["safety_subset"], 0.15)["recall"], 5
            ),
            "mr2_safety_delta_vs_onnx": round(safety["mr2"] - published["safety_subset"]["mr2"], 5),
            "inference_p50_ms": round(stages["inference"]["p50_ms"], 3),
            "inference_p99_ms": round(stages["inference"]["p99_ms"], 3),
            "pipeline_p50_ms": round(latency["end_to_end_ms"]["p50"], 3),
            "pipeline_p99_ms": round(latency["end_to_end_ms"]["p99"], 3),
            "reaction_p50_ms": round(latency["reaction_ms"]["p50"], 3),
            "reaction_p99_ms": round(latency["reaction_ms"]["p99"], 3),
            "reaction_events": latency["reaction_ms"]["n"],
            "can_interface": latency["can_interface"],
            "matches_onnx_on_coco": matched,
        })

    time.sleep(5.0)  # keep this run's own tail out of the "after" sample
    after = watch(5)
    cpus = os.cpu_count() or 1
    gpu = gpu_sample()
    summary = {
        "device": f"desktop {gpu['name'].replace('NVIDIA GeForce ', '')}, not a Jetson",
        "host": {
            "gpu": gpu["name"],
            "gpu_memory_mib": gpu["memory_total_mib"],
            "driver": gpu["driver"],
            "cuda_runtime": os.environ.get("CUDA_VERSION", "unknown"),
            "tensorrt": tensorrt.__version__,
            "pycuda": pycuda.VERSION_TEXT,
            "container_image": args.image,
            "under": "WSL2 Docker" if "microsoft" in os.uname().release.lower() else "Linux",
            "cpu_count": cpus,
        },
        "gpu_contention": {
            "idle_threshold_percent": IDLE_UTILISATION_PERCENT,
            "before": before,
            "after": after,
            "contended": any(
                w["utilisation_percent_max"] > IDLE_UTILISATION_PERCENT
                or w["compute_processes_max"] > 0
                for w in (before, after)
            ),
            "busy_cpu_threshold_load": cpus * BUSY_CPU_FRACTION,
            "cpu_busy": any(w["cpu_load_1min_max"] > cpus * BUSY_CPU_FRACTION
                            for w in (before, after)),
            "host_note": args.host_note,
        },
        "dataset": "COCO val2017",
        "images": published["images_scored"],
        "onnx_reference": {
            "recall_safety_015": sweep_row(published["safety_subset"], 0.15)["recall"],
            "mr2_safety": published["safety_subset"]["mr2"],
            "inference_p50_ms": round(
                next(b for b in published_latency["budget"] if b["stage"] == "inference")["p50_ms"],
                3,
            ),
            "pipeline_p50_ms": round(published_latency["end_to_end_ms"]["p50"], 3),
            "provider": "CPUExecutionProvider",
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if all(row["matches_onnx_on_coco"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
