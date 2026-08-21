#!/usr/bin/env python3
"""Draw the figures the README uses, from the published result files.

    python scripts/make_figures.py

Every figure is generated, never drawn by hand and every number on one comes
out of `results/published/`.  A diagram that is maintained separately from the
measurements drifts from them and a diagram of a pipeline is much more useful
when it also says how long each stage takes.

Produces, in `docs/figures/`:

    pipeline.png     the architecture, annotated with the measured per-stage
                     p50 from the latency run
    cell.png         plan view: zones, sensors and where people actually
                     walked, coloured by what the pipeline decided
    timeline.png     the demo run over time: state, detections, latency
    detection.png    miss rate against person height and the miss-rate curve
                     against false positives per image
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon as MplPolygon  # noqa: E402

from szp.data.synthetic import SceneConfig, SyntheticScene  # noqa: E402
from szp.fusion.extrinsics import load_extrinsics  # noqa: E402
from szp.zones.config import load_zones  # noqa: E402
from szp.zones.polygon import bounding_box, polygon_centroid  # noqa: E402

PUBLISHED = REPO_ROOT / "results" / "published"
FIGURES = REPO_ROOT / "docs" / "figures"

STATE_COLOUR = {
    "INIT": "#95a5a6",
    "CLEAR": "#27ae60",
    "WARN": "#e08e0b",
    "STOP": "#c0392b",
    "FAULT": "#8e44ad",
    "RECOVERING": "#2980b9",
}


def load(name: str) -> Optional[dict]:
    path = PUBLISHED / name
    if not path.exists():
        print(f"  skipping: no {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# -- pipeline ----------------------------------------------------------------


def figure_pipeline(latency: dict, path: Path) -> None:
    stages = [
        ("acquire", "RGB + depth + scan", "not measured;\nsee docs/latency.md"),
        ("detect_pre_post", "letterbox\nand decode", None),
        ("inference", "person\ndetector", None),
        ("fusion", "depth + lidar\nto 3D boxes", None),
        ("zones", "zone\noccupancy", None),
        ("safety", "fail-safe\nstate machine", None),
        ("publish", "CAN permit", None),
    ]
    measured = latency["stages_ms"]
    figure, axes = plt.subplots(figsize=(13.5, 3.6))
    axes.axis("off")

    width, height, gap = 1.62, 1.05, 0.32
    for index, (key, label, fallback) in enumerate(stages):
        x = index * (width + gap)
        row = measured.get(key)
        if row is not None:
            annotation = f"{row['p50']:.2f} ms p50\n{row['p99']:.2f} ms p99"
            share = row["mean"] / latency["end_to_end_ms"]["mean"]
            colour = plt.cm.Blues(0.20 + 0.65 * min(share, 1.0))
        else:
            annotation = fallback or ""
            colour = "#eeeeee"
        axes.add_patch(
            FancyBboxPatch(
                (x, 0.0),
                width,
                height,
                boxstyle="round,pad=0.05",
                facecolor=colour,
                edgecolor="#333333",
                linewidth=1.1,
            )
        )
        axes.text(x + width / 2, height * 0.62, label, ha="center", va="center", fontsize=9.5)
        axes.text(
            x + width / 2,
            height * 0.20,
            annotation,
            ha="center",
            va="center",
            fontsize=7.4,
            color="#333333",
        )
        if index:
            axes.add_patch(
                FancyArrowPatch(
                    (x - gap + 0.02, height / 2),
                    (x - 0.02, height / 2),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.1,
                    color="#333333",
                )
            )

    total_x = len(stages) * (width + gap)
    axes.add_patch(
        FancyArrowPatch(
            (total_x - gap + 0.02, height / 2),
            (total_x + 0.5, height / 2),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.1,
            color="#333333",
        )
    )
    e2e = latency["end_to_end_ms"]
    reaction = latency["reaction_ms"]
    axes.text(
        total_x + 0.6,
        height / 2,
        "PLC\n(permit, not command)",
        ha="left",
        va="center",
        fontsize=9.5,
    )

    axes.text(
        0.0,
        -0.42,
        f"frame arrival to the e-stop frame on the bus:  "
        f"p50 {e2e['p50']:.1f} ms   p95 {e2e['p95']:.1f} ms   p99 {e2e['p99']:.1f} ms"
        f"        measured at the PLC:  p99 {reaction['p99']:.1f} ms",
        fontsize=9.5,
        color="#222222",
    )
    axes.text(
        0.0,
        -0.75,
        f"{latency['detector_kind']} detector, {latency['can_interface']} bus, "
        f"{e2e['n']} frames.  Not a certified safety function -- "
        "see docs/SAFETY_DISCLAIMER.md",
        fontsize=8,
        color="#666666",
    )
    axes.set_xlim(-0.3, total_x + 3.6)
    axes.set_ylim(-1.0, height + 0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"  wrote {path}")


# -- cell --------------------------------------------------------------------


def figure_cell(timeline_rows: List[dict], path: Path) -> None:
    zone_set = load_zones(REPO_ROOT / "config" / "zones.yaml")
    extrinsics = load_extrinsics(REPO_ROOT / "config" / "extrinsics.yaml")
    scene = SyntheticScene(SceneConfig())

    figure, axes = plt.subplots(figsize=(9.5, 7.2))
    for zone in zone_set.zones:
        axes.add_patch(
            MplPolygon(
                zone.polygon,
                closed=True,
                facecolor="#c0392b" if zone.is_stop else "#e08e0b",
                edgecolor="black",
                alpha=0.28,
                linewidth=1.3,
                zorder=1,
            )
        )
        centre = polygon_centroid(zone.polygon)
        if zone.kind == "warn":
            centre = np.array([centre[0], bounding_box(zone.polygon)[3] - 0.2])
        axes.annotate(
            zone.id,
            centre,
            ha="center",
            va="center",
            fontsize=8,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, linewidth=0),
        )

    by_state: Dict[str, List[np.ndarray]] = {}
    for row in timeline_rows:
        t_s = float(row["t_s"])
        for position in scene.people_at(t_s):
            by_state.setdefault(row["state"], []).append(position)
    for state, positions in sorted(by_state.items()):
        stacked = np.asarray(positions)
        axes.scatter(
            stacked[:, 0],
            stacked[:, 1],
            s=9,
            color=STATE_COLOUR.get(state, "#000000"),
            label=f"{state} ({len(positions)} person-frames)",
            zorder=4,
            alpha=0.85,
        )

    for frame, marker in (("camera", "^"), ("lidar", "s"), ("base", "o")):
        if not extrinsics.connects("world", frame):
            continue
        position = extrinsics.T("world", frame)[:3, 3]
        axes.plot(position[0], position[1], marker, markersize=10, zorder=7, label=frame)

    axes.set_aspect("equal")
    axes.grid(True, linewidth=0.4, alpha=0.5)
    axes.set_xlabel("world x, metres")
    axes.set_ylabel("world y, metres")
    axes.set_title(
        "Where people walked and what the pipeline decided\n"
        "(true positions from the synthetic rig; colour is the pipeline's state that frame)",
        fontsize=11,
    )
    axes.legend(loc="upper left", fontsize=8, framealpha=0.9)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"  wrote {path}")


# -- timeline ----------------------------------------------------------------


def figure_timeline(timeline_rows: List[dict], path: Path) -> None:
    t = np.array([float(row["t_s"]) for row in timeline_rows])
    states = [row["state"] for row in timeline_rows]
    people = np.array([int(row["people_present"]) for row in timeline_rows])
    detections = np.array([int(row["detections"]) for row in timeline_rows])
    latency = np.array(
        [float(row["e2e_ms"]) if row["e2e_ms"] else np.nan for row in timeline_rows]
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11.5, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.4, 1.2]},
    )

    start = 0
    for index in range(1, len(states) + 1):
        if index == len(states) or states[index] != states[start]:
            axes[0].axvspan(
                t[start],
                t[min(index, len(t) - 1)],
                color=STATE_COLOUR.get(states[start], "#000000"),
                alpha=0.75,
            )
            start = index
    axes[0].set_yticks([])
    axes[0].set_ylabel("state")
    handles = [
        plt.Line2D([0], [0], color=colour, linewidth=8, label=state)
        for state, colour in STATE_COLOUR.items()
        if state in set(states)
    ]
    axes[0].legend(handles=handles, ncol=len(handles), fontsize=8, loc="upper center")

    axes[1].step(t, people, where="post", label="people in the cell (truth)", linewidth=1.4)
    axes[1].step(t, detections, where="post", label="detections", linewidth=1.4, alpha=0.8)
    axes[1].set_ylabel("count")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(True, linewidth=0.4, alpha=0.5)

    axes[2].plot(t, latency, linewidth=0.9)
    finite = latency[np.isfinite(latency)]
    if len(finite):
        axes[2].axhline(
            float(np.percentile(finite, 99)),
            linestyle="--",
            linewidth=1.0,
            color="#c0392b",
            label=f"p99 {np.percentile(finite, 99):.1f} ms",
        )
        axes[2].legend(fontsize=8, loc="upper left")
    axes[2].set_ylabel("frame arrival to\ne-stop frame, ms")
    axes[2].set_xlabel("scene time, seconds")
    axes[2].grid(True, linewidth=0.4, alpha=0.5)

    axes[0].set_title(
        "The reference run: three people walk through the cell, one at a time",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"  wrote {path}")


# -- detection ---------------------------------------------------------------


def figure_detection(detection: dict, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    conf = "0.15"
    buckets = detection["misses"][conf]["by_height"]
    order = sorted(buckets, key=lambda key: int(key.split("-")[0]))
    rates = [buckets[key]["miss_rate"] * 100 for key in order]
    counts = [buckets[key]["people"] for key in order]
    bars = axes[0].bar(range(len(order)), rates, color="#c0392b", alpha=0.8)
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(order, fontsize=9)
    axes[0].set_ylabel("missed, per cent")
    axes[0].set_xlabel("annotated person height, pixels")
    axes[0].set_title(f"Who gets missed, at confidence {conf}", fontsize=11)
    axes[0].grid(True, axis="y", linewidth=0.4, alpha=0.5)
    for bar, count, rate in zip(bars, counts, rates):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            rate + 1.5,
            f"{rate:.0f}%\nn={count}",
            ha="center",
            fontsize=7.5,
        )
    floor = detection["min_height_px"]
    axes[0].axvline(1.5, linestyle="--", color="#333333", linewidth=1.0)
    axes[0].text(
        1.6,
        max(rates) * 0.85,
        f"working range\nstarts at {floor:.0f} px",
        fontsize=8,
        color="#333333",
    )

    for subset, label, colour in (
        ("safety_subset", "people at least 50 px tall", "#27ae60"),
        ("all", "every annotated person", "#7f8c8d"),
    ):
        block = detection[subset]
        fppi = [row["false_positives_per_image"] for row in block["sweep"]]
        miss = [(1.0 - row["recall"]) * 100 for row in block["sweep"]]
        axes[1].plot(
            fppi, miss, "o-", color=colour, markersize=4,
            label=f"{label}  (MR-2 {block['mr2']:.3f})"
        )
        for row, x, y in zip(block["sweep"], fppi, miss):
            if row["conf"] in (0.15, 0.5):
                axes[1].annotate(
                    f"conf {row['conf']:.2f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=7.5,
                    color=colour,
                )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("false positives per image")
    axes[1].set_ylabel("missed, per cent")
    axes[1].set_title("Miss rate against nuisance stops (COCO val2017)", fontsize=11)
    axes[1].grid(True, which="both", linewidth=0.4, alpha=0.5)
    axes[1].legend(fontsize=8, loc="upper right")

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"  wrote {path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(FIGURES))
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    latency = load("latency_onnx.json")
    if latency:
        figure_pipeline(latency, out / "pipeline.png")

    timeline_path = PUBLISHED / "demo_timeline.csv"
    if timeline_path.exists():
        with open(timeline_path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        figure_cell(rows, out / "cell.png")
        figure_timeline(rows, out / "timeline.png")
    else:
        print(f"  skipping: no {timeline_path}")

    detection = load("detection_val2017.json")
    if detection:
        figure_detection(detection, out / "detection.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
