#!/usr/bin/env python3
"""Print and plot the cell, so a zone file can be checked by eye.

    python scripts/show_zones.py
    python scripts/show_zones.py --plot docs/figures/zones.png

Prints, for each zone: its kind, area, bounding box and the effective
clearance a detection needs at a typical and at a poor position uncertainty.
Then the sensor positions in world coordinates and the camera's field of view
projected onto the floor, because the most common zone-file mistake is a
polygon in a part of the cell the camera cannot see.

This is the tool ``docs/zones.md`` tells a commissioning engineer to run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from szp.fusion.extrinsics import load_extrinsics  # noqa: E402
from szp.geometry import backproject  # noqa: E402
from szp.zones.config import load_zones  # noqa: E402
from szp.zones.polygon import bounding_box, polygon_centroid  # noqa: E402

TYPICAL_SIGMA_M = 0.05
POOR_SIGMA_M = 0.40


def floor_footprint(
    extrinsics, samples: int = 60, max_range_m: float = 12.0
) -> Optional[np.ndarray]:
    """Where the camera's field of view meets the floor, in world coordinates.

    Clipped at ``max_range_m``.  Rays near the horizon meet the floor tens of
    metres away and an unclipped footprint is a long thin wedge that makes the
    plot useless; beyond the sensors' working range the camera sees the floor
    but cannot range anybody standing on it, so the clip is honest as well as
    convenient.
    """
    intrinsics = extrinsics.intrinsics.get("camera")
    if intrinsics is None:
        return None
    T = extrinsics.T("world", "camera")
    origin = T[:3, 3]
    if origin[2] <= 0:
        return None

    edge = []
    width, height = intrinsics.width, intrinsics.height
    for u, v in (
        *[(x, 0.0) for x in np.linspace(0, width, samples)],
        *[(float(width), y) for y in np.linspace(0, height, samples)],
        *[(x, float(height)) for x in np.linspace(width, 0, samples)],
        *[(0.0, y) for y in np.linspace(height, 0, samples)],
    ):
        ray_cam = backproject(intrinsics.K, np.array([[u, v]]), np.array([1.0]))[0]
        direction = T[:3, :3] @ ray_cam
        if direction[2] >= -1e-6:
            continue                      # this ray never reaches the floor
        t = -origin[2] / direction[2]
        point = origin + t * direction
        offset = point[:2] - origin[:2]
        distance = float(np.linalg.norm(offset))
        if distance > max_range_m:
            point[:2] = origin[:2] + offset * (max_range_m / distance)
        edge.append(point[:2].copy())
    return np.asarray(edge) if len(edge) >= 3 else None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zones", default=str(REPO_ROOT / "config" / "zones.yaml"))
    parser.add_argument("--extrinsics", default=str(REPO_ROOT / "config" / "extrinsics.yaml"))
    parser.add_argument("--plot", default=None, help="write a plan-view PNG here")
    parser.add_argument(
        "--max-range",
        type=float,
        default=12.0,
        help="clip the camera's floor footprint at this range, metres",
    )
    args = parser.parse_args(argv)

    zone_set = load_zones(args.zones)
    extrinsics = load_extrinsics(args.extrinsics)

    print(f"{args.zones}: {len(zone_set)} zones, frame '{zone_set.frame}'")
    if zone_set.meta:
        for key, value in zone_set.meta.items():
            print(f"  {key}: {str(value).strip()}")
    print()
    header = (
        f"{'bit':>3} {'id':<20}{'kind':<6}{'area m2':>9}{'margin':>8}"
        f"{'clearance @0.05':>17}{'@0.40':>8}   bounding box"
    )
    print(header)
    for zone in zone_set.zones:
        x0, y0, x1, y1 = bounding_box(zone.polygon)
        print(
            f"{zone.bit:>3} {zone.id:<20}{zone.kind:<6}{zone.area_m2:9.2f}"
            f"{zone.margin_m:8.2f}{zone.clearance_for(TYPICAL_SIGMA_M):17.2f}"
            f"{zone.clearance_for(POOR_SIGMA_M):8.2f}"
            f"   x [{x0:6.2f} {x1:6.2f}]  y [{y0:6.2f} {y1:6.2f}]"
            + ("" if zone.enabled else "   DISABLED")
        )

    print("\nsensors, in world coordinates:")
    for frame in ("camera", "lidar", "base"):
        if not extrinsics.connects("world", frame):
            continue
        position = extrinsics.T("world", frame)[:3, 3]
        print(f"  {frame:<8} ({position[0]:6.2f}, {position[1]:6.2f}, {position[2]:6.2f})")

    view = floor_footprint(extrinsics, max_range_m=args.max_range)
    if view is not None:
        vx0, vy0, vx1, vy1 = bounding_box(view)
        print(
            f"  camera sees the floor over x [{vx0:.2f} {vx1:.2f}]  y [{vy0:.2f} {vy1:.2f}]"
            f"  (clipped at {args.max_range:.1f} m)"
        )
        from szp.zones.polygon import polygons_intersect

        for zone in zone_set.zones:
            if not polygons_intersect(zone.polygon, view):
                print(
                    f"  WARNING: '{zone.id}' is entirely outside the camera's view of the "
                    "floor.  Only the lidar can see into it."
                )

    if args.plot:
        _plot(zone_set, extrinsics, view, Path(args.plot))
    return 0


def _plot(zone_set, extrinsics, view, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    figure, axes = plt.subplots(figsize=(9, 7))
    if view is not None:
        axes.add_patch(
            MplPolygon(view, closed=True, facecolor="#f0f4ff", edgecolor="#7f96c8",
                       linestyle="--", linewidth=1.0, zorder=0,
                       label="camera view of the floor")
        )
    colours = {"stop": "#c0392b", "warn": "#e08e0b"}
    for zone in zone_set.zones:
        axes.add_patch(
            MplPolygon(
                zone.polygon,
                closed=True,
                facecolor=colours[zone.kind],
                edgecolor="black",
                alpha=0.30 if zone.enabled else 0.08,
                linewidth=1.4,
                zorder=2,
            )
        )
        # Warning zones usually enclose the stop zone they protect, so their
        # centroids land on top of each other.  Label them on their top edge.
        centre = polygon_centroid(zone.polygon)
        if zone.kind == "warn":
            centre = np.array([centre[0], bounding_box(zone.polygon)[3] - 0.2])
        axes.annotate(
            zone.id,
            centre,
            ha="center",
            va="center",
            fontsize=8,
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, linewidth=0),
        )
    for frame, marker in (("camera", "^"), ("lidar", "s"), ("base", "o")):
        if not extrinsics.connects("world", frame):
            continue
        position = extrinsics.T("world", frame)[:3, 3]
        axes.plot(position[0], position[1], marker, markersize=9, zorder=5, label=frame)

    # Frame the picture on the zones and the sensors, not on the field of view.
    points = [zone.polygon for zone in zone_set.zones]
    for frame in ("camera", "lidar", "base"):
        if extrinsics.connects("world", frame):
            points.append(extrinsics.T("world", frame)[:3, 3][:2].reshape(1, 2))
    stacked = np.concatenate(points, axis=0)
    pad = 1.5
    axes.set_xlim(stacked[:, 0].min() - pad, stacked[:, 0].max() + pad)
    axes.set_ylim(stacked[:, 1].min() - pad, stacked[:, 1].max() + pad)

    axes.set_aspect("equal")
    axes.grid(True, linewidth=0.4, alpha=0.5)
    axes.set_xlabel("world x, metres")
    axes.set_ylabel("world y, metres")
    axes.set_title(
        "Safety zones, plan view  (red: stop, amber: warn)", fontsize=11
    )
    axes.legend(loc="upper right", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
