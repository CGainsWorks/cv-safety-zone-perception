"""Testing 3D boxes against zones, with hysteresis.

The test itself is: dilate the person's footprint by
``margin_m + sigma_k * sigma_xy`` and ask whether it meets the zone polygon.
That dilation is never constructed -- see ``zones.polygon`` for why -- it is
folded into a distance comparison instead.

The height band is checked as an interval overlap rather than a containment
test, because a person whose estimated height is wrong should still count.  A
box from 0 to 1.2 m overlaps a zone band from 0.3 to 2.2 m and trips it.

Hysteresis is asymmetric by design.  ``enter_frames`` defaults to one, so a
zone trips on the first frame that sees somebody in it and the reaction time is
the pipeline latency and nothing more.  ``exit_frames`` defaults to five, so
the zone stays tripped through a five-frame detector dropout.  The evaluator
holds that counter state, which is the only mutable state in the whole zone
stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from szp.types import Box3D, ZoneOccupancy
from szp.zones.config import Zone, ZoneSet
from szp.zones.polygon import bounding_box, polygon_distance, signed_gap

__all__ = ["OccupancyEvaluator", "box_footprint", "zone_mask"]


def box_footprint(box: Box3D) -> np.ndarray:
    """The plan-view outline of a box, as the fusion stage left it.

    Falls back to an axis-aligned rectangle when fusion did not attach one,
    which is the case for boxes constructed by hand in the tests.
    """
    if box.footprint is not None:
        return np.asarray(box.footprint, dtype=np.float64).reshape(-1, 2)
    cx, cy, _ = box.center
    hx, hy = 0.5 * box.size[0], 0.5 * box.size[1]
    return np.array(
        [[cx - hx, cy - hy], [cx + hx, cy - hy], [cx + hx, cy + hy], [cx - hx, cy + hy]],
        dtype=np.float64,
    )


def _height_overlaps(box: Box3D, zone: Zone) -> bool:
    half = 0.5 * box.size[2]
    low = box.center[2] - half
    high = box.center[2] + half
    return high >= zone.z_min_m and low <= zone.z_max_m


@dataclass
class _ZoneState:
    occupied: bool = False
    consecutive_clear: int = 0
    consecutive_occupied: int = 0


class OccupancyEvaluator:
    """Per-frame zone evaluation.  Stateful only in the hysteresis counters."""

    def __init__(self, zone_set: ZoneSet):
        self.zone_set = zone_set
        self._state: Dict[str, _ZoneState] = {z.id: _ZoneState() for z in zone_set.zones}
        self._zone_bounds: Dict[str, Tuple[float, float, float, float]] = {
            z.id: bounding_box(z.polygon) for z in zone_set.zones
        }

    @staticmethod
    def _bounds(polygon: np.ndarray) -> Tuple[float, float, float, float]:
        return (
            float(polygon[:, 0].min()),
            float(polygon[:, 1].min()),
            float(polygon[:, 0].max()),
            float(polygon[:, 1].max()),
        )

    def reset(self) -> None:
        for state in self._state.values():
            state.occupied = False
            state.consecutive_clear = 0
            state.consecutive_occupied = 0

    def evaluate(self, boxes: Sequence[Box3D]) -> List[ZoneOccupancy]:
        results: List[ZoneOccupancy] = []
        footprints = [box_footprint(box) for box in boxes]
        for zone in self.zone_set.zones:
            if not zone.enabled:
                results.append(
                    ZoneOccupancy(
                        zone_id=zone.id,
                        kind=zone.kind,
                        occupied=False,
                        nearest_m=float("inf"),
                        box_indices=(),
                    )
                )
                continue
            raw, hits, nearest, from_unranged = self._raw_occupancy(zone, boxes, footprints)
            occupied = self._debounce(zone, raw)
            results.append(
                ZoneOccupancy(
                    zone_id=zone.id,
                    kind=zone.kind,
                    occupied=occupied,
                    nearest_m=nearest,
                    box_indices=tuple(hits),
                    from_unranged=from_unranged and bool(hits),
                )
            )
        return results

    def _raw_occupancy(
        self,
        zone: Zone,
        boxes: Sequence[Box3D],
        footprints: Sequence[np.ndarray],
    ) -> Tuple[bool, List[int], float, bool]:
        hits: List[int] = []
        nearest = float("inf")
        only_unranged = True
        zx0, zy0, zx1, zy1 = self._zone_bounds[zone.id]
        for i, box in enumerate(boxes):
            if not _height_overlaps(box, zone):
                continue
            clearance = zone.clearance_for(box.sigma_xy)
            # Axis-aligned rejection before the exact test.  The wedge from an
            # unranged detection can be twelve metres long with a hundred
            # boundary samples on it and most of the time it is nowhere near
            # the zone.  This turns the common case into four comparisons.
            fx0, fy0, fx1, fy1 = self._bounds(footprints[i])
            if (
                fx1 + clearance < zx0
                or fx0 - clearance > zx1
                or fy1 + clearance < zy0
                or fy0 - clearance > zy1
            ):
                nearest = min(
                    nearest,
                    max(zx0 - fx1, fx0 - zx1, zy0 - fy1, fy0 - zy1) - clearance,
                )
                continue
            distance = polygon_distance(footprints[i], zone.polygon)
            gap = distance - clearance
            if distance <= 0.0:
                gap = signed_gap(footprints[i], zone.polygon) - clearance
            nearest = min(nearest, gap)
            if gap <= 0.0:
                hits.append(i)
                if not box.range_ambiguous:
                    only_unranged = False
        return bool(hits), hits, nearest, only_unranged

    def _debounce(self, zone: Zone, raw: bool) -> bool:
        state = self._state[zone.id]
        if raw:
            state.consecutive_occupied += 1
            state.consecutive_clear = 0
            if not state.occupied and state.consecutive_occupied >= zone.enter_frames:
                state.occupied = True
        else:
            state.consecutive_clear += 1
            state.consecutive_occupied = 0
            if state.occupied and state.consecutive_clear >= zone.exit_frames:
                state.occupied = False
        return state.occupied

    def state_of(self, zone_id: str) -> bool:
        return self._state[zone_id].occupied


def zone_mask(occupancy: Sequence[ZoneOccupancy], zone_set: ZoneSet) -> int:
    """Pack occupancy into the 16-bit field the CAN frame carries."""
    bits = {z.id: z.bit for z in zone_set.zones}
    mask = 0
    for entry in occupancy:
        if entry.occupied:
            bit = bits.get(entry.zone_id, -1)
            if 0 <= bit < 16:
                mask |= 1 << bit
    return mask


def nearest_range_m(occupancy: Sequence[ZoneOccupancy], boxes: Sequence[Box3D]) -> float:
    """Distance to the nearest person, for the CAN frame.

    Reported for every box, not only the ones inside a zone, because the PLC
    uses it to derate speed as somebody approaches.  For an unranged box the
    near end of its range interval is used: the pessimistic reading.
    """
    del occupancy
    best = float("inf")
    for box in boxes:
        if box.range_ambiguous and box.footprint is not None:
            # The nearest point of the wedge, not its centre.  A person who
            # could be anywhere between two and nine metres away is reported
            # at two.
            footprint = np.asarray(box.footprint, dtype=np.float64)
            candidate = float(np.hypot(footprint[:, 0], footprint[:, 1]).min())
        else:
            candidate = float(np.hypot(box.center[0], box.center[1]))
        best = min(best, candidate)
    return best


def summarise(
    occupancy: Sequence[ZoneOccupancy], zone_set: Optional[ZoneSet] = None
) -> Dict[str, object]:
    del zone_set
    return {
        "occupied": [o.zone_id for o in occupancy if o.occupied],
        "stop_occupied": [o.zone_id for o in occupancy if o.occupied and o.kind == "stop"],
        "warn_occupied": [o.zone_id for o in occupancy if o.occupied and o.kind == "warn"],
    }
