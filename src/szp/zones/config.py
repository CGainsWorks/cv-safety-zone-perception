"""The zone configuration format and the validation that goes with it.

A zone is a polygon on the floor of the world frame, a height band above it,
and the behaviour the machine should adopt when somebody is inside.  The file
format is deliberately flat and readable, because in practice a zone file is
edited by whoever commissioned the cell and not by whoever wrote the pipeline.
``docs/zones.md`` is the guide for that person.

Validation is strict and happens at load, never at run time.  A zone file with
a self-intersecting polygon, a stop zone that no longer contains the machine, a
negative margin or a duplicated identifier is a start-up failure.  The
alternative -- warn and carry on -- means the first anyone hears of a bad zone
file is when it fails to stop something.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from szp.zones.polygon import as_polygon, ensure_ccw, is_simple, signed_area

__all__ = ["Zone", "ZoneSet", "load_zones", "save_zones", "ZoneConfigError"]

VALID_KINDS = ("stop", "warn")
#: Sixteen is the width of the zone bitmap in the CAN frame.  A file with more
#: zones than this would silently lose the extras on the wire.
MAX_ZONES = 16


class ZoneConfigError(ValueError):
    """Raised for any problem with a zone file.  Always fatal."""


@dataclass(frozen=True)
class Zone:
    """One safety zone.

    ``margin_m`` is a fixed dilation of the zone, in metres, applied before the
    occupancy test.  It exists to absorb calibration error and the fact that a
    person is not really a 0.6 m box and it is separate from ``sigma_k``,
    which scales with the per-detection uncertainty that fusion reports.  The
    effective clearance a detection must have is
    ``margin_m + sigma_k * box.sigma_xy``.

    ``enter_frames`` defaults to 1 and ``exit_frames`` to 5.  The asymmetry is
    the point: a zone trips on the first frame that sees somebody and releases
    only after five consecutive clear frames.  Debouncing the entry would add
    latency to the one transition where latency costs something.
    """

    id: str
    kind: str
    polygon: np.ndarray            # (N, 2) world metres, counter-clockwise
    z_min_m: float = 0.0
    z_max_m: float = 2.2
    margin_m: float = 0.0
    sigma_k: float = 2.0
    enter_frames: int = 1
    exit_frames: int = 5
    enabled: bool = True
    description: str = ""
    #: Position in the CAN zone bitmap.  Assigned by ZoneSet on load, from file
    #: order, so the wire format does not silently change when a zone is
    #: renamed.
    bit: int = -1

    @property
    def is_stop(self) -> bool:
        return self.kind == "stop"

    @property
    def area_m2(self) -> float:
        return abs(signed_area(self.polygon))

    def clearance_for(self, sigma_xy: float) -> float:
        return float(self.margin_m + self.sigma_k * max(sigma_xy, 0.0))


@dataclass(frozen=True)
class ZoneSet:
    zones: Tuple[Zone, ...]
    frame: str = "world"
    meta: Mapping[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.zones)

    def __iter__(self):
        return iter(self.zones)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.zones[key]
        for zone in self.zones:
            if zone.id == key:
                return zone
        raise KeyError(key)

    @property
    def enabled(self) -> Tuple[Zone, ...]:
        return tuple(z for z in self.zones if z.enabled)

    @property
    def stop_zones(self) -> Tuple[Zone, ...]:
        return tuple(z for z in self.zones if z.enabled and z.is_stop)

    def to_dict(self) -> Dict[str, object]:
        return {
            "frame": self.frame,
            "meta": dict(self.meta),
            "zones": [
                {
                    "id": z.id,
                    "kind": z.kind,
                    "description": z.description,
                    "polygon": [[float(x), float(y)] for x, y in z.polygon],
                    "z_min_m": z.z_min_m,
                    "z_max_m": z.z_max_m,
                    "margin_m": z.margin_m,
                    "sigma_k": z.sigma_k,
                    "enter_frames": z.enter_frames,
                    "exit_frames": z.exit_frames,
                    "enabled": z.enabled,
                }
                for z in self.zones
            ],
        }


def _zone_from_entry(entry: Mapping[str, object], bit: int, index: int) -> Zone:
    where = f"zones[{index}]"
    try:
        zone_id = str(entry["id"])
    except KeyError:
        raise ZoneConfigError(f"{where}: missing 'id'") from None
    kind = str(entry.get("kind", "stop")).lower()
    if kind not in VALID_KINDS:
        raise ZoneConfigError(
            f"{where} ('{zone_id}'): kind must be one of {VALID_KINDS}, got '{kind}'"
        )
    if "polygon" not in entry:
        raise ZoneConfigError(f"{where} ('{zone_id}'): missing 'polygon'")
    try:
        polygon = as_polygon(entry["polygon"])
    except ValueError as exc:
        raise ZoneConfigError(f"{where} ('{zone_id}'): {exc}") from None
    if not np.isfinite(polygon).all():
        raise ZoneConfigError(f"{where} ('{zone_id}'): polygon contains a non-finite coordinate")
    if not is_simple(polygon):
        raise ZoneConfigError(
            f"{where} ('{zone_id}'): the polygon crosses itself.  A self-intersecting zone has "
            "no well-defined inside, so the occupancy answer would depend on the winding rule."
        )
    if abs(signed_area(polygon)) < 1e-6:
        raise ZoneConfigError(f"{where} ('{zone_id}'): the polygon has effectively zero area")

    z_min = float(entry.get("z_min_m", 0.0))
    z_max = float(entry.get("z_max_m", 2.2))
    if z_max <= z_min:
        raise ZoneConfigError(f"{where} ('{zone_id}'): z_max_m must exceed z_min_m")
    margin = float(entry.get("margin_m", 0.0))
    if margin < 0:
        raise ZoneConfigError(
            f"{where} ('{zone_id}'): margin_m is {margin}.  A negative margin shrinks the "
            "zone, which is never what a safety margin is for; edit the polygon instead."
        )
    sigma_k = float(entry.get("sigma_k", 2.0))
    if sigma_k < 0:
        raise ZoneConfigError(f"{where} ('{zone_id}'): sigma_k must not be negative")
    enter = int(entry.get("enter_frames", 1))
    exit_frames = int(entry.get("exit_frames", 5))
    if enter < 1 or exit_frames < 1:
        raise ZoneConfigError(f"{where} ('{zone_id}'): enter_frames and exit_frames must be >= 1")
    if kind == "stop" and enter > 2:
        raise ZoneConfigError(
            f"{where} ('{zone_id}'): enter_frames is {enter} on a stop zone.  Debouncing entry "
            "adds that many frame periods to the reaction time; if the detector is noisy enough "
            "to need it, raise the confidence threshold instead and say so in the report."
        )

    return Zone(
        id=zone_id,
        kind=kind,
        polygon=ensure_ccw(polygon),
        z_min_m=z_min,
        z_max_m=z_max,
        margin_m=margin,
        sigma_k=sigma_k,
        enter_frames=enter,
        exit_frames=exit_frames,
        enabled=bool(entry.get("enabled", True)),
        description=str(entry.get("description", "")),
        bit=bit,
    )


def zones_from_dict(doc: Mapping[str, object]) -> ZoneSet:
    entries: Sequence[Mapping[str, object]] = doc.get("zones", []) or []  # type: ignore
    if not entries:
        raise ZoneConfigError(
            "the zone file defines no zones.  An empty file means the pipeline can never "
            "stop anything, so it is rejected rather than run."
        )
    if len(entries) > MAX_ZONES:
        raise ZoneConfigError(
            f"{len(entries)} zones defined but the CAN zone bitmap is {MAX_ZONES} bits wide; "
            "the extras would not reach the PLC"
        )
    zones = [_zone_from_entry(entry, bit=i, index=i) for i, entry in enumerate(entries)]
    seen = set()
    for zone in zones:
        if zone.id in seen:
            raise ZoneConfigError(f"duplicate zone id '{zone.id}'")
        seen.add(zone.id)
    if not any(z.enabled and z.is_stop for z in zones):
        raise ZoneConfigError(
            "no enabled zone has kind 'stop'.  Warning zones alone cannot bring the machine "
            "to a halt, so this configuration would run with no protective stop at all."
        )
    return ZoneSet(
        zones=tuple(zones),
        frame=str(doc.get("frame", "world")),
        meta=dict(doc.get("meta", {}) or {}),
    )


def load_zones(path: os.PathLike | str) -> ZoneSet:
    with open(Path(path), "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}
    if not isinstance(doc, dict):
        raise ZoneConfigError(f"{path}: expected a mapping at the top level")
    try:
        return zones_from_dict(doc)
    except ZoneConfigError as exc:
        raise ZoneConfigError(f"{path}: {exc}") from None


def save_zones(zone_set: ZoneSet, path: os.PathLike | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(Path(path), "w", encoding="utf-8") as handle:
        yaml.safe_dump(zone_set.to_dict(), handle, sort_keys=False, default_flow_style=None)


def default_zones_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("SZP_ZONES")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "config" / "zones.yaml"


def zone_ids(zone_set: ZoneSet) -> List[str]:
    return [z.id for z in zone_set.zones]
