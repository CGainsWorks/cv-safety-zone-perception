"""The frame graph and the YAML it is loaded from.

``config/extrinsics.yaml`` lists edges, each one a transform ``T_parent_child``.
The graph is undirected for resolution purposes -- an edge can be traversed
backwards by inverting it -- so a file that gives ``base -> camera`` and
``world -> base`` answers a request for ``T("camera", "world")`` without anyone
writing the composition out by hand.

Two things this module refuses to do quietly:

* accept a rotation that is not orthonormal.  See ``geometry.se3``.
* resolve a frame pair that the graph does not connect.  A missing edge raises
  rather than falling back to identity, because an identity extrinsic between
  a lidar and a camera is a plausible-looking value that puts every person
  half a metre from where they are.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from szp.geometry import (
    is_rotation,
    quaternion_from_rotation,
    rotation_from_quaternion,
    rotation_from_rpy,
    rpy_from_rotation,
    se3,
    se3_inverse,
)
from szp.types import CameraIntrinsics


class FrameGraphError(KeyError):
    """Raised when two frames are not connected by any chain of edges."""


@dataclass
class ExtrinsicSet:
    """A set of rigid transforms plus the camera intrinsics that go with them."""

    edges: Dict[Tuple[str, str], np.ndarray]
    intrinsics: Dict[str, CameraIntrinsics]
    #: Free-form provenance: who calibrated this, when, with what residual.
    meta: Mapping[str, object]

    @property
    def frames(self) -> List[str]:
        names = set()
        for a, b in self.edges:
            names.add(a)
            names.add(b)
        return sorted(names)

    def add_edge(self, parent: str, child: str, T: np.ndarray) -> None:
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T_{parent}_{child} must be 4x4, got {T.shape}")
        if not is_rotation(T[:3, :3]):
            raise ValueError(f"T_{parent}_{child} has a non-orthonormal rotation")
        self.edges[(parent, child)] = T

    def T(self, target: str, source: str) -> np.ndarray:
        """``T_target_source``: the transform that maps a point in ``source``
        into ``target``."""
        if target == source:
            return np.eye(4)
        chain = self._path(target, source)
        out = np.eye(4)
        for step in chain:
            out = out @ step
        return out

    def _path(self, target: str, source: str) -> List[np.ndarray]:
        adjacency: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        for (parent, child), T in self.edges.items():
            adjacency.setdefault(parent, []).append((child, T))
            adjacency.setdefault(child, []).append((parent, se3_inverse(T)))
        # Breadth-first from target outwards, so the accumulated product is
        # already in T_target_x order when we arrive at source.
        seen = {target}
        queue: deque = deque([(target, [])])
        while queue:
            node, acc = queue.popleft()
            if node == source:
                return acc
            for nxt, T in adjacency.get(node, ()):
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append((nxt, acc + [T]))
        raise FrameGraphError(
            f"no chain of extrinsics connects '{source}' to '{target}'. "
            f"Known frames: {', '.join(self.frames) or '(none)'}"
        )

    def connects(self, target: str, source: str) -> bool:
        try:
            self._path(target, source)
        except FrameGraphError:
            return False
        return True

    def to_dict(self) -> Dict[str, object]:
        """Round-trippable through :func:`load_extrinsics`."""
        transforms = []
        for (parent, child), T in sorted(self.edges.items()):
            roll, pitch, yaw = rpy_from_rotation(T[:3, :3])
            transforms.append(
                {
                    "parent": parent,
                    "child": child,
                    "translation": [float(v) for v in T[:3, 3]],
                    "rpy": [float(roll), float(pitch), float(yaw)],
                    "quaternion": [float(v) for v in quaternion_from_rotation(T[:3, :3])],
                }
            )
        cameras = {
            name: {
                "fx": k.fx,
                "fy": k.fy,
                "cx": k.cx,
                "cy": k.cy,
                "width": k.width,
                "height": k.height,
                "distortion": list(k.distortion),
            }
            for name, k in sorted(self.intrinsics.items())
        }
        return {"meta": dict(self.meta), "cameras": cameras, "transforms": transforms}


def _rotation_from_entry(entry: Mapping[str, object]) -> np.ndarray:
    if "quaternion" in entry:
        return rotation_from_quaternion(entry["quaternion"])  # type: ignore[arg-type]
    if "rpy" in entry:
        roll, pitch, yaw = (float(v) for v in entry["rpy"])  # type: ignore[union-attr]
        return rotation_from_rpy(roll, pitch, yaw)
    if "rpy_deg" in entry:
        roll, pitch, yaw = (np.deg2rad(float(v)) for v in entry["rpy_deg"])  # type: ignore
        return rotation_from_rpy(roll, pitch, yaw)
    if "matrix" in entry:
        return np.asarray(entry["matrix"], dtype=np.float64)[:3, :3]
    raise ValueError(
        "transform entry needs one of 'quaternion', 'rpy', 'rpy_deg' or 'matrix'; "
        f"got keys {sorted(entry)}"
    )


def _translation_from_entry(entry: Mapping[str, object]) -> np.ndarray:
    if "translation" in entry:
        return np.asarray(entry["translation"], dtype=np.float64).reshape(3)
    if "matrix" in entry:
        return np.asarray(entry["matrix"], dtype=np.float64)[:3, 3]
    raise ValueError("transform entry needs 'translation' or 'matrix'")


def extrinsics_from_dict(doc: Mapping[str, object]) -> ExtrinsicSet:
    edges: Dict[Tuple[str, str], np.ndarray] = {}
    for entry in doc.get("transforms", ()) or ():  # type: ignore[union-attr]
        parent = str(entry["parent"])
        child = str(entry["child"])
        R = _rotation_from_entry(entry)
        t = _translation_from_entry(entry)
        key = (parent, child)
        if key in edges:
            raise ValueError(f"duplicate transform T_{parent}_{child}")
        edges[key] = se3(R, t)

    intrinsics: Dict[str, CameraIntrinsics] = {}
    for name, cam in (doc.get("cameras", {}) or {}).items():  # type: ignore[union-attr]
        intrinsics[str(name)] = CameraIntrinsics(
            fx=float(cam["fx"]),
            fy=float(cam["fy"]),
            cx=float(cam["cx"]),
            cy=float(cam["cy"]),
            width=int(cam["width"]),
            height=int(cam["height"]),
            distortion=tuple(float(v) for v in cam.get("distortion", ()) or ()),
        )
    return ExtrinsicSet(edges=edges, intrinsics=intrinsics, meta=dict(doc.get("meta", {}) or {}))


def load_extrinsics(path: os.PathLike | str) -> ExtrinsicSet:
    """Read ``config/extrinsics.yaml`` (or any file in the same schema)."""
    with open(Path(path), "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return extrinsics_from_dict(doc)


def save_extrinsics(extrinsics: ExtrinsicSet, path: os.PathLike | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(Path(path), "w", encoding="utf-8") as handle:
        yaml.safe_dump(extrinsics.to_dict(), handle, sort_keys=False, default_flow_style=False)


def required_frames(extrinsics: ExtrinsicSet, needed: Iterable[Sequence[str]]) -> None:
    """Fail fast at start-up rather than on the first frame.

    ``needed`` is an iterable of (target, source) pairs.  Every pair that the
    pipeline will ask for is checked here, at configuration load, so that a
    missing edge is a start-up error and not a mid-run exception on the frame
    where somebody happens to walk into shot.
    """
    missing = [pair for pair in needed if not extrinsics.connects(pair[0], pair[1])]
    if missing:
        pretty = ", ".join(f"T_{a}_{b}" for a, b in missing)
        raise FrameGraphError(f"extrinsics do not provide: {pretty}")


def default_extrinsics_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("SZP_EXTRINSICS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "config" / "extrinsics.yaml"
