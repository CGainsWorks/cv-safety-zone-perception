"""2D scan segmentation.

Segmentation is by adaptive breakpoint detection (Borges and Aldon, 2004).  The
threshold between consecutive returns is not a constant: two returns from one
surface are further apart in metres the further away and the more obliquely the
surface is viewed, so a fixed 10 cm gap over-segments a person at six metres
and merges two people at one metre.  The adaptive form

    d_max = r_{i-1} * sin(dphi) / sin(lambda - dphi) + 3 sigma_r

falls out of the sine rule for the worst-case surface tilt ``lambda`` and is
what this module uses.

Clusters are scored for how person-like they are but never dropped for failing
that test.  A cluster that is too wide might be two people standing together
and a cluster that is too narrow might be an arm; both are handed on with a
flag and it is the fusion stage that decides.  Discarding a return here is a
false negative that nothing downstream can recover from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["LidarConfig", "ScanCluster", "segment_scan", "scan_to_points"]


@dataclass(frozen=True)
class LidarConfig:
    #: Worst-case angle between the scan ray and the surface, radians.  Ten
    #: degrees is the value in the original paper and is deliberately
    #: pessimistic: a smaller value means a larger gap threshold and therefore
    #: fewer splits and merging is the safe failure here.
    lambda_rad: float = np.deg2rad(10.0)
    range_sigma_m: float = 0.02
    min_points: int = 2
    min_range_m: float = 0.05
    max_range_m: float = 20.0
    #: Plan-view width a single standing person subtends, metres.  Used only to
    #: tag clusters, never to drop them.
    person_width_min_m: float = 0.10
    person_width_max_m: float = 0.90


@dataclass(frozen=True)
class ScanCluster:
    """A run of consecutive returns that plausibly came from one object."""

    points: np.ndarray          # (N, 2) in the lidar frame, metres
    centroid: np.ndarray        # (2,)
    #: Distance from the first to the last point of the run.
    width_m: float
    min_range_m: float
    mean_range_m: float
    #: Angular extent covered by the run, radians.
    angular_width_rad: float
    person_like: bool
    index_range: tuple

    @property
    def n_points(self) -> int:
        return int(len(self.points))


def scan_to_points(ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Polar to Cartesian in the lidar frame (+x forward, +y left)."""
    r = np.asarray(ranges, dtype=np.float64).reshape(-1)
    a = np.asarray(angles, dtype=np.float64).reshape(-1)
    if r.shape != a.shape:
        raise ValueError(f"ranges {r.shape} and angles {a.shape} differ in length")
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


def _breakpoint_threshold(previous_range: float, dphi: float, config: LidarConfig) -> float:
    """The Borges-Aldon threshold for one pair of returns.

    ``segment_scan`` computes this for the whole scan at once; this scalar form
    is the readable statement of the formula and is what
    ``tests/test_lidar.py::test_vectorised_threshold_matches_scalar`` checks
    the array version against.
    """
    denom = np.sin(config.lambda_rad - dphi)
    if denom <= 1e-6:
        return float("inf")
    return float(previous_range * np.sin(dphi) / denom + 3.0 * config.range_sigma_m)


def segment_scan(
    ranges: np.ndarray,
    angles: np.ndarray,
    config: Optional[LidarConfig] = None,
) -> List[ScanCluster]:
    """Split one scan into clusters.

    Invalid returns (non-finite or outside the configured range window) act as
    hard breaks: an object seen either side of a dropout is two clusters, which
    is the conservative reading.
    """
    config = config or LidarConfig()
    r = np.asarray(ranges, dtype=np.float64).reshape(-1)
    a = np.asarray(angles, dtype=np.float64).reshape(-1)
    if r.shape != a.shape:
        raise ValueError(f"ranges {r.shape} and angles {a.shape} differ in length")
    valid = np.isfinite(r) & (r > config.min_range_m) & (r < config.max_range_m)

    # Vectorised break detection.  Consecutive valid returns are always
    # adjacent indices -- an invalid return forces a break -- so the "previous
    # point" in the adaptive threshold is always i - 1 and the whole decision
    # can be made as an array comparison rather than a Python loop over a
    # thousand beams.  scripts/benchmark.py times both forms; the array version
    # is several times faster on a 961-beam scan and the fusion stage is where
    # that shows up.
    dphi = np.abs(np.diff(a))
    previous_r = r[:-1]
    denom = np.sin(config.lambda_rad - dphi)
    with np.errstate(divide="ignore", invalid="ignore"):
        threshold = np.where(
            denom > 1e-6,
            previous_r * np.sin(dphi) / np.where(denom > 1e-6, denom, 1.0)
            + 3.0 * config.range_sigma_m,
            np.inf,
        )
    gap = np.abs(np.diff(r))
    unusable = (dphi <= 0.0) | (dphi >= config.lambda_rad)
    with np.errstate(invalid="ignore"):
        breaks = unusable | (gap > threshold)
    breaks = np.concatenate([[True], np.asarray(breaks, dtype=bool)])

    order = np.flatnonzero(valid)
    clusters: List[ScanCluster] = []
    if len(order) == 0:
        return clusters
    starts_a_run = np.empty(len(order), dtype=bool)
    starts_a_run[0] = True
    starts_a_run[1:] = (np.diff(order) != 1) | breaks[order[1:]]
    runs = np.split(order, np.flatnonzero(starts_a_run)[1:])

    for idx in runs:
        if len(idx) < config.min_points:
            continue
        pts = scan_to_points(r[idx], a[idx])
        span = float(np.linalg.norm(pts[-1] - pts[0]))
        angular = float(abs(a[idx[-1]] - a[idx[0]]))
        clusters.append(
            ScanCluster(
                points=pts,
                centroid=pts.mean(axis=0),
                width_m=span,
                min_range_m=float(r[idx].min()),
                mean_range_m=float(r[idx].mean()),
                angular_width_rad=angular,
                person_like=(
                    config.person_width_min_m <= span <= config.person_width_max_m
                ),
                index_range=(int(idx[0]), int(idx[-1])),
            )
        )
    return clusters


def fit_circle_centre(
    points: np.ndarray,
    radius: float,
    sensor_origin: Optional[np.ndarray] = None,
    iterations: int = 12,
) -> np.ndarray:
    """Centre of a circle of known radius through an arc of scan points.

    A lidar sees the near side of a person, so the centroid of the returns sits
    roughly one body radius in front of where the person actually is.  Using it
    as the position puts everybody a fifth of a metre closer to the sensor than
    they are, which is a systematic bias, not noise: it does not average away
    over frames and it does not shrink with more beams.

    Fitting a circle of fixed radius removes it.  The radius is fixed rather
    than estimated because an arc of thirty degrees does not constrain a radius
    and letting it float turns a small bias into a large variance.  Gauss-Newton
    on ``sum (|p - c| - r)^2``, started one radius behind the centroid along the
    view direction.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    centroid = p.mean(axis=0)
    origin = (
        np.zeros(2) if sensor_origin is None else np.asarray(sensor_origin, dtype=np.float64)
    )
    view = centroid - origin
    norm = float(np.linalg.norm(view))
    if norm < 1e-9:
        return centroid
    centre = centroid + radius * (view / norm)
    if len(p) < 3:
        return centre

    for _ in range(iterations):
        delta = centre - p
        distance = np.linalg.norm(delta, axis=1)
        good = distance > 1e-9
        if not good.any():
            break
        residual = distance[good] - radius
        jacobian = delta[good] / distance[good, None]
        step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        centre = centre + step
        if float(np.linalg.norm(step)) < 1e-9:
            break
    # A degenerate arc can send the fit somewhere silly.  Fall back to the
    # geometric estimate rather than reporting a person in the next room.
    if not np.all(np.isfinite(centre)) or float(np.linalg.norm(centre - centroid)) > 4 * radius:
        return centroid + radius * (view / norm)
    return centre


def clusters_to_array(clusters: Sequence[ScanCluster]) -> np.ndarray:
    """(N, 2) centroids, for the tests and the plots."""
    if not clusters:
        return np.zeros((0, 2))
    return np.stack([c.centroid for c in clusters], axis=0)
