"""Turning a 2D box plus a depth map into a range.

The naive answer -- take the median depth inside the box -- is wrong for a
person and wrong in the direction that matters.  A tight box around a standing
person still contains a lot of floor and wall between the arms and the torso,
often a majority of the pixels at longer ranges where the person is small.  The
median then lands on the background, the person is placed several metres
further away than they are and a person standing on the edge of a stop zone is
reported as safely outside it.

So the estimator here looks for the *nearest* dense cluster of depths inside the
box rather than the most common one, on the reasoning that nothing in the box is
in front of the person.  A histogram at ``bin_m`` resolution finds the modes,
any mode holding at least ``mode_fraction`` of the strongest mode's mass counts
as real and the nearest of those wins.  The final number is the median of the
returns within one bin of the winning mode, which takes the edge off the
quantisation.

Depth noise is modelled as growing with the square of range, which is what a
stereo or structured-light sensor does: constant disparity noise turns into
range noise proportional to z^2 / (f b).  That model is what puts a wide
``sigma_xy`` on a distant person and, through the footprint dilation in
``zones.occupancy``, makes a distant person more likely to trip a zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from szp.types import Detection2D

__all__ = ["DepthConfig", "DepthEstimate", "estimate_depth", "sigma_from_depth"]


@dataclass(frozen=True)
class DepthConfig:
    #: Fraction of the box width and height kept when sampling, per side.  0.2
    #: keeps the central 60 per cent, which drops most of the background that
    #: leaks in at the shoulders and between the legs.
    shrink: float = 0.2
    min_range_m: float = 0.3
    max_range_m: float = 12.0
    bin_m: float = 0.15
    #: A histogram mode counts as real at this fraction of the largest mode.
    mode_fraction: float = 0.35
    #: Below this fraction of valid pixels in the sampled window the depth map
    #: is not trusted and the detection is handed on unranged.
    min_valid_fraction: float = 0.10
    #: Fewest valid pixels, regardless of fraction.  A 12-pixel-tall person at
    #: the far end of the cell gives very few.
    min_valid_pixels: int = 12
    #: Stereo parameters for the noise model.  Defaults describe a short-range
    #: RGB-D camera: 50 mm baseline, 0.08 px disparity noise.
    baseline_m: float = 0.05
    disparity_noise_px: float = 0.08
    #: Floor on the reported sigma, covering everything the z^2 term does not:
    #: box jitter, the person not being a plane, extrinsic error.
    sigma_floor_m: float = 0.03


@dataclass(frozen=True)
class DepthEstimate:
    depth_m: float
    sigma_m: float
    #: Fraction of sampled pixels that carried a valid return.
    valid_fraction: float
    #: Fraction of valid pixels that fell in the winning mode.
    support_fraction: float
    n_modes: int


def sigma_from_depth(depth_m: float, config: DepthConfig, fx: float) -> float:
    """Range standard deviation at ``depth_m``, metres.

    ``sigma_z = z^2 * sigma_disparity / (f * b)``, the standard first-order
    propagation for a triangulating sensor, with a floor added in quadrature.
    """
    if config.baseline_m <= 0 or fx <= 0:
        return max(config.sigma_floor_m, 0.02 * depth_m)
    triangulation = depth_m**2 * config.disparity_noise_px / (fx * config.baseline_m)
    return float(np.hypot(triangulation, config.sigma_floor_m))


def _sample_window(
    detection: Detection2D, depth: np.ndarray, config: DepthConfig
) -> Tuple[np.ndarray, int]:
    h, w = depth.shape[:2]
    x0, y0, x1, y1 = detection.xyxy
    dx = (x1 - x0) * config.shrink
    dy = (y1 - y0) * config.shrink
    xa = int(np.floor(max(0.0, x0 + dx)))
    xb = int(np.ceil(min(float(w), x1 - dx)))
    ya = int(np.floor(max(0.0, y0 + dy)))
    yb = int(np.ceil(min(float(h), y1 - dy)))
    if xb <= xa or yb <= ya:
        # The shrink ate the box.  Fall back to the box itself, clipped.
        xa = int(np.floor(max(0.0, x0)))
        xb = int(np.ceil(min(float(w), x1)))
        ya = int(np.floor(max(0.0, y0)))
        yb = int(np.ceil(min(float(h), y1)))
    if xb <= xa or yb <= ya:
        return np.zeros(0, dtype=np.float64), 0
    window = np.asarray(depth[ya:yb, xa:xb], dtype=np.float64)
    total = int(window.size)
    finite = window[np.isfinite(window)]
    valid = finite[(finite > config.min_range_m) & (finite < config.max_range_m)]
    return valid, total


def estimate_depth(
    detection: Detection2D,
    depth: Optional[np.ndarray],
    config: DepthConfig,
    fx: float,
) -> Optional[DepthEstimate]:
    """Nearest-dense-cluster range for one detection or ``None``.

    ``None`` means "the depth map does not support a range for this box", not
    "there is nobody there".  The caller must keep the detection.
    """
    if depth is None or depth.size == 0:
        return None
    valid, total = _sample_window(detection, depth, config)
    if total == 0:
        return None
    valid_fraction = len(valid) / total
    if len(valid) < config.min_valid_pixels or valid_fraction < config.min_valid_fraction:
        return None

    lo, hi = float(valid.min()), float(valid.max())
    n_bins = max(1, int(np.ceil((hi - lo) / config.bin_m)) + 1)
    counts, edges = np.histogram(valid, bins=n_bins, range=(lo, lo + n_bins * config.bin_m))
    if counts.max() == 0:
        return None

    strong = np.flatnonzero(counts >= config.mode_fraction * counts.max())
    # Merge adjacent strong bins so one broad cluster is not counted as three.
    modes = []
    run_start = strong[0]
    previous = strong[0]
    for b in strong[1:]:
        if b != previous + 1:
            modes.append((run_start, previous))
            run_start = b
        previous = b
    modes.append((run_start, previous))

    first, last = modes[0]  # nearest, because histogram bins ascend in range
    centre = 0.5 * (edges[first] + edges[last + 1])
    half_width = 0.5 * (edges[last + 1] - edges[first]) + config.bin_m
    selected = valid[np.abs(valid - centre) <= half_width]
    if len(selected) < config.min_valid_pixels:
        selected = valid[valid <= edges[last + 1] + config.bin_m]
    if len(selected) == 0:
        return None

    depth_m = float(np.median(selected))
    return DepthEstimate(
        depth_m=depth_m,
        sigma_m=sigma_from_depth(depth_m, config, fx),
        valid_fraction=float(valid_fraction),
        support_fraction=float(len(selected) / len(valid)),
        n_modes=len(modes),
    )
