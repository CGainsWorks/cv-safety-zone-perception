# Design

Read [SAFETY_DISCLAIMER.md](SAFETY_DISCLAIMER.md) first.

## The shape of it

```
 RGB ─┐
depth ─┼─> detect ──> fuse ──> zones ──> FSM ──> CAN permit ──> PLC
 scan ─┘     │          │        │        │          │
             └──────────┴────────┴────────┴──────────┘
                     latency instrumentation
                              │
                          watchdog  <── heartbeat from the PLC
```

One thread, synchronous, no queues. A worker pool would raise throughput and
turn the end-to-end latency into a distribution over queue depths rather than a
property of the code. When the machine cannot keep up, the right answer is a
smaller model or a lower resolution, not a deeper queue: a queue converts a
throughput problem into a latency problem and hides it.

Each stage narrows the representation. Pixels become detections, detections
become metres, metres become a boolean per zone, booleans become one state and
that state becomes eight bytes on a bus. The narrowing is the design: by the
time anything reaches the machine interface there is very little left that can
be wrong in an interesting way.

## The one idea

**A missed person is the only failure that matters.** Everything else in this
document follows from that and where a choice traded recall against something
else, recall won and the cost is written down.

Concretely, in the order the frame moves through the system:

| Decision | Alternative | Why this one |
|---|---|---|
| Confidence threshold 0.15 | 0.25, the usual default | Measured: the safety-relevant miss rate at 0.25 is nearly double the rate at 0.15. The cost is a higher nuisance-stop rate, which is reported next to it |
| Nearest dense depth cluster | Median depth in the box | The median of a person-shaped box is the wall behind them. It puts people several metres too far away and too far away means outside the zone |
| Lidar association is a soft cost | A hard range gate | A depth camera that is systematically wrong would gate away every lidar cluster that contradicts it. The soft cost lets the disagreement surface |
| On disagreement, take the nearer | Take the more confident | There is no way to know which sensor is wrong and the nearer reading is the one that stops the machine |
| Unranged detections become wedges | Discard them | A detection with no range is still a person. The wedge is the floor they could be standing on and it usually overlaps a zone |
| Footprints dilated by `k * sigma` | Use the point estimate | Uncertainty must make a person occupy more floor, not less |
| Enter on one frame, exit after five | Symmetric debounce | Debouncing entry adds latency to the one transition where latency costs something |
| STOP latches | Clears when the zone clears | A person is what stops the machine. An operator is what restarts it |
| CAN frame is a permit | A stop command | A command that fails to arrive leaves the machine running. A permit that fails to arrive stops it |

## Coordinate frames

Defined once in `src/szp/types.py` and never re-argued.

| Frame | Convention |
|---|---|
| `camera` | OpenCV: +x right, +y down, +z forward |
| `depth` | the depth sensor; assumed registered to `camera` unless the extrinsics say otherwise |
| `lidar` | REP-103: +x forward, +y left, +z up; the scan is a horizontal ring at the lidar's z = 0 |
| `base` | the machine, REP-103, origin on the floor at the centre of the footprint |
| `world` | the fixed cell frame, +z up, floor at z = 0. Zone polygons live here and nowhere else |

`T_a_b` maps a point in `b` into `a`. The loader enforces it, refuses a
rotation that is not orthonormal to a part in a million and raises rather than
falling back to identity when two frames are not connected. An identity
extrinsic between a camera and a lidar is a plausible-looking value that puts
every person half a metre from where they are, which is exactly the sort of
failure that is hard to see and easy to prevent.

## Fusion

Three sources of horizontal position, in decreasing order of quality.

**Depth.** The box is sampled after shrinking it by 20 per cent per side. A
histogram at 15 cm resolution finds the modes and the *nearest* mode that
holds at least 35 per cent of the strongest mode's mass wins. The final number
is the median of the returns within one bin of that mode.

Two bias corrections, both derived rather than tuned:

- The depth estimate is the *front surface* of the person. Model the torso as a
  circle of radius `r` seen head on: the depth at image offset `x` is
  `z_c - sqrt(r^2 - x^2)` and pixels are uniform in `x`, so the median over
  the silhouette sits at `x = r/2` and the median depth is `z_c - r*sqrt(3)/2`.
  That much is added back.
- The lidar sees an *arc* and the centroid of an arc is about one radius in
  front of the centre. A circle of fixed radius is fitted to the cluster by
  Gauss-Newton instead. The radius is fixed because a thirty-degree arc does
  not constrain a radius and letting it float turns a small bias into a large
  variance.

Both corrections assume a body of the radius `FusionConfig.person_depth_m`
implies. `analysis/geometry_accuracy.py` has a `wrong_body_size` case that
quantifies what happens when that assumption is wrong and the README quotes it
next to the headline figure so the headline cannot be read as more than it is.

**Uncertainty.** `sigma_xy` travels with every box. Depth uncertainty is
`z^2 * sigma_disparity / (f * b)` with a floor added in quadrature, the standard
first-order propagation for a triangulating sensor. When both sensors agree
inside a Mahalanobis gate the two estimates are combined by inverse-variance
weighting; outside the gate the estimate goes to whichever is nearer and
`sigma_xy` is inflated by half the disagreement.

**Nothing.** A detection with neither becomes an `UNRANGED` box whose footprint
is the quadrilateral between the two viewing rays through the box's vertical
edges, from the near to the far end of the range interval implied by the
apparent height and a 1.40 to 2.05 m height prior. A box clipped by the image
border carries no height information at all, so its wedge runs the full working
range.

## Zones

Polygons on the floor, in `world`, with a height band and a behaviour. Concave
is allowed; the geometry is written out in `src/szp/zones/polygon.py` rather
than taken from Shapely, both to keep GEOS off a Jetson and because the margin
semantics have to be exactly right.

The occupancy test is: dilate the footprint by `margin_m + sigma_k * sigma_xy`
and ask whether it meets the zone. That dilation is never constructed --
dilating `A` by `r` and asking whether it meets `B` is the same question as
asking whether the distance from `A` to `B` is at most `r` and Minkowski
offsets of concave polygons are a good place for a bug to hide.

The height band is an interval *overlap*, not a containment test, so somebody
whose estimated height is wrong still counts.

Validation is strict and happens at load. A self-intersecting polygon, a
negative margin, a duplicated identifier, more zones than fit in the CAN
bitmap or a file with no stop zone at all: all of them are start-up failures.
Warn-and-continue means the first anyone hears of a bad zone file is when it
fails to stop something.

## The machine interface

`src/szp/safety/can_frames.py` is the whole protocol and has no dependency on
python-can, so a PLC-side implementation has one file to read and the encoding
can be tested exhaustively without a bus.

The frame is a **permit**. The PLC enables motion only while it is receiving a
`SZP_ESTOP` frame that is fresh, passes its CRC, has an ALIVE counter that has
advanced and carries `CLEAR` or `WARN`. Three failures are covered by that one
rule:

- the pipeline crashes, hangs, loses power or the cable is pulled: nothing
  arrives, the permit lapses, the machine stops;
- the payload is corrupted past the CAN CRC: the CRC-8 catches it and the test
  suite flips every single bit of every byte to prove it;
- the sender hangs with its last frame repeating out of a hardware buffer: the
  ALIVE counter stops advancing and the permit lapses.

The data identifier mixed into each CRC means a frame decoded under the wrong
arbitration ID fails its checksum rather than producing a plausible state.

## The state machine

```
INIT ──(healthy for N frames)──> CLEAR <──> WARN
  │                                │          │
  │                                └────┬─────┘
  │                                     │ stop zone occupied
  │                                     v
  └──(inputs unhealthy)──> FAULT     STOP
                              │         │
                              └────┬────┘
                                   │ cause gone AND operator reset
                                   v
                             RECOVERING ──(N clear frames)──> CLEAR
```

Four properties, each with a test named after it in `tests/test_safety_fsm.py`:
latching, fault dominance, no self-clearing on reset alone and monotone entry.

## What is deliberately absent

- **Tracking.** No Kalman filter, no data association across frames. It would
  smooth the position estimates and it would also let a track coast through a
  detection dropout, which sounds helpful and is the mechanism by which a
  person who has actually moved gets reported where they used to be. The
  five-frame zone exit debounce provides the small amount of temporal memory
  that is safe to have.
- **Segmentation.** Instance masks would give a better footprint than a box.
  They also cost several times the inference budget and the measurements say
  the box is not the limiting error.
- **Multi-camera.** The frame graph supports it; nothing else does. It is the
  obvious next step and it is not done.

## Layout

The original plan proposed `src/detection/`, `src/fusion/` and so on. This repository
uses `src/szp/detection/` and so on -- the same structure one level down, in a
package. A flat `src/` is not importable as a package without path games and
`szp` is short enough not to be a nuisance. That is the only deviation.
