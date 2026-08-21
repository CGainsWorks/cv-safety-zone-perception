# Calibration

Read [SAFETY_DISCLAIMER.md](SAFETY_DISCLAIMER.md) first.

Three things have to be measured before the pipeline means anything: the
camera's intrinsics, the transform from the camera to the lidar and the
transform from the camera to the world frame the zones are drawn in. Get the
second one wrong by a couple of degrees and every person is placed half a metre
from where they are, which looks exactly like sensor noise and is miserable to
find later. It is entry FM-06 in the FMEA.

## Camera intrinsics

Standard checkerboard calibration; use whatever you normally use. What matters
downstream:

- **Undistort or record the distortion.** `config/extrinsics.yaml` has a
  `distortion` list per camera. The reference configuration leaves it empty
  because the stream is rectified by the driver. If yours is not, put the
  coefficients in and undistort before the detector, not after: a detection box
  on a distorted image does not back-project to the right ray.
- **Record the image size.** The intrinsics are stored with the resolution they
  were calibrated at and `CameraIntrinsics.scaled` rescales them when a frame
  arrives at a different size. Without the size recorded, that rescale is a
  guess.

## Camera to lidar

A 2D lidar sees a line, not a surface, so the usual point-cloud registration is
not available. The method here is the planar-target constraint of Zhang and
Pless (IROS 2004), which needs nothing but a checkerboard:

> the camera gives the board plane in its own frame as a unit normal `n` and a
> distance `d`; every lidar return that lands on the board must satisfy
> `n . (R p + t) = d`.

### Capture

1. Hold a rigid checkerboard where **both** sensors can see it: inside the
   camera's field of view and crossing the lidar's scan plane. A board on a
   stand at knee height works; a hand-held board wobbles and the wobble goes
   straight into the residual.
2. Record a camera frame and a lidar scan simultaneously. Stationary is fine
   and much easier than synchronised.
3. Repeat for **at least five poses at genuinely different orientations**.
   Tilt the board forward, back and to each side; move it near and far.
   Rotating the board about the lidar's scan axis alone adds no information,
   and the solver will tell you so rather than fitting it -- see below.

### Segmenting the board returns

The one manual step. From each scan, keep only the returns that landed on the
board. A short run of points at roughly the right range, separated from the
background by a clear gap. Five poses is a handful of clicks; automating it
badly is worse than doing it by hand.

Write them out as JSON:

```json
[
  {
    "normal": [0.12, -0.34, 0.93],
    "distance": 2.41,
    "lidar_points": [[1.98, 0.31], [1.97, 0.33]]
  }
]
```

`normal` and `distance` are the board plane **in the camera frame**, from the
checkerboard pose estimate. `lidar_points` are in the **lidar frame**, metres.

### Solving

```bash
python scripts/calibrate.py \
    --observations calib/board_poses.json \
    --out config/extrinsics.yaml \
    --json results/calibration.json
```

Because the lidar's points all have `z = 0` in its own frame, the third column
of `R` never appears in the equations; it is recovered afterwards as
`r1 x r2`. That is a real property of the geometry, not a shortcut and it is
why the routine solves for nine unknowns rather than twelve.

The closed-form solution is a starting point. It is followed by a
Levenberg-Marquardt refinement on the same point-to-plane residual, minimally
parametrised over se(3) and it is that refinement's residual that gets
reported.

### What it refuses to do

- **Fewer than three poses**: refused outright.
- **Poses that do not constrain the problem**: the design matrix is rank
  checked and a rank below nine is an error with an explanation, not a
  confident-looking transform built from a rank-deficient system. That message
  is the single most useful thing this script does, because a degenerate
  calibration is otherwise indistinguishable from a good one until somebody
  gets hurt.

### Checking it

```bash
python scripts/calibrate.py --self-check
```

generates board observations from a transform it chose, hands them to the
solver and reports how well the solver got it back. That is the honest
demonstration: measured against a transform we know, not against a second
calibration. With seven poses, twenty-five points each and 10 mm of point
noise, it recovers the transform to under a centimetre and a tenth of a degree,
and the refinement takes the residual from tens of millimetres to single
figures. The exact numbers are printed and CI runs the same check.

Then look at the residual:

| RMS residual | Reading |
|---|---|
| under 10 mm | good |
| 10-30 mm | usable; check whether one pose dominates |
| over 30 mm | something is wrong: board wobble, a mis-segmented scan or a pose set that barely constrains the problem |

The residual is written into `config/extrinsics.yaml` under `meta`. An
extrinsic with no error bar is how a systematic ranging error becomes
invisible, so do not delete it.

## Camera to world

The floor fixes roll, pitch and height. Yaw and the origin within the floor
plane are **not observable** from a floor fit and they come from a surveyed
point instead.

```bash
python scripts/calibrate.py \
    --observations calib/board_poses.json \
    --floor-depth calib/floor_depth.npy \
    --out config/extrinsics.yaml
```

`floor_depth.npy` is one depth map of an **empty patch of floor**, saved as a
float32 array in metres. The routine fits a plane by RANSAC and refines it by
total least squares on the consensus set, then builds the transform that puts
the floor at `z = 0` with `+z` up. It refuses a depth map with no dominant
plane rather than fitting one to clutter.

Yaw and origin default to the camera's own footprint and optical axis. For a
real cell, survey a point on the floor -- a machine base bolt, a marked corner
-- and pass it, then check with `python scripts/show_zones.py`, which prints
the sensor positions in world coordinates and warns about any zone that lies
outside the camera's view of the floor.

## Depth to camera

The reference configuration assumes depth is registered to colour, which is
what an RGB-D camera's driver normally does. The identity transform is written
out explicitly in `config/extrinsics.yaml` rather than omitted, because an
unregistered depth stream is a common and silent misconfiguration and that line
is where you fix it.

## When to redo it

- After any knock to a sensor mount. Anything that could have moved a bracket.
- After changing lenses, resolution or the driver's rectification settings.
- Periodically, as part of the maintenance schedule. Nothing in this
  repository detects extrinsic drift at run time, which is exactly why FM-06
  scores a detection of 8. A production system wants a reference target in
  view and a periodic check against it.
