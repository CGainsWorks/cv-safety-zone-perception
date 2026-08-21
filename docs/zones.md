# Zone configuration guide

For whoever commissions a cell. You do not need to read any of the code.

Read [SAFETY_DISCLAIMER.md](SAFETY_DISCLAIMER.md) first: this pipeline is a
demonstrator and the zones you configure here do not constitute guarding.

## The file

`config/zones.yaml`. One entry per zone:

```yaml
frame: world

zones:
  - id: machine_envelope        # unique; appears in logs and on the CAN bus
    kind: stop                  # "stop" halts the machine, "warn" derates it
    description: reach envelope of the machine
    polygon:                    # plan view, metres, world frame
      - [2.60, -1.00]
      - [4.80, -1.00]
      - [4.80,  0.20]
      - [3.80,  0.20]
      - [3.80,  1.20]
      - [2.60,  1.20]
    z_min_m: 0.0                # height band the zone applies to
    z_max_m: 2.20
    margin_m: 0.55              # fixed dilation; see "sizing" below
    sigma_k: 2.0                # extra dilation per standard deviation of
                                # the pipeline's own position uncertainty
    enter_frames: 1             # frames inside before it trips
    exit_frames: 5              # consecutive clear frames before it releases
    enabled: true
```

Coordinates are metres in the **world** frame: the floor is `z = 0`, `+z` is up,
and `+x` and `+y` are wherever `config/extrinsics.yaml` put them. Get the frame
wrong and everything else is decoration, so start by checking it:

```bash
python scripts/show_zones.py --extrinsics config/extrinsics.yaml
```

which prints the camera and lidar positions in world coordinates alongside each
zone's bounding box, so you can see at a glance whether your polygon is where
you think it is.

## Sizing a zone: the part that matters

**A zone is not the danger area.** It is the danger area, plus the distance the
machine travels while it is stopping, plus the distance a person covers while
the system is reacting. ISO 13855 gives the minimum distance as

```
S = K * (t_reaction + t_stop) + C
```

where `K` is the approach speed, conventionally 1.6 m/s for a walking person,
`t_reaction` is the detection system's response time, `t_stop` is the machine's
stopping performance and `C` is an intrusion allowance for the detection
method.

For the reference cell in this repository:

| Term | Value | Where it comes from |
|---|---|---|
| `K` | 1.6 m/s | ISO 13855 walking speed |
| `t_reaction` | 0.10 s | p99 end to end, from `analysis/latency_budget.py` on the reference machine |
| `t_stop` | 0.25 s | **assumed.** Measure it on your machine |
| `S` | 0.56 m | `1.6 * 0.35` |
| `margin_m` | 0.55 | `S`, rounded |

Two things to be careful about.

**`t_reaction` is not the reaction time of your installation.** The number in
this repository is the pipeline's own latency on one machine. It does not
include camera exposure and driver latency, which on a real sensor is often
larger than everything else combined, nor the PLC's scan cycle. Measure both
and add them. `docs/latency.md` says which numbers to substitute where.

**`t_stop` must be measured, not assumed.** It is the largest single term in
most cells and it changes with load, wear and tooling. It is entry FM-22 in the
FMEA and it is the highest-severity item there with no software mitigation at
all.

`sigma_k` is on top of the margin and does a different job: it scales with the
pipeline's per-detection position uncertainty, so a person whose range is
poorly known is treated as occupying more floor. Two standard deviations is the
default. Lowering it below one is not sensible; the point of carrying the
uncertainty is to use it.

## Entry and exit

`enter_frames` defaults to 1 and the loader **refuses** a stop zone with more
than 2. Debouncing entry adds that many frame periods to the reaction time,
which then has to go back into the sizing calculation above. If the detector is
noisy enough that you want entry debounce, raise the confidence threshold
instead and record what that does to the miss rate.

`exit_frames` defaults to 5 and there is no upper limit. It is what keeps a
zone tripped through a short detector dropout. Longer is safer and only costs
uptime.

## Shapes

Concave polygons are supported, which is worth using. The reference cell's
`machine_envelope` is L-shaped because the operator station sits in the
near-left corner; a rectangle there would have nuisance-tripped every time
somebody stood at the panel and a guard that nuisance-trips gets bypassed.

What the loader rejects, always at start-up and never as a warning:

| Problem | Message |
|---|---|
| Polygon crosses itself | "the polygon crosses itself" |
| Fewer than 3 vertices or zero area | "a polygon needs at least 3 vertices" / "effectively zero area" |
| `margin_m` negative | "A negative margin shrinks the zone" |
| `z_max_m` not above `z_min_m` | "z_max_m must exceed z_min_m" |
| Duplicate `id` | "duplicate zone id" |
| More than 16 zones | "the CAN zone bitmap is 16 bits wide" |
| No enabled zone of kind `stop` | "no enabled zone has kind 'stop'" |
| `enter_frames` above 2 on a stop zone | "Debouncing entry adds that many frame periods" |

A geometrically valid but *wrong* polygon is not detectable by software. Walk
the cell with a tape measure.

## Zones on the wire

Zone order in the file fixes the bit position in the CAN `zone_mask`, so
renaming a zone does not change the wire format but reordering the file does.
If the PLC-side mapping is written by hand, reorder with care.

| Bit | Zone (reference cell) |
|---|---|
| 0 | `machine_envelope` |
| 1 | `feed_chute` |
| 2 | `approach_corridor` |

## Checking your work

```bash
python scripts/show_zones.py --zones config/zones.yaml     # geometry and a plot
python -m pytest tests/test_zones.py -q                    # the loader's rules
python scripts/run_demo.py --zones config/zones.yaml       # end to end
```

`analysis/geometry_accuracy.py` reports, per zone, how many frames the pipeline
said occupied when it was and when it was not. A zone that never trips in a
scene where people walk through it is a configuration error and a zone that
trips constantly in an empty cell is one too.
