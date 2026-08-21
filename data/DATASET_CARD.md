# Dataset card

Nothing described here is committed to the repository. Both datasets are either
downloaded on demand or generated on the fly.

## COCO val2017 (real images, detection accuracy)

| | |
|---|---|
| Source | https://cocodataset.org |
| Used for | measuring person-detection recall and miss rate. Evaluation only |
| Split | `val2017`, 5,000 images |
| Person annotations | 10,777 individual people, plus 227 crowd regions |
| Median annotated person height | 83 px |
| Size on disk | about 1.1 GB of images, 250 MB of annotations |
| Fetched by | `python scripts/fetch_data.py` |
| Stored at | `$SZP_COCO_ROOT` or `$SZP_DATA_ROOT/coco`, default `~/datasets/coco` |
| Annotation licence | CC BY 4.0, COCO Consortium |
| Image licence | Flickr photographs under a mixture of Creative Commons terms, made available by the COCO Consortium under their terms of use |
| Training | **none.** No model in this repository is trained or fine-tuned on anything |

### How it is used

`analysis/detection_metrics.py` runs the detector over every image and scores
person detections against the annotations. Two decisions in that script change
the number materially and are worth knowing about.

**Crowd regions are ignore regions.** A COCO `iscrowd` annotation is a polygon
over a mass of people with no individual boxes. Detections landing inside one
are dropped before scoring rather than counted as false positives and the
region itself is never counted as a missed person.

**The safety subset is people at least 50 px tall.** COCO is full of people who
are twelve pixels tall in the background of a photograph of something else. A
guarding system does not need to see them: at the working distance of the cell
in `config/zones.yaml` a person subtends more than that and anyone smaller is
beyond the range the sensors cover. People below the floor become ignore
regions rather than disappearing, so correctly detecting a small person is not
scored as a false positive. Both the full set and the subset are reported and
the README quotes both.

### What COCO does not represent

Everything about an industrial cell. Fixed overhead viewpoints, high-visibility
clothing, hard hats, crouching and kneeling postures, occlusion by machinery,
mixed artificial lighting, motion blur from a global-shutter camera at a short
exposure. A recall figure measured on photographs is evidence that the detector
works on photographs. It is not evidence that it works in a factory and
`docs/limitations.md` says so at more length.

## Synthetic cell (geometry, zones, safety, latency)

| | |
|---|---|
| Source | `src/szp/data/synthetic.py`, in this repository |
| Used for | fusion accuracy, zone confusion, state-machine behaviour, latency |
| Licence | MIT, same as the repository |
| Generated | on the fly, deterministically from a seed |
| Size on disk | nothing |

### Why it exists

The geometry numbers cannot come from COCO: it has no depth, no lidar and no
metric ground truth. No public dataset pairs a registered RGB-D camera, a 2D
lidar at knee height and surveyed 3D person positions in one industrial cell.
So the fusion error, zone confusion, fail-safe behaviour and end-to-end latency
are measured on a rig where the true position of every person is known exactly.

### What it contains

A room with a floor and four walls. People modelled as vertical cylinders
walking scripted paths, entering and leaving so that the state machine is driven
through clear, warning, stop and recovery rather than sitting in one state.
Depth maps and lidar scans are produced by closed-form ray intersection, so the
ground truth is exact to floating point rather than to a rasteriser.

Sensor imperfections are modelled rather than left out: depth noise growing with
the square of range, depth dropout as spatial holes rather than salt and pepper,
a hard maximum depth range, lidar range noise and beam dropout, a limited lidar
field of view and detector jitter, misses and false positives. Every one is a
parameter and `analysis/geometry_accuracy.py` sweeps them.

### What it cannot show

That a detector will find a person in a real factory. Rendered cylinders are not
people and no detection accuracy claim in this repository comes from this rig.
When `scripts/run_demo.py` is run with `--detector onnx` it runs the real model
on rendered frames and that is for **timing only** -- the detections are not
scored and should not be read as meaningful.

It also cannot show that the bias corrections in the fusion stage generalise.
The rig's people are cylinders of exactly the radius the corrections assume, so
the residual error in the best case is flatteringly small. The
`wrong_body_size` case in `analysis/geometry_accuracy.py` exists to put a number
on that and the README quotes it next to the headline figure.
