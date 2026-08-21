# SAFETY DISCLAIMER

## This is not a safety function

**Do not use this software to protect anybody from anything.**

This repository is an engineering demonstrator. It shows an architecture and a
latency budget for a perception pipeline that could inform the design of a
protective device. It is not a protective device and no part of it has been
built, verified or assessed to be one.

## What has not been done

Everything on this list is required before software may perform a safety
function on real machinery. None of it has happened here.

- **No functional safety assessment.** Nothing has been evaluated against
  IEC 61508, ISO 13849-1, IEC 62061, ISO 13855 or ISO 3691-4. No Performance
  Level and no Safety Integrity Level is claimed, achieved or achievable in the
  current form.
- **No independent channel.** A protective function at any meaningful
  Performance Level needs redundancy and diagnostic coverage. This pipeline is
  a single channel: one camera, one detector, one process, one bus. A fault in
  the detector is a fault in the whole function.
- **No certified components.** The camera, the lidar, the compute and the
  operating system are all standard commercial parts with no safety rating.
  Python, Linux, ONNX Runtime and CUDA are none of them suitable for a safety
  function.
- **No hard real-time guarantee.** The latency figures in this repository are
  measurements of what happened, not bounds on what can happen. A Python
  process on a general-purpose operating system with a garbage collector has no
  worst-case execution time.
- **No validation against a real machine.** The stopping distance,
  reaction time and zone dimensions in `config/zones.yaml` are worked examples
  for a fictional cell. They have never been checked against real plant.
- **No trained-and-validated detector for an industrial scene.** The model is a
  general-purpose COCO detector evaluated on photographs. An industrial cell
  has lighting, clothing, postures and occlusions that COCO does not represent.

## What a machine-learning detector cannot promise

A neural network gives no guarantee that it will detect a person. It gives a
statistical result on a test set and the test set is not your factory. The
measured miss rate in `analysis/detection_metrics.py` is the *observed* rate on
public photographs at a chosen operating point; it is not a bound, it does not
transfer to a different scene and it says nothing about the specific person
standing in front of your machine wearing something the training data never
contained.

This is the reason a camera-based system does not replace a light curtain, a
safety laser scanner or a physical guard. Those devices fail in ways that can
be analysed and bounded. A detector does not.

## What this repository is honestly good for

- Showing how the parts of such a system fit together and where the latency
  goes.
- Making the false-negative question measurable rather than rhetorical: what
  fraction of people are missed, which people and what happens downstream when
  one is.
- Demonstrating a fail-safe machine interface: a permit rather than a command,
  end-to-end protected, so that silence stops the machine.
- Being a starting point for a conversation with a functional safety engineer,
  who will tell you what else is needed.

## If you are building something real

Use certified protective equipment, sized and validated by someone competent to
do it and treat any camera-based perception as a *supplementary* layer that
improves productivity while the certified equipment does the protecting. Have
the risk assessment done. Have the stopping performance measured on the actual
machine, not assumed.

## Clean-room statement

This repository is an independent reimplementation of a problem class, built
from scratch on personal time using public and synthetic data and open-source
simulation. It is not derived from and contains no material belonging to any
employer or client: no employer source code, schematics, calibration data,
client data or internal documentation. Every number in it comes from a script
in the repository, run against public data or against the synthetic rig in
`src/szp/data/synthetic.py`.
