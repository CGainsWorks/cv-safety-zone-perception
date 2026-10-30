#!/usr/bin/env python3
"""Fetch the INT8 calibration images: a fixed list of COCO train2017 photographs.

    python scripts/fetch_calibration.py              # download the listed images
    python scripts/fetch_calibration.py --make-list  # regenerate the list itself

Why train2017.  Every accuracy number in this repository is measured on
val2017, so calibrating INT8 on val2017 would set the activation ranges on the
images the INT8 model is then scored on.  train2017 is the same distribution
and disjoint from val2017, which makes it the honest calibration set.  The
model was not trained by this repository, so nothing here has seen those
images either.

The list, ``data/calibration_train2017.txt``, is committed: 512 file names
drawn with a fixed seed from every train2017 image, people or not, because the
calibrator should see the scenes the detector runs on rather than only the
ones with a person in them.  Regenerating it needs ``instances_train2017.json``
from the annotations archive ``scripts/fetch_data.py`` already downloads.
Fetching the images needs only the list: each one comes from COCO's image
server, about 80 MB for the set, into ``$SZP_COCO_ROOT/calib_train2017``.

COCO images are Flickr photographs under their own terms (``data/DATASET_CARD.md``);
none is committed or redistributed here.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fetch_data import default_root  # noqa: E402

LIST = REPO_ROOT / "data" / "calibration_train2017.txt"
IMAGE_URL = "http://images.cocodataset.org/train2017/{name}"
SEED = 7
COUNT = 512


def default_calib_dir() -> Path:
    return default_root() / "calib_train2017"


def read_list(path: Path = LIST) -> list:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def make_list(annotations: Path, count: int, seed: int) -> list:
    images = json.loads(annotations.read_text(encoding="utf-8"))["images"]
    names = sorted(image["file_name"] for image in images)
    chosen = sorted(random.Random(seed).sample(names, count))
    header = (
        f"# {count} COCO train2017 images for INT8 calibration, drawn with\n"
        f"# random.Random({seed}).sample from all {len(names)} train2017 file names, sorted.\n"
        "# Regenerate with: python scripts/fetch_calibration.py --make-list\n"
    )
    with open(LIST, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(header + "\n".join(chosen) + "\n")
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=None, help="where the images go")
    parser.add_argument("--make-list", action="store_true")
    parser.add_argument("--annotations", default=None,
                        help="instances_train2017.json, for --make-list")
    args = parser.parse_args()

    if args.make_list:
        annotations = Path(args.annotations) if args.annotations else (
            default_root() / "annotations" / "instances_train2017.json"
        )
        chosen = make_list(annotations, COUNT, SEED)
        print(f"wrote {LIST.relative_to(REPO_ROOT).as_posix()} with {len(chosen)} names")
        return 0

    out = Path(args.out) if args.out else default_calib_dir()
    out.mkdir(parents=True, exist_ok=True)
    names = read_list()
    fetched = 0
    for name in names:
        destination = out / name
        if destination.exists() and destination.stat().st_size > 0:
            continue
        temporary = destination.with_suffix(".part")
        with urllib.request.urlopen(IMAGE_URL.format(name=name), timeout=60) as response, open(
            temporary, "wb"
        ) as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(destination)
        fetched += 1
    print(f"{len(names)} calibration images in {out} ({fetched} downloaded now)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
