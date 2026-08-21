#!/usr/bin/env python3
"""Download COCO val2017 and its annotations.

    python scripts/fetch_data.py

About 1.1 GB of images and 250 MB of annotations, into ``$SZP_DATA_ROOT/coco``
(default ``~/datasets/coco``).  Nothing is committed to this repository and
nothing is redistributed by it.

Licence, in short, with the long version in ``data/DATASET_CARD.md``: the COCO
*annotations* are CC BY 4.0; the *images* are Flickr photographs under a
mixture of Creative Commons terms, made available by the COCO consortium under
their own terms of use.  This repository uses them for evaluation only and
trains nothing.  Read the card before you use them for anything else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

FILES = {
    "annotations_trainval2017.zip": (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        252_907_541,
    ),
    "val2017.zip": ("http://images.cocodataset.org/zips/val2017.zip", 815_585_330),
}


def default_root() -> Path:
    env = os.environ.get("SZP_COCO_ROOT")
    if env:
        return Path(env)
    root = os.environ.get("SZP_DATA_ROOT")
    if root:
        return Path(root) / "coco"
    return Path.home() / "datasets" / "coco"


def download(url: str, destination: Path, expected_size: Optional[int]) -> None:
    if destination.exists() and (
        expected_size is None or destination.stat().st_size == expected_size
    ):
        print(f"have {destination.name} ({destination.stat().st_size / 1e6:.0f} MB)")
        return
    print(f"downloading {destination.name} from {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, open(destination, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    size = destination.stat().st_size
    print(f"  wrote {size / 1e6:.0f} MB")
    if expected_size is not None and size != expected_size:
        raise SystemExit(
            f"{destination.name} is {size} bytes, expected {expected_size}.  "
            "The download was truncated or upstream has changed; delete it and retry."
        )


def sha256(path: Path, limit: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()[:16]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None)
    parser.add_argument("--keep-zips", action="store_true")
    parser.add_argument("--no-size-check", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else default_root()
    root.mkdir(parents=True, exist_ok=True)
    print(f"data root: {root}")

    for name, (url, size) in FILES.items():
        archive = root / name
        download(url, archive, None if args.no_size_check else size)
        marker = root / ("annotations" if "annotations" in name else "val2017")
        if marker.exists() and any(marker.iterdir()):
            print(f"already extracted {marker.name}")
        else:
            print(f"extracting {name}")
            with zipfile.ZipFile(archive) as archive_file:
                archive_file.extractall(root)
        if not args.keep_zips:
            archive.unlink(missing_ok=True)

    annotations = root / "annotations" / "instances_val2017.json"
    if not annotations.exists():
        raise SystemExit(f"expected {annotations} after extraction; something went wrong")

    card = {
        "root": str(root),
        "split": "val2017",
        "annotations": annotations.name,
        "annotations_sha256_prefix": sha256(annotations),
        "images": len(list((root / "val2017").glob("*.jpg"))),
        "annotations_licence": "CC BY 4.0",
        "images_licence": "mixed Flickr terms; see data/DATASET_CARD.md",
        "used_for": "evaluation only; nothing in this repository is trained",
    }
    (root / "szp_dataset_card.json").write_text(
        json.dumps(card, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(card, indent=2))
    print("\nrun: python analysis/detection_metrics.py --out results/detection.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
