"""COCO val2017, person class only.

Parsed with the standard library rather than pycocotools, which is a build
dependency that fails on Windows more often than it is worth for reading one
JSON file.

Two things this loader is careful about, both of which change the recall number
if you get them wrong:

*crowd regions are ignored, not counted as misses*
    A COCO ``iscrowd`` annotation is a polygon over a mass of people with no
    individual boxes.  Counting it as one instance and expecting one detection
    is wrong in both directions.  Detections that land inside a crowd region
    are removed before scoring rather than counted as false positives and the
    region itself is never counted as a missed person.

*the safety subset is the number that means anything*
    COCO contains a great many people who are 8 pixels tall in the background
    of a photograph of something else.  A safety system does not care about
    them: at the working distance of the cell in ``config/zones.yaml``, a
    person subtends at least ``min_height_px`` pixels and anyone smaller is
    outside the range the sensor suite covers.  ``DetectionSubset`` makes that
    cut explicit and the analysis reports both the full set and the subset, so
    the reader can see how much of the headline number comes from people the
    system was never meant to see.

Licence: the COCO *annotations* are CC BY 4.0.  The *images* are Flickr
photographs under a mixture of Creative Commons terms and are redistributed by
the COCO consortium under their terms of use.  See ``data/DATASET_CARD.md``.
Nothing from this dataset is committed to this repository.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["CocoPersonDataset", "CocoImage", "default_coco_root"]

PERSON_CATEGORY_ID = 1


def default_coco_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("SZP_COCO_ROOT")
    if env:
        return Path(env)
    root = os.environ.get("SZP_DATA_ROOT")
    if root:
        return Path(root) / "coco"
    return Path.home() / "datasets" / "coco"


@dataclass(frozen=True)
class CocoImage:
    """One image and its person annotations."""

    image_id: int
    file_name: str
    width: int
    height: int
    #: (N, 4) xyxy boxes of individually annotated people.
    boxes: np.ndarray
    #: (N,) segmentation areas in pixels, as COCO reports them.
    areas: np.ndarray
    #: (M, 4) xyxy boxes of crowd regions.  Neither expected nor penalised.
    crowd_boxes: np.ndarray
    path: Path = field(default=Path("."))

    @property
    def n_people(self) -> int:
        return int(len(self.boxes))

    @property
    def heights(self) -> np.ndarray:
        if len(self.boxes) == 0:
            return np.zeros(0)
        return self.boxes[:, 3] - self.boxes[:, 1]

    def touches_border(self, margin: float = 2.0) -> np.ndarray:
        if len(self.boxes) == 0:
            return np.zeros(0, dtype=bool)
        b = self.boxes
        return (
            (b[:, 0] <= margin)
            | (b[:, 1] <= margin)
            | (b[:, 2] >= self.width - margin)
            | (b[:, 3] >= self.height - margin)
        )


class CocoPersonDataset:
    """Lazy over images, eager over annotations.

    The annotation file is 450 MB of JSON for train2017 and 25 MB for val2017;
    it is parsed once at construction and the images are read only when
    iterated.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        split: str = "val2017",
        *,
        include_empty: bool = True,
        limit: Optional[int] = None,
    ):
        self.root = default_coco_root(root)
        self.split = split
        self.images_dir = self.root / split
        annotation_file = self.root / "annotations" / f"instances_{split}.json"
        if not annotation_file.exists():
            raise FileNotFoundError(
                f"no COCO annotations at {annotation_file}.  Run "
                "'python scripts/fetch_data.py' or set SZP_COCO_ROOT."
            )
        with open(annotation_file, "r", encoding="utf-8") as handle:
            doc = json.load(handle)

        by_id: Dict[int, dict] = {img["id"]: img for img in doc["images"]}
        boxes: Dict[int, List[List[float]]] = {}
        areas: Dict[int, List[float]] = {}
        crowds: Dict[int, List[List[float]]] = {}
        for annotation in doc["annotations"]:
            if annotation["category_id"] != PERSON_CATEGORY_ID:
                continue
            x, y, w, h = annotation["bbox"]
            if w <= 0 or h <= 0:
                continue
            target = crowds if annotation.get("iscrowd", 0) else boxes
            target.setdefault(annotation["image_id"], []).append([x, y, x + w, y + h])
            if not annotation.get("iscrowd", 0):
                areas.setdefault(annotation["image_id"], []).append(float(annotation["area"]))

        self._images: List[CocoImage] = []
        for image_id in sorted(by_id):
            info = by_id[image_id]
            person_boxes = np.asarray(boxes.get(image_id, []), dtype=np.float64).reshape(-1, 4)
            if not include_empty and len(person_boxes) == 0:
                continue
            self._images.append(
                CocoImage(
                    image_id=image_id,
                    file_name=info["file_name"],
                    width=int(info["width"]),
                    height=int(info["height"]),
                    boxes=person_boxes,
                    areas=np.asarray(areas.get(image_id, []), dtype=np.float64),
                    crowd_boxes=np.asarray(
                        crowds.get(image_id, []), dtype=np.float64
                    ).reshape(-1, 4),
                    path=self.images_dir / info["file_name"],
                )
            )
        if limit is not None:
            self._images = self._images[:limit]

    def __len__(self) -> int:
        return len(self._images)

    def __iter__(self) -> Iterator[CocoImage]:
        return iter(self._images)

    def __getitem__(self, index: int) -> CocoImage:
        return self._images[index]

    @property
    def images(self) -> Sequence[CocoImage]:
        return tuple(self._images)

    def with_people(self, min_height_px: float = 0.0) -> List[CocoImage]:
        return [
            image
            for image in self._images
            if image.n_people and (image.heights >= min_height_px).any()
        ]

    def counts(self) -> Tuple[int, int, int]:
        """(images, annotated people, crowd regions)."""
        people = sum(image.n_people for image in self._images)
        crowds = sum(len(image.crowd_boxes) for image in self._images)
        return len(self._images), people, crowds

    def summary(self) -> Dict[str, object]:
        images, people, crowds = self.counts()
        heights = np.concatenate(
            [image.heights for image in self._images if image.n_people] or [np.zeros(0)]
        )
        return {
            "split": self.split,
            "root": str(self.root),
            "images": images,
            "people": people,
            "crowd_regions": crowds,
            "images_with_people": sum(1 for image in self._images if image.n_people),
            "median_person_height_px": float(np.median(heights)) if len(heights) else 0.0,
        }
