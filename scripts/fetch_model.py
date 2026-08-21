#!/usr/bin/env python3
"""Download the YOLOv8n checkpoint and export it to ONNX.

Run once:

    python scripts/fetch_model.py

This is the only place ultralytics is used and ultralytics is AGPL-3.0.  The
runtime never imports it -- ``requirements.txt`` does not list it and nothing
under ``src/`` touches it -- so the deployed pipeline links against ONNX
Runtime and numpy only.  ``docs/licences.md`` spells out what that means if you
intend to use any of this commercially.

The exported file lands in ``models/`` which is gitignored.  Its SHA-256 is
printed and written into ``models/model_card.json`` so every result file can
name the exact weights it was produced with.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1_000_000:
        print(f"already have {destination.name} ({destination.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, open(destination, "wb") as out:
        shutil.copyfileobj(response, out)
    print(f"wrote {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


def export(weights: Path, imgsz: int, opset: int) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed.  It is an export-time dependency only:\n"
            "    pip install -r requirements-export.txt\n"
            "Read docs/licences.md first -- ultralytics is AGPL-3.0."
        ) from None
    model = YOLO(str(weights))
    produced = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=False)
    return Path(produced)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "models"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--url", default=WEIGHTS_URL)
    parser.add_argument("--force", action="store_true", help="re-export even if the ONNX exists")
    args = parser.parse_args()

    models = Path(args.models_dir)
    models.mkdir(parents=True, exist_ok=True)
    weights = models / "yolov8n.pt"
    onnx_path = models / "yolov8n.onnx"

    if onnx_path.exists() and not args.force:
        print(f"already have {onnx_path}; pass --force to rebuild")
    else:
        download(args.url, weights)
        produced = export(weights, args.imgsz, args.opset)
        if produced.resolve() != onnx_path.resolve():
            shutil.move(str(produced), str(onnx_path))
        print(f"exported {onnx_path}")

    card = {
        "source_url": args.url,
        "checkpoint": weights.name,
        "checkpoint_sha256": sha256_file(weights) if weights.exists() else None,
        "onnx": onnx_path.name,
        "onnx_sha256": sha256_file(onnx_path),
        "imgsz": args.imgsz,
        "opset": args.opset,
        "licence": "AGPL-3.0 (ultralytics); see docs/licences.md",
        "classes": "COCO 80; this pipeline uses class 0 (person) only",
    }
    (models / "model_card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(card, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
