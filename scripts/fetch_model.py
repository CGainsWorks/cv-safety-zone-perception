#!/usr/bin/env python3
"""Download Megvii's YOLOX-Tiny COCO checkpoint and export it to ONNX.

Run once:

    python scripts/fetch_model.py            # FP32, on the CPU
    python scripts/fetch_model.py --fp16     # plus the FP16 graph, needs a GPU

YOLOX is Apache-2.0, code and released weights alike, so the exported graph
carries no copyleft obligation; ``docs/licences.md`` has the detail.  torch and
yolox are export-time dependencies only (``requirements-export.txt``) and the
runtime never imports either, which keeps two gigabytes of wheel out of the
deployed image.

The checkpoint is used as shipped: COCO's 80 classes, of which the pipeline
reads class 0, person.  Nothing is fine-tuned, so the detection numbers in the
README are Megvii's model measured by this repository's code.

The exported files land in ``models/``, which is gitignored.  Their SHA-256 is
printed and written into ``models/model_card.json`` so every result file can
name the exact weights it was produced with.  The download is checked against
the digest below and refused on a mismatch.

What is exported is :class:`yolox_export.DeployYOLOX`: the RGB 0..1 input
folded into the stem convolution and the grid decode and ``obj * cls`` scoring
done in the graph, so the output is ``(1, 84, 8400)`` at 640x640.  Opset 13,
because per-channel INT8 needs the ``axis`` attribute on QuantizeLinear, which
arrived in 13.  FP16 is a separate export from a half-precision module rather
than a conversion of the FP32 file, so its input is ``float16``.
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

WEIGHTS_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth"
)
WEIGHTS_SHA256 = "9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0"
ARCHITECTURE = "yolox_tiny"
NUM_CLASSES = 80


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(destination) == expected_sha:
        print(f"already have {destination.name} ({destination.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"downloading {url}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(temporary, "wb") as out:
        shutil.copyfileobj(response, out)
    actual = sha256_file(temporary)
    if actual != expected_sha:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"{destination.name}: sha256 {actual} does not match {expected_sha}")
    temporary.replace(destination)
    print(f"wrote {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


def export(weights: Path, destination: Path, imgsz: int, opset: int, half: bool) -> None:
    try:
        import onnx
        import onnxslim
        import torch

        from yolox_export import deploy_module
    except ImportError as exc:
        raise SystemExit(
            f"the export toolchain is not installed ({exc}).  It is export-time only:\n"
            "    pip install -r requirements-export.txt\n"
            "    pip install --no-deps --no-build-isolation -r requirements-yolox.txt"
        ) from None
    if half and not torch.cuda.is_available():
        raise SystemExit("the FP16 export traces a half-precision module and needs a GPU")

    module = deploy_module(weights, ARCHITECTURE, NUM_CLASSES, imgsz, imgsz)
    device = torch.device("cuda" if half else "cpu")
    module = module.to(device)
    dummy = torch.zeros((1, 3, imgsz, imgsz), device=device)
    if half:
        module, dummy = module.half(), dummy.half()
    with torch.no_grad():
        torch.onnx.export(
            module,
            (dummy,),
            str(destination),
            dynamo=False,
            opset_version=opset,
            input_names=["images"],
            output_names=["output0"],
            do_constant_folding=True,
        )
    onnx.save(onnxslim.slim(onnx.load(str(destination))), str(destination))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "models"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--url", default=WEIGHTS_URL)
    parser.add_argument("--fp16", action="store_true", help="also export the FP16 graph")
    parser.add_argument("--force", action="store_true", help="re-export even if the ONNX exists")
    args = parser.parse_args()

    models = Path(args.models_dir)
    models.mkdir(parents=True, exist_ok=True)
    weights = models / "yolox_tiny.pth"
    outputs = {"fp32": models / "yolox_tiny.onnx"}
    if args.fp16:
        outputs["fp16"] = models / "yolox_tiny_fp16.onnx"

    for precision, onnx_path in outputs.items():
        if onnx_path.exists() and not args.force:
            print(f"already have {onnx_path}; pass --force to rebuild")
            continue
        download(args.url, weights, WEIGHTS_SHA256)
        export(weights, onnx_path, args.imgsz, args.opset, half=precision == "fp16")
        print(f"exported {onnx_path}")

    card_path = models / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
    card.update({
        "source_url": args.url,
        "architecture": ARCHITECTURE,
        "checkpoint": weights.name,
        "checkpoint_sha256": sha256_file(weights) if weights.exists() else None,
        "onnx": outputs["fp32"].name,
        "onnx_sha256": sha256_file(outputs["fp32"]),
        "imgsz": args.imgsz,
        "opset": args.opset,
        "licence": "Apache-2.0 (Megvii YOLOX, code and weights); see docs/licences.md",
        "classes": "COCO 80; this pipeline uses class 0 (person) only",
    })
    if "fp16" in outputs:
        card["onnx_fp16"] = outputs["fp16"].name
        card["onnx_fp16_sha256"] = sha256_file(outputs["fp16"])
    with open(card_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
