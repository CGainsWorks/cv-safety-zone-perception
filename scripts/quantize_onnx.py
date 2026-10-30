#!/usr/bin/env python3
"""Build the INT8 ONNX graph from the FP32 export.

    python scripts/quantize_onnx.py

Static QDQ quantisation through ONNX Runtime, calibrated on the fixed list of
COCO train2017 photographs in ``data/calibration_train2017.txt`` (fetched by
``scripts/fetch_calibration.py``), never on val2017, which is what the result
is scored on.  INT8 is derived from the FP32 graph rather than exported
separately, so the only difference between the two files is the arithmetic.

Convolutions only, with the detection head's nine output convolutions left in
float.  The head ends in a Sigmoid over objectness and class logits, an Exp on
the box size and the grid arithmetic that turns offsets into pixels; quantising
those to uint8 puts one coarse step across tensors whose useful range spans
several orders of magnitude, box centres up to 640 pixels beside scores that
live between 0 and 1.  The output convolutions feeding them are found by name
rather than listed by hand, so the exclusion keeps matching if the network
changes and fails loudly if it stops matching anything.

This is the ONNX Runtime INT8 path, measured on the CPU.  TensorRT's INT8 is a
different implementation with its own calibrator
(``scripts/export_tensorrt.py --precision int8``) and the two are scored
separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fetch_calibration import default_calib_dir, read_list  # noqa: E402
from szp.detection.preprocess import letterbox  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CalibrationReader:
    """Calibration frames through the pipeline's own letterbox."""

    def __init__(self, images, input_name: str, input_size: tuple) -> None:
        import cv2

        self._cv2 = cv2
        self.images = list(images)
        self.input_name = input_name
        self.input_size = input_size
        self.index = 0

    def get_next(self):
        if self.index >= len(self.images):
            return None
        path = self.images[self.index]
        self.index += 1
        image = self._cv2.imread(str(path))
        if image is None:
            raise ValueError(f"could not decode calibration image {path}")
        tensor, _ = letterbox(image, self.input_size)
        return {self.input_name: tensor}

    def rewind(self) -> None:
        self.index = 0


def head_output_nodes(model_path: Path) -> list:
    """YOLOX's nine head output convolutions: cls, reg and obj on each stride."""
    import onnx

    branches = ("cls_preds.", "reg_preds.", "obj_preds.")
    names = [n.name for n in onnx.load(str(model_path)).graph.node if n.op_type == "Conv"]
    outputs = sorted(
        name for name in names
        if "/head/" in name and any(part.startswith(branches) for part in name.split("/"))
    )
    if len(outputs) != 9:
        raise SystemExit(
            f"expected nine head output convolutions and found {len(outputs)}: {outputs}.  "
            "Quantising without the exclusion would quantise the head."
        )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=str(REPO_ROOT / "models" / "yolox_tiny.onnx"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--calib-dir", default=None)
    parser.add_argument("--calib-images", type=int, default=256)
    parser.add_argument("--input-size", type=int, default=640)
    args = parser.parse_args()

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType
    from onnxruntime.quantization import quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    source = Path(args.model)
    destination = Path(args.out) if args.out else source.with_name(f"{source.stem}_int8.onnx")
    calib_dir = Path(args.calib_dir) if args.calib_dir else default_calib_dir()
    images = [calib_dir / name for name in read_list()[: args.calib_images]]
    missing = [p.name for p in images if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} calibration images missing; run "
                         "scripts/fetch_calibration.py")

    prepared = source.with_name(source.stem + "_prepared.onnx")
    quant_pre_process(str(source), str(prepared), skip_symbolic_shape=False)
    exclude = head_output_nodes(source)
    input_name = onnx.load(str(source)).graph.input[0].name
    size = (args.input_size, args.input_size)
    quantize_static(
        model_input=str(prepared),
        model_output=str(destination),
        calibration_data_reader=CalibrationReader(images, input_name, size),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv"],
        nodes_to_exclude=exclude,
        extra_options={"AddQDQPairToWeight": False},
    )
    prepared.unlink(missing_ok=True)

    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    output = session.run(None, {input_name: np.zeros((1, 3, *size[::-1]), np.float32)})[0]
    nodes = onnx.load(str(destination)).graph.node
    qdq = sum(1 for n in nodes if n.op_type in ("QuantizeLinear", "DequantizeLinear"))
    if qdq == 0:
        raise SystemExit(f"{destination.name} has no QDQ nodes; it is not quantised")
    record = {
        "onnx": destination.name,
        "source": source.name,
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "size_ratio": round(destination.stat().st_size / source.stat().st_size, 4),
        "qdq_nodes": qdq,
        "output_shape": list(output.shape),
        "quant_format": "QDQ",
        "per_channel": True,
        "calibrate_method": "MinMax",
        "op_types_to_quantize": ["Conv"],
        "nodes_excluded": exclude,
        "calibration_split": "train2017",
        "calibration_images": len(images),
    }
    card = source.with_name("quantisation.json")
    with open(card, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
