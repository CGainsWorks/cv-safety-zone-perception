#!/usr/bin/env python3
"""Build a TensorRT engine from the exported ONNX graph.

    python scripts/export_tensorrt.py --fp16

Needs a machine with an NVIDIA GPU and TensorRT installed.  It does not run in
CI, which has no GPU.  ``make tensorrt-desktop`` runs it inside NVIDIA's
TensorRT container on a desktop RTX 4060; ``docs/tensorrt.md`` and
``results/published/tensorrt.json`` say exactly which machine and versions and
what came out.  Running it somewhere else will give different numbers, which
is the whole point of writing them down.

Engines are **never committed**.  A serialised TensorRT plan is tied to the
GPU architecture, the driver and the TensorRT version that built it and an
engine from elsewhere either refuses to deserialise or, worse, deserialises and
behaves differently.  ``.gitignore`` and ``.gitattributes`` both cover
``*.engine``; this script writes next to the ONNX file and prints the build
environment so a result can be traced to it.

FP16 is the default because it roughly halves the inference time on a Jetson
and, for a detector at this operating point, changes the detections by less
than the box jitter -- but "less than the jitter" is a claim, so
``tests/test_backends.py::test_tensorrt_matches_onnx`` checks it against ONNX
Runtime rather than asserting it.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from szp.detection.base import default_model_path  # noqa: E402


def build(
    onnx_path: Path,
    engine_path: Path,
    *,
    fp16: bool,
    workspace_mb: int,
    verbose: bool,
) -> dict:
    try:
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise SystemExit(
            f"tensorrt is not installed: {exc}\n"
            "On a Jetson it comes with JetPack.  On a desktop, install the TensorRT "
            "wheel that matches your CUDA version.  See docs/tensorrt.md."
        ) from None

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as handle:
        if not parser.parse(handle.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise SystemExit("ONNX parse failed:\n  " + "\n  ".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if fp16:
        if not builder.platform_has_fast_fp16:
            print("warning: this platform has no fast FP16; building anyway")
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"building {engine_path.name} from {onnx_path.name} ...", flush=True)
    started = time.perf_counter()
    serialised = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - started
    if serialised is None:
        raise SystemExit("TensorRT returned no engine; run again with --verbose")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialised)

    return {
        "engine": engine_path.name,
        "onnx": onnx_path.name,
        "fp16": fp16,
        "workspace_mb": workspace_mb,
        "build_seconds": round(elapsed, 1),
        "engine_bytes": engine_path.stat().st_size,
        "tensorrt": trt.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "note": "engines are machine-specific and are never committed",
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=None, help="path to the ONNX file")
    parser.add_argument("--engine", default=None, help="where to write the plan")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--workspace-mb", type=int, default=2048)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    onnx_path = default_model_path(args.model)
    if not onnx_path.exists():
        raise SystemExit(
            f"no ONNX file at {onnx_path}.  Run 'python scripts/fetch_model.py' first."
        )
    engine_path = Path(args.engine) if args.engine else onnx_path.with_suffix(".engine")

    info = build(
        onnx_path,
        engine_path,
        fp16=args.fp16,
        workspace_mb=args.workspace_mb,
        verbose=args.verbose,
    )
    card = engine_path.with_suffix(".engine.json")
    card.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"\nwrote {engine_path} and {card}")
    print("check it against ONNX Runtime:  python -m pytest tests/test_backends.py -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
