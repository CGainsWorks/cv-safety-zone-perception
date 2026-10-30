"""The TensorRT code path against the real TensorRT Python API.

Skipped unless ``tensorrt`` is importable, which in CI means the
``tensorrt-api`` job, where ``requirements-tensorrt.txt`` installs the pinned
wheel on a runner with no GPU.

What a GPU-less runner can check and what it cannot.  The wheel imports and
its classes and enums can be inspected, so every TensorRT symbol that
``szp.detection.tensorrt_backend`` and ``scripts/export_tensorrt.py`` touch is
looked up here against the pinned version.  ``trt.Builder`` needs a CUDA
driver and aborts the process without one, so creating a network definition,
parsing the ONNX graph and building an engine stay a manual step
(``make tensorrt-desktop``, in ``docs/tensorrt.md``).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

trt = pytest.importorskip("tensorrt")

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "src" / "szp" / "detection" / "tensorrt_backend.py",
    ROOT / "scripts" / "export_tensorrt.py",
)

# Methods and properties called on TensorRT objects rather than on the module.
OBJECT_API = {
    "Runtime": ("deserialize_cuda_engine",),
    "ICudaEngine": ("num_io_tensors", "get_tensor_name", "get_tensor_mode",
                    "get_tensor_shape", "get_tensor_dtype", "create_execution_context"),
    "IExecutionContext": ("get_tensor_shape", "set_tensor_address", "execute_async_v3"),
    "Builder": ("create_network", "create_builder_config", "build_serialized_network",
                "platform_has_fast_fp16"),
    "IBuilderConfig": ("set_memory_pool_limit", "set_flag"),
    "OnnxParser": ("parse", "num_errors", "get_error"),
}


def module_chains(path: Path) -> set:
    """Every dotted ``trt.X.Y`` chain a source file uses, longest form only."""
    chains = set()

    def dotted(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            head = dotted(node.value)
            return f"{head}.{node.attr}" if head else None
        return None

    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Attribute):
            chain = dotted(node)
            if chain and chain.startswith("trt."):
                chains.add(chain)
    return {c for c in chains if not any(o.startswith(c + ".") for o in chains)}


def test_the_wheel_is_the_pinned_version():
    pin = re.search(r"tensorrt-cu12==([\d.]+)",
                    (ROOT / "requirements-tensorrt.txt").read_text(encoding="utf-8"))
    assert pin, "requirements-tensorrt.txt has no tensorrt-cu12 pin"
    assert trt.__version__ == pin.group(1)


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.name)
def test_every_module_symbol_exists(source):
    chains = module_chains(source)
    assert chains, f"found no trt.* usage in {source.name}; has the import alias changed?"
    missing = []
    for chain in sorted(chains):
        node = trt
        for part in chain.split(".")[1:]:
            if not hasattr(node, part):
                missing.append(chain)
                break
            node = getattr(node, part)
    assert not missing, f"{source.name} uses symbols this TensorRT lacks: {missing}"


@pytest.mark.parametrize("cls", sorted(OBJECT_API))
def test_every_object_method_exists(cls):
    klass = getattr(trt, cls)
    missing = [name for name in OBJECT_API[cls] if not hasattr(klass, name)]
    assert not missing, f"trt.{cls} lacks {missing}"


def test_the_backend_module_imports():
    from szp.detection import tensorrt_backend

    assert tensorrt_backend.TensorRTDetector.name == "tensorrt"
