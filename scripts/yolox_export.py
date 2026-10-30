"""YOLOX-Tiny as this repository exports it.  Export time only.

Imports torch and yolox, so it lives in ``scripts/`` next to the one script
that uses it, ``scripts/fetch_model.py`` and nothing under ``src/`` can reach
it.  The runtime image has neither package and the ``docker`` CI job checks.

YOLOX is Megvii's detector.  The code and the released COCO checkpoint are
both Apache-2.0, which is the reason it is here; ``docs/licences.md`` has the
reasoning and the alternatives that were weighed.

Two things change between the network Megvii released and the graph this
repository exports and both are exact rather than approximate.

The input.  YOLOX takes BGR in 0..255 with no normalisation.  The pipeline
feeds RGB in 0..1, because that is what :func:`szp.detection.preprocess.letterbox`
produces and what both backends share.  Rather than add a second preprocessing
path, :func:`fold_input_into_stem` scales the first convolution's weights by
255 and reverses their colour order.  The Focus stem in front of that
convolution only slices the image into four phases, so the fold is the same
arithmetic in a different order and adds nothing to the graph.

The output.  YOLOX's head emits per-anchor box offsets, an objectness score and
per-class scores.  :class:`DeployYOLOX` decodes the offsets against the stride
8, 16 and 32 grids, multiplies objectness into the class scores and transposes
to ``(1, 4 + 80, 8400)`` at 640x640.  That is the layout
:func:`szp.detection.postprocess.decode_head` reads: centre, size and one
score per class in letterboxed pixels, already through the sigmoid.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

__all__ = [
    "ARCHITECTURES",
    "build_yolox",
    "load_checkpoint",
    "fold_input_into_stem",
    "DeployYOLOX",
    "deploy_module",
]

#: depth and width multipliers, from Megvii's exps/default/*.py
ARCHITECTURES = {
    "yolox_nano": (0.33, 0.25),
    "yolox_tiny": (0.33, 0.375),
    "yolox_s": (0.33, 0.50),
}

STRIDES = (8, 16, 32)


def build_yolox(architecture: str, num_classes: int) -> nn.Module:
    """The YOLOX network, as ``yolox.exp.Exp.get_model`` builds it."""
    from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

    depth, width = ARCHITECTURES[architecture]
    in_channels = [256, 512, 1024]
    depthwise = architecture == "yolox_nano"
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, depthwise=depthwise, act="silu")
    head = YOLOXHead(num_classes, width, in_channels=in_channels, depthwise=depthwise, act="silu")
    model = YOLOX(backbone, head)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03
    return model


def load_checkpoint(path: str | Path) -> dict:
    """Read a Megvii checkpoint without executing anything in it.

    ``weights_only=True``: the file carries the model and optimiser state as
    plain tensors, which is all a weights-only load allows.
    """
    return torch.load(str(path), map_location="cpu", weights_only=True)


def fold_input_into_stem(model: nn.Module) -> None:
    """Make the network take RGB in 0..1 instead of BGR in 0..255, in place.

    The stem's convolution sees twelve channels: four spatial phases of three
    colours each.  Reversing each group of three and scaling by 255 gives a
    convolution that computes on RGB/255 exactly what the original computed on
    BGR.  Zero padding is zero in both encodings, so the borders agree as well.
    """
    conv = model.backbone.backbone.stem.conv.conv
    weight = conv.weight.data
    out_channels, in_channels, kh, kw = weight.shape
    if in_channels != 12:
        raise ValueError(f"expected a Focus stem with 12 input channels, got {in_channels}")
    folded = weight.view(out_channels, 4, 3, kh, kw).flip(2) * 255.0
    conv.weight.data = folded.reshape(out_channels, in_channels, kh, kw).contiguous()


def _grids(width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    grids, strides = [], []
    for stride in STRIDES:
        h, w = height // stride, width // stride
        yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        grids.append(torch.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(torch.full((h * w, 1), float(stride)))
    return torch.cat(grids).float()[None], torch.cat(strides)[None]


class DeployYOLOX(nn.Module):
    """RGB 0..1 in, ``(N, 4 + num_classes, anchors)`` out.

    Written without in-place slice assignment, which is how YOLOX's own
    ``decode_outputs`` works and which traces to ScatterND.  Slice, Add, Mul,
    Exp, Concat and a Transpose is a graph ONNX Runtime and TensorRT both
    handle and one the INT8 quantiser can be told exactly which parts to leave
    alone.
    """

    def __init__(self, model: nn.Module, width: int, height: int) -> None:
        super().__init__()
        if width % 32 or height % 32:
            raise ValueError(f"input must be a multiple of 32 in both axes, got {width}x{height}")
        self.model = model
        self.model.head.decode_in_inference = False
        grid, stride = _grids(width, height)
        self.register_buffer("grid", grid, persistent=False)
        self.register_buffer("stride", stride, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.model(x)
        xy = (raw[..., 0:2] + self.grid) * self.stride
        wh = torch.exp(raw[..., 2:4]) * self.stride
        scores = raw[..., 4:5] * raw[..., 5:]
        return torch.cat((xy, wh, scores), dim=-1).permute(0, 2, 1)


def deploy_module(
    checkpoint: str | Path, architecture: str, num_classes: int, width: int, height: int
) -> DeployYOLOX:
    """Megvii's checkpoint as the module the export traces.

    Every tensor in the checkpoint must match the architecture: this is the
    released COCO model used as shipped, so a single skipped tensor means the
    wrong file or the wrong architecture and that is an error.
    """
    model = build_yolox(architecture, num_classes)
    state = load_checkpoint(checkpoint)["model"]
    own = model.state_dict()
    mismatched = [k for k, v in state.items() if k not in own or own[k].shape != v.shape]
    missing = sorted(set(own) - set(state))
    if mismatched or missing:
        raise RuntimeError(
            f"{checkpoint} does not match {architecture} with {num_classes} classes: "
            f"mismatched {mismatched[:5]}, missing {missing[:5]}"
        )
    model.load_state_dict(state)
    model.eval()
    fold_input_into_stem(model)
    return DeployYOLOX(model, width, height).eval()
