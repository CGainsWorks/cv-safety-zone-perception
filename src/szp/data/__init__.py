"""Dataset loaders and the synthetic rig."""

from szp.data.coco import CocoImage, CocoPersonDataset, default_coco_root
from szp.data.synthetic import SceneConfig, SimulatedDetector, SyntheticScene, build_scene

__all__ = [
    "CocoPersonDataset",
    "CocoImage",
    "default_coco_root",
    "SyntheticScene",
    "SceneConfig",
    "SimulatedDetector",
    "build_scene",
]
