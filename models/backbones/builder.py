"""
models/backbones/builder.py
=============================
`build_backbone(cfg)` — the single place that maps a config's
`backbone.name` string to an actual nn.Module. Add a new backbone by
registering it in `_REGISTRY`; nothing else in the repo needs to change.
"""
from __future__ import annotations

import torch.nn as nn

from models.backbones.r2plus1d import R2Plus1D18
from models.backbones.resnet3d import ResNet3D18, ResNet3D50
from models.backbones.s3d import S3D
from models.backbones.video_swin import VideoSwinT

_REGISTRY = {
    "resnet3d_18": ResNet3D18,
    "resnet3d_50": ResNet3D50,
    "r2plus1d_18": R2Plus1D18,
    "s3d": S3D,
    "video_swin_t": VideoSwinT,
}


def build_backbone(cfg) -> nn.Module:
    """cfg is the `backbone:` sub-config (name / pretrained / pretrained_path)."""
    name = cfg["name"]
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](
        pretrained=cfg.get("pretrained", False),
        pretrained_path=cfg.get("pretrained_path", None),
    )
