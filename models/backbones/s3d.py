"""
models/backbones/s3d.py
=========================
S3D: separable 3D convolutions (spatial + temporal factorized per-layer,
Xie et al., 2018) — cheaper than plain 3D conv while still capturing
repetitive motion well, which is why the design doc calls it out
specifically for periodicity. Wraps torchvision's `s3d`.

torchvision's S3D classifier head is a Conv3d (not a Linear layer like the
ResNet-style backbones), so it's swapped out slightly differently: replaced
with an Identity + adaptive pool so the output is still a flat (B, out_dim)
feature vector. We also replace torchvision's fixed-kernel `avgpool`
(hard-coded for its Kinetics input size) with `AdaptiveAvgPool3d(1)`, so this
backbone works at whatever `data.clip_len` / `data.frame_size` a config
picks instead of silently requiring near-224px, multi-second clips.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import s3d


class S3D(nn.Module):
    def __init__(self, pretrained: bool = False, pretrained_path: str = None):
        super().__init__()
        weights = "KINETICS400_V1" if pretrained else None
        net = s3d(weights=weights)
        self.features = net.features
        self.avgpool = nn.AdaptiveAvgPool3d(1)  # robust to any (T, H, W), unlike net.avgpool
        self.out_dim = 1024  # S3D's feature-extractor output channels
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            self.load_state_dict(state, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)
