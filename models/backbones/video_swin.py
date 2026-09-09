"""
models/backbones/video_swin.py
================================
Video Swin Transformer (Liu et al., 2022) — 3D shifted-window self-attention,
used when a task needs longer-range temporal context than a convolutional
backbone's fixed receptive field gives it. Wraps torchvision's `swin3d_t`
(the "tiny" variant; swap in `swin3d_s`/`swin3d_b` for more capacity by
editing `_VARIANTS` below if a config ever needs it).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import swin3d_b, swin3d_s, swin3d_t

_VARIANTS = {"t": swin3d_t, "s": swin3d_s, "b": swin3d_b}


class VideoSwinT(nn.Module):
    def __init__(self, pretrained: bool = False, pretrained_path: str = None, variant: str = "t"):
        super().__init__()
        builder = _VARIANTS[variant]
        weights = "KINETICS400_V1" if pretrained else None
        self.net = builder(weights=weights)
        self.out_dim = self.net.head.in_features
        self.net.head = nn.Identity()
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            self.net.load_state_dict(state, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
