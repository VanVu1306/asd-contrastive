"""
models/backbones/r2plus1d.py
==============================
R(2+1)D-18: factorizes each 3D conv into a 2D spatial conv followed by a 1D
temporal conv (Tran et al., 2018). Thin wrapper around torchvision's
`r2plus1d_18`, dropping the classification head like the other backbones.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18


class R2Plus1D18(nn.Module):
    def __init__(self, pretrained: bool = False, pretrained_path: str = None):
        super().__init__()
        weights = "KINETICS400_V1" if pretrained else None
        self.net = r2plus1d_18(weights=weights)
        self.out_dim = self.net.fc.in_features
        self.net.fc = nn.Identity()
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            self.net.load_state_dict(state, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
