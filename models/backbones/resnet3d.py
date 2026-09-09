"""
models/backbones/resnet3d.py
==============================
3D-ResNet baseline backbones. ResNet3D-18 is torchvision's `r3d_18` as-is.
torchvision doesn't ship a 50-layer 3D-ResNet builder, so ResNet3D-50 is
assembled here from the same building blocks torchvision uses internally
(`VideoResNet` + `Bottleneck` + `Conv3DSimple` + `BasicStem`), using the
standard ResNet-50 stage depths [3, 4, 6, 3].

Both variants expose `.out_dim` and drop the classification `fc` layer,
returning pooled (B, out_dim) features — the input to `models/heads.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torchvision.models.video.resnet import BasicStem, Bottleneck, Conv3DSimple, VideoResNet


def _strip_classifier(model: nn.Module) -> int:
    out_dim = model.fc.in_features
    model.fc = nn.Identity()
    return out_dim


class ResNet3D18(nn.Module):
    def __init__(self, pretrained: bool = False, pretrained_path: str = None):
        super().__init__()
        weights = "KINETICS400_V1" if pretrained else None
        self.net = r3d_18(weights=weights)
        self.out_dim = _strip_classifier(self.net)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            self.net.load_state_dict(state, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W) -> (B, out_dim)
        return self.net(x)


class ResNet3D50(nn.Module):
    """3D-ResNet-50 = torchvision's VideoResNet scaffolding with Bottleneck
    blocks and stage depths [3, 4, 6, 3] (the standard ResNet-50 recipe)."""

    def __init__(self, pretrained: bool = False, pretrained_path: str = None):
        super().__init__()
        if pretrained:
            raise ValueError(
                "No torchvision-hosted pretrained weights exist for 3D-ResNet-50; "
                "pass backbone.pretrained_path to a local checkpoint instead."
            )
        self.net = VideoResNet(
            block=Bottleneck,
            conv_makers=[Conv3DSimple] * 4,
            layers=[3, 4, 6, 3],
            stem=BasicStem,
        )
        self.out_dim = _strip_classifier(self.net)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            self.net.load_state_dict(state, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
