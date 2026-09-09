"""
transforms/spatial_transforms.py
=================================
Pixel-level (spatial) augmentation applied identically to every frame of a
clip — the crop box, flip decision, and colour jitter factors are sampled
*once per clip* and re-used across all T frames. If we resampled per-frame
the augmentation itself would inject artificial motion/flicker, which would
corrupt exactly the periodicity signal this project is trying to learn.

Operates on a torch tensor of shape (T, C, H, W), float32 in [0, 1]
(datasets/base_dataset.py converts raw uint8 frames to this layout before
handing them to a spatial transform).

`SpatialAugment_A` / `SpatialAugment_B` from the design doc are just two
independently-seeded instances of `VideoSpatialAugment` — see
datasets/ssl_dataset.py for how the two views are built from them.

`RandomErasingVideo` (temporally-consistent cutout) is an optional extra
decorrelation step, off by default — see its own docstring for why it
exists (in-batch leakage between same-video clips).
"""
from __future__ import annotations

import random
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class RandomResizedCropVideo:
    """Same crop box + resize for every frame in the clip."""

    def __init__(self, size: int, scale: Tuple[float, float] = (0.5, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        _, _, h, w = clip.shape
        area = h * w
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(3 / 4, 4 / 3)
            crop_w = int(round((target_area * aspect_ratio) ** 0.5))
            crop_h = int(round((target_area / aspect_ratio) ** 0.5))
            if 0 < crop_w <= w and 0 < crop_h <= h:
                top = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)
                clip = clip[:, :, top : top + crop_h, left : left + crop_w]
                return F.interpolate(
                    clip, size=(self.size, self.size), mode="bilinear", align_corners=False
                )
        # fallback: center crop to the smaller spatial dim, then resize
        crop = min(h, w)
        top, left = (h - crop) // 2, (w - crop) // 2
        clip = clip[:, :, top : top + crop, left : left + crop]
        return F.interpolate(clip, size=(self.size, self.size), mode="bilinear", align_corners=False)


class RandomHorizontalFlipVideo:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            clip = torch.flip(clip, dims=[-1])
        return clip


class ColorJitterVideo:
    """Brightness/contrast/saturation/hue jitter with one sampled factor set
    shared across the whole clip (per-frame factors would flicker)."""

    def __init__(self, strength: float = 0.4):
        self.brightness = self.contrast = self.saturation = strength
        self.hue = min(0.1, strength / 4)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        b = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        c = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        s = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        h = random.uniform(-self.hue, self.hue)
        frames = [TF.adjust_hue(
            TF.adjust_saturation(
                TF.adjust_contrast(TF.adjust_brightness(frame, b), c), s
            ), h
        ) for frame in clip]
        return torch.stack(frames, dim=0)
    

class RandomErasingVideo:
    """Temporally-consistent cutout: one randomly placed rectangle, erased
    (filled with its per-channel mean, like torchvision's RandomErasing) at
    the *same* pixel location across every frame of the clip.

    This exists specifically for the "reduce in-batch leakage" case: with
    `data.clips_per_video > 1`, several clips from the same source video can
    still end up sharing a batch even with VideoDiverseBatchSampler capping
    it at a small number — cutout forces the encoder to not rely on any one
    fixed patch of background/appearance being present, on top of what
    RandomResizedCrop + ColorJitter already do. Off by default
    (`p=0.0`) so it never changes existing configs unless opted in.
    """

    def __init__(self, p: float = 0.0, scale: Tuple[float, float] = (0.02, 0.15)):
        self.p = p
        self.scale = scale

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or random.random() >= self.p:
            return clip
        _, _, h, w = clip.shape
        area = h * w
        target_area = random.uniform(*self.scale) * area
        aspect_ratio = random.uniform(3 / 4, 4 / 3)
        box_h = min(h, max(1, int(round((target_area * aspect_ratio) ** 0.5))))
        box_w = min(w, max(1, int(round((target_area / aspect_ratio) ** 0.5))))
        top = random.randint(0, h - box_h)
        left = random.randint(0, w - box_w)

        fill = clip.mean(dim=(0, 2, 3), keepdim=True)  # per-channel mean over the whole clip
        clip = clip.clone()
        clip[:, :, top : top + box_h, left : left + box_w] = fill
        return clip


class NormalizeVideo:
    """ImageNet-style per-channel normalization, broadcast over T."""

    def __init__(self, mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        return (clip - self.mean.to(clip.device)) / self.std.to(clip.device)


class VideoSpatialAugment:
    """Bundles the four transforms above into the single augmentation
    pipeline referenced as `SpatialAugment_A` / `SpatialAugment_B` in the
    design doc. `train=False` skips crop/flip/jitter and only resizes +
    normalizes (used by eval_dataset.py)."""

    def __init__(
        self,
        size: int = 112,
        scale: Tuple[float, float] = (0.5, 1.0),
        color_jitter: float = 0.4,
        h_flip_prob: float = 0.5,
        random_erasing_prob: float = 0.0,
        random_erasing_scale: Tuple[float, float] = (0.02, 0.15),
        train: bool = True,
    ):
        self.train = train
        self.size = size
        self.crop = RandomResizedCropVideo(size, scale)
        self.flip = RandomHorizontalFlipVideo(h_flip_prob)
        self.jitter = ColorJitterVideo(color_jitter)
        self.erase = RandomErasingVideo(random_erasing_prob, random_erasing_scale)
        self.normalize = NormalizeVideo()

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (T, C, H, W) float32 in [0, 1]
        if self.train:
            clip = self.crop(clip)
            clip = self.flip(clip)
            clip = self.jitter(clip)
            clip = self.erase(clip)
        else:
            clip = F.interpolate(clip, size=(self.size, self.size), mode="bilinear", align_corners=False)
        clip = self.normalize(clip)
        # (T, C, H, W) -> (C, T, H, W) to match the backbone's expected layout
        return clip.permute(1, 0, 2, 3).contiguous()