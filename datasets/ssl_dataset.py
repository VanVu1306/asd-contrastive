"""
datasets/ssl_dataset.py
========================
Unlabeled dataset for MoCo pretraining. Manifest is a plain text file with
one clip path per line (see configs/ssl_moco.yaml: data.train_split).

Per the design doc, each sample returns three views built from one raw
window of `raw_clip_len` frames:

    x_anchor      : contiguous T-frame crop + SpatialAugment_A
    x_pos_warp    : SpeedWarp (0.8x or 1.5x) + SpatialAugment_B
    x_neg_shuffle : FrameShuffle + SpatialAugment_A

Batches of different clips' shuffled views double as the "used-as-negatives"
side of InfoNCE inside MoCo's queue — see losses/moco_nce_loss.py and
models/moco_wrapper.py for how anchor/warp/shuffle map onto MoCo's
query/key/queue roles.
"""
from __future__ import annotations

from typing import Tuple, List, Optional

import numpy as np
import torch

from datasets.base_dataset import BaseVideoDataset, frames_to_float_tensor
from datasets.multi_window import WindowPlan, expand_manifest, jitter_within_segment
from transforms.spatial_transforms import VideoSpatialAugment
from transforms.temporal_transforms import FrameShuffle, SpeedWarp, TemporalCrop


class SSLDataset(BaseVideoDataset):
    def __init__(
        self,
        root: str,
        split_file: str,
        clip_len: int = 16,
        raw_clip_len: int = 64,
        frame_size: int = 112,
        frame_source: str = "auto",
        speed_warp_rates=(0.8, 1.5),
        frame_shuffle_ratio: float = 1.0,
        random_crop_scale=(0.5, 1.0),
        color_jitter: float = 0.4,
        h_flip_prob: float = 0.5,
        random_erasing_prob: float = 0.0,
        random_erasing_scale=(0.02, 0.15),
        clips_per_video: int = 1,
    ):
        super().__init__(root, split_file, frame_source)
        self.raw_clip_len = raw_clip_len
        self.clips_per_video = max(1, clips_per_video)

        self._window_plans: Optional[List[WindowPlan]] = None
        if self.clips_per_video > 1:
            plans, stats = expand_manifest(
                rows=self.samples, resolve_path=self._resolve, frame_source=frame_source,
                window_len=raw_clip_len, clips_per_video=self.clips_per_video,
            )
            self._window_plans = plans
            self.video_index: List[int] = [p.source_index for p in plans]
            print(
                f"[SSLDataset] {stats['num_source_videos']} videos -> {stats['num_expanded_windows']} "
                f"windows (clips_per_video={self.clips_per_video}); "
                f"{stats['videos_with_forced_overlap']}/{stats['num_source_videos']} videos "
                f"({stats['forced_overlap_ratio']:.1%}) were too short to avoid overlap entirely."
            )
        else:
            # video_index is trivially identity here — every sample already
            # is its own "video" as far as VideoDiverseBatchSampler cares.
            self.video_index = list(range(len(self.samples)))

        self.raw_crop = TemporalCrop(raw_clip_len, random_start=True)
        self.anchor_crop = TemporalCrop(clip_len, random_start=True)
        self.speed_warp = SpeedWarp(clip_len, rates=speed_warp_rates)
        self.frame_shuffle = FrameShuffle(ratio=frame_shuffle_ratio)

        aug_kwargs = dict(
            size=frame_size, scale=random_crop_scale, color_jitter=color_jitter,
            h_flip_prob=h_flip_prob, random_erasing_prob=random_erasing_prob,
            random_erasing_scale=random_erasing_scale, train=True,
        )
        # Two independently-seeded augmenters == SpatialAugment_A / _B from
        # the design doc. Independent instances matter only in that random
        # draws are *not* shared between them (each call to __call__ samples
        # its own crop box / jitter factors regardless), documented here
        # mainly so the anchor/shuffle-share-A vs warp-uses-B pairing below
        # reads clearly against the spec.
        self.spatial_aug_a = VideoSpatialAugment(**aug_kwargs)
        self.spatial_aug_b = VideoSpatialAugment(**aug_kwargs)

    def __len__(self) -> int:
        return len(self._window_plans) if self._window_plans is not None else len(self.samples)
    
    def _get_raw_window(self, index: int) -> np.ndarray:
        """Returns the (raw_clip_len, H, W, C) window for dataset index
        `index` — either a fixed low-overlap slot (clips_per_video > 1) or a
        fresh uniformly-random crop (the original, default behavior)."""
        if self._window_plans is None:
            raw_frames = self._load(self.samples[index])
            return self.raw_crop(raw_frames)
    
        plan = self._window_plans[index]
        raw_frames = self._load(self.samples[plan.source_index])
        num_frames = raw_frames.shape[0]
        if num_frames <= self.raw_clip_len:
            # Video shorter than the raw window itself — this is exactly the
            # case TemporalCrop's own wrap-around (frame-looping) logic
            # exists for, regardless of what multi_window planned.
            return self.raw_crop(raw_frames)
    
        start = jitter_within_segment(plan.start, plan.jitter_radius, num_frames, self.raw_clip_len)
        return raw_frames[start : start + self.raw_clip_len]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_window = self._get_raw_window(index)  # (raw_clip_len, H, W, C)

        anchor_np = self.anchor_crop(raw_window)
        warp_np = self.speed_warp(raw_window)
        shuffle_np = self.frame_shuffle(anchor_np)

        x_anchor = self.spatial_aug_a(frames_to_float_tensor(anchor_np))
        x_pos_warp = self.spatial_aug_b(frames_to_float_tensor(warp_np))
        x_neg_shuffle = self.spatial_aug_a(frames_to_float_tensor(shuffle_np))

        return x_anchor, x_pos_warp, x_neg_shuffle
