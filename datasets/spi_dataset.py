"""
datasets/spi_dataset.py
=========================
Third, independent SSL pretext branch — Synthetic Periodicity Injection
(SPI). Complementary to, not a replacement for, the existing MoCo branch
(datasets/ssl_dataset.py) or SupCon branch (datasets/supcon_dataset.py):
neither of those files is touched by this one.

Manifest format is identical to SSLDataset's (plain text, one clip path per
line, no labels) — the same unlabeled data can feed both branches; nothing
about SPI requires a separate dataset.

Each sample returns 4 items:
    x_periodic_phaseA, x_periodic_phaseB : two T-frame windows from
        different phases (repeat-unit offsets) of ONE synthesized periodic
        clip — the positive pair for SPIConLoss (see its docstring for why
        no cross-video exclusion is used here, unlike real SupCon).
    x_nonperiodic                         : a plain, continuous T-frame crop
        of the same raw window, no SPI applied — the negative pair partner
        (periodic vs non-periodic).
    period_label                          : log(L) — L = the synthesized
        cycle length in frames — ground truth for the optional
        PeriodRegressionHead auxiliary loss.

`clips_per_video` (default 1) works exactly as in SSLDataset — see
datasets/multi_window.py.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from datasets.base_dataset import BaseVideoDataset, frames_to_float_tensor
from datasets.multi_window import WindowPlan, expand_manifest, jitter_within_segment
from transforms.spatial_transforms import VideoSpatialAugment
from transforms.temporal_transforms import SyntheticPeriodicityInjection, TemporalCrop


def _ensure_len(frames: np.ndarray, target_len: int) -> np.ndarray:
    """Wrap-around loop, matching TemporalCrop's own too-short-clip handling
    — needed here only in the extreme edge case where the raw window itself
    is shorter than clip_len, so the synthesized periodic clip is too."""
    t = frames.shape[0]
    if t >= target_len:
        return frames[:target_len]
    idx = np.arange(target_len) % t
    return frames[idx]


class SPIDataset(BaseVideoDataset):
    def __init__(
        self,
        root: str,
        split_file: str,
        clip_len: int = 16,
        raw_clip_len: int = 64,
        frame_size: int = 112,
        frame_source: str = "auto",
        fps: float = 30.0,
        cycle_duration_range_sec: Tuple[float, float] = (0.3, 2.0),
        n_repeats_range: Tuple[int, int] = (3, 5),
        speed_jitter: float = 0.05,
        color_jitter_strength: float = 0.1,
        random_crop_scale=(0.5, 1.0),
        color_jitter: float = 0.4,
        h_flip_prob: float = 0.5,
        random_erasing_prob: float = 0.0,
        random_erasing_scale=(0.02, 0.15),
        clips_per_video: int = 1,
    ):
        super().__init__(root, split_file, frame_source)
        self.raw_clip_len = raw_clip_len
        self.clip_len = clip_len

        # cycle_duration_range_sec (real-world seconds/cycle) -> frame counts,
        # so `L` stays anchored to a plausible stimming frequency instead of
        # an arbitrary frame count that happens to look reasonable at 30fps
        # but not at, say, 15fps or 60fps source footage.
        cycle_len_range = (
            max(2, round(fps * cycle_duration_range_sec[0])),
            max(3, round(fps * cycle_duration_range_sec[1])),
        )
        self.spi = SyntheticPeriodicityInjection(
            cycle_len_range=cycle_len_range, n_repeats_range=n_repeats_range,
            speed_jitter=speed_jitter, color_jitter_strength=color_jitter_strength,
        )

        self._window_plans: Optional[List[WindowPlan]] = None
        self.clips_per_video = max(1, clips_per_video)
        if self.clips_per_video > 1:
            plans, stats = expand_manifest(
                rows=self.samples, resolve_path=self._resolve, frame_source=frame_source,
                window_len=raw_clip_len, clips_per_video=self.clips_per_video,
            )
            self._window_plans = plans
            self.video_index: List[int] = [p.source_index for p in plans]
            print(
                f"[SPIDataset] {stats['num_source_videos']} videos -> {stats['num_expanded_windows']} "
                f"windows (clips_per_video={self.clips_per_video}); "
                f"{stats['videos_with_forced_overlap']}/{stats['num_source_videos']} "
                f"({stats['forced_overlap_ratio']:.1%}) too short to avoid overlap entirely."
            )
        else:
            self.video_index = list(range(len(self.samples)))

        self.raw_crop = TemporalCrop(raw_clip_len, random_start=True)
        self.nonperiodic_crop = TemporalCrop(clip_len, random_start=True)

        aug_kwargs = dict(
            size=frame_size, scale=random_crop_scale, color_jitter=color_jitter,
            h_flip_prob=h_flip_prob, random_erasing_prob=random_erasing_prob,
            random_erasing_scale=random_erasing_scale, train=True,
        )
        # Three independently-drawn augmenters — phaseA/phaseB/non-periodic
        # each get their own crop box, flip, jitter draw, same rationale as
        # SpatialAugment_A/_B in ssl_dataset.py.
        self.spatial_aug_a = VideoSpatialAugment(**aug_kwargs)
        self.spatial_aug_b = VideoSpatialAugment(**aug_kwargs)
        self.spatial_aug_c = VideoSpatialAugment(**aug_kwargs)

    def __len__(self) -> int:
        return len(self._window_plans) if self._window_plans is not None else len(self.samples)

    def _get_raw_window(self, index: int) -> np.ndarray:
        """Identical logic to SSLDataset._get_raw_window — duplicated rather
        than imported so this file has zero coupling to ssl_dataset.py and
        can never affect it."""
        if self._window_plans is None:
            raw_frames = self._load(self.samples[index])
            return self.raw_crop(raw_frames)

        plan = self._window_plans[index]
        raw_frames = self._load(self.samples[plan.source_index])
        num_frames = raw_frames.shape[0]
        if num_frames <= self.raw_clip_len:
            return self.raw_crop(raw_frames)

        start = jitter_within_segment(plan.start, plan.jitter_radius, num_frames, self.raw_clip_len)
        return raw_frames[start : start + self.raw_clip_len]

    def _pick_phase_starts(self, total_len: int, L: int) -> Tuple[int, int]:
        """Two window-start positions favoring different repeat-unit offsets
        (i.e. genuinely different phases of the cycle) when there's room to
        choose distinct units; falls back to independent random starts
        otherwise (short synthesized clip, or L bigger than the available
        range) rather than failing."""
        max_start = max(0, total_len - self.clip_len)
        if max_start <= 0:
            return 0, 0

        num_units = max(1, total_len // L)
        if num_units >= 2 and L <= max_start:
            unit_a = random.randint(0, num_units - 1)
            unit_b = random.randint(0, num_units - 1)
            while unit_b == unit_a:
                unit_b = random.randint(0, num_units - 1)
            return min(unit_a * L, max_start), min(unit_b * L, max_start)

        return random.randint(0, max_start), random.randint(0, max_start)

    def __getitem__(self, index: int):
        raw_window = self._get_raw_window(index)  # (raw_clip_len, H, W, C)

        periodic_clip, L, _N = self.spi(raw_window)
        total_len = periodic_clip.shape[0]

        phase_a_start, phase_b_start = self._pick_phase_starts(total_len, L)
        phase_a_np = _ensure_len(periodic_clip[phase_a_start:], self.clip_len) if total_len - phase_a_start < self.clip_len \
            else periodic_clip[phase_a_start : phase_a_start + self.clip_len]
        phase_b_np = _ensure_len(periodic_clip[phase_b_start:], self.clip_len) if total_len - phase_b_start < self.clip_len \
            else periodic_clip[phase_b_start : phase_b_start + self.clip_len]

        nonperiodic_np = self.nonperiodic_crop(raw_window)

        x_periodic_a = self.spatial_aug_a(frames_to_float_tensor(phase_a_np))
        x_periodic_b = self.spatial_aug_b(frames_to_float_tensor(phase_b_np))
        x_nonperiodic = self.spatial_aug_c(frames_to_float_tensor(nonperiodic_np))

        period_label = torch.tensor(math.log(max(L, 1)), dtype=torch.float32)
        return x_periodic_a, x_periodic_b, x_nonperiodic, period_label
