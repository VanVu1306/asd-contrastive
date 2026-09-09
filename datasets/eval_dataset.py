"""
datasets/eval_dataset.py
=========================
Sliding-window loader used by eval.py for both eval modes:

  - anomaly_scoring : every window of every test video needs a score, so we
    enumerate *all* windows (stride = eval.window_stride) instead of taking
    one random crop per video like the training datasets do.
  - linear_probing  : reuses the same (clip, label) shape as SupConDataset,
    just without the train-time augmentation (crop/flip/jitter off).

Two manifest shapes are supported, matching configs/eval.yaml:
  - normal_split (centroid mining): plain text, one clip path per line
    (no labels needed — see models/heads.py CentroidScoringHead).
  - test_split (scoring) / linear_probe splits: CSV with
    clip_path,video_id,frame_labels_path — frame_labels_path points to a
    .npy of per-frame 0/1 ground truth, used only for computing metrics,
    never for training.
"""
from __future__ import annotations

import csv
import os
from typing import List, NamedTuple, Optional

import numpy as np
import torch

from datasets.base_dataset import BaseVideoDataset, frames_to_float_tensor, load_raw_frames
from transforms.spatial_transforms import VideoSpatialAugment
from transforms.temporal_transforms import SlidingWindow


class WindowSample(NamedTuple):
    clip: torch.Tensor
    video_index: int
    start_frame: int


class CentroidMiningDataset(BaseVideoDataset):
    """One random T-frame crop per normal-only clip — Stage 2 of the
    non-parametric anomaly-scoring pipeline (K-Means over encoder features,
    zero backprop)."""

    def __init__(self, root: str, split_file: str, clip_len: int, frame_size: int, frame_source: str = "auto"):
        super().__init__(root, split_file, frame_source)
        from transforms.temporal_transforms import TemporalCrop

        self.crop = TemporalCrop(clip_len, random_start=True)
        self.spatial_aug = VideoSpatialAugment(size=frame_size, train=False)

    def __getitem__(self, index: int) -> torch.Tensor:
        raw = self._load(self.samples[index])
        clip_np = self.crop(raw)
        return self.spatial_aug(frames_to_float_tensor(clip_np))


class SlidingWindowTestDataset:
    """Not a torch Dataset in the usual per-sample sense: __getitem__ returns
    *all* windows for one video at once, since Stage 3 scoring and the
    frame-level metrics (utils/metrics.py) both need to reassemble a
    per-frame score curve per video, not a flat pool of unrelated windows."""

    def __init__(
        self,
        root: str,
        split_file: str,
        clip_len: int,
        stride: int,
        frame_size: int,
        frame_source: str = "auto",
    ):
        self.root = root
        self.frame_source = frame_source
        self.window = SlidingWindow(clip_len, stride)
        self.spatial_aug = VideoSpatialAugment(size=frame_size, train=False)
        self.entries = self._read_csv(split_file)

    @staticmethod
    def _read_csv(split_file: str):
        rows = []
        with open(split_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def __len__(self) -> int:
        return len(self.entries)

    def _resolve(self, path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(self.root, path)

    def get_video(self, index: int):
        """Returns (clips: (N, C, T, H, W) tensor, starts: (N,) array,
        frame_labels: (T_total,) array-or-None, video_id: str)."""
        row = self.entries[index]
        frames = load_raw_frames(self._resolve(row["clip_path"]), self.frame_source)
        t_total = frames.shape[0]
        starts = self.window.window_starts(t_total)

        clips = []
        for s in starts:
            window_np = self.window(frames, int(s))
            clips.append(self.spatial_aug(frames_to_float_tensor(window_np)))
        clips = torch.stack(clips, dim=0)  # (N, C, T, H, W)

        frame_labels: Optional[np.ndarray] = None
        labels_path = row.get("frame_labels_path")
        if labels_path:
            frame_labels = np.load(self._resolve(labels_path))

        video_id = row.get("video_id", str(index))
        return clips, starts, frame_labels, video_id
