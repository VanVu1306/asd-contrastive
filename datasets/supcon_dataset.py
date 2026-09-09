"""
datasets/supcon_dataset.py
============================
Labeled dataset for Supervised Contrastive pretraining.

Manifest is a CSV with columns: clip_path,video_id,label
    label: 1 = ASD/stimming, 0 = non-ASD    (see design doc: y in {0, 1})

`__getitem__` returns (x_clip, label). The *pairing* logic the design doc
describes — "positives are any two same-label clips in the batch, but from
two different video IDs, to cancel subject/context bias" — is not something
a single `__getitem__` call can enforce (it's a property of the whole
mini-batch), so it lives in two places:
    1. `GroupBalancedBatchSampler` below tries to put >=2 distinct
       video_ids per class into every batch, so cross-video positives
       actually exist to be sampled from.
    2. `losses/supcon_loss.py` takes the video_ids as an extra argument and
       masks out any same-label-same-video_id pair from the positive set,
       so even if two clips from the same video *do* land in a batch
       together, they're never treated as a positive pair.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Sampler

from datasets.base_dataset import BaseVideoDataset, frames_to_float_tensor
from datasets.multi_window import WindowPlan, expand_manifest, jitter_within_segment
from transforms.spatial_transforms import VideoSpatialAugment
from transforms.temporal_transforms import TemporalCrop


class SupConDataset(BaseVideoDataset):
    def __init__(
        self,
        root: str,
        split_file: str,
        clip_len: int = 16,
        frame_size: int = 112,
        frame_source: str = "auto",
        random_crop_scale=(0.5, 1.0),
        color_jitter: float = 0.4,
        h_flip_prob: float = 0.5,
        random_erasing_prob: float = 0.0,
        random_erasing_scale=(0.02, 0.15),
        clips_per_video: int = 1,
        train: bool = True,
    ):
        # Skip BaseVideoDataset's plain-text manifest reader — SupCon needs
        # the richer (path, video_id, label) CSV format instead.
        self.root = root
        self.split_file = split_file
        self.frame_source = frame_source
        self.clip_len = clip_len
        rows: List[Tuple[str, str, int]] = self._read_csv_manifest(split_file)

        self._window_plans: Optional[List[WindowPlan]] = None
        self.clips_per_video = max(1, clips_per_video)
        if self.clips_per_video > 1 and train:
            plans, stats = expand_manifest(
                rows=rows, resolve_path=lambda r: self._resolve(r[0]), frame_source=frame_source,
                window_len=clip_len, clips_per_video=self.clips_per_video,
            )
            self._window_plans = plans
            # Each expanded window inherits its source row's (path, video_id, label).
            self.samples = [rows[p.source_index] for p in plans]
            print(
                f"[SupConDataset] {stats['num_source_videos']} rows -> {stats['num_expanded_windows']} "
                f"windows (clips_per_video={self.clips_per_video}); "
                f"{stats['videos_with_forced_overlap']}/{stats['num_source_videos']} "
                f"({stats['forced_overlap_ratio']:.1%}) too short to avoid overlap entirely."
            )
        else:
            self.samples = rows

        self.crop = TemporalCrop(clip_len, random_start=train)
        self.spatial_aug = VideoSpatialAugment(
            size=frame_size, scale=random_crop_scale, color_jitter=color_jitter,
            h_flip_prob=h_flip_prob, random_erasing_prob=random_erasing_prob,
            random_erasing_scale=random_erasing_scale, train=train,
        )
        # DataLoader batches must be plain tensors, so string video_ids get
        # mapped to ints once here; losses/supcon_loss.py uses this id only
        # to tell "same video" apart from "different video", not as a label.
        unique_vids = sorted({vid for _, vid, _ in self.samples})
        self._vid_to_int = {vid: i for i, vid in enumerate(unique_vids)}

    @staticmethod
    def _read_csv_manifest(split_file: str) -> List[Tuple[str, str, int]]:
        rows = []
        with open(split_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"clip_path", "video_id", "label"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"{split_file} must have columns {sorted(required)}, "
                    f"found {reader.fieldnames}"
                )
            for row in reader:
                rows.append((row["clip_path"], row["video_id"], int(row["label"])))
        return rows

    @property
    def video_ids(self) -> List[str]:
        return [vid for _, vid, _ in self.samples]

    @property
    def labels(self) -> List[int]:
        return [label for _, _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, video_id, label = self.samples[index]
        raw_frames = self._load(path)
        
        if self._window_plans is not None:
            plan = self._window_plans[index]
            num_frames = raw_frames.shape[0]
            if num_frames <= self.clip_len:
                clip_np = self.crop(raw_frames)  # too-short-video fallback: existing wrap-around crop
            else:
                start = jitter_within_segment(plan.start, plan.jitter_radius, num_frames, self.clip_len)
                clip_np = raw_frames[start : start + self.clip_len]
        else:
            clip_np = self.crop(raw_frames)
        
        x_clip = self.spatial_aug(frames_to_float_tensor(clip_np))
        return (
            x_clip,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(self._vid_to_int[video_id], dtype=torch.long),
        )


class GroupBalancedBatchSampler(Sampler[List[int]]):
    """Yields batches that, for each class present, try to include samples
    from at least `min_positives_per_class` distinct video_ids — otherwise a
    batch could easily end up with same-label pairs that are *all* from one
    video, which the loss would then have nothing to form a valid positive
    pair from (see supcon_dataset.py module docstring)."""

    def __init__(
        self,
        video_ids: List[str],
        labels: List[int],
        batch_size: int,
        min_positives_per_class: int = 2,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.min_positives_per_class = min_positives_per_class
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self.by_class: Dict[int, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        for idx, (vid, label) in enumerate(zip(video_ids, labels)):
            self.by_class[label][vid].append(idx)
        self.classes = sorted(self.by_class.keys())

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch (before iterating) so every DDP rank shards
        and reshuffles in sync — same contract as torch's DistributedSampler."""
        self.epoch = epoch

    def __iter__(self):
        from utils.distributed import get_rank, get_world_size
        
        rank, world_size = get_rank(), get_world_size()
        rng = random.Random(self.seed + self.epoch)
        
        # Pool of (class -> shuffled list-of-video_id-buckets), each bucket
        # itself shuffled, so every epoch sees a different pairing.
        pools = {}
        for c in self.classes:
            vid_buckets = [list(v) for v in self.by_class[c].values()]
            for bucket in vid_buckets:
                rng.shuffle(bucket)
            rng.shuffle(vid_buckets)
            per_rank = len(vid_buckets) // world_size
            if per_rank == 0:
                # Fewer video_ids than ranks for this class — every rank
                # gets the same (small) leftover rather than starving some
                # ranks of this class entirely.
                pools[c] = vid_buckets
            else:
                sharded = vid_buckets[: per_rank * world_size]
                pools[c] = sharded[rank::world_size]

        n_batches = sum(len(b) for buckets in pools.values() for b in buckets) // self.batch_size
        for _ in range(n_batches):
            batch: List[int] = []
            per_class_budget = max(1, self.batch_size // max(1, len(self.classes)))
            for c in self.classes:
                buckets = pools[c]
                taken_vids = 0
                for bucket in buckets:
                    if not bucket or taken_vids >= max(self.min_positives_per_class, per_class_budget):
                        continue
                    batch.append(bucket.pop())
                    taken_vids += 1
                    if len(batch) >= self.batch_size:
                        break
            if not batch:
                break
            rng.shuffle(batch)
            yield batch[: self.batch_size]

    def video_ids_flat(self):
        return [i for buckets in self.by_class.values() for ids in buckets.values() for i in ids]

    def __len__(self):
        from utils.distributed import get_world_size
        
        return (len(self.video_ids_flat()) // get_world_size()) // self.batch_size