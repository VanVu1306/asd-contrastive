"""
datasets/samplers.py
======================
`VideoDiverseBatchSampler` — caps how many clips from the *same source
video* can land in one training batch (`max_per_video`, default 1: at most
one clip per video per batch). This is the batch-construction half of the
"reduce leakage" request: `datasets/multi_window.py` can now hand out
several low-overlap clips per video, but without this sampler nothing stops
several of them from ending up in the same batch — which is exactly the
situation where a contrastive loss can cheat by matching on shared
background/appearance instead of the augmented views' actual motion
content, since two same-video clips in one batch look like an "easy"
negative (or, worse, share generic negatives in a way that leaks the
video's identity through appearance alone).

Unlike `datasets/supcon_dataset.GroupBalancedBatchSampler`, this needs no
labels — it only needs a `video_index` per sample (which video a clip came
from), so it works for the unlabeled SSL pipeline too, not just SupCon.

DDP: rank-sharding is done *inside* this sampler (interleaved shard of a
shuffled index order, mirroring `DistributedSampler`), so it can be passed
straight as `batch_sampler=` to a DataLoader under
`torch.distributed.run` without a separate `DistributedSampler` — the two
aren't composable in PyTorch anyway (`sampler` and `batch_sampler` are
mutually exclusive). Call `.set_epoch(epoch)` once per epoch so every rank
reshuffles in sync, exactly like `DistributedSampler.set_epoch`.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Sequence

from torch.utils.data import Sampler

from utils.distributed import get_rank, get_world_size


class VideoDiverseBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        video_index: Sequence[int],
        batch_size: int,
        max_per_video: int = 1,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.video_index = list(video_index)
        self.batch_size = batch_size
        self.max_per_video = max(1, max_per_video)
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch (before iterating) so every DDP rank reshuffles
        in sync — same contract as torch's DistributedSampler."""
        self.epoch = epoch

    def _local_shard(self) -> List[int]:
        rank, world_size = get_rank(), get_world_size()
        rng = random.Random(self.seed + self.epoch)
        n = len(self.video_index)
        order = list(range(n))
        rng.shuffle(order)
        per_rank = n // world_size
        order = order[: per_rank * world_size]  # even shards, mirrors DistributedSampler(drop_last=True)
        return order[rank::world_size]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch + 1)  # +1: distinct stream from the shard shuffle
        local = self._local_shard()

        buckets = defaultdict(list)
        for idx in local:
            buckets[self.video_index[idx]].append(idx)
        for v in buckets:
            rng.shuffle(buckets[v])

        while sum(len(b) for b in buckets.values()) >= (self.batch_size if self.drop_last else 1):
            batch: List[int] = []
            used_this_batch = defaultdict(int)
            active = [v for v in buckets if buckets[v]]
            progressed = True

            while len(batch) < self.batch_size and active and progressed:
                progressed = False
                rng.shuffle(active)
                for v in active:
                    if len(batch) >= self.batch_size:
                        break
                    if used_this_batch[v] >= self.max_per_video:
                        continue
                    room = min(
                        self.max_per_video - used_this_batch[v],
                        len(buckets[v]),
                        self.batch_size - len(batch),
                    )
                    if room > 0:
                        batch.extend(buckets[v][:room])
                        buckets[v] = buckets[v][room:]
                        used_this_batch[v] += room
                        progressed = True
                active = [v for v in active if buckets[v] and used_this_batch[v] < self.max_per_video]

            if not batch or (len(batch) < self.batch_size and self.drop_last):
                break
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        local_n = len(self._local_shard())
        # Upper bound: exact count depends on how evenly clips are spread
        # across videos, which varies per epoch — this is only used for
        # progress bars/logging, not for correctness.
        return local_n // self.batch_size if self.drop_last else -(-local_n // self.batch_size)