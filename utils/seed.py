"""
utils/seed.py
===============
Single source of truth for reproducibility. Three distinct gaps existed
before this module and are each addressed here:

1. Global seed (random/numpy/torch) — was already set once in
   trainers/base_trainer.py, but eval.py never called it at all, so every
   eval.py run (centroid mining's random crops, linear-probe's augmented
   training, DataLoader shuffling) was fully unseeded.

2. DataLoader worker processes — PyTorch's default worker seeding only
   reseeds *torch's* RNG per worker; it does NOT reseed Python's `random`
   module or numpy's global RNG. Since almost every augmentation in
   transforms/ uses `random.*` or `np.random.*` (not torch's RNG), training
   with `num_workers > 0` — the actual default (data.num_workers: 8) — was
   silently non-reproducible even with a fixed top-level seed. `worker_init_fn`
   below fixes this.

3. cudnn determinism — cudnn's autotuner can pick different algorithms
   (and cudnn convolution algorithms themselves can be non-deterministic)
   across runs even with every RNG seeded identically. Off by default here
   (it costs real throughput) but available via hardware.deterministic.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seeds Python's random, numpy, and torch (CPU + all CUDA devices).
    Call this once, as early as possible (before any dataset/model
    construction) — both train.py's trainers and eval.py's main() do this.

    `deterministic=True` additionally forces cudnn onto deterministic
    (but slower) algorithms — needed for bit-exact repeatability across
    runs on GPU, not needed just to get *a* fixed, reasonable seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # When not requested, deliberately leave cudnn's flags alone (don't
    # force benchmark=True either) — respect whatever the environment/user
    # already had set, rather than silently changing throughput behavior.


def worker_init_fn(worker_id: int) -> None:
    """Pass this to every DataLoader(..., worker_init_fn=worker_init_fn).

    Derives a per-worker seed from torch's own per-worker base seed
    (`torch.initial_seed()`, which PyTorch already varies per worker per
    epoch) and uses it to also seed Python's `random` and numpy — the two
    RNGs PyTorch's own default worker seeding does NOT touch, and the two
    almost every transform in this repo actually uses.
    """
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
