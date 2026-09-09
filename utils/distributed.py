"""
utils/distributed.py
======================
Small collection of DDP utilities. Every function here degrades gracefully
to a single-process no-op when `torch.distributed` isn't initialized, so the
exact same trainer code path runs whether you launch with
`python train.py` (1 GPU/CPU) or `torch.distributed.run --nproc_per_node=N`.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_ddp(backend: str = "nccl") -> None:
    """Expects to be launched via `torch.distributed.run` (sets RANK,
    WORLD_SIZE, LOCAL_RANK env vars). No-ops if those aren't present, so it's
    always safe to call this at the top of train.py."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    if not torch.cuda.is_available():
        backend = "gloo"  # nccl requires CUDA
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else None


def cleanup_ddp() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def convert_to_sync_bn(model: nn.Module) -> nn.Module:
    if is_dist_avail_and_initialized():
        return nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather a tensor from every rank and concat along dim 0. This is
    what makes the MoCo memory queue see keys produced by *every* GPU's
    key-encoder forward pass before `dequeue_and_enqueue()` runs, per the
    design doc's DDP guideline for the memory queue."""
    if not is_dist_avail_and_initialized():
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


@torch.no_grad()
def batch_shuffle_ddp(x: torch.Tensor):
    """MoCo's "shuffling BN": shuffle the key-encoder's mini-batch across all
    GPUs before the forward pass, so BatchNorm running-stats can't leak
    same-batch information between the query and key encoders (which would
    let the model cheat by matching on batch statistics instead of content).
    Single-GPU / non-distributed: returns the input unchanged with an
    identity unshuffle index.
    """
    if not is_dist_avail_and_initialized():
        idx_unshuffle = torch.arange(x.shape[0], device=x.device)
        return x, idx_unshuffle

    world_size = get_world_size()
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]
    batch_size_this = x.shape[0]

    idx_shuffle = torch.randperm(batch_size_all, device=x.device)
    dist.broadcast(idx_shuffle, src=0)
    idx_unshuffle = torch.argsort(idx_shuffle)

    gpu_idx = get_rank()
    idx_this = idx_shuffle.view(world_size, -1)[gpu_idx]
    return x_gather[idx_this], idx_unshuffle


@torch.no_grad()
def batch_unshuffle_ddp(x: torch.Tensor, idx_unshuffle: torch.Tensor) -> torch.Tensor:
    if not is_dist_avail_and_initialized():
        return x
    world_size = get_world_size()
    x_gather = concat_all_gather(x)
    gpu_idx = get_rank()
    idx_this = idx_unshuffle.view(world_size, -1)[gpu_idx]
    return x_gather[idx_this]
