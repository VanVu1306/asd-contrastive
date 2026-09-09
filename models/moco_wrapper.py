"""
models/moco_wrapper.py
========================
MoCo (Momentum Contrast) key-encoder + memory-queue manager, wired up to
whatever 3D backbone `configs/*.yaml: backbone.name` selects.

Two encoders share the same architecture (backbone + ProjectionHead):
    encoder_q — updated by backpropagation every step.
    encoder_k — a momentum (EMA) copy of encoder_q, never receives gradients
                directly; see `_momentum_update_key_encoder`.

The FIFO memory queue (`self.queue`) holds the last `queue_size` key vectors
across steps, giving InfoNCE a large, consistent set of negatives without
needing a huge batch size — that's the entire point of MoCo over plain
end-to-end contrastive learning. See losses/moco_nce_loss.py for how
(q, k, queue) become one InfoNCE loss.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.builder import build_backbone
from models.heads import ProjectionHead
from utils.distributed import batch_shuffle_ddp, batch_unshuffle_ddp, concat_all_gather


class _Encoder(nn.Module):
    """backbone -> pooled features -> ProjectionHead -> L2-normalized embedding."""

    def __init__(self, backbone_cfg=None, feature_dim: int = 256, hidden_dim: int = 512, backbone: nn.Module = None):
        super().__init__()
        self.backbone = backbone if backbone is not None else build_backbone(backbone_cfg)
        self.projector = ProjectionHead(self.backbone.out_dim, hidden_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(x))


class MoCoWrapper(nn.Module):
    def __init__(
        self,
        backbone_cfg,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        queue_size: int = 8192,
        momentum: float = 0.999,
        shared_backbone: nn.Module = None,
    ):
        super().__init__()
        self.momentum = momentum
        self.queue_size = queue_size

        self.encoder_q = _Encoder(backbone_cfg, feature_dim, hidden_dim, backbone=shared_backbone)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False  # key encoder only ever moves via EMA

        # FIFO queue of negative keys, L2-normalized, plus a write pointer.
        self.register_buffer("queue", F.normalize(torch.randn(feature_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.mul_(self.momentum).add_(p_q.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        keys = concat_all_gather(keys)  # gather keys from every GPU first (DDP guideline)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr : ptr + batch_size] = keys.t()
        else:
            # wrap around the ring buffer
            tail = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:tail].t()
            self.queue[:, : batch_size - tail] = keys[tail:].t()
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, x_anchor: torch.Tensor, x_pos_warp: torch.Tensor, x_neg_shuffle: torch.Tensor = None):
        """x_anchor plays MoCo's "query" role, x_pos_warp plays "key" (its
        temporally-warped view of the same clip, i.e. the one positive).

        x_neg_shuffle (the frame-shuffled view from datasets/ssl_dataset.py)
        is *not* pushed into the FIFO queue — it's specific to this sample,
        not a generic negative worth keeping around for future batches.
        Instead it's returned as an explicit "hard negative" embedding,
        one per sample, that losses/moco_nce_loss.py appends next to the
        queue's negatives in the InfoNCE denominator: because it shares
        every frame with the anchor and differs only in temporal order, it's
        a much harder negative than a random queue entry, which is exactly
        what forces the encoder to represent motion/order and not just
        appearance.

        Returns (q, k, queue.clone(), k_neg) — k_neg is None if
        x_neg_shuffle wasn't provided (e.g. a caller only wants the plain
        MoCo positive/queue-negative logits).
        """
        q = self.encoder_q(x_anchor)
        q = F.normalize(q, dim=-1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            x_shuffled, idx_unshuffle = batch_shuffle_ddp(x_pos_warp)
            k = self.encoder_k(x_shuffled)
            k = F.normalize(k, dim=-1)
            k = batch_unshuffle_ddp(k, idx_unshuffle)

            k_neg = None
            if x_neg_shuffle is not None:
                # Same key encoder, encoded independently — the shuffling-BN
                # trick isn't needed here since a hard negative doesn't risk
                # being mistaken for its own positive, only queue negatives
                # need not to be picked from the same in-batch statistics.
                k_neg = F.normalize(self.encoder_k(x_neg_shuffle), dim=-1)

        queue = self.queue.clone().detach()
        self._dequeue_and_enqueue(k)
        return q, k, queue, k_neg
