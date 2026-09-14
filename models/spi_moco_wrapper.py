"""
models/spi_moco_wrapper.py
=============================
MoCo-style memory-queue wrapper for the SPI (Synthetic Periodicity
Injection) branch's OWN periodic/non-periodic contrastive task — an
alternative to SPIConLoss's in-batch-only negatives (see
losses/spicon_loss.py), added on request: a persistent FIFO queue of
non-periodic embeddings gives many more, and progressively harder,
negatives than whatever happens to be non-periodic in the current
mini-batch, reducing the risk of the task saturating early on a
small/uniform batch.

Positive pair : (x_periodic_phaseA, x_periodic_phaseB) — same synthesized
                 clip, different phase, exactly as in the in-batch mode.
Negatives     : x_nonperiodic (this step's own non-periodic view, an
                 explicit hard negative, mirroring MoCoWrapper's own
                 k_neg treatment) + the FIFO queue's accumulated
                 non-periodic embeddings from many past steps.

Deliberately does NOT push periodic embeddings into the queue: the queue's
entire job here is to be a large, diverse pool of the *negative* class
(non-periodic) specifically, since that's what the design doc's suggestion
was trying to enrich — the positive side (phase-invariance) doesn't need a
queue, it's already anchored to one specific synthesized clip per step.

losses/moco_nce_loss.MoCoNCELoss is reused as-is for the (q, k, queue,
k_neg) -> InfoNCE part — its interface is already generic enough that
nothing needed to change there.
"""
from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.builder import build_backbone
from models.heads import ProjectionHead
from utils.distributed import batch_shuffle_ddp, batch_unshuffle_ddp, concat_all_gather


class _QueueEncoder(nn.Module):
    """backbone -> pooled features -> ProjectionHead -> L2-normalized
    embedding, exposing the pre-projection pooled features too — needed by
    SPITrainer's optional period-regression head, which predicts from raw
    backbone features, not the contrastive embedding."""

    def __init__(self, backbone_cfg, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = build_backbone(backbone_cfg)
        self.projector = ProjectionHead(self.backbone.out_dim, hidden_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        z = F.normalize(self.projector(feats), dim=-1)
        return z, feats


class SPIMoCoWrapper(nn.Module):
    def __init__(
        self,
        backbone_cfg,
        feature_dim: int = 128,
        hidden_dim: int = 512,
        queue_size: int = 2048,
        momentum: float = 0.999,
    ):
        super().__init__()
        self.momentum = momentum
        self.queue_size = queue_size

        self.encoder_q = _QueueEncoder(backbone_cfg, feature_dim, hidden_dim)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False  # key encoder only ever moves via EMA

        # FIFO queue of non-periodic ("negative class") embeddings.
        self.register_buffer("queue", F.normalize(torch.randn(feature_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.mul_(self.momentum).add_(p_q.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr : ptr + batch_size] = keys.t()
        else:
            tail = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:tail].t()
            self.queue[:, : batch_size - tail] = keys[tail:].t()
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, x_periodic_a: torch.Tensor, x_periodic_b: torch.Tensor, x_nonperiodic: torch.Tensor):
        """Returns (q, k, queue.clone(), k_neg, feats_a).

        feats_a: encoder_q's pooled backbone features for phaseA (carries
        gradient) — for the optional period-regression head. Only phaseA is
        exposed (not also phaseB) to avoid a second full backbone forward
        pass for something the period target doesn't distinguish by phase
        anyway (L is the same regardless of which phase you're looking at).
        """
        q, feats_a = self.encoder_q(x_periodic_a)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            x_shuffled, idx_unshuffle = batch_shuffle_ddp(x_periodic_b)
            k, _ = self.encoder_k(x_shuffled)
            k = batch_unshuffle_ddp(k, idx_unshuffle)

            # This step's own non-periodic view, encoded via encoder_k for
            # consistency with how MoCoWrapper treats its own hard negative
            # (never a target to pull toward, so no need for it to flow
            # through the query encoder / receive gradient).
            k_neg, _ = self.encoder_k(x_nonperiodic)

        queue = self.queue.clone().detach()
        self._dequeue_and_enqueue(k_neg)  # today's non-periodic embeddings become tomorrow's negatives too
        return q, k, queue, k_neg, feats_a