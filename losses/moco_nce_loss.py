"""
losses/moco_nce_loss.py
=========================
InfoNCE loss for MoCo. Given:
    q      : (B, D) query embeddings
    k      : (B, D) positive key embeddings (q[i] pairs with k[i])
    queue  : (D, Q) bank of negative keys, shared across the batch
    k_neg  : optional (B, D) per-sample hard negative (frame-shuffled view)

logits are [pos | queue_negatives | hard_negative] per row, scaled by
1/temperature, and the label for every row is always index 0 (the positive
column) — this is exactly cross-entropy over a (1 + Q [+ 1]) - way
classification problem, which is the standard way to implement InfoNCE.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        queue: torch.Tensor,
        k_neg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        l_pos = torch.einsum("bd,bd->b", [q, k]).unsqueeze(-1)      # (B, 1)
        l_queue = torch.einsum("bd,dk->bk", [q, queue])              # (B, Q)

        logits = [l_pos, l_queue]
        if k_neg is not None:
            l_hard = torch.einsum("bd,bd->b", [q, k_neg]).unsqueeze(-1)  # (B, 1)
            logits.append(l_hard)

        logits = torch.cat(logits, dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)
