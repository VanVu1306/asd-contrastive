"""
losses/supcon_loss.py
=======================
Supervised Contrastive Loss (Khosla et al., 2020), extended with an optional
`group_ids` argument so a positive pair must come from two different
video_ids — the design doc's explicit requirement for cancelling out
subject/context bias (see datasets/supcon_dataset.py module docstring).

For anchor i, positives P(i) = {p != i : label[p] == label[i] AND (group_ids
is None OR group_ids[p] != group_ids[i])}, and the loss is the standard
multi-positive InfoNCE-style objective averaged over P(i):

    L_i = -1/|P(i)| * sum_{p in P(i)} log( exp(z_i . z_p / tau)
                                            / sum_{a != i} exp(z_i . z_a / tau) )

Anchors that end up with an empty P(i) (e.g. no cross-video same-label
sample in this particular batch) are skipped for that step rather than
forced into a divide-by-zero — this is why
datasets/supcon_dataset.GroupBalancedBatchSampler exists, to keep that from
happening often.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,          # (B, D), expected L2-normalized
        labels: torch.Tensor,             # (B,)
        group_ids: Optional[torch.Tensor] = None,  # (B,) e.g. video_id ints
    ) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        labels = labels.view(-1, 1)
        same_label = torch.eq(labels, labels.t())

        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        pos_mask = same_label & ~self_mask

        if group_ids is not None:
            group_ids = group_ids.view(-1, 1)
            same_group = torch.eq(group_ids, group_ids.t())
            pos_mask = pos_mask & ~same_group  # "2 clips from 2 different video IDs"

        sim = torch.matmul(features, features.t()) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # numerical stability only
        exp_sim = torch.exp(sim) * (~self_mask)  # never let a sample be its own denominator term
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_counts = pos_mask.sum(dim=1)
        has_positives = pos_counts > 0
        if not has_positives.any():
            # Degenerate batch (no cross-video same-label pair at all) — return
            # a zero loss rather than NaN so a single bad batch doesn't derail
            # training; this should be rare once GroupBalancedBatchSampler is used.
            return features.new_tensor(0.0, requires_grad=True)

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[has_positives] / pos_counts[has_positives]
        return -mean_log_prob_pos.mean()
