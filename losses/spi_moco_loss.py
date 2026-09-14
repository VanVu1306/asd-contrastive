"""
losses/spi_moco_loss.py
==========================
Loss for the SPI branch's queue-based training mode
(`spi.training_mode: "queue"` in configs/spi_periodicity.yaml). Wraps
losses/moco_nce_loss.MoCoNCELoss (reused as-is, unmodified) around the
periodic/non-periodic MoCo-style forward pass from
models/spi_moco_wrapper.SPIMoCoWrapper, plus the same optional
period-regression auxiliary term SPIConLoss has.

This mirrors SPIConLoss's shape (forward returns (total_loss, logs_dict))
for consistency, but is its own separate class rather than a shared one —
SPIConLoss is tightly coupled to its own in-batch SupCon path, and forcing
the two training modes through one class would make both harder to read
for a smaller benefit than just duplicating the ~5-line period-regression
block.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from losses.moco_nce_loss import MoCoNCELoss


class SPIMoCoLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, period_loss_weight: float = 0.0):
        super().__init__()
        self.moco_nce = MoCoNCELoss(temperature=temperature)
        self.period_loss_weight = period_loss_weight
        self.period_criterion = nn.SmoothL1Loss()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        queue: torch.Tensor,
        k_neg: torch.Tensor,
        pred_log_period: Optional[torch.Tensor] = None,
        true_log_period: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        contrastive = self.moco_nce(q, k, queue, k_neg)
        total = contrastive
        logs = {"spimoco_contrastive": contrastive.item(), "spimoco_period_reg": 0.0}

        if self.period_loss_weight > 0 and pred_log_period is not None and true_log_period is not None:
            period_loss = self.period_criterion(pred_log_period, true_log_period)
            total = total + self.period_loss_weight * period_loss
            logs["spimoco_period_reg"] = period_loss.item()

        return total, logs