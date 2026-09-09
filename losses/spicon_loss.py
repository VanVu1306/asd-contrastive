"""
losses/spicon_loss.py
=======================
Loss for the SPI (Synthetic Periodicity Injection) pretext branch — see
datasets/spi_dataset.py for how the pseudo-labels and features feeding this
are built.

Per the design doc, this reuses `losses/supcon_loss.SupConLoss`'s existing
label-grouping mechanism, just with a different label *source*: pseudo-label
1 = periodic (synthesized), 0 = non-periodic (plain raw clip) — instead of
real ASD/non-ASD labels.

One deliberate difference from how SupConLoss is used for the real ASD
labels: no `group_ids` (cross-video exclusion) here. For the real SupCon
task, two same-label clips sharing a video is a *bias* to cancel out. Here,
the two periodic views of one synthesized clip (x_periodic_phaseA/_phaseB)
sharing a source video is *exactly the intended signal* — the whole point of
the phase-invariance pair is "same repeating pattern, different phase", and
the mandatory per-repeat jitter in SyntheticPeriodicityInjection is what
already guards against a trivial pixel-matching shortcut, so the group_ids
exclusion mechanism isn't needed (or wanted) for this task.

An optional auxiliary term regresses log(period length) via
models/heads.PeriodRegressionHead, so the embedding space encodes cycle
*length*, not just a periodic/non-periodic binary.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from losses.supcon_loss import SupConLoss


class SPIConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, period_loss_weight: float = 0.0):
        super().__init__()
        self.supcon = SupConLoss(temperature=temperature)
        self.period_loss_weight = period_loss_weight
        self.period_criterion = nn.SmoothL1Loss()  # robust to the occasional very long/short cycle

    def forward(
        self,
        features: torch.Tensor,               # (B, D), L2-normalized, periodic+non-periodic mixed in one batch
        pseudo_labels: torch.Tensor,            # (B,) in {0, 1} — 1 = periodic, 0 = non-periodic
        pred_log_period: Optional[torch.Tensor] = None,  # (B_periodic,) — PeriodRegressionHead output, periodic samples only
        true_log_period: Optional[torch.Tensor] = None,  # (B_periodic,) — log(L) ground truth for the same samples
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        contrastive = self.supcon(features, pseudo_labels)  # no group_ids — see module docstring
        total = contrastive
        logs = {"spicon_contrastive": contrastive.item(), "spicon_period_reg": 0.0}

        if self.period_loss_weight > 0 and pred_log_period is not None and true_log_period is not None:
            period_loss = self.period_criterion(pred_log_period, true_log_period)
            total = total + self.period_loss_weight * period_loss
            logs["spicon_period_reg"] = period_loss.item()

        return total, logs
