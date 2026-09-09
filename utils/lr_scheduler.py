"""
utils/lr_scheduler.py
=======================
Per-step (not per-epoch) LR schedules. Contrastive-learning setups
(MoCo/SupCon alike) consistently converge better with a linear warmup
before cosine decay than with a bare step schedule, so that's the default
(`optim.scheduler: cosine_warmup` in the configs) — plain cosine and StepLR
are kept available for ablations.

All three are plain callables: `scheduler(step) -> lr`. `build_scheduler`
returns one of them pre-configured from `optim:` config values, and the
trainer calls it every optimizer step (see trainers/base_trainer.py).
"""
from __future__ import annotations

import math


class WarmupCosineSchedule:
    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 0.0):
        self.base_lr = base_lr
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.min_lr = min_lr

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine


class CosineSchedule:
    def __init__(self, base_lr: float, total_steps: int, min_lr: float = 0.0):
        self.base_lr = base_lr
        self.total_steps = max(1, total_steps)
        self.min_lr = min_lr

    def __call__(self, step: int) -> float:
        progress = min(max(step / self.total_steps, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine


class StepSchedule:
    def __init__(self, base_lr: float, steps_per_epoch: int, drop_epochs=(120, 160), gamma: float = 0.1):
        self.base_lr = base_lr
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.drop_steps = sorted(e * steps_per_epoch for e in drop_epochs)
        self.gamma = gamma

    def __call__(self, step: int) -> float:
        drops = sum(1 for s in self.drop_steps if step >= s)
        return self.base_lr * (self.gamma ** drops)


def build_scheduler(optim_cfg, steps_per_epoch: int):
    name = optim_cfg.get("scheduler", "cosine_warmup")
    base_lr = optim_cfg["lr"]
    total_steps = optim_cfg["epochs"] * steps_per_epoch
    warmup_steps = optim_cfg.get("warmup_epochs", 0) * steps_per_epoch

    if name == "cosine_warmup":
        return WarmupCosineSchedule(base_lr, warmup_steps, total_steps)
    if name == "cosine":
        return CosineSchedule(base_lr, total_steps)
    if name == "step":
        return StepSchedule(base_lr, steps_per_epoch)
    raise ValueError(f"Unknown scheduler '{name}'")


def set_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
