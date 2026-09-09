"""
trainers/base_trainer.py
==========================
Generic trainer setup shared by ssl_trainer.py and supcon_trainer.py:
device/DDP initialization, optimizer construction, checkpointing, and
logging. Neither subclass duplicates any of this — they only implement
`build_dataloader`, `build_model_and_loss`, and `train_step`.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn

from utils.distributed import (
    convert_to_sync_bn,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    setup_ddp,
)
from utils.logger import MetricLogger
from utils.lr_scheduler import build_scheduler, set_lr
from utils.seed import set_seed

class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        setup_ddp()
        # self.seed: the *base* seed, identical across every DDP rank — the
        # samplers (VideoDiverseBatchSampler, GroupBalancedBatchSampler) need
        # every rank to compute the exact same shuffle before slicing off
        # their own shard, so they must never see a rank-offset seed.
        # The global RNGs (random/numpy/torch), by contrast, deliberately
        # DO get a rank-offset seed here — augmentation randomness should
        # differ across ranks (there's no shared-shuffle-then-slice
        # requirement for it), and letting every rank draw identical
        # augmentations from cloned RNG state would just waste diversity.
        self.seed = cfg.get("seed", 42)
        deterministic = cfg.get("hardware", {}).get("deterministic", False)
        set_seed(self.seed + get_rank(), deterministic=deterministic)

        want_cuda = cfg.get("hardware", {}).get("device", "cuda") == "cuda"
        self.device = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")
        if want_cuda and not torch.cuda.is_available() and is_main_process():
            print("[trainer] CUDA requested but unavailable — falling back to CPU.")

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.output_dir = os.path.join(cfg.get("output_dir", "./runs"), cfg.get("experiment_name", "run"))
        if is_main_process():
            os.makedirs(self.output_dir, exist_ok=True)

        log_cfg = cfg.get("logging", {})
        self.logger = MetricLogger(
            log_dir=self.output_dir,
            backends=log_cfg.get("backend", ["console"]),
            wandb_project=log_cfg.get("wandb_project"),
            run_name=cfg.get("experiment_name"),
        )

        self.amp_enabled = cfg.get("hardware", {}).get("amp", False) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        self.global_step = 0
        self.start_epoch = 0
        self.resume_path = None  # set from train.py's --resume before .train() runs

    # ---- to be implemented by subclasses -----------------------------------
    def build_dataloader(self):
        raise NotImplementedError

    def build_model_and_loss(self):
        raise NotImplementedError

    def train_step(self, model, criterion, batch) -> torch.Tensor:
        """Runs the forward pass + loss computation for one batch and
        returns the scalar loss tensor (still attached to the graph — the
        trainer's main loop calls .backward() on it, so AMP scaling wraps
        cleanly around every subclass without duplicating that logic)."""
        raise NotImplementedError

    # ---- shared plumbing ----------------------------------------------------
    def wrap_for_ddp(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        if self.cfg.get("hardware", {}).get("sync_bn", True):
            model = convert_to_sync_bn(model)
        if is_dist_avail_and_initialized():
            device_ids = [self.local_rank] if self.device.type == "cuda" else None
            model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
        return model

    def build_optimizer(self, params) -> torch.optim.Optimizer:
        optim_cfg = self.cfg["optim"]
        name = optim_cfg.get("name", "sgd").lower()
        params = [p for p in params if p.requires_grad]
        if name == "sgd":
            return torch.optim.SGD(
                params, lr=optim_cfg["lr"], momentum=optim_cfg.get("momentum", 0.9),
                weight_decay=optim_cfg.get("weight_decay", 0.0),
            )
        if name in ("adam", "adamw"):
            cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
            return cls(params, lr=optim_cfg["lr"], weight_decay=optim_cfg.get("weight_decay", 0.0))
        raise ValueError(f"Unknown optimizer '{name}'")

    def unwrap(self, model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, extra: dict = None) -> None:
        if not is_main_process():
            return
        state = {
            "model": self.unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "config": dict(self.cfg),
            "rng_state": dict(self.cfg),
        }
        if extra:
            state.update(extra)
        torch.save(state, os.path.join(self.output_dir, "ckpt_last.pth"))
        torch.save(state, os.path.join(self.output_dir, f"ckpt_epoch{epoch}.pth"))

    def load_checkpoint(self, path: str, model: nn.Module, optimizer: torch.optim.Optimizer = None) -> int:
        state = torch.load(path, map_location=self.device)
        self.unwrap(model).load_state_dict(state["model"])
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        self.global_step = state.get("global_step", 0)
        return state.get("epoch", 0)

    def run_optimizer_step(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor, scheduler=None) -> None:
        if scheduler is not None:
            set_lr(optimizer, scheduler(self.global_step))
        optimizer.zero_grad(set_to_none=True)
        if self.amp_enabled:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
        self.global_step += 1

    def make_scheduler(self, optimizer, steps_per_epoch: int):
        return build_scheduler(self.cfg["optim"], steps_per_epoch)

    def maybe_resume(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Called by each subclass's train() right after model/optimizer are
        built, so a resumed run continues training the *same* model instance
        instead of a throwaway one built just to read a checkpoint."""
        if self.resume_path:
            self.start_epoch = self.load_checkpoint(self.resume_path, model, optimizer) + 1
            if is_main_process():
                print(f"[trainer] resumed from {self.resume_path} at epoch {self.start_epoch}")