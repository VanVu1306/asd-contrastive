"""
trainers/spi_trainer.py
=========================
Drives the SPI (Synthetic Periodicity Injection) pretext branch — a third,
independent SSL pipeline. Neither ssl_trainer.py nor supcon_trainer.py is
imported or modified by this file; it only builds on the same shared
BaseTrainer/model/loss infrastructure every trainer in this repo already
uses.

`spi.training_mode` in configs/spi_periodicity.yaml selects between three
ways to train the SPI task, all independently selectable and none of which
modifies another's code path:

  "in_batch" (default): backbone -> spi_projector -> SPIConLoss over
      (periodic_a, periodic_b, non_periodic) pseudo-labels, optionally +
      PeriodRegressionHead. Negatives = whatever's non-periodic in the
      current mini-batch only. This is the mode most thoroughly tested.

  "queue": the SAME periodic/non-periodic task, but through
      models.spi_moco_wrapper.SPIMoCoWrapper — a MoCo-style momentum
      encoder + persistent FIFO queue of non-periodic embeddings, so
      negatives accumulate across many past steps instead of being limited
      to the current batch. Added specifically to address a concern that
      in-batch-only negatives risk saturating/collapsing early on a small
      or low-diversity batch. `multi_task.moco_loss_weight` (the combined
      mode below) is ignored in this mode — it would mean training two
      independent MoCo-style objectives at once, which isn't what either
      mode is for.

  combined (`training_mode: "in_batch"` + `multi_task.moco_loss_weight > 0`):
      ONE shared backbone (via models.moco_wrapper.MoCoWrapper's
      `shared_backbone` parameter) feeds both SPIConLoss and a full,
      separate MoCoWrapper/InfoNCE objective; the two losses are
      weighted-summed and backpropagated together each step. This is a
      different mechanism from "queue" mode above: here MoCo runs as its
      own independent task (its own projector, its own queue) alongside
      SPIConLoss, rather than replacing SPIConLoss's own negative sampling.
      Runs without crashing (smoke-tested) but its joint-optimization
      dynamics haven't been validated the way "in_batch" has.

See the README's SPI section for what "tested" means for each mode.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.samplers import VideoDiverseBatchSampler
from datasets.spi_dataset import SPIDataset
from losses.moco_nce_loss import MoCoNCELoss
from losses.spi_moco_loss import SPIMoCoLoss
from losses.spicon_loss import SPIConLoss
from models.backbones.builder import build_backbone
from models.heads import PeriodRegressionHead, ProjectionHead
from models.moco_wrapper import MoCoWrapper
from models.spi_moco_wrapper import SPIMoCoWrapper
from trainers.base_trainer import BaseTrainer
from utils.distributed import is_main_process
from utils.logger import AverageMeter
from utils.seed import worker_init_fn


class SPIEncoder(nn.Module):
    """backbone + ProjectionHead (+ optional PeriodRegressionHead), trained
    end-to-end — the SPI-branch analogue of trainers/supcon_trainer.
    SupConEncoder. In combined mode, `backbone` is the same instance also
    wrapped by a MoCoWrapper elsewhere in this trainer (see
    build_model_and_loss) — PyTorch's own parameter deduplication makes
    that safe to hand to a single optimizer with no extra bookkeeping."""

    def __init__(
        self, backbone_cfg, feature_dim: int, hidden_dim: int,
        period_head_hidden_dim: int = 128, predict_period: bool = False,
        backbone: nn.Module = None,
    ):
        super().__init__()
        self.backbone = backbone if backbone is not None else build_backbone(backbone_cfg)
        self.projector = ProjectionHead(self.backbone.out_dim, hidden_dim, feature_dim)
        self.period_head = (
            PeriodRegressionHead(self.backbone.out_dim, period_head_hidden_dim) if predict_period else None
        )

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        z = self.projector(feats)
        log_period = self.period_head(feats) if self.period_head is not None else None
        return z, log_period


class SPITrainer(BaseTrainer):
    def build_dataloader(self):
        cfg = self.cfg
        dataset = SPIDataset(
            root=cfg["data"]["root"],
            split_file=cfg["data"]["train_split"],
            clip_len=cfg["data"]["clip_len"],
            raw_clip_len=cfg["data"]["raw_clip_len"],
            frame_size=cfg["data"]["frame_size"],
            frame_source=cfg["data"].get("frame_source", "auto"),
            fps=cfg["data"].get("fps", 30.0),
            cycle_duration_range_sec=cfg["spi"]["cycle_duration_range_sec"],
            n_repeats_range=cfg["spi"]["n_repeats_range"],
            speed_jitter=cfg["spi"].get("speed_jitter", 0.05),
            color_jitter_strength=cfg["spi"].get("color_jitter_strength", 0.1),
            random_crop_scale=cfg["spatial_transform"]["random_crop_scale"],
            color_jitter=cfg["spatial_transform"].get("color_jitter", 0.4),
            h_flip_prob=cfg["spatial_transform"].get("h_flip_prob", 0.5),
            random_erasing_prob=cfg["spatial_transform"].get("random_erasing_prob", 0.0),
            random_erasing_scale=cfg["spatial_transform"].get("random_erasing_scale", (0.02, 0.15)),
            clips_per_video=cfg["data"].get("clips_per_video", 1),
            fixed_L=cfg["spi"].get("fixed_L"),
            fixed_N=cfg["spi"].get("fixed_N"),
            stats_path=cfg["spi"].get("stats_path"),
        )
        batch_size = cfg["optim"]["batch_size"]
        max_per_video = cfg["data"].get("max_per_video_per_batch") or batch_size
        batch_sampler = VideoDiverseBatchSampler(
            video_index=dataset.video_index, batch_size=batch_size,
            max_per_video=max_per_video, drop_last=True, seed=self.seed,
        )
        loader = DataLoader(
            dataset, batch_sampler=batch_sampler,
            num_workers=cfg["data"].get("num_workers", 4),
            pin_memory=cfg["data"].get("pin_memory", True),
            worker_init_fn=worker_init_fn,
        )
        return loader, batch_sampler

    def build_model_and_loss(self):
        cfg = self.cfg
        training_mode = cfg["spi"].get("training_mode", "in_batch")
        predict_period = cfg["spi"].get("period_loss_weight", 0.0) > 0

        if training_mode == "queue":
            # New: MoCo-style persistent queue for this task's own negatives
            # (see models/spi_moco_wrapper.py). Independent of, and never
            # combined with, multi_task.moco_loss_weight's separate
            # combined-mode mechanism below.
            spi_moco = SPIMoCoWrapper(
                backbone_cfg=cfg["backbone"], feature_dim=cfg["spi"]["feature_dim"],
                hidden_dim=cfg["spi"]["hidden_dim"], queue_size=cfg["spi"].get("queue_size", 2048),
                momentum=cfg["spi"].get("queue_momentum", 0.999),
            )
            modules = {"spi_moco": spi_moco}
            if predict_period:
                # Standalone head (not inside SPIMoCoWrapper) consuming
                # encoder_q's exposed backbone features — see
                # SPIMoCoWrapper.forward's feats_a return value.
                modules["period_head"] = PeriodRegressionHead(
                    spi_moco.encoder_q.backbone.out_dim, cfg["spi"].get("period_head_hidden_dim", 128),
                )
            model = self.wrap_for_ddp(nn.ModuleDict(modules))
            criterion = SPIMoCoLoss(
                temperature=cfg["spi"]["temperature"], period_loss_weight=cfg["spi"].get("period_loss_weight", 0.0),
            )
            return model, criterion

        # --- training_mode == "in_batch" (default) ---
        moco_weight = cfg.get("multi_task", {}).get("moco_loss_weight", 0.0)

        # Combined mode builds the backbone once, up front, and hands the
        # SAME instance to both SPIEncoder and MoCoWrapper. Standalone mode
        # (the default) never constructs a second model at all.
        shared_backbone = build_backbone(cfg["backbone"]) if moco_weight > 0 else None
        spi_encoder = SPIEncoder(
            backbone_cfg=cfg["backbone"], feature_dim=cfg["spi"]["feature_dim"],
            hidden_dim=cfg["spi"]["hidden_dim"],
            period_head_hidden_dim=cfg["spi"].get("period_head_hidden_dim", 128),
            predict_period=predict_period, backbone=shared_backbone,
        )

        modules = {"spi": spi_encoder}
        moco_wrapper = None
        if moco_weight > 0:
            moco_wrapper = MoCoWrapper(
                backbone_cfg=cfg["backbone"], feature_dim=cfg["moco"]["feature_dim"],
                hidden_dim=cfg["moco"]["hidden_dim"], queue_size=cfg["moco"]["queue_size"],
                momentum=cfg["moco"]["momentum"], shared_backbone=spi_encoder.backbone,
            )
            modules["moco"] = moco_wrapper

        model = self.wrap_for_ddp(nn.ModuleDict(modules))

        spicon_criterion = SPIConLoss(
            temperature=cfg["spi"]["temperature"], period_loss_weight=cfg["spi"].get("period_loss_weight", 0.0),
        )
        moco_criterion = MoCoNCELoss(temperature=cfg["moco"]["temperature"]) if moco_wrapper is not None else None
        return model, (spicon_criterion, moco_criterion)

    def train_step(self, model, criterion, batch):
        cfg = self.cfg
        training_mode = cfg["spi"].get("training_mode", "in_batch")
        x_periodic_a, x_periodic_b, x_nonperiodic, period_label = (
            t.to(self.device, non_blocking=True) for t in batch
        )
        m = self.unwrap(model)

        if training_mode == "queue":
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                q, k, queue, k_neg, feats_a = m["spi_moco"](x_periodic_a, x_periodic_b, x_nonperiodic)
        
                pred_log_period, true_log_period = None, None
                if "period_head" in m:
                    pred_log_period = m["period_head"](feats_a)
                    true_log_period = period_label
        
                loss, logs = criterion(q, k, queue, k_neg, pred_log_period, true_log_period)
        
            self._last_logs = logs
            return loss

        # --- training_mode == "in_batch" (default) ---
        spicon_criterion, moco_criterion = criterion
        with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
            z_a, log_p_a = m["spi"](x_periodic_a)
            z_b, log_p_b = m["spi"](x_periodic_b)
            z_np, _ = m["spi"](x_nonperiodic)

            features = torch.cat([z_a, z_b, z_np], dim=0)
            batch_size = x_periodic_a.shape[0]
            pseudo_labels = torch.cat([
                torch.ones(2 * batch_size, dtype=torch.long, device=self.device),
                torch.zeros(batch_size, dtype=torch.long, device=self.device),
            ])

            pred_log_period, true_log_period = None, None
            if log_p_a is not None:
                pred_log_period = torch.cat([log_p_a, log_p_b], dim=0)
                true_log_period = torch.cat([period_label, period_label], dim=0)

            spi_loss, spi_logs = spicon_criterion(features, pseudo_labels, pred_log_period, true_log_period)
            loss = cfg.get("multi_task", {}).get("spi_loss_weight", 1.0) * spi_loss

            moco_logs = {}
            if "moco" in m:
                # Combined mode: an independent InfoNCE signal through the
                # SAME shared backbone, on top of (not instead of) SPIConLoss.
                # x_periodic_a/_b/_nonperiodic double as MoCo's
                # anchor/positive/hard-negative views for this extra step —
                # they're already three augmented views of one raw window,
                # exactly the shape MoCoWrapper.forward expects.
                q, k, queue, k_neg = m["moco"](x_periodic_a, x_periodic_b, x_nonperiodic)
                moco_loss = moco_criterion(q, k, queue, k_neg)
                moco_weight = cfg.get("multi_task", {}).get("moco_loss_weight", 0.0)
                loss = loss + moco_weight * moco_loss
                moco_logs = {"moco_infonce": moco_loss.item()}

        self._last_logs = {**spi_logs, **moco_logs}
        return loss

    def train(self):
        cfg = self.cfg
        loader, batch_sampler = self.build_dataloader()
        model, criterion = self.build_model_and_loss()
        optimizer = self.build_optimizer(model.parameters())
        self.maybe_resume(model, optimizer)
        scheduler = self.make_scheduler(optimizer, steps_per_epoch=len(loader))

        epochs = cfg["optim"]["epochs"]
        log_every = cfg["logging"].get("log_every", 20)
        ckpt_every = cfg["logging"].get("ckpt_every", 5)

        for epoch in range(self.start_epoch, epochs):
            batch_sampler.set_epoch(epoch)
            model.train()
            loss_meter = AverageMeter()
            t0 = time.time()

            for step, batch in enumerate(loader):
                loss = self.train_step(model, criterion, batch)
                self.run_optimizer_step(optimizer, loss, scheduler)
                loss_meter.update(loss.item())

                if step % log_every == 0 and is_main_process():
                    metrics = {"loss": loss_meter.avg, "lr": optimizer.param_groups[0]["lr"]}
                    metrics.update(getattr(self, "_last_logs", {}))
                    self.logger.log(metrics, step=self.global_step)

            if is_main_process():
                print(f"[spi] epoch {epoch} done in {time.time() - t0:.1f}s, avg_loss={loss_meter.avg:.4f}")
            if (epoch + 1) % ckpt_every == 0 or epoch == epochs - 1:
                self.save_checkpoint(model, optimizer, epoch)

        self.logger.close()
