"""
trainers/supcon_trainer.py
=============================
Drives Supervised Contrastive pretraining: a single backbone+ProjectionHead
(no momentum encoder — SupCon, unlike MoCo, contrasts within one batch, no
queue needed) trained with SupConLoss over the GroupBalancedBatchSampler's
video-id-diverse batches.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.supcon_dataset import GroupBalancedBatchSampler, SupConDataset
from losses.supcon_loss import SupConLoss
from models.backbones.builder import build_backbone
from models.heads import ProjectionHead
from trainers.base_trainer import BaseTrainer
from utils.distributed import is_main_process
from utils.logger import AverageMeter
from utils.seed import worker_init_fn


class SupConEncoder(nn.Module):
    """backbone + ProjectionHead, trained end-to-end (unlike MoCo, there is
    no frozen/momentum copy — every parameter here gets gradients)."""

    def __init__(self, backbone_cfg, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = build_backbone(backbone_cfg)
        self.projector = ProjectionHead(self.backbone.out_dim, hidden_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(x))


class SupConTrainer(BaseTrainer):
    def build_dataloader(self):
        cfg = self.cfg
        dataset = SupConDataset(
            root=cfg["data"]["root"],
            split_file=cfg["data"]["train_split"],
            clip_len=cfg["data"]["clip_len"],
            frame_size=cfg["data"]["frame_size"],
            frame_source=cfg["data"].get("frame_source", "auto"),
            random_crop_scale=cfg["spatial_transform"]["random_crop_scale"],
            color_jitter=cfg["spatial_transform"].get("color_jitter", 0.4),
            h_flip_prob=cfg["spatial_transform"].get("h_flip_prob", 0.5),
            random_erasing_prob=cfg["spatial_transform"].get("random_erasing_prob", 0.0),
            random_erasing_scale=cfg["spatial_transform"].get("random_erasing_scale", (0.02, 0.15)),
            clips_per_video=cfg["data"].get("clips_per_video", 1),
            train=True,
        )
        batch_sampler = GroupBalancedBatchSampler(
            video_ids=dataset.video_ids,
            labels=dataset.labels,
            batch_size=cfg["optim"]["batch_size"],
            min_positives_per_class=cfg["supcon"].get("min_positives_per_class", 2),
            seed=self.seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg["data"].get("num_workers", 4),
            pin_memory=cfg["data"].get("pin_memory", True),
            worker_init_fn=worker_init_fn,
        )
        return loader, batch_sampler

    def build_model_and_loss(self):
        model = SupConEncoder(
            backbone_cfg=self.cfg["backbone"],
            feature_dim=self.cfg["supcon"]["feature_dim"],
            hidden_dim=self.cfg["supcon"]["hidden_dim"],
        )
        model = self.wrap_for_ddp(model)
        criterion = SupConLoss(temperature=self.cfg["supcon"]["temperature"])
        return model, criterion

    def train_step(self, model, criterion, batch):
        x_clip, labels, video_ids = (t.to(self.device, non_blocking=True) for t in batch)
        with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
            features = model(x_clip)
            loss = criterion(features, labels, group_ids=video_ids)
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
            batch_sampler.set_epoch(epoch)  # keeps all DDP ranks' shard+shuffle in sync
            model.train()
            loss_meter = AverageMeter()
            t0 = time.time()

            for step, batch in enumerate(loader):
                loss = self.train_step(model, criterion, batch)
                self.run_optimizer_step(optimizer, loss, scheduler)
                loss_meter.update(loss.item())

                if step % log_every == 0 and is_main_process():
                    self.logger.log(
                        {"loss": loss_meter.avg, "lr": optimizer.param_groups[0]["lr"]},
                        step=self.global_step,
                    )

            if is_main_process():
                print(f"[supcon] epoch {epoch} done in {time.time() - t0:.1f}s, avg_loss={loss_meter.avg:.4f}")
            if (epoch + 1) % ckpt_every == 0 or epoch == epochs - 1:
                self.save_checkpoint(model, optimizer, epoch)

        self.logger.close()