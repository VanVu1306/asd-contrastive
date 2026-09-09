"""
trainers/ssl_trainer.py
=========================
Drives MoCo pretraining: builds the SSLDataset (unlabeled clips ->
anchor/warp/shuffle triplets), the MoCoWrapper, and MoCoNCELoss, then runs
the standard epoch/step loop from BaseTrainer.
"""
from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from datasets.samplers import VideoDiverseBatchSampler
from datasets.ssl_dataset import SSLDataset
from losses.moco_nce_loss import MoCoNCELoss
from models.moco_wrapper import MoCoWrapper
from trainers.base_trainer import BaseTrainer
from utils.distributed import is_main_process
from utils.logger import AverageMeter
from utils.seed import worker_init_fn


class SSLTrainer(BaseTrainer):
    def build_dataloader(self):
        cfg = self.cfg
        dataset = SSLDataset(
            root=cfg["data"]["root"],
            split_file=cfg["data"]["train_split"],
            clip_len=cfg["data"]["clip_len"],
            raw_clip_len=cfg["data"]["raw_clip_len"],
            frame_size=cfg["data"]["frame_size"],
            frame_source=cfg["data"].get("frame_source", "auto"),
            speed_warp_rates=cfg["temporal_transform"]["speed_warp_rates"],
            frame_shuffle_ratio=cfg["temporal_transform"].get("frame_shuffle_ratio", 1.0),
            random_crop_scale=cfg["spatial_transform"]["random_crop_scale"],
            color_jitter=cfg["spatial_transform"].get("color_jitter", 0.4),
            h_flip_prob=cfg["spatial_transform"].get("h_flip_prob", 0.5),
            random_erasing_prob=cfg["spatial_transform"].get("random_erasing_prob", 0.0),
            random_erasing_scale=cfg["spatial_transform"].get("random_erasing_scale", (0.02, 0.15)),
            clips_per_video=cfg["data"].get("clips_per_video", 1),
        )
        batch_size = cfg["optim"]["batch_size"]
        # max_per_video_per_batch=None (default) -> no cap, i.e. identical to
        # the plain-shuffle behavior this replaces when clips_per_video=1.
        # DDP rank-sharding lives inside the sampler itself (see
        # datasets/samplers.py), so no separate DistributedSampler is used.
        max_per_video = cfg["data"].get("max_per_video_per_batch") or batch_size
        batch_sampler = VideoDiverseBatchSampler(
            video_index=dataset.video_index, batch_size=batch_size,
            max_per_video=max_per_video, drop_last=True, seed=self.seed,
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
        model = MoCoWrapper(
            backbone_cfg=self.cfg["backbone"],
            feature_dim=self.cfg["moco"]["feature_dim"],
            hidden_dim=self.cfg["moco"]["hidden_dim"],
            queue_size=self.cfg["moco"]["queue_size"],
            momentum=self.cfg["moco"]["momentum"],
        )
        model = self.wrap_for_ddp(model)
        criterion = MoCoNCELoss(temperature=self.cfg["moco"]["temperature"])
        return model, criterion

    def train_step(self, model, criterion, batch):
        x_anchor, x_pos_warp, x_neg_shuffle = (t.to(self.device, non_blocking=True) for t in batch)
        with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
            q, k, queue, k_neg = model(x_anchor, x_pos_warp, x_neg_shuffle)
            loss = criterion(q, k, queue, k_neg)
        return loss

    def train(self):
        cfg = self.cfg
        loader, batch_sampler = self.build_dataloader()
        model, criterion = self.build_model_and_loss()
        # Only encoder_q is trained by gradient descent; encoder_k only ever
        # moves via the EMA update inside MoCoWrapper._momentum_update_key_encoder.
        optimizer = self.build_optimizer(self.unwrap(model).encoder_q.parameters())
        self.maybe_resume(model, optimizer)
        scheduler = self.make_scheduler(optimizer, steps_per_epoch=len(loader))

        epochs = cfg["optim"]["epochs"]
        log_every = cfg["logging"].get("log_every", 20)
        ckpt_every = cfg["logging"].get("ckpt_every", 5)

        for epoch in range(self.start_epoch, epochs):
            batch_sampler.set_epoch(epoch)  # keeps all DDP ranks' shuffles in sync
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
                print(f"[ssl] epoch {epoch} done in {time.time() - t0:.1f}s, avg_loss={loss_meter.avg:.4f}")
            if (epoch + 1) % ckpt_every == 0 or epoch == epochs - 1:
                self.save_checkpoint(model, optimizer, epoch)

        self.logger.close()