#!/usr/bin/env python3
"""
train.py
=========
Single entry point for both training pipelines. The config file's `mode:`
field (set in configs/ssl_moco.yaml / configs/supcon.yaml) selects which
trainer runs — that's the only thing that differs between the two `python -m
torch.distributed.run ... train.py --config ...` invocations in the design
doc's usage guide.

Examples
--------
Single process (CPU or 1 GPU):
    python train.py --config configs/ssl_moco.yaml
    python train.py --config configs/supcon.yaml

Multi-GPU (DDP):
    python -m torch.distributed.run --nproc_per_node=4 train.py --config configs/ssl_moco.yaml
"""
from __future__ import annotations

import argparse

from trainers.ssl_trainer import SSLTrainer
from trainers.supcon_trainer import SupConTrainer
from trainers.spi_trainer import SPITrainer
from utils.config import apply_overrides, load_config
from utils.distributed import cleanup_ddp

_TRAINERS = {
    "ssl_moco": SSLTrainer,
    "supcon": SupConTrainer,
    "spi_periodicity": SPITrainer,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the video-periodicity contrastive framework.")
    parser.add_argument("--config", required=True, help="Path to a YAML config (e.g. configs/ssl_moco.yaml)")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from")
    parser.add_argument(
        "--opts", nargs="*", default=[],
        help="Override config values on the fly, e.g. --opts optim.lr=0.01 optim.epochs=5",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.opts:
        cfg = apply_overrides(cfg, args.opts)

    mode = cfg.get("mode")
    if mode not in _TRAINERS:
        raise ValueError(f"config.mode must be one of {list(_TRAINERS)}, got '{mode}'")

    trainer = _TRAINERS[mode](cfg)
    trainer.resume_path = args.resume

    try:
        trainer.train()
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()