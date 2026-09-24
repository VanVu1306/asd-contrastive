#!/usr/bin/env python3
"""
scripts/select_k_bic.py
==========================
Standalone BIC-based K search, with NO time limit — for the case
`eval.py --mode anomaly_scoring` (with `eval.centroids.k_selection: "bic"`)
reports that its inline, time-boxed sweep timed out. Run this separately,
offline, then copy the reported best K into `eval.centroids.k` in your
config by hand (set `k_selection: "fixed"` there too, so eval.py doesn't
try the inline sweep again next time).

This needs a trained checkpoint already — it extracts normal-clip features
the exact same way eval.py's Stage 2 does, then runs the same BIC sweep
utils/model_selection.select_k_by_bic uses internally, just without a
wall-clock budget.

Usage:
    python scripts/select_k_bic.py --config configs/eval.yaml \
        --k-min 2 --k-max 60 --k-step 2
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from datasets.eval_dataset import CentroidMiningDataset
from eval import extract_features, load_backbone_from_checkpoint
from utils.config import apply_overrides, load_config
from utils.model_selection import format_sweep_report, select_k_by_bic
from utils.seed import set_seed, worker_init_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone, unbounded BIC search for Stage 2's K.")
    parser.add_argument("--config", required=True, help="Path to an eval config (e.g. configs/eval.yaml)")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=60)
    parser.add_argument("--k-step", type=int, default=2)
    parser.add_argument("--opts", nargs="*", default=[], help="Same --opts overrides eval.py accepts")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.opts:
        cfg = apply_overrides(cfg, args.opts)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["hardware"]["device"] == "cuda" else "cpu")
    centroid_cfg = cfg["eval"]["centroids"]

    print(f"[select_k_bic] loading backbone from {cfg['checkpoint_path']}")
    backbone = load_backbone_from_checkpoint(cfg["backbone"], cfg["checkpoint_path"]).to(device)
    backbone.eval()

    normal_ds = CentroidMiningDataset(
        root=cfg["data"]["root"], split_file=centroid_cfg["normal_split"],
        clip_len=cfg["data"]["clip_len"], frame_size=cfg["data"]["frame_size"],
        frame_source=cfg["data"].get("frame_source", "auto"),
    )
    normal_loader = DataLoader(
        normal_ds, batch_size=cfg["eval"]["batch_size"], shuffle=False, worker_init_fn=worker_init_fn,
    )
    print(f"[select_k_bic] extracting features for {len(normal_ds)} normal clips...")
    normal_features = extract_features(backbone, normal_loader, device)

    k_candidates = list(range(args.k_min, args.k_max + 1, args.k_step))
    print(f"[select_k_bic] running unbounded BIC sweep over {k_candidates}...")
    result = select_k_by_bic(
        normal_features, k_candidates, seed=centroid_cfg.get("kmeans_seed", 42), time_budget_sec=float("inf"),
    )
    print(format_sweep_report(result))
    print()
    print(f"==> Set eval.centroids.k: {result.best_k} (and eval.centroids.k_selection: \"fixed\") in your config.")


if __name__ == "__main__":
    main()
