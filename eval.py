#!/usr/bin/env python3
"""
eval.py
========
Two evaluation modes, selected by --mode (or eval.mode in the config):

  anomaly_scoring : Stage 2 (K-Centroid mining over normal-only clips) +
                     Stage 3 (sliding-window distance scoring) of the
                     non-parametric pipeline described in the design doc,
                     followed by frame-level AUC-ROC / EER / FAR-ratio.

  linear_probing  : freeze (by default) the pretrained encoder and train
                     only ClassificationHead on labeled clips — the standard
                     way to sanity-check how good the self-supervised or
                     SupCon representation actually is.

Usage
-----
    python eval.py --config configs/eval.yaml --mode anomaly_scoring
    python eval.py --config configs/eval.yaml --mode linear_probing
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.eval_dataset import CentroidMiningDataset, SlidingWindowTestDataset
from datasets.supcon_dataset import SupConDataset
from models.backbones.builder import build_backbone
from models.heads import CentroidScoringHead, ClassificationHead
from utils.config import apply_overrides, load_config
from utils.metrics import classification_metrics, frame_level_metrics
from utils.seed import set_seed, worker_init_fn
from utils.visualize import plot_roc_curve, plot_score_vs_groundtruth


def resolve_results_path(cfg, filename: str) -> str:
    """Where to save a results JSON. Deliberately NOT derived from
    checkpoint_path's directory: on Kaggle (and most shared-checkpoint
    setups) that directory is a read-only input mount, not something eval.py
    can write into — see the 'read-only file system' crash this replaces.
    Defaults to the same output_dir/experiment_name layout trainers use,
    which is always writable, and can be overridden with eval.results_dir.
    """
    results_dir = cfg.get("eval", {}).get("results_dir") or os.path.join(
        cfg.get("output_dir", "./runs"), cfg.get("experiment_name", "eval")
    )
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, filename)


def save_results_json(path: str, results: dict) -> None:
    """Best-effort save: a results file failing to write should never take
    down a run that already computed and printed its metrics."""
    try:
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[eval] results saved to {path}")
    except OSError as e:
        print(f"[eval] WARNING: could not save results to {path} ({e}); "
              f"metrics were still computed and printed above.")


def load_backbone_from_checkpoint(backbone_cfg, checkpoint_path: str) -> nn.Module:
    """Every trainer checkpoints a *different* wrapping module around the
    same backbone: MoCoWrapper.encoder_q.backbone.* (ssl_moco),
    SupConEncoder.backbone.* (supcon), or SPIEncoder's spi.backbone.*
    (spi_periodicity, standalone mode — combined mode also writes
    moco.encoder_q.backbone.* for the same shared weights, so the "moco."
    prefix below already covers it too). This strips whichever prefix
    matches instead of assuming one trainer."""
    backbone = build_backbone(backbone_cfg)
    state = torch.load(checkpoint_path, map_location="cpu",weights_only=False)["model"]

    for prefix in ("encoder_q.backbone.", "backbone."):
        matched = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if matched:
            missing, unexpected = backbone.load_state_dict(matched, strict=False)
            if missing or unexpected:
                print(f"[eval] backbone load — missing={missing}, unexpected={unexpected}")
            return backbone

    raise ValueError(
        f"Could not find backbone weights under 'encoder_q.backbone.*' or 'backbone.*' "
        f"in {checkpoint_path}; found top-level keys: {sorted(state.keys())[:5]}..."
    )


@torch.no_grad()
def extract_features(backbone: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    backbone.eval()
    feats = []
    for batch in loader:
        clips = batch.to(device) if torch.is_tensor(batch) else batch[0].to(device)
        feats.append(backbone(clips).cpu())
    return torch.cat(feats, dim=0)


def smooth_scores(scores: np.ndarray, method: str, window: int) -> np.ndarray:
    if method == "none" or window <= 1:
        return scores
    if method == "moving_average":
        kernel = np.ones(window) / window
        return np.convolve(scores, kernel, mode="same")
    if method == "median":
        from scipy.ndimage import median_filter

        return median_filter(scores, size=window, mode="nearest")
    raise ValueError(f"Unknown smoothing method '{method}'")


def run_anomaly_scoring(cfg, device: torch.device):
    backbone = load_backbone_from_checkpoint(cfg["backbone"], cfg["checkpoint_path"]).to(device)
    backbone.eval()

    # ---- Stage 2: Centroid Mining (single pass, zero backprop) ----------
    centroid_cfg = cfg["eval"]["centroids"]
    normal_ds = CentroidMiningDataset(
        root=cfg["data"]["root"], split_file=centroid_cfg["normal_split"],
        clip_len=cfg["data"]["clip_len"], frame_size=cfg["data"]["frame_size"],
        frame_source=cfg["data"].get("frame_source", "auto"),
    )
    normal_loader = DataLoader(
        normal_ds, batch_size=cfg["eval"]["batch_size"], shuffle=False, worker_init_fn=worker_init_fn,
    )
    normal_features = extract_features(backbone, normal_loader, device)

    scoring_head = CentroidScoringHead(feature_dim=backbone.out_dim, k=centroid_cfg["k"])
    scoring_head.fit(normal_features, seed=centroid_cfg.get("kmeans_seed", 42))
    os.makedirs(os.path.dirname(centroid_cfg["save_path"]), exist_ok=True)
    torch.save(scoring_head.centroids, centroid_cfg["save_path"])
    print(f"[eval] mined {centroid_cfg['k']} centroids from {len(normal_ds)} normal clips "
          f"-> {centroid_cfg['save_path']}")
    scoring_head = scoring_head.to(device)

    # ---- Stage 3: sliding-window inference + frame-level scoring ---------
    test_ds = SlidingWindowTestDataset(
        root=cfg["data"]["root"], split_file=cfg["eval"]["test_split"],
        clip_len=cfg["data"]["clip_len"], stride=cfg["eval"]["window_stride"],
        frame_size=cfg["data"]["frame_size"], frame_source=cfg["data"].get("frame_source", "auto"),
    )

    post_cfg = cfg["eval"]["post_process"]
    viz_cfg = cfg.get("visualize", {})
    all_scores, all_labels = [], []

    for i in range(len(test_ds)):
        clips, starts, frame_labels, video_id = test_ds.get_video(i)
        n_frames = (frame_labels.shape[0] if frame_labels is not None
                    else int(starts[-1]) + cfg["data"]["clip_len"])

        window_scores = []
        batch_size = cfg["eval"]["batch_size"]
        with torch.no_grad():
            for b in range(0, clips.shape[0], batch_size):
                chunk = clips[b : b + batch_size].to(device)
                feats = backbone(chunk)
                window_scores.append(scoring_head(feats).cpu().numpy())
        window_scores = np.concatenate(window_scores, axis=0)

        # Overlap-average window scores back onto a per-frame curve.
        score_sum = np.zeros(n_frames, dtype=np.float64)
        score_count = np.zeros(n_frames, dtype=np.float64)
        clip_len = cfg["data"]["clip_len"]
        for s, w_score in zip(starts, window_scores):
            score_sum[s : s + clip_len] += w_score
            score_count[s : s + clip_len] += 1
        frame_scores = np.divide(score_sum, np.maximum(score_count, 1))
        frame_scores = smooth_scores(frame_scores, post_cfg.get("smoothing", "none"), post_cfg.get("window", 1))

        if viz_cfg.get("enabled", False):
            plot_score_vs_groundtruth(frame_scores, frame_labels, video_id, viz_cfg["out_dir"])

        if frame_labels is not None:
            all_scores.append(frame_scores)
            all_labels.append(frame_labels)

    results = {}
    if all_scores:
        scores_cat = np.concatenate(all_scores)
        labels_cat = np.concatenate(all_labels)
        results = frame_level_metrics(
            scores_cat,
            labels_cat,
            cfg["metrics"]["frame_level"],
            scores_by_video=all_scores,
            labels_by_video=all_labels,
        )
        print(f"[eval] frame-level metrics over {len(all_scores)} videos: {results}")
        if viz_cfg.get("enabled", False) and len(np.unique(labels_cat)) > 1:
            plot_roc_curve(scores_cat, labels_cat, viz_cfg["out_dir"])
    else:
        print("[eval] no frame_labels_path found in test_split — skipping metrics (scores/plots still saved).")

    out_path = resolve_results_path(cfg, "anomaly_scoring_results.json")
    save_results_json(out_path, results)
    return results


def run_linear_probing(cfg, device: torch.device):
    probe_cfg = cfg["eval"]["linear_probe"]
    backbone = load_backbone_from_checkpoint(cfg["backbone"], cfg["checkpoint_path"]).to(device)
    if probe_cfg.get("freeze_backbone", True):
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

    head = ClassificationHead(backbone.out_dim, probe_cfg["num_classes"]).to(device)
    # Frozen (or barely-trained) backbone features have arbitrary scale —
    # feeding them straight into a Linear+CrossEntropy is numerically
    # unstable (loss can blow up within a couple of steps). A parameter-free
    # BatchNorm1d in front of the classifier is the standard linear-eval
    # fix: it normalizes feature scale using running statistics, with no
    # extra learnable capacity that could leak label information into the
    # "linear" probe.
    feat_norm = nn.BatchNorm1d(backbone.out_dim, affine=False).to(device)

    train_ds = SupConDataset(
        root=cfg["data"]["root"], split_file=probe_cfg["train_split"],
        clip_len=cfg["data"]["clip_len"], frame_size=cfg["data"]["frame_size"],
        frame_source=cfg["data"].get("frame_source", "auto"), train=True,
    )
    val_ds = SupConDataset(
        root=cfg["data"]["root"], split_file=probe_cfg["val_split"],
        clip_len=cfg["data"]["clip_len"], frame_size=cfg["data"]["frame_size"],
        frame_source=cfg["data"].get("frame_source", "auto"), train=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=probe_cfg["batch_size"], shuffle=True, drop_last=True, worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=probe_cfg["batch_size"], shuffle=False, worker_init_fn=worker_init_fn,
    )

    trainable = list(head.parameters()) if probe_cfg.get("freeze_backbone", True) else (
        list(head.parameters()) + list(backbone.parameters())
    )
    optimizer = torch.optim.SGD(trainable, lr=probe_cfg["lr"], momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(probe_cfg["epochs"]):
        head.train()
        feat_norm.train()
        if not probe_cfg.get("freeze_backbone", True):
            backbone.train()
        total_loss = 0.0
        for x_clip, labels, _video_ids in train_loader:
            x_clip, labels = x_clip.to(device), labels.to(device)
            with torch.set_grad_enabled(True):
                feats = backbone(x_clip) if not probe_cfg.get("freeze_backbone", True) else backbone(x_clip).detach()
                feats = feat_norm(feats)
                logits = head(feats)
                loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[linear_probe] epoch {epoch}: train_loss={total_loss / max(1, len(train_loader)):.4f}")

    head.eval()
    feat_norm.eval()
    backbone.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x_clip, labels, _video_ids in val_loader:
            x_clip = x_clip.to(device)
            logits = head(feat_norm(backbone(x_clip)))
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    logits_cat = np.concatenate(all_logits, axis=0)
    labels_cat = np.concatenate(all_labels, axis=0)
    results = classification_metrics(logits_cat, labels_cat, cfg["metrics"]["classification"])
    print(f"[linear_probe] val metrics: {results}")

    out_path = resolve_results_path(cfg, "linear_probing_results.json")
    save_results_json(out_path, results)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained encoder.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["anomaly_scoring", "linear_probing"], default=None)
    parser.add_argument("--opts", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.opts:
        cfg = apply_overrides(cfg, args.opts)
    mode = args.mode or cfg["eval"]["mode"]

    set_seed(cfg.get("seed", 42), deterministic=cfg.get("hardware", {}).get("deterministic", False))

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["hardware"]["device"] == "cuda" else "cpu")

    if mode == "anomaly_scoring":
        run_anomaly_scoring(cfg, device)
    elif mode == "linear_probing":
        run_linear_probing(cfg, device)
    else:
        raise ValueError(f"Unknown eval mode '{mode}'")


if __name__ == "__main__":
    main()