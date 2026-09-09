#!/usr/bin/env python3
"""
scripts/make_dummy_data.py
============================
Generates a tiny synthetic dataset (random-noise .npy clips + manifests in
every format the repo expects) so the full pipeline — both training modes
and both eval modes — can be smoke-tested end-to-end without any real
videos. This is what was used to verify the repo before delivery; it is
NOT a substitute for real data (random noise has no periodicity or
behavior signal for the model to actually learn), only a way to check that
every file path, tensor shape, and training loop actually runs.

Usage:
    python scripts/make_dummy_data.py --out ./data --frame-size 32

Then, e.g.:
    python train.py --config configs/ssl_moco.yaml --opts \\
        data.root=. data.train_split=./data/splits/unlabeled_train.txt \\
        data.clip_len=8 data.raw_clip_len=20 data.frame_size=32 \\
        hardware.device=cpu hardware.amp=false
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


def make_clip(rng, t, h, w):
    return rng.integers(0, 256, size=(t, h, w, 3), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./data", help="Output data root")
    parser.add_argument("--frame-size", type=int, default=48, help="H=W of generated clips")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = args.out
    for sub in ("unlabeled", "labeled", "normal", "test", "splits"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    # --- Unlabeled clips for SSL ---
    unlabeled_paths = []
    for i in range(12):
        t = rng.integers(24, 40)
        arr = make_clip(rng, t, args.frame_size, args.frame_size)
        rel = f"unlabeled/clip_{i:02d}.npy"
        np.save(os.path.join(root, rel), arr)
        unlabeled_paths.append(rel)
    with open(os.path.join(root, "splits/unlabeled_train.txt"), "w") as f:
        f.write("\n".join(unlabeled_paths) + "\n")

    # --- Labeled clips for SupCon: 6 distinct video_ids spread across 2 labels ---
    rows = []
    for i in range(24):
        label = i % 2
        video_id = f"vid_{i % 6}"
        t = rng.integers(20, 30)
        arr = make_clip(rng, t, args.frame_size, args.frame_size)
        rel = f"labeled/clip_{i:02d}.npy"
        np.save(os.path.join(root, rel), arr)
        rows.append((rel, video_id, label))

    with open(os.path.join(root, "splits/labeled_train.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_path", "video_id", "label"])
        w.writerows(rows[:18])
    with open(os.path.join(root, "splits/labeled_val.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_path", "video_id", "label"])
        w.writerows(rows[18:])

    # --- Normal-only clips for centroid mining ---
    normal_paths = []
    for i in range(10):
        arr = make_clip(rng, 24, args.frame_size, args.frame_size)
        rel = f"normal/clip_{i:02d}.npy"
        np.save(os.path.join(root, rel), arr)
        normal_paths.append(rel)
    with open(os.path.join(root, "splits/normal_train.txt"), "w") as f:
        f.write("\n".join(normal_paths) + "\n")

    # --- Test videos with per-frame labels for anomaly scoring ---
    test_rows = []
    for i in range(3):
        t_total = 60
        arr = make_clip(rng, t_total, args.frame_size, args.frame_size)
        clip_rel = f"test/video_{i:02d}.npy"
        np.save(os.path.join(root, clip_rel), arr)
        labels = np.zeros(t_total, dtype=np.int64)
        labels[30:45] = 1  # a fake "anomalous" segment, purely for shape/plumbing testing
        labels_rel = f"test/video_{i:02d}_labels.npy"
        np.save(os.path.join(root, labels_rel), labels)
        test_rows.append((clip_rel, f"test_vid_{i}", labels_rel))
    with open(os.path.join(root, "splits/test.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_path", "video_id", "frame_labels_path"])
        w.writerows(test_rows)

    print(f"Dummy dataset written under {root}/ (manifests under {root}/splits/)")


if __name__ == "__main__":
    main()
