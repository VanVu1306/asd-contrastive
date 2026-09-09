"""
utils/visualize.py
====================
Plotting helpers for eval.py's anomaly-scoring mode: one PNG per test video,
showing the per-frame anomaly score curve against the ground-truth anomalous
segments, plus a small ROC-curve helper for the whole test split.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


def plot_score_vs_groundtruth(
    scores: np.ndarray,
    labels: Optional[np.ndarray],
    video_id: str,
    out_dir: str,
    threshold: Optional[float] = None,
) -> str:
    """Saves `{out_dir}/{video_id}_score.png` and returns its path."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(scores, color="tab:blue", label="anomaly score")

    if labels is not None:
        anomalous = np.where(labels > 0)[0]
        if len(anomalous) > 0:
            ax.fill_between(
                np.arange(len(labels)), 0, 1, where=labels > 0,
                transform=ax.get_xaxis_transform(), color="tab:red", alpha=0.15,
                label="ground-truth anomaly",
            )

    if threshold is not None:
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1, label="threshold")

    ax.set_title(f"Anomaly score — {video_id}")
    ax.set_xlabel("frame index")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)

    out_path = os.path.join(out_dir, f"{video_id}_score.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_roc_curve(scores: np.ndarray, labels: np.ndarray, out_dir: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    os.makedirs(out_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Frame-level ROC")
    ax.legend(loc="lower right")

    out_path = os.path.join(out_dir, "roc_curve.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
