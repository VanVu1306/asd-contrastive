"""
utils/metrics.py
==================
Thin wrappers around scikit-learn for the two metric families eval.py needs:

  Frame-level (anomaly scoring):  AUC-ROC, Equal Error Rate, FAR at a fixed
                                    operating threshold.
  Classification (linear probing / action classification): Top-1/Top-5
                                    accuracy, Precision/Recall/F1.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.metrics import roc_auc_score as _sklearn_auc


def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """scores: higher = more anomalous. labels: 1 = anomalous, 0 = normal."""
    if len(np.unique(labels)) < 2:
        return float("nan")  # AUC is undefined with only one class present
    return float(_sklearn_auc(labels, scores))


def macro_auc_roc(scores_by_video: Sequence[np.ndarray], labels_by_video: Sequence[np.ndarray]) -> float:
    """Mean per-video AUROC, ignoring videos with only one label class."""
    per_video = [
        auc_roc(scores, labels)
        for scores, labels in zip(scores_by_video, labels_by_video)
    ]
    valid = [value for value in per_video if not np.isnan(value)]
    return float(np.mean(valid)) if valid else float("nan")


def normalize_scores(scores: np.ndarray, method: str = "none") -> np.ndarray:
    """Per-video score normalization, applied BEFORE pooling videos together
    for micro-AUC. Different videos' raw K-Centroid distances can sit on 
    different baselines/scales — pooling raw scores across videos then
    lets that baseline mismatch dominate the ranking micro-AUC measures,
    even when each video is internally well-ordered. Both methods here are
    monotonic *within* a single video, so they never change that video's
    own internal ranking.

    "zscore"      : (x - mean) / std. Sensitive to outliers — a handful of
                     unusually extreme scores (plausible with a small,
                     ~45-clip centroid-mining set) can compress everything
                     else toward 0, understating genuine separation.
    "percentile"  : rank-based (x's rank among this video's own frames,
                     scaled to [0, 1]). Depends only on ORDER, not
                     magnitude, so a few outlier frames can't distort the
                     rest of the video's scale — generally the more robust
                     choice for a small/noisy centroid-mining set.
    "none"        : scores passed through unchanged (the previous, only,
                     behavior — default, so existing configs are unaffected).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if method in (None, "none"):
        return scores
    if method == "zscore":
        std = scores.std()
        if std < 1e-8:
            return scores - scores.mean()  # degenerate (constant score): just center, avoid /0
        return (scores - scores.mean()) / std
    if method == "percentile":
        if len(scores) <= 1:
            return scores
        lo, hi = np.percentile(scores, [5, 95])
        if hi - lo < 1e-8:
            return scores - scores.mean()
        return np.clip((scores - lo) / (hi - lo), 0, 1)
    raise ValueError(f"Unknown score normalization method '{method}' (use 'none', 'zscore', or 'percentile')")


def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> float:
    """EER: the point on the ROC curve where the false-positive rate equals
    the false-negative rate (1 - TPR). Lower is better."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def far_ratio(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    """False Alarm Rate at a fixed operating threshold: fraction of normal
    (label==0) frames whose score exceeds `threshold` and get (wrongly)
    flagged as anomalous."""
    normal_mask = labels == 0
    if normal_mask.sum() == 0:
        return float("nan")
    false_alarms = (scores[normal_mask] >= threshold).sum()
    return float(false_alarms / normal_mask.sum())


def frame_level_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    metric_names: Sequence[str],
    scores_by_video: Sequence[np.ndarray] | None = None,
    labels_by_video: Sequence[np.ndarray] | None = None,
    normalization: str = "none",
) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    fns = {"auc_roc": auc_roc, "eer": equal_error_rate, "far_ratio": far_ratio}
    out = {name: fns[name](scores, labels) for name in metric_names if name in fns and name != "auc_roc"}

    if "auc_roc" in metric_names or "micro_auc_roc" in metric_names:
        normalized = [normalize_scores(s, normalization) for s in scores_by_video]
        pooled_norm_scores = np.concatenate(normalized)
        pooled_labels = np.concatenate(labels_by_video)
        assert pooled_labels.shape == labels.shape and np.array_equal(pooled_labels, labels), (
            "labels does not match concat(labels_by_video) — there is a risk of mismatch in order/data source "
            "between `labels` and `labels_by_video`. Please verify the function call before relying on micro_auc_roc."
        )

        micro = auc_roc(pooled_norm_scores, pooled_labels)
        out["micro_auc_roc"] = micro
        # Keep the old key usable for existing config files and consumers.
        if "auc_roc" in metric_names:
            out["auc_roc"] = micro

    if "macro_auc_roc" in metric_names or "auc_roc" in metric_names:
        if scores_by_video is None or labels_by_video is None:
            raise ValueError("Per-video scores and labels are required for macro AUROC")
        out["macro_auc_roc"] = macro_auc_roc(scores_by_video, labels_by_video)

    return out


def top_k_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    top_k_preds = np.argsort(-logits, axis=1)[:, :k]
    hits = (top_k_preds == labels[:, None]).any(axis=1)
    return float(hits.mean())


def classification_metrics(logits: np.ndarray, labels: np.ndarray, metric_names: Sequence[str]) -> Dict[str, float]:
    preds = np.argmax(logits, axis=1)
    out: Dict[str, float] = {}
    if "top1" in metric_names:
        out["top1"] = top_k_accuracy(logits, labels, k=1)
    if "top5" in metric_names and logits.shape[1] >= 5:
        out["top5"] = top_k_accuracy(logits, labels, k=5)
    if "precision" in metric_names:
        out["precision"] = float(precision_score(labels, preds, average="macro", zero_division=0))
    if "recall" in metric_names:
        out["recall"] = float(recall_score(labels, preds, average="macro", zero_division=0))
    if "f1" in metric_names:
        out["f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    return out