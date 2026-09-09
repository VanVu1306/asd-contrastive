"""
models/heads.py
=================
Small heads attached on top of a backbone's pooled (B, D) feature vector.

  ProjectionHead        : Linear -> BatchNorm1d -> ReLU -> Linear, L2-normalized.
                           Used by both MoCo (query/key projector) and SupCon.
  ClassificationHead     : Linear(D, num_classes). Downstream action
                           classification / linear probing.
  PeriodRegressionHead    : auxiliary head predicting a clip's cycle length
                           (log P) for the SPI pretext branch — see
                           datasets/spi_dataset.py / losses/spicon_loss.py.
  AnomalyScoringHead      : two interchangeable scoring strategies —
      - parametric mode: a small MLP/Conv1d regresses S in [0, 1] directly
        (needs labeled/weak-labeled anomaly supervision to train).
      - non-parametric mode ("K-Centroids"): no gradient at all. K cluster
        centroids (torch buffer, not a Parameter) are fit once via k-means
        over normal-only features (see models/moco_wrapper.py's encoder +
        eval.py's Stage-2 centroid mining), then every subsequent score is
        just "cosine distance to the nearest centroid" — Eq. in the design
        doc: S_raw(x) = min_k (1 - cos_sim(z, c_k)).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)  # L2-normalize for cosine similarity


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # raw logits; caller applies CrossEntropyLoss


class PeriodRegressionHead(nn.Module):
    """Auxiliary head for the SPI (Synthetic Periodicity Injection) pretext
    task, see datasets/spi_dataset.py and losses/spicon_loss.py: predicts a
    clip's cycle length P directly (a real-valued regression target, not a
    class), as a training signal on top of SPIConLoss's periodic/
    non-periodic contrastive loss — so the embedding space encodes *how*
    periodic a clip is, not just whether it is.

    Predicts log(P) rather than P directly: cycle lengths span roughly one
    order of magnitude (a few frames to a few dozen), and squared-error on
    the raw frame count would let long-cycle clips dominate the loss.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # predicted log(P), shape (B,)


class ParametricAnomalyHead(nn.Module):
    """Small temporal-aware MLP: Linear -> ReLU -> Linear -> Sigmoid,
    producing S in [0, 1] directly. Requires (at least weak) anomaly labels
    to train — use this branch only if such labels are available; otherwise
    prefer `CentroidScoringHead`."""

    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


class CentroidScoringHead(nn.Module):
    """Non-parametric K-Centroids scorer. `fit()` is called once, offline,
    during eval.py's Stage 2 (centroid mining) — everything after that is a
    zero-backprop nearest-centroid lookup at inference time (Stage 3)."""

    def __init__(self, feature_dim: int, k: int = 20):
        super().__init__()
        self.k = k
        # Registered as a buffer (not nn.Parameter): it's never updated by
        # gradient descent, only overwritten wholesale by `fit()` / `load()`.
        self.register_buffer("centroids", torch.zeros(k, feature_dim))
        self._fitted = False

    @torch.no_grad()
    def fit(self, normal_features: torch.Tensor, seed: int = 42, n_iters: int = 100) -> None:
        """K-Means over L2-normalized normal-only features. Implemented with
        plain torch ops (no extra clustering dependency) since it only needs
        to run once, offline, over a modest number of clips."""
        feats = F.normalize(normal_features, dim=-1)
        n = feats.shape[0]
        if n < self.k:
            raise ValueError(f"Need at least k={self.k} normal clips to mine centroids, got {n}.")

        g = torch.Generator(device=feats.device).manual_seed(seed)
        init_idx = torch.randperm(n, generator=g)[: self.k]
        centroids = feats[init_idx].clone()

        for _ in range(n_iters):
            sims = feats @ centroids.t()                  # (N, K) cosine sim (both L2-normalized)
            assignments = sims.argmax(dim=1)               # nearest centroid per sample
            new_centroids = centroids.clone()
            moved = False
            for k in range(self.k):
                members = feats[assignments == k]
                if members.numel() == 0:
                    continue  # keep previous centroid for an empty cluster rather than NaN-ing it
                mean = F.normalize(members.mean(dim=0), dim=0)
                if not torch.allclose(mean, centroids[k], atol=1e-6):
                    moved = True
                new_centroids[k] = mean
            centroids = new_centroids
            if not moved:
                break

        self.centroids.copy_(centroids)
        self._fitted = True

    def load(self, centroids: torch.Tensor) -> None:
        assert centroids.shape == self.centroids.shape, (
            f"expected centroids shaped {tuple(self.centroids.shape)}, got {tuple(centroids.shape)}"
        )
        self.centroids.copy_(centroids)
        self._fitted = True

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D) encoder features (need not be pre-normalized).
        Returns S_raw: (B,) = min_k (1 - cos_sim(z, c_k))."""
        if not self._fitted:
            raise RuntimeError("CentroidScoringHead.fit() (or .load()) must run before scoring.")
        z = F.normalize(z, dim=-1)
        c = F.normalize(self.centroids, dim=-1)
        cos_sim = z @ c.t()               # (B, K)
        best_sim, _ = cos_sim.max(dim=1)  # nearest centroid = highest cosine similarity
        return 1.0 - best_sim


class AnomalyScoringHead(nn.Module):
    """Thin dispatcher so trainers/eval.py can request either scoring mode
    through one interface, matching the design doc's "Supervised Mode
    (Parametric)" vs "Non-parametric Mode (K-Centroids)" split."""

    def __init__(self, feature_dim: int, mode: str = "centroid", k: int = 20, hidden_dim: int = 256):
        super().__init__()
        self.mode = mode
        if mode == "parametric":
            self.impl: nn.Module = ParametricAnomalyHead(feature_dim, hidden_dim)
        elif mode == "centroid":
            self.impl = CentroidScoringHead(feature_dim, k)
        else:
            raise ValueError(f"Unknown AnomalyScoringHead mode '{mode}' (use 'parametric' or 'centroid').")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.impl(z)
