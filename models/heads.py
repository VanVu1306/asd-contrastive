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
      - "centroid" mode ("K-Centroids"): no gradient at all. K cluster
        centroids (torch buffer, not a Parameter) are fit once via k-means
        over normal-only features (see models/moco_wrapper.py's encoder +
        eval.py's Stage-2 centroid mining), then every subsequent score is
        just "cosine distance to the nearest centroid" — Eq. in the design
        doc: S_raw(x) = min_k (1 - cos_sim(z, c_k)).
      - "gmm" mode: a Gaussian Mixture Model (diagonal covariance) fit once
        over the same normal-only features, offering three ways to turn
        that density model into a score — see GMMScoringHead below.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
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


class GMMScoringHead(nn.Module):
    """Non-parametric alternative to CentroidScoringHead: fits a Gaussian
    Mixture Model (diagonal covariance) over L2-normalized normal-only
    features instead of hard K-Means clusters, then scores a test clip by
    how poorly it's explained by that mixture. `fit()` runs once, offline,
    same as CentroidScoringHead — no gradient at inference time either.

    `covariance_type="diag"` specifically (not "full"): full covariance
    needs roughly N >> D samples per component to stay non-singular, which
    the small (tens-of-clips) centroid-mining sets used throughout this
    project don't have; diagonal covariance only estimates D variances per
    component instead of a D×D matrix, staying stable at small N.

    Three scoring modes (`mode`):
      "nll"      : negative log-likelihood under the fitted mixture — the
                    classic density-based novelty score. Low density under
                    the "normal" distribution -> high anomaly score.
      "entropy"  : entropy of a *temperature-scaled* soft-assignment
                    (posterior responsibility) over the K components. A
                    clip that doesn't confidently belong to any one
                    "normal" mode (high entropy across components) sits in
                    an ambiguous/boundary region of the mixture — a signal
                    NLL alone can miss (a point can sit near several
                    components' means, so no single component scores it as
                    far away, and yet it isn't a confident match to any one
                    behavior pattern either).

                    Dividing log-likelihoods by a temperature > 1 before the
                    softmax smooths this out — but empirically (see
                    tests run while building this), even a large
                    temperature only fixes the boundary-point case; a point
                    genuinely far from *every* component still tends to get
                    assigned almost entirely to whichever component happens
                    to be relatively closest, so its entropy reads as LOW
                    (falsely "confident") despite being a strong anomaly by
                    density standards — the same "overconfident far from
                    training data" effect well documented for softmax
                    classifiers. Net effect: "entropy" alone is a detector
                    of ambiguous/boundary points *among the existing normal
                    modes*, not a general-purpose replacement for "nll" —
                    use "nll" or "combined" as the default choice unless the
                    specific question is about blended/borderline behavior,
                    since "entropy" alone can rank a genuine far-outlier
                    BELOW a normal point.
      "combined" : both signals combined AFTER independently z-score
                    normalizing each against the *fitting* (normal) set's
                    own distribution of that signal. This is necessary, not
                    optional — NLL and entropy live on unrelated numeric
                    scales (NLL scales with feature_dim and the mixture's
                    log-density range; entropy is bounded in [0, log K]) —
                    summing them raw would just let whichever happens to
                    have the larger typical magnitude dominate the sum.
    """

    def __init__(
        self,
        feature_dim: int,
        k: int = 20,
        mode: str = "nll",
        combined_weights: tuple = (0.5, 0.5),
        entropy_temperature: float = 5.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.k = k
        self.mode = mode
        self.combined_weights = combined_weights
        self.entropy_temperature = entropy_temperature
        self._gmm = None  # sklearn.mixture.GaussianMixture, set by fit()/load()
        # Normalization stats for "combined" mode, computed once on the
        # fitting (normal) set itself — buffers so they travel with the
        # checkpoint/state like everything else here, even though the GMM
        # object itself (not a tensor) is stored separately (see
        # export_state/import_state).
        self.register_buffer("nll_mean", torch.zeros(1))
        self.register_buffer("nll_std", torch.ones(1))
        self.register_buffer("entropy_mean", torch.zeros(1))
        self.register_buffer("entropy_std", torch.ones(1))
        self._fitted = False

    def _weighted_log_prob(self, x_np: np.ndarray) -> np.ndarray:
        """(N, K) log(weight_k * N(x; mean_k, diag(var_k))), computed
        manually from the GMM's public means_/covariances_/weights_ rather
        than a private sklearn method — needed so entropy_temperature can
        scale it before the softmax below (predict_proba doesn't expose
        that). Numerically the same quantity predict_proba normalizes."""
        means = self._gmm.means_             # (K, D)
        covars = self._gmm.covariances_      # (K, D) for covariance_type="diag"
        weights = self._gmm.weights_         # (K,)
        d = means.shape[1]

        diff2 = (x_np[:, None, :] - means[None, :, :]) ** 2   # (N, K, D)
        quad = (diff2 / covars[None, :, :]).sum(axis=2)       # (N, K)
        log_det = np.log(covars).sum(axis=1)                  # (K,)
        log_gauss = -0.5 * (d * np.log(2 * np.pi) + log_det[None, :] + quad)
        return log_gauss + np.log(weights)[None, :]

    def _soft_responsibilities(self, x_np: np.ndarray, temperature: float) -> np.ndarray:
        wlp = self._weighted_log_prob(x_np) / max(temperature, 1e-6)
        wlp = wlp - wlp.max(axis=1, keepdims=True)  # numerical stability, doesn't change softmax result
        exp_wlp = np.exp(wlp)
        return exp_wlp / exp_wlp.sum(axis=1, keepdims=True)

    @staticmethod
    def _responsibility_entropy(resp: np.ndarray) -> np.ndarray:
        """Shannon entropy of each row of posterior responsibilities (N, K)."""
        return -(resp * np.log(resp + 1e-12)).sum(axis=1)

    def fit(self, normal_features: torch.Tensor, seed: int = 42, max_iter: int = 200) -> None:
        """Fits the GMM on L2-normalized normal-only features (same
        normalization convention as CentroidScoringHead, for consistency
        between the two scoring modes)."""
        from sklearn.mixture import GaussianMixture

        feats = F.normalize(normal_features, dim=-1)
        n = feats.shape[0]
        if n < self.k:
            raise ValueError(f"Need at least k={self.k} normal clips to fit a {self.k}-component GMM, got {n}.")
        x_np = feats.detach().cpu().numpy().astype(np.float64)

        gmm = GaussianMixture(
            n_components=self.k, covariance_type="diag", random_state=seed,
            max_iter=max_iter, reg_covar=1e-6,
        )
        gmm.fit(x_np)
        self._gmm = gmm

        # Normalization stats for "combined" mode, from the fitting set itself.
        nll = -gmm.score_samples(x_np)
        resp = self._soft_responsibilities(x_np, self.entropy_temperature)
        ent = self._responsibility_entropy(resp)
        self.nll_mean = torch.tensor([nll.mean()], dtype=torch.float32)
        self.nll_std = torch.tensor([nll.std() + 1e-8], dtype=torch.float32)
        self.entropy_mean = torch.tensor([ent.mean()], dtype=torch.float32)
        self.entropy_std = torch.tensor([ent.std() + 1e-8], dtype=torch.float32)
        self._fitted = True

    def export_state(self) -> dict:
        """The fitted sklearn GaussianMixture isn't a tensor/Parameter, so it
        can't live in this module's own state_dict() — eval.py saves this
        dict with plain torch.save() instead (which pickles arbitrary
        picklable Python objects, sklearn models included) alongside the
        buffers above, and import_state() reloads it."""
        if not self._fitted:
            raise RuntimeError("GMMScoringHead.fit() must run before export_state().")
        return {
            "kind": "gmm",
            "gmm": self._gmm,
            "k": self.k,
            "mode": self.mode,
            "combined_weights": self.combined_weights,
            "entropy_temperature": self.entropy_temperature,
            "nll_mean": self.nll_mean.clone(),
            "nll_std": self.nll_std.clone(),
            "entropy_mean": self.entropy_mean.clone(),
            "entropy_std": self.entropy_std.clone(),
        }

    def import_state(self, state: dict) -> None:
        if state.get("kind") != "gmm":
            raise ValueError(f"export_state()/import_state() mismatch: expected a 'gmm' state, got {state.get('kind')}")
        self._gmm = state["gmm"]
        self.entropy_temperature = state.get("entropy_temperature", self.entropy_temperature)
        self.nll_mean = state["nll_mean"]
        self.nll_std = state["nll_std"]
        self.entropy_mean = state["entropy_mean"]
        self.entropy_std = state["entropy_std"]
        self._fitted = True

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D) encoder features (need not be pre-normalized).
        Returns S_raw: (B,) — higher means more anomalous, in every mode."""
        if not self._fitted:
            raise RuntimeError("GMMScoringHead.fit() (or .import_state()) must run before scoring.")
        device = z.device
        x_np = F.normalize(z, dim=-1).detach().cpu().numpy().astype(np.float64)

        nll = -self._gmm.score_samples(x_np)
        if self.mode == "nll":
            score = nll
        elif self.mode in ("entropy", "combined"):
            resp = self._soft_responsibilities(x_np, self.entropy_temperature)
            ent = self._responsibility_entropy(resp)
            if self.mode == "entropy":
                score = ent
            else:
                nll_norm = (nll - self.nll_mean.item()) / self.nll_std.item()
                ent_norm = (ent - self.entropy_mean.item()) / self.entropy_std.item()
                w_nll, w_ent = self.combined_weights
                score = w_nll * nll_norm + w_ent * ent_norm
        else:
            raise ValueError(f"Unknown GMMScoringHead mode '{self.mode}' (use 'nll', 'entropy', or 'combined').")

        return torch.from_numpy(score).to(dtype=torch.float32, device=device)


class AnomalyScoringHead(nn.Module):
    """Thin dispatcher so trainers/eval.py can request any scoring mode
    through one interface, matching the design doc's "Supervised Mode
    (Parametric)" vs "Non-parametric Mode (K-Centroids)" split, plus the
    GMM alternative added alongside it."""

    def __init__(
        self, 
        feature_dim: int, 
        mode: str = "centroid", 
        k: int = 20, 
        hidden_dim: int = 256,
        gmm_mode: str = "nll",
        gmm_combined_weights: tuple = (0.5, 0.5),
        gmm_entropy_temperature: float = 5.0,
    ):
        super().__init__()
        self.mode = mode
        if mode == "parametric":
            self.impl: nn.Module = ParametricAnomalyHead(feature_dim, hidden_dim)
        elif mode == "centroid":
            self.impl = CentroidScoringHead(feature_dim, k)
        elif mode == "gmm":
            self.impl = GMMScoringHead(feature_dim, k, gmm_mode, gmm_combined_weights, gmm_entropy_temperature)
        else:
            raise ValueError(f"Unknown AnomalyScoringHead mode '{mode}' (use 'parametric', 'centroid', or 'gmm').")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.impl(z)