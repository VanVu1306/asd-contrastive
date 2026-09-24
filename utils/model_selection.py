"""
utils/model_selection.py
===========================
BIC-based selection of K (number of centroids/components) for Stage 2
(Centroid Mining), as an alternative to a fixed, manually-chosen `k` in
configs/eval.yaml.

BIC (Bayesian Information Criterion) is native to the GMM branch — it's a
direct byproduct of `sklearn.mixture.GaussianMixture.bic()`, penalizing
extra components unless they improve the fit enough to justify their added
parameters. It is NOT native to K-Means (which has no likelihood to
penalize), so when `eval.centroids.method: "kmeans"`, this module still
picks K via a GMM sweep (a principled proxy for "how many behavior modes
does the normal data support") and hands that K to K-Means — this is a
deliberate, disclosed approximation, not a claim that K-Means itself
optimizes BIC.

Time-boxed by design: fitting one GMM per candidate K is cheap for the
small (tens-to-low-hundreds of clips) centroid-mining sets this project
expects, but nothing here assumes that — `select_k_by_bic()` checks the
wall-clock budget between fits and stops early (returning what it has so
far) rather than running an unbounded sweep. See scripts/select_k_bic.py
for the same search with no time limit, meant to be run manually/offline
when the inline budget isn't enough.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class BICSweepResult:
    best_k: Optional[int]              # None if not even one candidate finished in budget
    scored: List[Tuple[int, float]]      # (k, bic) for every candidate that finished, in order tried
    elapsed_sec: float
    timed_out: bool                        # True if the budget was hit before finishing all candidates


def select_k_by_bic(
    normal_features: torch.Tensor,
    k_candidates: Sequence[int],
    seed: int = 42,
    time_budget_sec: float = 60.0,
) -> BICSweepResult:
    """Fits a diagonal-covariance GMM for each K in `k_candidates` (skipping
    any K >= number of samples, which can't be fit), tracks each one's BIC,
    and returns the K with the lowest BIC — stopping early if
    `time_budget_sec` is exceeded partway through.
    """
    from sklearn.mixture import GaussianMixture

    x_np = F.normalize(normal_features, dim=-1).detach().cpu().numpy().astype(np.float64)
    n = x_np.shape[0]

    scored: List[Tuple[int, float]] = []
    start = time.time()
    timed_out = False

    for k in k_candidates:
        if k >= n:
            continue  # can't fit more components than samples
        if time.time() - start > time_budget_sec:
            timed_out = True
            break
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed, reg_covar=1e-6)
        gmm.fit(x_np)
        scored.append((k, float(gmm.bic(x_np))))

    elapsed = time.time() - start
    best_k = min(scored, key=lambda kv: kv[1])[0] if scored else None
    return BICSweepResult(best_k=best_k, scored=scored, elapsed_sec=elapsed, timed_out=timed_out)


def format_sweep_report(result: BICSweepResult) -> str:
    """Human-readable summary for logging / saving alongside a run."""
    lines = [f"BIC sweep over {len(result.scored)} candidate K value(s), {result.elapsed_sec:.1f}s elapsed:"]
    for k, bic in result.scored:
        marker = "  <- best" if k == result.best_k else ""
        lines.append(f"  K={k:>4d}  BIC={bic:.2f}{marker}")
    if result.timed_out:
        lines.append(
            f"  NOTE: stopped early at the {result.elapsed_sec:.1f}s time budget — "
            f"not every candidate K was tried. Run scripts/select_k_bic.py separately "
            f"(no time limit) for a complete sweep, then set eval.centroids.k manually."
        )
    return "\n".join(lines)
