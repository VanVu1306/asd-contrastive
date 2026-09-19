"""
transforms/temporal_transforms.py
==================================
Transforms that operate on the *time axis* of a raw clip, i.e. they choose
which frame indices (out of a longer raw window) end up in the final T-frame
tensor, and in what order/spacing.

All transforms here work on a numpy array of frames with shape
(T_raw, H, W, C) — uint8, values 0-255 — and index-based logic. Pixel-level
(spatial) augmentation happens afterwards, in spatial_transforms.py, so these
classes never touch pixel values.

Used by:
    - datasets/ssl_dataset.py to build (x_anchor, x_pos_warp, x_neg_shuffle)
    - datasets/eval_dataset.py (TemporalCrop only, for sliding-window clips)
    - datasets/spi_dataset.py (SyntheticPeriodicityInjection) — a third,
      independent SSL pretext branch; see that module's docstring
"""
from __future__ import annotations

import atexit
import csv
import os
import random
from collections import Counter
from typing import Optional, Sequence, Tuple

import numpy as np


class TemporalCrop:
    """Sample `clip_len` contiguous frame indices out of a longer raw clip.

    If the raw clip is shorter than clip_len, frames are looped (wrapped
    around) rather than padded with black frames, so downstream backbones
    always see a full-length, motion-bearing clip.
    """

    def __init__(self, clip_len: int, random_start: bool = True):
        self.clip_len = clip_len
        self.random_start = random_start

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        t_raw = frames.shape[0]
        if t_raw >= self.clip_len:
            max_start = t_raw - self.clip_len
            start = random.randint(0, max_start) if self.random_start else max_start // 2
            idx = np.arange(start, start + self.clip_len)
        else:
            # wrap-around looping for short raw clips
            idx = np.arange(self.clip_len) % t_raw
        return frames[idx]


class SpeedWarp:
    """Resample the clip at a different playback speed, then crop/pad back to
    `clip_len` frames. This is the SSL "positive" temporal transform: a
    genuinely periodic action still looks periodic when sped up or slowed
    down, just at a different frequency — a property a plain frame shuffle
    would not preserve.

    rate < 1.0  -> slow motion (stretches T_raw over more output frames,
                   effectively sub-sampling less densely)
    rate > 1.0  -> fast forward (skips frames)
    """

    def __init__(self, clip_len: int, rates: Sequence[float] = (0.8, 1.5)):
        self.clip_len = clip_len
        self.rates = rates

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        t_raw = frames.shape[0]
        rate = random.choice(self.rates)

        # Indices into the raw clip, spaced `rate` apart, starting from a
        # random offset so the same rate doesn't always sample the same
        # phase of a periodic motion.
        span_needed = self.clip_len * rate
        max_start = max(0, t_raw - span_needed)
        start = random.uniform(0, max_start) if max_start > 0 else 0.0
        raw_idx = start + np.arange(self.clip_len) * rate
        raw_idx = np.clip(raw_idx, 0, t_raw - 1).astype(np.int64)
        return frames[raw_idx]


class FrameShuffle:
    """Randomly permute a fraction of the frame *positions* in the clip.
    This is the SSL "negative"/hard-negative temporal transform: it destroys
    the temporal ordering (and hence any periodic structure) while keeping
    the exact same set of frames, so a shortcut based on appearance alone
    can't tell the anchor and the shuffled view apart — the model is forced
    to encode motion/order, not just static content.
    """

    def __init__(self, ratio: float = 1.0):
        # ratio=1.0 -> fully shuffled; lower ratio only permutes a random
        # subset of positions and leaves the rest in their original order.
        self.ratio = ratio

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        t = frames.shape[0]
        n_shuffle = max(2, int(round(t * self.ratio)))
        chosen = np.sort(np.random.choice(t, size=n_shuffle, replace=False))

        permuted = chosen.copy()
        while True:
            np.random.shuffle(permuted)
            # guard against an identity permutation leaving the "negative"
            # view identical to the anchor's frame order on the chosen slots
            if n_shuffle <= 1 or not np.array_equal(permuted, chosen):
                break

        # positions[i] = which original frame index ends up at output slot i
        positions = np.arange(t)
        positions[chosen] = permuted
        return frames[positions]


class SlidingWindow:
    """Non-overlapping-by-`stride` sliding window over a *full* video's
    frames, used only at inference/evaluation time (eval_dataset.py) so every
    frame gets an anomaly score, not just one random crop per video.
    """

    def __init__(self, clip_len: int, stride: int):
        self.clip_len = clip_len
        self.stride = stride

    def window_starts(self, t_total: int) -> np.ndarray:
        if t_total <= self.clip_len:
            return np.array([0])
        last_start = t_total - self.clip_len
        starts = np.arange(0, last_start + 1, self.stride)
        if starts[-1] != last_start:
            starts = np.append(starts, last_start)
        return starts

    def __call__(self, frames: np.ndarray, start: int) -> np.ndarray:
        return frames[start : start + self.clip_len]


class SyntheticPeriodicityInjection:
    """Synthesizes a clip with a *known* period from a raw window, by cutting
    a short segment of `L` frames and repeating it `N` times — the "SPI"
    pretext task from the proposal doc: this operator itself is neither a
    positive nor a negative, it just manufactures periodic content with a
    free pseudo-label (P = L). Positive/negative pairing happens one layer
    up, in datasets/spi_dataset.py.

    Jitter every repeat is mandatory, not optional: identical repeated
    frames would let a contrastive loss "solve" the positive-pair task with
    literal pixel matching instead of learning to recognize periodicity —
    exactly the shortcut this pretext task exists to avoid.

    Optional fixed_L / fixed_N values preserve the legacy random defaults
    when the input is None, and can be used for controlled experiments
    without modifying the current sampling logic elsewhere.
    """

    def __init__(
        self,
        cycle_len_range: Tuple[int, int] = (6, 20),
        n_repeats_range: Tuple[int, int] = (3, 5),
        speed_jitter: float = 0.05,
        color_jitter_strength: float = 0.1,
        fixed_L: Optional[int] = None,
        fixed_N: Optional[int] = None,
        stats_path: Optional[str] = None,
    ):
        if speed_jitter <= 0.0 and color_jitter_strength <= 0.0:
            # The doc is explicit that jitter is mandatory — rather than
            # silently degrading into the degenerate case, fall back to a
            # small default so every repeat is still guaranteed distinct.
            color_jitter_strength = 0.02
        self.cycle_len_range = (max(2, cycle_len_range[0]), max(3, cycle_len_range[1]))
        self.n_repeats_range = (max(1, n_repeats_range[0]), max(n_repeats_range[0], n_repeats_range[1]))
        self.speed_jitter = speed_jitter
        self.color_jitter_strength = color_jitter_strength

        # Non-breaking experiment hooks: keep legacy random behavior when
        # fixed_L/fixed_N are left as None.
        self.fixed_L = fixed_L
        self.fixed_N = fixed_N

        # Distribution bookkeeping for L and N sampled over the run.
        self.l_counts = Counter()
        self.n_counts = Counter()
        self.pair_counts = Counter()
        self.stats_path = stats_path

        if self.stats_path is not None:
            # Register a write-on-exit hook so the user can inspect the
            # actual distribution of L/N without having to log every sample.
            atexit.register(self.dump_stats)

    def sample_periodicity_params(self, t_raw: int) -> Tuple[int, int]:
        """Choose L and N, with fixed-value override when configured.

        This preserves the original random path by default and only swaps in
        fixed values when users set `spi.fixed_L` or `spi.fixed_N` in config.
        """
        lo, hi = self.cycle_len_range
        effective_hi = min(hi, t_raw)

        if self.fixed_L is not None:
            L = int(self.fixed_L)
        elif effective_hi < lo:
            # Raw window shorter than even the minimum configured cycle
            # length — nothing sensible to sample within range; use
            # everything available rather than requesting more frames than
            # the video has (which would silently under-fill the segment).
            L = t_raw
        else:
            L = random.randint(lo, effective_hi)

        if self.fixed_N is not None:
            N = int(self.fixed_N)
        else:
            N = random.randint(*self.n_repeats_range)

        return L, N

    def update_stats(self, L: int, N: int) -> None:
        """Update distribution counters for a sampled L/N pair."""
        self.l_counts[L] += 1
        self.n_counts[N] += 1
        self.pair_counts[(L, N)] += 1

    def dump_stats(self, path: Optional[str] = None) -> None:
        """Persist sampled L/N distributions in three CSV views:
            - pair_counts: rows (L, N, count)
            - L_counts: rows (L, count)
            - N_counts: rows (N, count)

        If no output path is supplied, use the configured stats_path; if
        the caller never supplies a path, the method is a no-op.
        """
        out_path = path or self.stats_path
        if out_path is None:
            return

        base, ext = os.path.splitext(out_path)
        pair_path = out_path
        l_path = f"{base}_L{ext}"
        n_path = f"{base}_N{ext}"

        try:
            os.makedirs(os.path.dirname(pair_path) or ".", exist_ok=True)
        except Exception:
            pass

        with open(pair_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["L", "N", "count"])
            for (L, N), count in sorted(self.pair_counts.items()):
                writer.writerow([L, N, count])

        with open(l_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["L", "count"])
            for L, count in sorted(self.l_counts.items()):
                writer.writerow([L, count])

        with open(n_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["N", "count"])
            for N, count in sorted(self.n_counts.items()):
                writer.writerow([N, count])

    def __call__(self, frames: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """frames: (T_raw, H, W, C) uint8. Returns (periodic_clip, L, N)
        where periodic_clip has exactly L*N frames."""
        t_raw = frames.shape[0]
        L, N = self.sample_periodicity_params(t_raw)
        self.update_stats(L, N)

        max_start = max(0, t_raw - L)
        start = random.randint(0, max_start)
        base_segment = frames[start : start + L]

        repeats = []
        for _ in range(N):
            rep = base_segment
            if self.speed_jitter > 0:
                # Independent per-repeat playback-rate jitter: resample this
                # repeat's L frames from the base segment at a slightly
                # different rate, so consecutive cycles are never
                # frame-for-frame identical.
                rate = 1.0 + random.uniform(-self.speed_jitter, self.speed_jitter)
                idx = np.clip((np.arange(L) * rate).astype(np.int64), 0, L - 1)
                rep = rep[idx]
            if self.color_jitter_strength > 0:
                # One shared brightness factor per repeat (not per frame) —
                # this is what breaks pixel-identical cycles while keeping
                # every frame *within* a repeat internally consistent.
                factor = 1.0 + random.uniform(-self.color_jitter_strength, self.color_jitter_strength)
                rep = np.clip(rep.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            repeats.append(rep)

        periodic_clip = np.concatenate(repeats, axis=0)  # (L * N, H, W, C)
        return periodic_clip, L, N