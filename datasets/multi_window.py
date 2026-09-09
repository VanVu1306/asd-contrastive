"""
datasets/multi_window.py
==========================
Turns "1 manifest row per source video" into "N clips per source video",
placed to overlap as little as possible — this is what backs
`data.clips_per_video` in configs/ssl_moco.yaml / configs/supcon.yaml.

Why not just call TemporalCrop(random_start=True) N times per video? Because
nothing then stops two of those N random windows from landing on almost the
same frames — with a short video and a large N, heavy overlap is the likely
outcome, not the exception. Instead this uses a **segment partition**
(the sampling strategy from Temporal Segment Networks, Wang et al. 2016):
split the video into `num_windows` contiguous, non-overlapping segments and
place exactly one window per segment. Two windows can only ever overlap if a
single video is too short to fit `num_windows` windows of length
`window_len` side by side — and even then, the overlap is the *minimum*
physically possible, not a matter of luck.

This module only computes *where* the windows go (a list of start-frame
indices per video); it does not read pixel data — `probe_num_frames()` in
base_dataset.py gets each video's length cheaply (header/metadata only) so
planning windows for a whole manifest is fast even for large datasets.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from datasets.base_dataset import probe_num_frames


@dataclass
class WindowPlan:
    """One planned clip: which source row it comes from, and where."""
    source_index: int    # index into the original (pre-expansion) manifest/row list
    start: int             # first (nominal) frame index of this window
    jitter_radius: int      # +/- frames this window's start may move at
                              # __getitem__ time without crossing into a
                              # neighboring window's slot or going out of
                              # bounds — see jitter_within_segment below.


def build_segment_windows(num_frames: int, window_len: int, num_windows: int) -> List[Tuple[int, int]]:
    """Places `num_windows` windows of `window_len` frames each, spread as
    evenly as possible across [0, num_frames). Returns (start, jitter_radius)
    pairs — jitter_radius is how far `jitter_within_segment` may nudge that
    window's start (each way) while keeping it in-bounds and near its slot.

    Every window is always fully in-bounds (start + window_len <= num_frames)
    — this only degrades to *controlled* overlap between neighboring windows
    when the video is too short to fit `num_windows` of them side by side; it
    never lets a window run off the end of the video.
    """
    if num_windows <= 1 or num_frames <= window_len:
        return [(0, 0)]  # nothing to spread; caller's own TemporalCrop/loop
                            # logic already handles a too-short clip.

    valid_range = num_frames - window_len  # last legal start position
    if num_windows == 1:
        starts = [valid_range // 2]
    else:
        starts = [round(i * valid_range / (num_windows - 1)) for i in range(num_windows)]

    spacing = valid_range / (num_windows - 1)
    windows = []
    for i, start in enumerate(starts):
        # Half the gap to each neighbor (or to the array bounds at the ends)
        # is how far this window can jitter without crossing into a
        # neighbor's slot or going out of bounds.
        left_room = spacing / 2 if i > 0 else start
        right_room = spacing / 2 if i < len(starts) - 1 else (valid_range - start)
        jitter_radius = int(min(left_room, right_room))
        windows.append((start, jitter_radius))
    return windows


def expand_manifest(
    rows: Sequence,
    resolve_path,
    frame_source: str,
    window_len: int,
    clips_per_video: int,
) -> Tuple[List[WindowPlan], dict]:
    """rows: the original manifest (list of paths, or list of (path, ...) tuples —
    this function only needs `resolve_path(row) -> filesystem path` to probe length).

    Returns (plans, stats). `stats` reports how many of the resulting windows
    had to overlap because their source video was too short for
    `clips_per_video` non-overlapping windows of `window_len` frames — worth
    checking once per dataset, since a high overlap rate means
    `clips_per_video` is set too high for your clip lengths.
    """
    plans: List[WindowPlan] = []
    forced_overlap_videos = 0

    for source_index, row in enumerate(rows):
        path = resolve_path(row)
        try:
            num_frames = probe_num_frames(path, frame_source)
        except Exception as e:
            raise RuntimeError(f"Could not probe frame count for '{path}': {e}") from e

        segments = build_segment_windows(num_frames, window_len, clips_per_video)
        if num_frames < clips_per_video * window_len:
            forced_overlap_videos += 1  # not enough room to avoid overlap entirely

        for start, jitter_radius in segments:
            plans.append(WindowPlan(source_index=source_index, start=start, jitter_radius=jitter_radius))

    stats = {
        "num_source_videos": len(rows),
        "num_expanded_windows": len(plans),
        "videos_with_forced_overlap": forced_overlap_videos,
        "forced_overlap_ratio": forced_overlap_videos / max(1, len(rows)),
    }
    return plans, stats


def jitter_within_segment(start: int, jitter_radius: int, num_frames: int, window_len: int) -> int:
    """Adds epoch-to-epoch randomness back on top of a fixed window slot:
    nudges `start` by up to +/- `jitter_radius` frames, so repeated epochs
    don't always sample the exact same frames for a given window index —
    while `jitter_radius` (from build_segment_windows) already guarantees
    the result can't cross into a neighboring window's slot or run past the
    video's bounds, so no extra clamping is needed here for correctness;
    the clamp against [0, num_frames - window_len] is only a defensive
    fallback in case this is called with a hand-built plan.
    """
    if jitter_radius <= 0:
        return start
    jittered = start + random.randint(-jitter_radius, jitter_radius)
    return max(0, min(jittered, num_frames - window_len))