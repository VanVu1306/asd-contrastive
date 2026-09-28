"""
datasets/base_dataset.py
=========================
Abstract video loader. Every other dataset (ssl_dataset, supcon_dataset,
eval_dataset) subclasses `BaseVideoDataset` and only implements
`__getitem__` / the sampling logic around it — the actual "get me the raw
frames for clip X" part lives here so it's implemented once.

Supports two on-disk layouts, auto-detected per-path from the extension:
    - a single .mp4/.avi/... file           -> decoded with OpenCV
    - a directory of frames.jpg / frames.npy -> loaded with PIL / np.load

`frame_source` in the config can force one or the other ("video"/"frames")
if auto-detection would be ambiguous for a given data layout.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _read_video_file(path: str) -> np.ndarray:
    """Decode a video file to a (T, H, W, C) uint8 numpy array."""
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise OSError(f"Could not open video file: {path}")

    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"Video file contains no readable frames: {path}")
    return np.stack(frames, axis=0)


def _read_frame_folder(path: str) -> np.ndarray:
    """Load frame-folder metadata and decode frames only when indexed.

    Keeping a lazy sequence here avoids materializing thousands of full-size
    JPEGs before temporal transforms select a short training window.
    """
    if path.endswith(".npy"):
        arr = np.load(path)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    files = sorted(
        f for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
    )
    if not files:
        raise FileNotFoundError(f"No frame images found under {path}")

    return _LazyFrameSequence(path, files)


class _LazyFrameSequence:
    """Array-like frame folder that decodes only requested frame indices."""

    def __init__(self, path: str, files: List[str]):
        from PIL import Image

        self.path = path
        self.files = files
        with Image.open(os.path.join(path, files[0])) as image:
            height, width = image.height, image.width
        self.shape = (len(files), height, width, 3)

    def __getitem__(self, index):
        from PIL import Image

        if isinstance(index, (int, np.integer)):
            file_names = [self.files[int(index)]]
            scalar = True
        else:
            indices = np.arange(len(self.files))[index]
            indices = np.atleast_1d(indices)
            file_names = [self.files[int(i)] for i in indices]
            scalar = False

        frames = []
        for file_name in file_names:
            with Image.open(os.path.join(self.path, file_name)) as image:
                frames.append(np.array(image.convert("RGB")))
        if scalar:
            return frames[0]
        return np.stack(frames, axis=0)


def load_raw_frames(path: str, frame_source: str = "auto") -> np.ndarray:
    """Entry point used by every dataset subclass. Returns (T, H, W, C) uint8."""
    if frame_source == "video":
        return _read_video_file(path)
    if frame_source == "frames":
        return _read_frame_folder(path)

    # auto-detect
    ext = os.path.splitext(path)[1].lower()
    if ext in _VIDEO_EXTS:
        return _read_video_file(path)
    if ext == ".npy" or os.path.isdir(path):
        return _read_frame_folder(path)
    raise ValueError(f"Could not infer frame source for '{path}' — pass frame_source explicitly.")


def _resolve_frame_source(path: str, frame_source: str) -> str:
    if frame_source in ("video", "frames"):
        return frame_source
    ext = os.path.splitext(path)[1].lower()
    if ext in _VIDEO_EXTS:
        return "video"
    if ext == ".npy" or os.path.isdir(path):
        return "frames"
    raise ValueError(f"Could not infer frame source for '{path}' — pass frame_source explicitly.")


def _seek_capture(capture, start: int) -> None:
    """Seek an already-open cv2.VideoCapture to frame index `start`.

    OpenCV's CAP_PROP_POS_FRAMES seek is not reliably frame-accurate for
    every codec/container (some backends round to the nearest keyframe).
    Rather than trusting it blindly, read the position back after seeking
    and, if the backend under-shot, close the remaining gap with `grab()`
    calls — `grab()` advances the decoder's read pointer without allocating
    or returning a full decoded frame, so this stays cheap even when the
    gap is a few hundred frames.
    """
    import cv2

    if start <= 0:
        return
    capture.set(cv2.CAP_PROP_POS_FRAMES, float(start))
    actual = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    if actual < start:
        for _ in range(start - actual):
            if not capture.grab():
                break


def _read_video_window(path: str, start: int, num_frames: int) -> np.ndarray:
    """Seek directly to frame `start` and decode exactly `num_frames` frames.

    This is the memory-safe counterpart to `_read_video_file`: peak memory
    for one call is always O(num_frames), never O(video length) — the whole
    point for videos that can run to 13,000-20,000+ frames. Uses OpenCV
    (`cv2.VideoCapture`), the same decoder already used elsewhere in this
    module, just with a seek instead of a sequential full-file read.
    """
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise OSError(f"Could not open video file: {path}")

    frames = []
    try:
        _seek_capture(capture, start)
        while len(frames) < num_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"Could not read any frames starting at frame {start} from '{path}'.")

    window = np.stack(frames, axis=0)
    if window.shape[0] < num_frames:
        # Ran past EOF — only expected if a caller asks for a window that
        # runs off the end of the video (shouldn't happen with windows
        # planned via probe_num_frames, but stay defensive rather than
        # returning a ragged clip that breaks every downstream shape
        # assumption). Wrap around exactly like TemporalCrop's own
        # too-short-clip handling.
        reps = int(np.ceil(num_frames / window.shape[0]))
        window = np.tile(window, (reps, 1, 1, 1))[:num_frames]
    return window


def _read_frame_window_from_folder(path: str, start: int, num_frames: int) -> np.ndarray:
    """Same seek&read contract as `_read_video_window`, for `.npy` stacks and
    frame-image folders — only the requested rows/files are ever decoded or
    copied into RAM."""
    if path.endswith(".npy"):
        arr = np.load(path, mmap_mode="r")  # header-only; pixel data stays on disk until indexed
        total = arr.shape[0]
        idx = np.arange(start, start + num_frames) % total
        window = np.asarray(arr[idx])  # copies only the requested rows into RAM
        if window.dtype != np.uint8:
            window = np.clip(window, 0, 255).astype(np.uint8)
        return window

    files = sorted(
        f for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
    )
    if not files:
        raise FileNotFoundError(f"No frame images found under {path}")
    total = len(files)
    idx = np.arange(start, start + num_frames) % total
    seq = _LazyFrameSequence(path, files)
    return seq[idx]


def load_frame_window(path: str, start: int, num_frames: int, frame_source: str = "auto") -> np.ndarray:
    """Seek&read entry point: returns exactly `(num_frames, H, W, C)` uint8
    starting at frame `start`, decoding only those frames regardless of how
    long the source is. Use this (together with `probe_num_frames` to learn
    the source's length up front) instead of `load_raw_frames` whenever a
    dataset already knows *where* its window starts — this is what keeps
    memory flat whether the source video is 200 or 20,000 frames long.

    `start` is assumed in-bounds (`start + num_frames <= total_frames`),
    which is guaranteed by the window-planning callers use
    (`datasets/multi_window.py`, or a plain `random.randint(0, total -
    num_frames)`); a defensive wrap-around still applies if it isn't.
    """
    resolved = _resolve_frame_source(path, frame_source)
    if resolved == "video":
        return _read_video_window(path, start, num_frames)
    return _read_frame_window_from_folder(path, start, num_frames)


def frames_to_float_tensor(frames: np.ndarray) -> torch.Tensor:
    """(T, H, W, C) uint8 -> (T, C, H, W) float32 in [0, 1], the layout every
    spatial transform in transforms/spatial_transforms.py expects."""
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float() / 255.0
    return tensor


def probe_num_frames(path: str, frame_source: str = "auto") -> int:
    """Cheap frame-count lookup, used by datasets/multi_window.py to plan
    non-overlapping windows *without* decoding every frame of every source
    video just to find out how long it is:
        - .npy stack   : read only the header via mmap, never loads pixel data
        - frame folder : just counts files, no decoding
        - video file   : reads the frame count through OpenCV metadata,
                  falling back to sequential frame grabbing only if needed
    """
    ext = os.path.splitext(path)[1].lower()
    if frame_source == "frames" or (frame_source == "auto" and (ext == ".npy" or os.path.isdir(path))):
        if path.endswith(".npy"):
            return int(np.load(path, mmap_mode="r").shape[0])
        return len([f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in _IMAGE_EXTS])

    if frame_source == "video" or (frame_source == "auto" and ext in _VIDEO_EXTS):
        import cv2

        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise OSError(f"Could not open video file: {path}")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 0:
                return frame_count

            # Some codecs do not expose frame-count metadata. Grab frames
            # without decoding them into arrays as a low-memory fallback.
            frame_count = 0
            while capture.grab():
                frame_count += 1
            if frame_count == 0:
                raise ValueError(f"Video file contains no readable frames: {path}")
            return frame_count
        finally:
            capture.release()

    raise ValueError(f"Could not infer frame source for '{path}' — pass frame_source explicitly.")


class BaseVideoDataset(Dataset):
    """Shared plumbing: read a manifest (list of clip paths, one per line, or
    a subclass-provided richer manifest), and expose raw-frame loading.
    Subclasses implement `__getitem__`.
    """

    def __init__(self, root: str, split_file: str, frame_source: str = "auto"):
        self.root = root
        self.split_file = split_file
        self.frame_source = frame_source
        self.samples: List[str] = self._read_manifest(split_file)

    def _read_manifest(self, split_file: str) -> List[str]:
        if split_file is None or not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Split file '{split_file}' not found. Point data.train_split / "
                f"data.val_split at a real manifest before training."
            )
        with open(split_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        return lines

    def _resolve(self, rel_or_abs_path: str) -> str:
        return rel_or_abs_path if os.path.isabs(rel_or_abs_path) else os.path.join(self.root, rel_or_abs_path)

    def _load(self, rel_or_abs_path: str) -> np.ndarray:
        return load_raw_frames(self._resolve(rel_or_abs_path), self.frame_source)

    def _load_window(self, rel_or_abs_path: str, start: int, num_frames: int) -> np.ndarray:
        """Seek&read version of `_load`: decodes only `num_frames` frames
        starting at `start`, never the whole source. See `load_frame_window`
        in this module for the memory-safety rationale."""
        return load_frame_window(self._resolve(rel_or_abs_path), start, num_frames, self.frame_source)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):  # pragma: no cover - abstract
        raise NotImplementedError
