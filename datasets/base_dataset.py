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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):  # pragma: no cover - abstract
        raise NotImplementedError
