"""
utils/logger.py
=================
- `AverageMeter`: running mean of a scalar over a window (loss, GPU mem, ...).
- `MetricLogger`: fans a dict of metrics out to any combination of
  console / tensorboard / wandb / csv, selected via `logging.backend` in a
  config. tensorboard/wandb are imported lazily so a console-only run never
  needs those packages installed.
"""
from __future__ import annotations

import csv
import os
import time
from typing import Dict, Iterable, List, Optional

from utils.distributed import is_main_process


class AverageMeter:
    """Tracks a running average of a scalar (e.g. loss per step)."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


class MetricLogger:
    def __init__(
        self,
        log_dir: str,
        backends: Iterable[str] = ("console",),
        wandb_project: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        self.backends: List[str] = list(backends)
        self.log_dir = log_dir
        self._only_main = True  # non-main DDP ranks stay silent
        self._writer = None
        self._wandb = None
        self._csv_path = None
        self._csv_header_written = False

        if not is_main_process():
            return

        os.makedirs(log_dir, exist_ok=True)

        if "tensorboard" in self.backends:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=log_dir)

        if "wandb" in self.backends:
            import wandb

            wandb.init(project=wandb_project, name=run_name, dir=log_dir)
            self._wandb = wandb

        if "csv" in self.backends:
            self._csv_path = os.path.join(log_dir, "metrics.csv")

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if not is_main_process():
            return

        if "console" in self.backends:
            msg = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"[step {step}] {msg}", flush=True)

        if self._writer is not None:
            for k, v in metrics.items():
                self._writer.add_scalar(k, v, step)

        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

        if self._csv_path is not None:
            row = {"step": step, "time": time.time(), **metrics}
            write_header = not self._csv_header_written and not os.path.exists(self._csv_path)
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                    self._csv_header_written = True
                writer.writerow(row)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._wandb is not None:
            self._wandb.finish()
