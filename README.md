# Video Periodicity & Contrastive Learning Framework

A modular PyTorch framework for periodicity analysis, stimming/stereotypical-behavior detection, and anomaly scoring in video — built around three contrastive learning strategies that share the same backbones, data pipeline, and evaluation tooling:

- **Self-Supervised Learning (SSL):** MoCo (Momentum Contrast) over temporal augmentations (Speed Warp + Frame Shuffle) — no labels needed.
- **Supervised Contrastive Learning (SupCon):** representation learning driven by behavior-group labels (`1 = ASD/stimming`, `0 = non-ASD`).
- **SPI (Synthetic Periodicity Injection):** a third, independent SSL branch that manufactures clips with a *known* period and trains on periodic-vs-non-periodic pseudo-labels — see [SPI](#spi-synthetic-periodicity-injection) below. Fully additive: it shares infrastructure with (but never modifies) the other two.

All three feed into two downstream tasks: **Action Classification** (linear probing / fine-tuning) and **Anomaly Scoring**, via a non-parametric K-Centroids scorer that needs zero backprop at inference time.

## Repository Structure

```text
video-periodicity-contrastive/
├── configs/
│   ├── base.yaml              # backbone, clip geometry, hardware, seed — shared by everything else
│   ├── ssl_moco.yaml          # SSL: queue size, temperature, speed-warp rates
│   ├── supcon.yaml            # SupCon: temperature, class-balancing knobs
│   ├── spi_periodicity.yaml   # SPI: cycle-duration range, repeat count, optional MoCo combination
│   └── eval.yaml              # eval: mode, checkpoint path, centroid/probe settings
├── datasets/
│   ├── base_dataset.py        # abstract video loader (.mp4 via torchvision.io, or frame folders / .npy)
│   ├── multi_window.py        # segment-based (TSN-style) low/no-overlap multi-clip-per-video planning
│   ├── samplers.py            # VideoDiverseBatchSampler — caps same-video clips per batch, DDP-aware
│   ├── ssl_dataset.py         # -> (x_anchor, x_pos_warp, x_neg_shuffle)
│   ├── supcon_dataset.py      # -> (x_clip, label, video_id) + a video-id-balanced batch sampler
│   ├── spi_dataset.py         # -> (x_periodic_a, x_periodic_b, x_nonperiodic, period_label)
│   └── eval_dataset.py        # centroid-mining loader + sliding-window test loader
├── transforms/
│   ├── spatial_transforms.py  # RandomResizedCrop/Flip/ColorJitter/RandomErasing, shared across a clip's frames
│   └── temporal_transforms.py # TemporalCrop, SpeedWarp, FrameShuffle, SlidingWindow, SyntheticPeriodicityInjection
├── models/
│   ├── backbones/             # resnet3d_18/50, r2plus1d_18, s3d, video_swin_t + builder factory
│   ├── heads.py               # ProjectionHead, ClassificationHead, PeriodRegressionHead, AnomalyScoringHead
│   └── moco_wrapper.py        # query/key encoders, momentum update, FIFO memory queue, optional shared backbone
├── losses/
│   ├── supcon_loss.py         # SupCon with a cross-video-only positive mask
│   ├── moco_nce_loss.py       # InfoNCE over (positive, queue negatives, hard negative)
│   └── spicon_loss.py         # SupCon-style loss over periodic/non-periodic pseudo-labels + period regression
├── trainers/
│   ├── base_trainer.py        # DDP setup, optimizer/checkpoint/logging/seeding plumbing
│   ├── ssl_trainer.py         # MoCo training loop
│   ├── supcon_trainer.py      # SupCon training loop
│   └── spi_trainer.py         # SPI training loop (standalone, or combined with MoCo on a shared backbone)
├── utils/
│   ├── config.py              # YAML loader with `_base_` inheritance + CLI overrides
│   ├── seed.py                # set_seed(), DataLoader worker_init_fn, optional cudnn determinism
│   ├── distributed.py         # all_gather, SyncBN, MoCo shuffling-BN, DDP setup/teardown
│   ├── lr_scheduler.py        # cosine annealing w/ linear warmup, plain cosine, step
│   ├── logger.py              # AverageMeter + console/tensorboard/wandb/csv logging
│   ├── metrics.py             # AUC-ROC, EER, FAR-ratio, classification metrics
│   └── visualize.py           # anomaly-score-vs-ground-truth and ROC plots
├── scripts/
│   └── make_dummy_data.py     # generates a synthetic dataset for smoke-testing the whole pipeline
├── eval.py                    # evaluation entry point (anomaly_scoring | linear_probing)
├── train.py                   # training entry point (ssl_moco | supcon | spi_periodicity)
└── requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

`tensorboard`/`wandb` are only needed if selected in a config's `logging.backend`, and `av` is only needed if your clips are `.mp4`/`.avi` files rather than frame folders or `.npy` stacks.

## Input tensor convention

Every backbone consumes `X ∈ R^{B×C×T×H×W}` — batch, RGB channels, temporal frames, height, width. All datasets and transforms in this repo produce/consume that layout consistently, so swapping a backbone never requires touching the data pipeline.

## Data format

Point `data.root` at wherever clips live, and each split file at a manifest relative to that root:

| Use case                         | Manifest format                                                                                                             |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| SSL training (unlabeled)         | plain text, one clip path per line                                                                                          |
| SupCon training / linear probing | CSV:`clip_path,video_id,label`                                                                                            |
| Centroid mining (normal-only)    | plain text, one clip path per line                                                                                          |
| Anomaly-scoring test set         | CSV:`clip_path,video_id,frame_labels_path` (labels optional — omit the column/value if you only want scores, no metrics) |

A "clip path" can be an `.mp4`/`.avi`/etc. file, a directory of `.jpg`/`.png` frames, or a single `.npy` stack shaped `(T, H, W, C)` — auto-detected per path (`datasets/base_dataset.py`).

## Usage

### 1. Training

```bash
# Self-Supervised Learning (MoCo + temporal Speed-Warp/Frame-Shuffle)
python -m torch.distributed.run --nproc_per_node=gpu train.py --config configs/ssl_moco.yaml

# Supervised Contrastive Learning (SupCon)
python -m torch.distributed.run --nproc_per_node=gpu train.py --config configs/supcon.yaml

# SPI (Synthetic Periodicity Injection) — standalone by default
python -m torch.distributed.run --nproc_per_node=gpu train.py --config configs/spi_periodicity.yaml
```

Single-GPU or CPU: drop the `torch.distributed.run` wrapper and just run `python train.py --config ...` — every trainer degrades gracefully to single-process (see `utils/distributed.py`).

Resume from a checkpoint: `--resume runs/<experiment_name>/ckpt_last.pth`.

Override any config value from the CLI without editing YAML:

```bash
python train.py --config configs/ssl_moco.yaml --opts optim.lr=0.01 optim.epochs=50
```

### 2. Evaluation (`eval.py`)

```bash
# Anomaly Scoring: Stage 2 (K-Centroid mining) + Stage 3 (sliding-window scoring),
# reports frame-level AUC-ROC / EER / FAR-ratio.
python eval.py --config configs/eval.yaml --mode anomaly_scoring

# Linear Probing: freeze the pretrained encoder, train only ClassificationHead.
python eval.py --config configs/eval.yaml --mode linear_probing
```

Both modes read `checkpoint_path` from `configs/eval.yaml` and work against a checkpoint from *any* trainer — `eval.py` auto-detects whether the checkpoint's backbone weights live under `encoder_q.backbone.*` (MoCo), `backbone.*` (SupCon), or `spi.backbone.*` / `moco.encoder_q.backbone.*` (SPI, standalone or combined mode).

## Pipelines at a glance

### SSL (MoCo) pretraining

```
raw clip (T_raw frames)
   ├── TemporalCrop -> x_anchor      -> SpatialAugment_A -> encoder_q -> q
   ├── SpeedWarp     -> x_pos_warp    -> SpatialAugment_B -> encoder_k -> k        (EMA, shuffled-BN)
   └── FrameShuffle  -> x_neg_shuffle -> SpatialAugment_A -> encoder_k -> k_neg    (hard negative)

InfoNCE( q against [k | FIFO queue | k_neg] )  ->  backward through encoder_q only
encoder_k <- momentum * encoder_k + (1 - momentum) * encoder_q   (no gradient)
k          -> pushed into the FIFO queue, oldest entries dequeued
```

### SupCon pretraining

```
labeled clip -> TemporalCrop -> SpatialAugment -> encoder (backbone+ProjectionHead) -> z

Positive pairs (i, j) in a batch:  same label  AND  different video_id
Negative pairs: everything else (in particular, any ASD/non-ASD pair)

GroupBalancedBatchSampler spreads video_ids per class across each batch so cross-video positives actually exist for SupConLoss to use.
```

### Downstream: Action Classification

```
clip -> encoder.backbone (frozen, or small LR if fine-tuning)
     -> BatchNorm1d(affine=False)   # normalizes arbitrary feature scale before...
     -> ClassificationHead -> CrossEntropyLoss
```

### Downstream: Anomaly Scoring (non-parametric K-Centroids)

```
Stage 1 (already done by SSL pretraining): encoder learns continuity/periodicity.

Stage 2 — Centroid Mining (single pass, zero backprop):
    normal-only clips -> frozen encoder -> Z_normal (N x D)
    K-Means (K≈10~50) over L2-normalized Z_normal -> K centroids -> centroids.pth

Stage 3 — Inference (sliding window over a full test video):
    x_test -> frozen encoder -> z
    S_raw(x) = min_k ( 1 - cos_sim(z, c_k) )
    per-window scores -> overlap-averaged onto a per-frame curve -> smoothing
    -> frame-level AUC-ROC / EER / FAR-ratio (if ground-truth labels are available)
```

### SPI (Synthetic Periodicity Injection)

A third, independent SSL pretext branch. Instead of relying on a real periodic action being present in the raw footage, it *manufactures* one with a known period, giving a free pseudo-label with no manual annotation:

```
raw clip (T_raw frames)
   └── cut a short L-frame segment, repeat it N times (N>=3),
       each repeat independently speed-/color-jittered (mandatory, not optional —
       see SyntheticPeriodicityInjection's docstring for why)
       -> synthetic periodic_clip, L*N frames, true period = L

   ├── window at repeat-unit phase A -> x_periodic_phaseA -> SpatialAugment_A -> encoder -> z_a  (pseudo-label 1)
   ├── window at repeat-unit phase B -> x_periodic_phaseB -> SpatialAugment_B -> encoder -> z_b  (pseudo-label 1)
   └── plain continuous crop of the SAME raw window -> x_nonperiodic -> SpatialAugment_C -> encoder -> z_np  (pseudo-label 0)

SPIConLoss = SupConLoss(z_a, z_b, z_np; pseudo-labels)   # same mechanism as supcon_loss.py
                        + period_loss_weight * SmoothL1(log P̂, log L)   # optional, via PeriodRegressionHead
```

Standalone by default (`python train.py --config configs/spi_periodicity.yaml`) — this is the mode that's been fully end-to-end tested. It can optionally also train a full MoCo/InfoNCE objective on a **shared backbone** in the same step (`multi_task.moco_loss_weight > 0` in `configs/spi_periodicity.yaml`); this combined mode runs and checkpoints correctly (smoke-tested — see `models/moco_wrapper.MoCoWrapper`'s `shared_backbone` parameter) but its joint loss-weighting dynamics haven't been tuned/validated the way the standalone path has, so treat it as an experimental option rather than a validated recipe.

Before trusting SPI's representation for Stage 2 Centroid Mining, validate it the same way as the MoCo branch: run `eval.py --mode linear_probing` against a SPI checkpoint and confirm the learned periodicity axis actually correlates with real ASD/non-ASD labels, since SPI never sees real labels during pretraining.

## Data density & anti-leakage (multi-window sampling)

For small datasets, one clip per source video badly underuses the available footage and can make a MoCo queue dominated by near-duplicate, false-negative encodings of the very clips currently being trained on. `data.clips_per_video: N`(any of `ssl_moco.yaml` / `supcon.yaml` / `spi_periodicity.yaml`) turns "1 window per video" into "N low/no-overlap windows per video", placed via a segment partition (`datasets/multi_window.py`, the sampling strategy from Temporal Segment Networks): split the video into N contiguous segments and take one window per segment, so overlap between two windows of the same video is zero whenever the video is long enough, and the *minimum physically possible* otherwise — never more, and never out of bounds (this is planned from each video's frame count, not by trial and error).

This alone doesn't stop several of a video's N windows from landing in the same training batch, which is exactly the situation a contrastive loss can exploit via shared background/appearance instead of learning real motion content. Two complementary defenses:

- `data.max_per_video_per_batch: 1` (SSL/SPI) — `datasets/samplers.VideoDiverseBatchSampler` caps how many clips from one video can land in a batch (DDP-aware rank-sharding built in). SupCon's own `GroupBalancedBatchSampler` already guarantees this implicitly (one pop per video-id bucket per batch), so it has no separate knob.
- `spatial_transform.random_erasing_prob: 0.25-0.5` — a temporally-consistent cutout (`transforms/spatial_transforms.RandomErasingVideo`: same erased box across every frame of one clip, different per clip) as extra decorrelation between same-video clips that still end up sharing a batch. Off by default.

Both are off/uncapped by default, so existing configs behave identically unless opted in — enabling `clips_per_video` logs a one-line summary (`[SSLDataset] 45 videos -> 168 windows ...; 12/45 videos (26.7%) were too short to avoid overlap entirely`) so you can tell whether the setting is well-matched to your clip lengths.

## Reproducibility

`seed:` in any config (inherited from `base.yaml`) governs every source of randomness in this repo — `utils/seed.py`'s `set_seed()` covers Python's `random`, numpy, and torch (CPU + CUDA); every `DataLoader` also gets `worker_init_fn` from the same module, since PyTorch's own default worker seeding only reseeds torch's RNG, not `random`/numpy — which is what almost every augmentation in `transforms/`actually uses. `hardware.deterministic: true` additionally forces cudnn onto deterministic (slower) algorithms, for bit-exact GPU reproducibility; leave it `false` (the default) for normal training. Verified end-to-end: identical seed + identical `num_workers` (0 or >0) reproduces bit-exact loss curves across repeated runs, for every trainer and `eval.py`.

## Extending the framework

The design goal throughout is "swap one piece, nothing else breaks":

- **New backbone:** add a wrapper in `models/backbones/`, register it in `models/backbones/builder.py`'s `_REGISTRY`. Every trainer/eval script picks it up automatically via `backbone.name` in the config.
- **New contrastive method:** add a `losses/your_loss.py` and a `trainers/your_trainer.py` following the `BaseTrainer` pattern (implement `build_dataloader` / `build_model_and_loss` / `train_step`); register its mode string in `train.py`'s `_TRAINERS` dict.
- **New augmentation:** add it to `transforms/spatial_transforms.py` or `temporal_transforms.py` and wire it into the relevant dataset's `__init__`.
- **New input modality (keypoints / optical flow):** implement a new `datasets/*_dataset.py` that yields the same `(B, C, T, H, W)`-per-sample convention (or a modality-appropriate equivalent) — the backbones/heads/losses don't need to know where the tensor came from.

## Verifying your setup

`scripts/make_dummy_data.py` generates a small random-noise dataset in every manifest format the repo expects, so you can confirm your environment/install is wired correctly before pointing it at real data:

```bash
python scripts/make_dummy_data.py --out ./data --frame-size 32

python train.py --config configs/ssl_moco.yaml --opts \
    data.root=./data data.train_split=./data/splits/unlabeled_train.txt \
    data.clip_len=8 data.raw_clip_len=20 data.frame_size=32 \
    optim.batch_size=4 optim.epochs=2 hardware.device=cpu hardware.amp=false
```

Since the data is random noise, loss values and metrics from this smoke test are meaningless as *research* results — they only confirm that every shape, configpath, and training/eval loop actually runs. Swap in `configs/supcon.yaml` or `configs/spi_periodicity.yaml` and `eval.py --mode anomaly_scoring|linear_probing` (pointing `checkpoint_path` at the checkpoint the smoke-test training run just produced) to exercise the rest of the pipeline the same way. Adding `--opts seed=123` (or any fixed value) to two otherwise-identical runs is also a quick way to confirm reproducibility end-to-end in your own environment.

## Notes on the design doc's proposal

- `resnet3d_50` isn't a torchvision-provided builder; it's assembled here from torchvision's own `VideoResNet`/`Bottleneck`/`Conv3DSimple` building blocks with the standard ResNet-50 stage depths `[3, 4, 6, 3]`.
- `video_swin_t` wraps torchvision's `swin3d_t`; `swin3d_s`/`swin3d_b` are one line away in `models/backbones/video_swin.py` if a config ever needs more capacity.
- Pretrained (ImageNet/Kinetics) weights require network access to torchvision's weight URLs at first use; set `backbone.pretrained_path` to a local checkpoint instead if you're training somewhere offline.
- `x_neg_shuffle` (the frame-shuffled SSL view) is used as an explicit per-sample hard negative appended to MoCo's InfoNCE denominator (`models/moco_wrapper.py`, `losses/moco_nce_loss.py`) — it shares every frame with the anchor and differs only in temporal order, which is a much harder negative than a random queue entry and pushes the encoder toward representing motion, not just appearance.
