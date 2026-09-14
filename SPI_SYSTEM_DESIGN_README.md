# SPI System Design Readme

This document explains the current SPI branch implementation in the repository, following the code paths in:

- `datasets/spi_dataset.py`
- `transforms/temporal_transforms.py`
- `transforms/spatial_transforms.py`
- `trainers/spi_trainer.py`
- `losses/spicon_loss.py`
- `models/heads.py`
- `models/moco_wrapper.py`

The design here is a standalone SPI pretext branch that can optionally be combined with a MoCo branch through a weighted multi-task loss. The default config shipped in the repo is the standalone mode, where `multi_task.moco_loss_weight = 0.0`.

---

## 1. High-level goal

The SPI branch synthesizes a pseudo-periodic clip from a raw temporal window by:

1. picking a cycle length `L` in frames,
2. sampling a contiguous base segment of `L` frames,
3. repeating that base segment `N` times with independent per-repeat jitter,
4. building a positive pair from two phase offsets of that synthetic periodic clip,
5. building a negative pair from a plain non-periodic crop of the same raw window,
6. teaching the encoder to separate periodic views from non-periodic views with a SupCon-like contrastive loss, and optionally regressing the cycle length `log(L)`.

This is different from MoCo/SSL, which builds positive/negative views through temporal warp and frame shuffle, and uses a queue-based InfoNCE loss.

---

## 2. Training mode choices

The repository supports three operating patterns in practice, all expressed by the same `SPITrainer` and configurable by `multi_task` and the `spi`/`moco` blocks:

### 2.1 Standalone SPI mode

This is the default configuration in the shipped file `configs/spi_periodicity.yaml`:

```yaml
multi_task:
  moco_loss_weight: 0.0
  spi_loss_weight: 1.0
```

In this case:

- the model creates only `SPIEncoder` and only one `SPIConLoss` object;
- no `MoCoWrapper` object is created;
- no memory queue is allocated;
- the loss is purely:

```python
loss = spi_loss_weight * SPIConLoss(features, pseudo_labels, pred_log_period, true_log_period)
```

This is the cleanest source of truth when you want to evaluate the SPI branch by itself.

### 2.2 Joint SPI + MoCo mode

This is the optional extension of SPI into a multi-task objective:

```yaml
multi_task:
  moco_loss_weight: > 0.0
  spi_loss_weight: 1.0
```

In this case:

- the same backbone is reused as a shared feature extractor (`shared_backbone`);
- `SPIEncoder` is built first;
- `MoCoWrapper` is created by passing `shared_backbone=spi_encoder.backbone`;
- the model contains two modules: `{"spi": spi_encoder, "moco": moco_wrapper}`;
- the loss is:

```python
loss = spi_loss_weight * SPIConLoss(...) + moco_loss_weight * MoCoNCELoss(...)
```

This requires both the SPI view tuple `(x_periodic_a, x_periodic_b, x_nonperiodic)` and the MoCo `query/key/hard-negative` interpretation.

### 2.3 Pure MoCo + period-regression mode (not natively supported as a config)

A pure MoCo branch plus the auxiliary `PeriodRegressionHead` can be described conceptually as:

- use the SPI dataset to generate the same tuple `(x_periodic_a, x_periodic_b, x_nonperiodic)`;
- feed `x_periodic_a` and `x_periodic_b` to MoCo's query/key route;
- feed `x_nonperiodic` as a hard negative route;
- keep the regression head on `log_p_a` or `log_p_b` and supervise it with `log(L)`;
- suppress the `SPIConLoss` branch completely.

This requires a trainer branch change, because the current `SPITrainer` hard-wires the `SPIConLoss` call inside `train_step()` and always supplies `pseudo_labels` and the `features` tensor for the SupCon-like contrastive term.

The practical config knob available now is only:

```yaml
multi_task:
  moco_loss_weight: 0.0  # SPI-only mode
  moco_loss_weight: > 0.0  # hybrid SPI + MoCo joint mode
```

whereas the choice between `SPIConLoss` and `MoCoNCELoss` is enforced by the current `train_step()` code, not by the YAML alone.

---

## 3. Loss design choices

The branch currently has three possible objective packets:

### 3.1 SPI contrastive loss `SPIConLoss`

Implementation: `losses/spicon_loss.py`

```python
class SPIConLoss(nn.Module):
    self.supcon = SupConLoss(temperature=temperature)
    self.period_criterion = nn.SmoothL1Loss()
```

Forward:

```python
contrastive = self.supcon(features, pseudo_labels)
```

with:

- `features`: concatenated embeddings of the two periodic views and the one non-periodic view
- `pseudo_labels`: label map
  - `1` for periodic views
  - `0` for non-periodic view

This is the contrastive component of SPI.

### 3.2 Period regression loss

This is optional and governed by:

```yaml
spi:
  period_loss_weight: 0.5
```

The loss adds:

```python
period_loss = nn.SmoothL1Loss(pred_log_period, true_log_period)
```

where:

```python
true_log_period = torch.cat([period_label, period_label], dim=0)
```

and `period_label = log(L)` is stored in the dataset sample. This head teaches the backbone to represent approximate cycle-length information, not just a binary periodic/non-periodic class.

### 3.3 MoCo InfoNCE loss

Implementation: `losses/moco_nce_loss.py`

```python
class MoCoNCELoss(nn.Module):
    logits = [l_pos, l_queue, l_hard]
    labels = zeros(logits.shape[0])
    return F.cross_entropy(logits, labels)
```

The query `q` and key `k` are outputs of `MoCoWrapper`, which uses an encoder-q and encoder-k pair. The queue itself is a memory bank of cached keys.

---

## 4. Memory queue design (`MoCoWrapper`)

The memory module is implemented in `models/moco_wrapper.py`.

### 4.1 Components

```python
class MoCoWrapper(nn.Module):
    self.encoder_q = _Encoder(backbone_cfg, feature_dim, hidden_dim, backbone=shared_backbone)
    self.encoder_k = copy.deepcopy(self.encoder_q)
    self.queue = F.normalize(torch.randn(feature_dim, queue_size), dim=0)
    self.queue_ptr = torch.zeros(1, dtype=torch.long)
```

Important details:

- `encoder_q` is updated by gradient descent.
- `encoder_k` is an EMA momentum copy of `encoder_q` and receives no gradients.
- `queue` is a FIFO memory bank of negative keys.
- `queue_ptr` is a circular write pointer for the memory bank.

### 4.2 How the queue is updated

`MoCoWrapper.forward()` does:

```python
q = encoder_q(x_anchor)
k = encoder_k(x_pos_warp)
k_neg = encoder_k(x_neg_shuffle)
```

and in the `with torch.no_grad()` block:

```python
self._momentum_update_key_encoder()
```

then:

```python
queue = self.queue.clone().detach()
self._dequeue_and_enqueue(k)
```

The queue stores normalized negative keys from recent mini-batches.

### 4.3 Queue usage

`MoCoNCELoss` turns `(q, k, queue, k_neg)` into logits by:

```python
l_pos = q · k
l_queue = q · queue
l_hard = q · k_neg
logits = [l_pos, l_queue, l_hard]
```

where the first column is always the positive index and the rest are negatives.

---

## 5. How to choose your training recipe

A safe decision map is:

- Want pure SPI branch? Set `multi_task.moco_loss_weight = 0.0` and run `SPITrainer` directly.
- Want a hybrid/shared-backbone training? Set `multi_task.moco_loss_weight > 0` and reuse the same backbone for both `SPIEncoder` and `MoCoWrapper`.
- Want only a pure MoCo branch plus cycle regression? You must special-case the current trainer and remove `SPIConLoss` from the `train_step()` in [trainers/spi_trainer.py](trainers/spi_trainer.py).

The current repo does not ship a ready-made config for the third recipe because `SPITrainer` is coded around the SPI contrastive loss as the primary loss.

---

## 6. Summary of derivative choices

If you want to translate the training design into one sentence:

- `SPIConLoss` is the “periodic vs non-periodic” contrastive part,
- `PeriodRegressionHead` supplies the auxiliary supervised target `log(L)`,
- `MoCoNCELoss` is the optional queue-memory contrastive branch over the same shared backbone,
- `MoCoWrapper` holds the memory queue and momentum-key encoder,
- `SPITrainer` chooses which objective combination to run in a single optimizer step.

The current SPI config is in:

- `configs/spi_periodicity.yaml`

Important fields:

```yaml
spi:
  cycle_duration_range_sec: [0.3, 2.0]
  n_repeats_range: [3, 5]
  speed_jitter: 0.05
  color_jitter_strength: 0.1
  feature_dim: 128
  hidden_dim: 512
  temperature: 0.05
  period_loss_weight: 0.5
```

and:

```yaml
multi_task:
  moco_loss_weight: 0.0
  spi_loss_weight: 1.0
```

This means the default branch is SPI-only (standalone), and the MoCo branch is not active unless `moco_loss_weight > 0`.

---

## 3. Full data flow

### 3.1 Input manifest

The dataset reads a plain text manifest of unlabeled clips, one path per line, same format as the SSL/MoCo branch:

```text
<root>/<video-or-frame-folder-path>
```

The dataset class is `SPIDataset`, built from `BaseVideoDataset`:

```python
SPIDataset(
    root=cfg["data"]["root"],
    split_file=cfg["data"]["train_split"],
    clip_len=cfg["data"]["clip_len"],
    raw_clip_len=cfg["data"]["raw_clip_len"],
    frame_size=cfg["data"]["frame_size"],
    fps=cfg["data"].get("fps", 30.0),
    cycle_duration_range_sec=cfg["spi"]["cycle_duration_range_sec"],
    n_repeats_range=cfg["spi"]["n_repeats_range"],
)
```

### 3.2 Raw window loading

`BaseVideoDataset` uses `load_raw_frames()` and returns a numpy array:

```python
(T_raw, H, W, C)  # uint8
```

The repository converts raw RGB frames to a tensor via:

```python
def frames_to_float_tensor(frames):
    return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float() / 255.0
```

So tensor layout becomes:

```python
(T, C, H, W)  # float32 in [0, 1]
```

### 3.3 Raw window sampling from dataset

`SPIDataset._get_raw_window(index)` chooses a random `raw_clip_len` crop from a given raw video:

- If `clips_per_video == 1`:
  - one raw window generated by `TemporalCrop(raw_clip_len, random_start=True)`.

- If `clips_per_video > 1`:
  - `expand_manifest()` creates several overlapping windows across the same source video.

This raw window is the base input to SPI:

```python
raw_window = self._get_raw_window(index)  # shape: (raw_clip_len, H, W, C)
```

The raw window is then passed to `SyntheticPeriodicityInjection`.

---

## 4. SyntheticPeriodicityInjection: how L and N are produced

The transformer class `SyntheticPeriodicityInjection` appears in `transforms/temporal_transforms.py`:

```python
class SyntheticPeriodicityInjection:
    def __call__(self, frames):
        return periodic_clip, L, N
```

### 4.1 L selection

At init time, SPI maps the real-world duration range `[0.3, 2.0]` seconds to a frame-count range using `fps`.

In `SPIDataset.__init__`:

```python
cycle_len_range = (
    max(2, round(fps * cycle_duration_range_sec[0])),
    max(3, round(fps * cycle_duration_range_sec[1])),
)
```

If `fps = 30.0`, the config gives:

```python
round(30.0 * 0.3) = 9
round(30.0 * 2.0) = 60
```

so:

```python
L in [9, 60]
```

Inside `SyntheticPeriodicityInjection.__call__()`:

```python
lo, hi = self.cycle_len_range
effective_hi = min(hi, t_raw)
if effective_hi < lo:
    L = t_raw
else:
    L = random.randint(lo, effective_hi)
```

This means:

- if the source raw window is long enough to support the configured period range, `L` is sampled uniformly from integer range `[lo, effective_hi]`;
- if the raw window is shorter than the lower bound, the code falls back to `L = t_raw` instead of sampling an impossible length.

### 4.2 N selection

`N` is sampled directly from the configured repeat-range:

```python
N = random.randint(*self.n_repeats_range)
```

For default config:

```yaml
n_repeats_range: [3, 5]
```

so:

```python
N in {3, 4, 5}
```

### 4.3 Base segment sampling

A start index `start` is sampled from the raw window:

```python
max_start = max(0, t_raw - L)
start = random.randint(0, max_start)
base_segment = frames[start:start+L]
```

So the `base_segment` dtype and shape are:

```python
base_segment: (L, H, W, C)  # uint8
```

### 4.4 Repeat construction with jitter

For each repeat `r = 1..N`:

1. `rep = base_segment`
2. apply speed jitter by resampling `L` frames at a different playback rate:

```python
rate = 1.0 + random.uniform(-speed_jitter, speed_jitter)
idx = np.clip((np.arange(L) * rate).astype(np.int64), 0, L - 1)
rep = rep[idx]
```

3. apply a brightness/color jitter factor per repeat:

```python
factor = 1.0 + random.uniform(-color_jitter_strength, color_jitter_strength)
rep = np.clip(rep.astype(np.float32) * factor, 0, 255).astype(np.uint8)
```

The result is a stack:

```python
periodic_clip = concatenate([rep_1, rep_2, ..., rep_N], axis=0)
```

Shape:

```python
periodic_clip: (L * N, H, W, C)
```

This is a synthesized periodic clip with exactly `L*N` frames.

---

## 5. SPI dataset output tuple

`SPIDataset.__getitem__` receives `raw_window` and returns:

```python
x_periodic_a, x_periodic_b, x_nonperiodic, period_label
```

Where:

```python
x_periodic_a: a spatially-augmented clip of shape (B, C, T, H, W)
x_periodic_b: same
x_nonperiodic: same
period_label: log(L)
```

Internally, the code:

```python
periodic_clip, L, _N = self.spi(raw_window)
```

Then it selects two phase slices from the periodic clip:

```python
phase_a_start, phase_b_start = self._pick_phase_starts(total_len, L)
phase_a_np = periodic_clip[phase_a_start:phase_a_start + clip_len]
phase_b_np = periodic_clip[phase_b_start:phase_b_start + clip_len]
```

And it samples one non-periodic clip from the raw window:

```python
nonperiodic_np = self.nonperiodic_crop(raw_window)
```

These arrays are then sent through three different `VideoSpatialAugment` instances:

```python
x_periodic_a = spatial_aug_a(frames_to_float_tensor(phase_a_np))
x_periodic_b = spatial_aug_b(frames_to_float_tensor(phase_b_np))
x_nonperiodic = spatial_aug_c(frames_to_float_tensor(nonperiodic_np))
```

Because `VideoSpatialAugment.__call__()` permutes the frame axis from `(T, C, H, W)` to `(C, T, H, W)` for the backbone, the tensor entering the backbone is:

```python
(B, C, T, H, W)
```

with `B = batch_size` and `T = clip_len` (`cfg["data"]["clip_len"]` in training config; default often 16 frames).

Then `period_label = torch.tensor(math.log(max(L, 1)), dtype=torch.float32)` is used as the regression target.

---

## 6. Encoder and branch structure

The trainer builds a `SPIEncoder` in `trainers/spi_trainer.py`:

```python
class SPIEncoder(nn.Module):
    def __init__(...):
        self.backbone = backbone if backbone is not None else build_backbone(backbone_cfg)
        self.projector = ProjectionHead(self.backbone.out_dim, hidden_dim, feature_dim)
        self.period_head = PeriodRegressionHead(self.backbone.out_dim, period_head_hidden_dim)
```

The forward path is:

```python
feats = backbone(x)
z = projector(feats)
z = normalize(z)
log_period = period_head(feats)
```

So the output shapes are:

- `feats`: `(B, D_backbone)`
- `z`: `(B, feature_dim)`
- `log_period`: `(B,)`

`ProjectionHead` applies a linear projection into the hidden dimension and final normalized embedding in `feature_dim`:

```python
z = F.normalize(z, dim=-1)
```

`PeriodRegressionHead` outputs a scalar prediction for `log(L)`:

```python
pred_log_period: (B,)
```

---

## 7. SPI contrastive loss path

In the training loop `train_step()`:

```python
x_periodic_a, x_periodic_b, x_nonperiodic, period_label = batch

z_a, log_p_a = m["spi"](x_periodic_a)
z_b, log_p_b = m["spi"](x_periodic_b)
z_np, _ = m["spi"](x_nonperiodic)
```

Concatenate embeddings:

```python
features = torch.cat([z_a, z_b, z_np], dim=0)
```

This yields:

```python
features: (3B, feature_dim)
```

Construct pseudo-labels:

```python
pseudo_labels = torch.cat([
    torch.ones(2 * batch_size),
    torch.zeros(batch_size),
])
```

So:

```python
pseudo_labels: (3B,)
```

with labels:

- `1` for the two periodic views
- `0` for the one non-periodic view

This is fed into `SPIConLoss`:

```python
spi_loss, spi_logs = spicon_criterion(
    features,
    pseudo_labels,
    pred_log_period,
    true_log_period,
)
```

The loss object itself is in `losses/spicon_loss.py`:

```python
class SPIConLoss(nn.Module):
    self.supcon = SupConLoss(temperature=temperature)
    self.period_criterion = nn.SmoothL1Loss()
```

and the `forward()` path does:

```python
contrastive = self.supcon(features, pseudo_labels)
if period_loss_weight > 0:
    period_loss = SmoothL1Loss(pred_log_period, true_log_period)
    total = contrastive + period_loss_weight * period_loss
```

This is the SPI contrastive part:

- `SupConLoss` on `features` with pseudo-labels
- `identity` of positives = periodic views `A/B`
- `identity` of negatives = non-periodic view `x_nonperiodic`

The regression branch is optional:

```python
pred_log_period = cat([log_p_a, log_p_b])
true_log_period = cat([period_label, period_label])
```

This matches the periodic view pairs and teaches the encoder to predict log-period from the same backbone features.

---

## 8. Optional joint training (MoCo branch)

When `multi_task.moco_loss_weight > 0`, the same backbone may be wired into a `MoCoWrapper` and a MoCo InfoNCE loss. In that case, the code creates:

```python
moco_wrapper = MoCoWrapper(
    backbone_cfg=cfg["backbone"],
    feature_dim=cfg["moco"]["feature_dim"],
    hidden_dim=cfg["moco"]["hidden_dim"],
    queue_size=cfg["moco"]["queue_size"],
    momentum=cfg["moco"]["momentum"],
    shared_backbone=spi_encoder.backbone,
)
```

and the current batch passes through:

```python
q, k, queue, k_neg = m["moco"](x_periodic_a, x_periodic_b, x_nonperiodic)
```

where:

- `q` is the encoder-q query embedding from `x_periodic_a`
- `k` is the encoder-k positive key from `x_periodic_b`
- `k_neg` is the hard negative from `x_nonperiodic`
- `queue` is the MoCo FIFO memory queue

The combined loss is:

```python
loss = spi_loss_weight * spi_loss + moco_loss_weight * moco_loss
```

This is what the repo calls “joint-training”. The important constraint: the MoCo branch receives the same shared backbone; it is not a separate training loop.

---

## 9. Shape summary

The end-to-end tensor shape chain is:

```text
Raw frames:           (T_raw, H, W, C)
Sampled raw window:   (raw_clip_len, H, W, C)
SPITemporal clip:     (L * N, H, W, C)
Phase crop A:         (clip_len, H, W, C)
Phase crop B:         (clip_len, H, W, C)
Nonperiodic crop:     (clip_len, H, W, C)
```

After converting to tensor and then spatially augmenting:

```text
x_periodic_a:         (B, C, clip_len, H, W)
x_periodic_b:         (B, C, clip_len, H, W)
x_nonperiodic:        (B, C, clip_len, H, W)
period_label:          (B,) -> scalar log(L) per sample
```

After `SPIEncoder` forward:

```text
z_a:                  (B, feature_dim)
z_b:                  (B, feature_dim)
z_np:                 (B, feature_dim)
features:              (3B, feature_dim)
pseudo_labels:         (3B,)
```

And if period regression is enabled:

```text
pred_log_period:      (2B,)
true_log_period:      (2B,)
```

---

## 10. Expected training loop

In `SPITrainer.train()`:

```python
loader, batch_sampler = self.build_dataloader()
model, criterion = self.build_model_and_loss()
optimizer = self.build_optimizer(model.parameters())

for epoch in range(...):
    for step, batch in enumerate(loader):
        loss = self.train_step(model, criterion, batch)
        self.run_optimizer_step(optimizer, loss, scheduler)
```

So training is single-stream: one dataset, one batch, one shared backbone, and one optimizer step that receives either:

- `SPIConLoss` only (standalone), or
- `SPIConLoss + MoCoNCELoss` together (joint/multi-task mode).

---

## 11. Practical interpretation

The SPI pretext branch is a synthetic periodicity generator plus a contrastive learner:

- the synthetic periodic signal defines the positive pair (`x_periodic_a`, `x_periodic_b`);
- the raw non-periodic crop defines the negative sample (`x_nonperiodic`);
- the regression head uses `log(L)` as a self-supervised target;
- the loss is made of a SupCon-like contrastive component and an optional period regression term.

It is therefore a distinct SSL-style branch from MoCo/InfoNCE, although it shares the same general philosophy of using unlabeled data and hand-crafted temporal transformations to create a pretext signal.
