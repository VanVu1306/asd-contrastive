# Video Periodicity & Contrastive Learning Framework — Tài liệu hệ thống

Tài liệu này mô tả đầy đủ pipeline 3 giai đoạn của hệ thống, từ input đến output: **Stage 1** (pretrain encoder bằng contrastive learning — 3 nhánh độc lập), **Stage 2** (Centroid Mining), **Stage 3** (Inference & Anomaly Scoring). Mọi con số, shape, công thức trong tài liệu này được lấy trực tiếp từ code hiện tại của repo (`video-periodicity-contrastive`), không phải mô tả lý thuyết chung chung.

---

## 0. Quy ước tensor xuyên suốt hệ thống

Mọi backbone và pipeline đều thống nhất một quy ước tensor duy nhất:

```
X ∈ R^{B × C × T × H × W}
```

| Ký hiệu | Ý nghĩa                                                 | Giá trị mặc định          |
| --------- | --------------------------------------------------------- | ------------------------------ |
| B         | batch size                                                | tùy`optim.batch_size`       |
| C         | số kênh màu (RGB)                                      | 3                              |
| T         | số frame trong 1 clip đưa vào backbone (`clip_len`) | 16                             |
| H, W      | độ phân giải không gian                              | 112 (mặc định`base.yaml`) |

Một khái niệm thứ hai luôn xuất hiện song song: **raw window** (`raw_clip_len`, mặc định 64 frame) — một đoạn dài hơn được lấy ra từ video gốc trước, rồi từ đó các temporal transform (crop/warp/shuffle/SPI) mới cắt ra `clip_len` frame thực sự đưa vào mô hình. Việc tách 2 khái niệm này (raw window vs. clip đưa vào model) là điều cho phép SpeedWarp, FrameShuffle, SPI hoạt động — chúng cần "vùng nguyên liệu" rộng hơn để biến đổi.

---

## 1. Input Pipeline — nạp dữ liệu & kỹ thuật sampling

### 1.1 Định dạng input được hỗ trợ (`datasets/base_dataset.py`)

Một "clip path" trong manifest có thể là:

- File video (`.mp4`, `.avi`, ...) — giải mã qua `torchvision.io`.
- Thư mục ảnh frame (`.jpg`/`.png`) — đọc qua PIL, sort theo tên file.
- File `.npy` — mảng `(T, H, W, C)` uint8 đã stack sẵn.

Loại được tự nhận diện qua phần mở rộng file (`frame_source: "auto"`), hoặc ép buộc qua config nếu cần.

**`probe_num_frames()`** — hàm đọc số lượng frame **không cần decode toàn bộ** video: dùng `mmap_mode='r'` cho `.npy` (chỉ đọc header), đếm file cho thư mục ảnh, và cố lấy timestamp container (không decode) cho video file, fallback sang decode toàn bộ chỉ khi thực sự cần. Mục đích: lập kế hoạch multi-window (mục 1.3) cho hàng trăm video mà không tốn chi phí decode.

### 1.2 `TemporalCrop` — lấy mẫu cơ bản

```
TemporalCrop(clip_len, random_start=True)(frames) -> (clip_len, H, W, C)
```

- Nếu video đủ dài (`T_total >= clip_len`): chọn `start` ngẫu nhiên trong `[0, T_total-clip_len]`.
- Nếu video ngắn hơn `clip_len`: **loop (lặp vòng)** chỉ số `idx = arange(clip_len) % T_total` — không pad khung đen, đảm bảo backbone luôn thấy đủ chuyển động.

Đây là cơ chế lấy mẫu mặc định (`clips_per_video: 1`) — mỗi video chỉ đóng góp 1 clip ngẫu nhiên mỗi lần gọi `__getitem__`.

### 1.3 Multi-window sampling — tăng mật độ, giảm overlap (`datasets/multi_window.py`)

**Mục đích:** với dataset nhỏ, 1 clip/video dùng dữ liệu rất lãng phí, và khiến MoCo's queue (mục 3.1) chứa đầy bản sao gần giống của chính clip đang train (false negative). `data.clips_per_video: N` biến "1 window/video" thành "N window/video", đặt theo kiểu **phân đoạn (segment-based, giống Temporal Segment Networks)**:

```
build_segment_windows(num_frames, window_len, num_windows):
    valid_range = num_frames - window_len
    starts[i] = round(i * valid_range / (num_windows - 1))   for i in 0..num_windows-1
    jitter_radius[i] = min(khoảng cách tới neighbor bên trái, bên phải) / 2
```

- Nếu video đủ dài: các `starts[i]` cách đều nhau ≥ `window_len` → **overlap = 0** tuyệt đối.
- Nếu video quá ngắn để chứa N window không chồng lấp: overlap vẫn xảy ra nhưng là **mức tối thiểu về mặt toán học** (các start đặt cách đều nhất có thể) — không bao giờ vượt biên video.
- Mỗi lần `__getitem__` được gọi, `jitter_within_segment()` dịch `start` ngẫu nhiên trong `[-jitter_radius, +jitter_radius]` — vẫn có tính ngẫu nhiên qua các epoch mà không phá vỡ đảm bảo không-chồng-lấp.

Khi bật, dataset in ra 1 dòng chẩn đoán, ví dụ:

```
[SSLDataset] 45 videos -> 168 windows (clips_per_video=4); 12/45 videos (26.7%) were too short to avoid overlap entirely.
```

### 1.4 Batch construction chống leakage

Có clip đa dạng hơn (mục 1.3) không có nghĩa nhiều clip cùng 1 video sẽ không lọt chung 1 batch — nếu xảy ra, contrastive loss có thể "ăn gian" bằng cách khớp background/appearance chung thay vì học chuyển động thật. Hai cơ chế:

- **`VideoDiverseBatchSampler`** (`datasets/samplers.py`, dùng cho SSL & SPI): giới hạn số clip cùng video trong 1 batch qua `max_per_video`. Có DDP-aware rank-sharding tích hợp sẵn (shuffle 1 lần, mọi rank tính giống nhau, rồi mỗi rank lấy 1 lát cắt riêng — không cần `DistributedSampler` riêng).
- **`GroupBalancedBatchSampler`** (`datasets/supcon_dataset.py`, dùng cho SupCon): tổ chức theo `(class, video_id)` — mỗi lượt build batch chỉ lấy tối đa 1 sample/`video_id`/class → tự động đảm bảo không leakage mà không cần tham số riêng.

### 1.5 Transform — biến đổi pixel & thời gian

**Spatial (`transforms/spatial_transforms.py`)** — áp dụng đồng nhất cho toàn bộ T frame của 1 clip (crop box/jitter được sample 1 lần/clip, không phải 1 lần/frame — tránh tạo chuyển động giả):

| Transform                     | Tác dụng                                                                                                                                                                                                                                                                                                                  |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RandomResizedCropVideo`    | crop 1 vùng ngẫu nhiên (scale mặc định 0.5-1.0), resize về`frame_size`                                                                                                                                                                                                                                             |
| `RandomHorizontalFlipVideo` | lật ngang, p=0.5                                                                                                                                                                                                                                                                                                           |
| `ColorJitterVideo`          | brightness/contrast/saturation/hue, 1 hệ số dùng chung cho cả clip                                                                                                                                                                                                                                                      |
| `RandomErasingVideo`        | **cutout nhất quán theo thời gian** — che 1 ô ngẫu nhiên tại **cùng vị trí trên mọi frame** của 1 clip (khác vị trí giữa các clip). Mặc định tắt (`p=0.0`); bật (0.25-0.5) khi dùng `clips_per_video>1` để tăng decorrelation nếu vẫn có 2 clip cùng video lọt chung batch |
| `NormalizeVideo`            | chuẩn hóa kiểu ImageNet (mean/std)                                                                                                                                                                                                                                                                                       |

Output cuối: `(C, T, H, W)` — permute lại để khớp input backbone.

**Temporal (`transforms/temporal_transforms.py`)**:

| Transform                         | Input → Output                                                         | Dùng ở đâu |
| --------------------------------- | ----------------------------------------------------------------------- | -------------- |
| `TemporalCrop`                  | `(T_raw,H,W,C) → (clip_len,H,W,C)`                                   | mọi nhánh    |
| `SpeedWarp`                     | resample với tốc độ`rate ∈ {0.8, 1.5}` (mặc định)             | SSL/MoCo       |
| `FrameShuffle`                  | hoán vị 1 phần vị trí frame (`ratio=1.0` = xáo trộn toàn bộ) | SSL/MoCo       |
| `SlidingWindow`                 | liệt kê toàn bộ window trượt qua 1 video dài (`stride`)        | Stage 3        |
| `SyntheticPeriodicityInjection` | cắt đoạn L frame, lặp N lần kèm jitter                            | SPI            |

---

## 2. Stage 1 — Pretrain Encoder bằng Contrastive Learning (3 nhánh độc lập)

Cả 3 nhánh dùng chung backbone factory (`models/backbones/builder.py`: `resnet3d_18` out_dim=512, `resnet3d_50` out_dim=2048, `r2plus1d_18` out_dim=512, `s3d` out_dim=1024, `video_swin_t` out_dim=768) và chung `ProjectionHead` (`Linear → BatchNorm1d → ReLU → Linear`, rồi L2-normalize) — nhưng **cách tạo positive/negative pair, kiến trúc, và loss khác nhau hoàn toàn**.

### 2.1 Nhánh SSL / MoCo (`ssl_dataset.py` + `moco_wrapper.py` + `moco_nce_loss.py`)

**Không cần nhãn.** Positive/negative được tạo hoàn toàn từ biến đổi thời gian trên chính 1 raw window.

```
raw window (raw_clip_len, H, W, C)
   ├── TemporalCrop(clip_len)         → x_anchor       → SpatialAugment_A
   ├── SpeedWarp(clip_len, rate)      → x_pos_warp     → SpatialAugment_B
   └── FrameShuffle(x_anchor_raw)     → x_neg_shuffle  → SpatialAugment_A
```

- **Positive pair**: `(x_anchor, x_pos_warp)` — cùng nội dung, khác tốc độ phát lại. Ý tưởng: hành vi lặp lại thật vẫn "là chính nó" dù tua nhanh/chậm.
- **Negative pair**: 2 nguồn
  1. `x_neg_shuffle` — **hard negative tường minh**: cùng từng frame với anchor, chỉ khác thứ tự thời gian →ép encoder học chuyển động/thứ tự, không chỉ học appearance tĩnh.
  2. **FIFO Memory Queue** (`queue_size`, mặc định 512) — kho các `key` embedding từ nhiều step trước, tồn tại xuyên suốt quá trình train.

**Kiến trúc (`MoCoWrapper`)**:

```
encoder_q = backbone + ProjectionHead      # nhận gradient mỗi step
encoder_k = deepcopy(encoder_q)            # KHÔNG nhận gradient, chỉ update EMA
```

Momentum update mỗi step:

```
θ_k ← m · θ_k + (1 − m) · θ_q          (m = momentum, mặc định 0.999)
```

`encoder_k` được gọi qua "shuffling BN" (`batch_shuffle_ddp`) khi chạy DDP để tránh rò rỉ batch-statistics giữa query và key.

**Forward pass — tensor shape:**

| Bước                                  | Shape                                             |
| --------------------------------------- | ------------------------------------------------- |
| `x_anchor, x_pos_warp, x_neg_shuffle` | mỗi cái`(B, 3, 16, 112, 112)`                 |
| `q = encoder_q(x_anchor)`             | `(B, feature_dim)`, feature_dim mặc định 256 |
| `k = encoder_k(x_pos_warp)`           | `(B, 256)`                                      |
| `k_neg = encoder_k(x_neg_shuffle)`    | `(B, 256)`                                      |
| `queue`                               | `(256, queue_size)`                             |

**Loss (`MoCoNCELoss`, InfoNCE):**

```
l_pos   = q·k                (B, 1)     — cosine sim, vì q,k đã L2-normalize
l_queue = q·queue            (B, Q)
l_hard  = q·k_neg             (B, 1)
logits  = [l_pos | l_queue | l_hard] / τ      (τ mặc định 0.07)
loss    = CrossEntropy(logits, label=0)        # cột 0 luôn là positive
```

Sau mỗi step, `k` được đẩy vào queue (`_dequeue_and_enqueue`), đẩy phần tử cũ nhất ra — **không** đẩy `k_neg` vào queue (nó chỉ là negative riêng của sample đó).

### 2.2 Nhánh SupCon (`supcon_dataset.py` + `supcon_loss.py`)

**Cần nhãn hành vi** (`1 = ASD/stimming`, `0 = non-ASD`), lấy từ manifest CSV `clip_path,video_id,label`.

- **Positive pair**: 2 clip **cùng nhãn** nhưng **khác `video_id`** — bắt buộc, để loại bỏ bias theo subject/bối cảnh (2 clip cùng người/cùng video dễ giống nhau vì lý do không liên quan đến hành vi).
- **Negative pair**: bất kỳ cặp khác nhãn nào trong batch (1 ASD với 1 non-ASD).

**Kiến trúc**: `SupConEncoder = backbone + ProjectionHead` (feature_dim mặc định 128) — **không có momentum encoder**, toàn bộ tham số train end-to-end bằng gradient, vì SupCon lấy negative hoàn toàn trong batch, không cần "kho nhớ" bền vững.

**Cơ chế đảm bảo positive hợp lệ**: `GroupBalancedBatchSampler` cố gắng đưa ≥2 `video_id` khác nhau/class vào mỗi batch (mục 1.4).

**Loss (`SupConLoss`)** — với `pos_mask(i,j) = same_label(i,j) ∧ i≠j ∧ diff_video(i,j)`:

```
sim(i,j) = z_i · z_j / τ                    (τ mặc định 0.1)
L_i = -1/|P(i)| · Σ_{j∈P(i)} log( exp(sim(i,j)) / Σ_{a≠i} exp(sim(i,a)) )
loss = mean_i L_i    (chỉ tính trên các i có |P(i)|>0; nếu cả batch không có positive nào → trả 0, không NaN)
```

Tensor shape: `x_clip (B,3,16,112,112) → z (B,128)`.

### 2.3 Nhánh SPI — Synthetic Periodicity Injection (`spi_dataset.py`, `spicon_loss.py`, `spi_moco_wrapper.py`, `spi_moco_loss.py`, `spi_trainer.py`)

**Không cần nhãn thật** — tự "chế tạo" tính chu kỳ, cho pseudo-label miễn phí.

**Bước 1 — tổng hợp clip có chu kỳ đã biết (`SyntheticPeriodicityInjection`)**:

```
raw window (raw_clip_len, H, W, C)
   └── chọn L ∈ [L_min, L_max]  (quy đổi từ fps × cycle_duration_range_sec, vd 0.3-2.0s/chu kỳ)
       cắt đoạn L frame, LẶP N lần (N ∈ [3,5])
       mỗi lần lặp: jitter tốc độ (±5%) + jitter màu (±10%) ĐỘC LẬP — BẮT BUỘC, không tùy chọn
   → periodic_clip (L·N, H, W, C), period thật = L
```

Jitter bắt buộc vì nếu không, N lần lặp giống hệt pixel-by-pixel → positive pair (mục dưới) bị "giải" bằng cách so khớp pixel thay vì học nhận diện chu kỳ.

**Bước 2 — cắt 2 "pha" khác nhau + 1 view không chu kỳ**:

```
   ├── window tại repeat-unit A  → x_periodic_phaseA  → SpatialAugment_A
   ├── window tại repeat-unit B  → x_periodic_phaseB  → SpatialAugment_B
   └── TemporalCrop liên tục trên raw window gốc → x_nonperiodic → SpatialAugment_C
period_label = log(L)
```

- **Positive pair**: `(x_periodic_phaseA, x_periodic_phaseB)` — cùng 1 clip tổng hợp, khác pha → bất biến theo pha (phase-invariance).
- **Negative**: `x_nonperiodic` — pseudo-label 0 (không chu kỳ) so với pseudo-label 1 (có chu kỳ) của cả 2 pha trên.

Nhánh SPI hỗ trợ **3 cách train** (chọn qua `spi.training_mode`, không can thiệp lẫn nhau):

#### (a) `in_batch` (mặc định)

```
features = [z_a; z_b; z_np]              (3B, feature_dim)
pseudo_labels = [1,...,1 (2B), 0,...,0 (B)]
loss = SupConLoss(features, pseudo_labels)     # KHÔNG dùng group_ids
```

Khác biệt cố ý so với SupCon thật (2.2): **không loại trừ theo video_id** — vì ở đây phaseA/phaseB *cùng nguồn* chính là tín hiệu mong muốn (bất biến pha), không phải bias cần khử. Vì không dùng `group_ids`, **mọi cặp periodic trong cả batch được coi là positive lẫn nhau**, không chỉ riêng cặp phaseA/phaseB của cùng 1 sample. Negative chỉ đến từ những gì "non-periodic" **có trong batch hiện tại**.

#### (b) `queue` — MoCo-style queue cho chính task này

Được thêm để giải quyết đúng hạn chế của (a): negative giới hạn trong batch dễ bão hòa/kém đa dạng trên batch nhỏ. Kiến trúc (`SPIMoCoWrapper`) là 1 MoCo hoàn chỉnh **riêng cho task periodic/non-periodic**:

```
q = encoder_q(x_periodic_a)                         # nhận gradient
k = encoder_k(x_periodic_b)  [momentum, shuffle-BN]  # positive key
k_neg = encoder_k(x_nonperiodic)  [momentum]          # negative của step này
queue: chỉ lưu embedding NON-PERIODIC (đẩy k_neg vào sau mỗi step)
loss = MoCoNCELoss(q, k, queue, k_neg)     # tái dùng nguyên loss của nhánh SSL, công thức y hệt mục 2.1
```

Điểm khác biệt bản chất so với (c) bên dưới: queue ở đây phục vụ **trực tiếp** chính cặp positive/negative periodic-vs-nonperiodic, không phải 1 task InfoNCE độc lập chạy song song.

#### (c) `combined` — kết hợp với 1 nhánh MoCo riêng (multi_task.moco_loss_weight > 0)

```
loss = spi_loss_weight · SPIConLoss(...)  +  moco_loss_weight · MoCoNCELoss(MoCoWrapper(x_a, x_b, x_np))
```

`MoCoWrapper` ở đây dùng **backbone dùng chung** (`shared_backbone`) với nhánh SPI (qua PyTorch's tự khử trùng tham số khi gọi `.parameters()`) nhưng có `ProjectionHead` và `queue` **hoàn toàn riêng** — tức là 1 objective InfoNCE độc lập cộng thêm vào, không thay thế cơ chế lấy negative của SPIConLoss.

**Nhánh phụ (auxiliary) — Period Regression** (dùng chung cho cả 3 cách train ở trên, qua `PeriodRegressionHead`):

```
log_P̂ = PeriodRegressionHead(backbone_features)     # Linear→ReLU→Linear, dự đoán log(P)
loss_period = SmoothL1(log_P̂, log(L))
loss_total = loss_contrastive + period_loss_weight · loss_period    (mặc định weight=0, tắt)
```

Mục đích: bơm thêm thông tin *định lượng* (P dài bao nhiêu) vào embedding, không chỉ phân loại nhị phân có/không chu kỳ.

---

## 3. Stage 2 — Centroid Mining (`eval.py: run_anomaly_scoring`, Stage 2)

**Không backprop, chạy 1 lần duy nhất (single pass)** trên tập clip "bình thường" (`eval.centroids.normal_split`), dùng backbone đã đóng băng (`.eval()`) từ checkpoint Stage 1 (tự nhận diện checkpoint đến từ nhánh nào qua prefix: `encoder_q.backbone.` / `backbone.` / `spi.backbone.` / `spi_moco.encoder_q.backbone.`).

```
normal clips → backbone (frozen) → Z_normal ∈ R^{N × D}
K-Means (K≈10-50, mặc định 20) trên Z_normal đã L2-normalize:
    1. Khởi tạo: chọn ngẫu nhiên K sample làm centroid (seed cố định, torch.Generator riêng)
    2. Lặp tối đa 100 vòng:
       assignment[i] = argmax_k cos_sim(z_i, c_k)
       c_k ← normalize( mean(z_i : assignment[i]=k) )      # cụm rỗng: giữ nguyên centroid cũ
       dừng sớm nếu không có centroid nào di chuyển
→ C = {c_1, ..., c_K} ∈ R^{K × D}, lưu vào centroids.pth
```

Đây là bước duy nhất tạo ra tri thức "thế nào là bình thường" — hoàn toàn không cần nhãn bất thường ở mức frame.

---

## 4. Stage 3 — Inference & Anomaly Scoring (`eval.py: run_anomaly_scoring`, Stage 3)

**Input**: video test đầy đủ (`eval.test_split`), có thể kèm nhãn bất thường mức-frame để tính metric (tùy chọn).

```
SlidingWindow(clip_len, stride) liệt kê TOÀN BỘ vị trí start có thể trên video
   → mỗi window: (clip_len, H, W, C) → backbone (frozen) → z ∈ R^D
S_raw(z) = min_k (1 − cos_sim(z, c_k)) = 1 − max_k cos_sim(z, c_k)      # càng xa mọi centroid "bình thường" càng cao
```

**Ghép điểm window → điểm mức-frame** (overlap-averaging): mỗi window đóng góp điểm của nó cho toàn bộ `clip_len` frame nó bao phủ; frame nằm trong nhiều window được lấy **trung bình** các đóng góp:

```
score_sum[t] += S_raw(window)   với mọi t ∈ [start, start+clip_len)
frame_score[t] = score_sum[t] / count[t]
```

**Hậu xử lý (`post_process.smoothing`)**: `moving_average` (convolve với kernel đều) hoặc `median` (scipy median_filter) hoặc tắt hẳn — làm mượt đường điểm theo thời gian.

**Metric** (`utils/metrics.py`, tính trên toàn bộ frame của toàn bộ video test gộp lại):

- **Micro/Macro-AUC.**
- **EER** — điểm trên đường ROC nơi FPR = FNR (1−TPR)
- **FAR-ratio** — tỷ lệ frame "bình thường" (label=0) có điểm ≥ ngưỡng cố định 0.5 bị báo động nhầm

*(Lưu ý đã rút ra từ thực nghiệm: `far_ratio` với ngưỡng cố định 0.5 dễ không mang thông tin nếu thang điểm `S_raw` thực tế không trải đều trong [0,1] — nên cân nhắc ngưỡng theo percentile thay vì cố định khi diễn giải kết quả.)*

---

## 5. Downstream Task 2 — Action Classification / Linear Probing (`eval.py: run_linear_probing`)

Song song với Anomaly Scoring, mỗi checkpoint Stage 1 cũng dùng được cho phân loại có giám sát:

```
backbone (đóng băng hoặc fine-tune nhẹ)
   → BatchNorm1d(affine=False)     # chuẩn hóa scale đặc trưng thô trước khi vào Linear — 
                                     # cần thiết vì feature thô không chuẩn hóa từng gây loss 
                                     # nổ (loss ~10^8) khi test thực tế
   → ClassificationHead (Linear(D, num_classes))
   → CrossEntropyLoss
```

`freeze_backbone: true` (mặc định) → chỉ train `ClassificationHead`; `false` → fine-tune toàn bộ với LR nhỏ. Đây cũng là cách để **kiểm chứng chất lượng embedding** trước khi tin tưởng dùng nó cho Stage 2/3 (đặc biệt quan trọng với nhánh SPI, vốn không thấy nhãn thật trong lúc pretrain).

---

## 6. Reproducibility (`utils/seed.py`)

`seed:` trong config chi phối **mọi** nguồn ngẫu nhiên: Python `random`, `numpy`, `torch` (CPU+CUDA) qua `set_seed()`, và mỗi `DataLoader` đều nhận `worker_init_fn` riêng (vì cơ chế seed mặc định của PyTorch cho worker process chỉ seed lại RNG của `torch`, không đụng đến `random`/`numpy` — trong khi hầu hết augmentation của hệ thống này dùng đúng 2 module đó). `hardware.deterministic: true` ép cudnn chạy thuật toán tất định (chậm hơn) để tái lập bit-exact trên GPU. Đã kiểm chứng: cùng seed cho kết quả loss giống hệt nhau qua nhiều lần chạy, kể cả khi bật `num_workers>0`.

---

## 7. Sơ đồ tổng thể

```
                        ┌────────────────────────────────────────────┐
                        │              STAGE 1 (chọn 1)              │
                        │                                            │
   unlabeled clips ───► │  SSL/MoCo   SupCon (cần nhãn)   SPI        │──► checkpoint
                        │  (2.1)      (2.2)              (2.3: a/b/c)│    (backbone
                        └────────────────────────────────────────────┘     đã train)
                                                │
                        ┌───────────────────────┴────────────────────┐
                        ▼                                            ▼
              STAGE 2: Centroid Mining                    Downstream: Linear Probing
              (K-Means trên clip normal,                  (kiểm chứng chất lượng
               zero backprop) → K centroids                embedding bằng nhãn thật)
                        │
                        ▼
              STAGE 3: Sliding-window scoring
              S_raw = 1 − cos_sim(z, centroid gần nhất)
              → làm mượt → AUC-ROC / EER / FAR-ratio
```

---

## 8. Định hướng mở rộng

**Ngắn hạn, ít rủi ro:**

- Bật `backbone.pretrained=true` (Kinetics/ImageNet init) — đòn bẩy lớn nhất cho dataset nhỏ, đến giờ mới chỉ đề xuất, chưa từng bật thử trong các lần train thực tế.
- Tinh chỉnh `clips_per_video` + `max_per_video_per_batch` đồng bộ theo quy mô dataset thật (hiện mới test trên dữ liệu synthetic).
- Đánh giá theo cross-validation xoay vòng video test thay vì 1 split cố định (bộ test hiện tại quá nhỏ để tin 1 con số AUC đơn lẻ).
- Thay ngưỡng cố định trong `far_ratio` bằng ngưỡng theo percentile của phân phối điểm thực tế.

**Trung hạn:**

- Ablation có hệ thống giữa 3 nhánh Stage 1 (và 3 cách train của SPI) trên cùng 1 bộ test, dùng Linear Probing làm tiêu chí sàng lọc trước khi đầu tư vào Stage 2/3.
- Thử `parametric` mode của `AnomalyScoringHead` (MLP dự đoán S trực tiếp) nếu có nhãn bất thường yếu/bán giám sát, thay vì chỉ dùng K-Centroids không giám sát.
- Mở rộng `SyntheticPeriodicityInjection` với nhiều "hình thái" chu kỳ hơn (không chỉ lặp nguyên khối, mà biến đổi biên độ/hình dạng chuyển động mỗi lần lặp) để pseudo-task gần hành vi thật hơn.
- Kết hợp `period_loss_weight>0` với đánh giá tương quan giữa embedding-theo-chu-kỳ và nhãn thật (hiện auxiliary loss này mới kiểm tra đúng cơ chế backprop, chưa đánh giá nó có thực sự cải thiện Stage 2/3 hay không).

**Dài hạn / thay đổi kiến trúc:**

- Thêm modality mới (keypoints/optical flow) qua 1 `datasets/*_dataset.py` mới, giữ nguyên convention `(B,C,T,H,W)` hoặc tương đương — các backbone/head/loss hiện tại không cần biết tensor đến từ đâu.
- Thêm backbone mới chỉ cần đăng ký vào `models/backbones/builder.py`'s `_REGISTRY`, không cần sửa nơi khác.
- Cân nhắc kiến trúc chia sẻ backbone giữa cả 3 nhánh Stage 1 cùng lúc (hiện `shared_backbone` mới hỗ trợ SPI↔MoCo cục bộ trong 1 nhánh) nếu muốn 1 encoder duy nhất học từ cả 3 tín hiệu supervision đồng thời — sẽ cần thiết kế lại quy trình checkpoint/loading, vì hiện mỗi nhánh sinh 1 checkpoint riêng.
- Nếu `training_mode: "queue"` (SPI) chứng minh hiệu quả, cân nhắc thêm lựa chọn tương tự cho nhánh SupCon (queue lưu embedding non-ASD) — hiện SupCon chỉ có in-batch negatives.
