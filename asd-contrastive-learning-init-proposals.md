# Video Periodicity & Contrastive Learning Framework

Một PyTorch framework thiết kế dạng modular dành cho bài toán phân tích tính lặp lại (periodicity), phát hiện hành vi lặp lại (stimming behavior), và chấm điểm bất thường (anomaly scoring) trong video.

Repository hỗ trợ đồng thời 2 chiến lược Contrastive Learning chính cùng quy trình đánh giá chuẩn mực:

* **Self-Supervised Learning (SSL):** Kết hợp MoCo (Momentum Contrast) với biến đổi chuỗi thời gian (Speed Warping & Frame Shuffling).
* **Supervised Contrastive Learning (SupCon):** Tối ưu hóa biểu diễn đặc trưng theo nhãn nhóm hành vi.

---

## Thư mục mã nguồn (Repository Structure)

```text
video-periodicity-contrastive/
├── configs/
│   ├── base.yaml              # Cấu hình chung (Backbone, Resolution, Hardware)
│   ├── ssl_moco.yaml          # Cấu hình SSL (Queue size, Temperature, Speed rates)
│   ├── supcon.yaml            # Cấu hình SupCon (Temperature, Class mapping)
│   └── eval.yaml              # Cấu hình Eval (Eval mode, Checkpoint path, Stride)
├── datasets/
│   ├── base_dataset.py        # Abstract Video Loader (đọc .mp4 hoặc JPEG/NPY frames)
│   ├── ssl_dataset.py         # Trả về: (x_anchor, x_pos_warp, x_neg_shuffle)
│   ├── supcon_dataset.py      # Trả về: (x_clip, label)
│   └── eval_dataset.py        # Sliding-window loader cho Inference/Evaluation
├── transforms/
│   ├── spatial_transforms.py  # Spatial augmentations (RandomCrop, Flip, ColorJitter)
│   └── temporal_transforms.py # Temporal Speed Warp, Temporal Crop, Frame Shuffle
├── models/
│   ├── backbones/             # Đa dạng 3D Backbones
│   │   ├── resnet3d.py        # 3D-ResNet18 / 3D-ResNet50
│   │   ├── r2plus1d.py        # R(2+1)D
│   │   ├── s3d.py             # Separable 3D CNN (S3D)
│   │   ├── video_swin.py      # Video Swin Transformer
│   │   └── builder.py         # Factory function khởi tạo backbone theo config
│   ├── heads.py               # ProjectionHead, AnomalyScoringHead, ClassificationHead
│   └── moco_wrapper.py        # MoCo Key-Encoder & Memory Queue Manager
├── losses/
│   ├── supcon_loss.py         # Supervised Contrastive Loss (Khosla et al.)
│   └── moco_nce_loss.py       # InfoNCE, sau này có thể custom
├── trainers/
│   ├── base_trainer.py        # Generic trainer setup, DDP initialization, Logging
│   ├── ssl_trainer.py         # SSL training loop & Memory Queue update
│   └── supcon_trainer.py      # SupCon training loop
├── utils/
│   ├── distributed.py         # All-gather utilities, SyncBN helpers
│   ├── lr_scheduler.py        # Cosine Annealing, Linear Warmup, StepLR
│   ├── logger.py              # Metric trackers, WandB, TensorBoard, Console Logger, AverageMeter
│   ├── metrics.py             # AUC-ROC, EER, Frame-level Precision/Recall
│   └── visualize.py           # Plot Anomaly Score vs Ground Truth,... 
├── eval.py                    # Evaluation pipeline (Linear Probe / Anomaly Scoring)
├── train.py                   # Training entry point
└── requirements.txt
```

---

## Chuẩn Tensor Đầu Vào (Input Specification)

Toàn bộ các 3D Backbones trong repo thống nhất định dạng Input Tensor theo chuẩn PyTorch Video:

$$
\mathbf{X} \in \mathbb{R}^{B \times C \times T \times H \times W}
$$

* **$B$ (Batch Size):** Số lượng video clip trong một GPU sub-batch.
* **$C$ (Channels):** 3 (RGB).
* **$T$ (Temporal Frames):** Số lượng frame chuẩn của một clip (ví dụ: $T = 16$ hoặc $T = 32$).
* **$H, W$ (Spatial Resolution):** $224 \times 224$ (hoặc $112 \times 112$).

---

## Quản lý Data Pipelines (`datasets/` & `transforms/`)

### 1. SSL Dataset (`ssl_dataset.py`)

Mỗi mẫu dữ liệu trả về 3 views từ một đoạn video thô $T_{raw}$:

* **`x_anchor`:** Sampling $T$ frames liên tục + `SpatialAugment_A`(mỗi batch sẽ khác nhau để tránh leakage).
* **`x_pos_warp`:** Biến đổi thời gian bằng `SpeedWarp` (tỷ lệ $0.8 \times$ hoặc $1.5 \times$) + `SpatialAugment_B`.
* **`x_neg_shuffle`:** Biến đổi thời gian bằng `FrameShuffle` (xáo trộn vị trí index của $T$ frames) + `SpatialAugment_A`.

### 2. SupCon Dataset (`supcon_dataset.py`)

* Positive Pairs: Bất kỳ 2 clip nào có cùng nhãn **$y \in \{0, 1\}$** trong mini-batch (**$1 = \text{ASD/Stimming}$**, **$0 = \text{Non-ASD}$**). Lưu ý là 2 clip từ 2 video ID khác nhau để triệt tiêu bias bối cảnh/người thực hiện.
* Negative Pairs: 1 clip ASD 1 clip non-ASD.
* Trả về cặp `(x_clip, label_id)`.

---

## Kiến trúc Mô hình (`models/`)

### 1. Backbones hỗ trợ (`models/backbones/`)

Hệ thống sử dụng hàm khởi tạo linh hoạt `build_backbone(cfg)` để tải các kiến trúc trích xuất đặc trưng thời gian - không gian:

* **`s3d.py` (S3D):** Sử dụng các lớp 3D Convolution phân tách (Spatial + Temporal), giúp tối ưu chi phí tính toán và capture chuyển động lặp hiệu quả.
* **`r2plus1d.py` (R(2+1)D):** Phân tách 3D Conv thành 2D Spatial Conv và 1D Temporal Conv.
* **`resnet3d.py` (3D-ResNet18/50):** Baseline chuẩn cho xử lý video 3D.
* **`video_swin.py` (Video Swin Transformer):** Dành cho các bài toán yêu cầu biểu diễn ngữ cảnh dài hạn với cơ chế 3D Shifted Windows.

### 2. Heads phục vụ đa nhiệm (`models/heads.py`)

* **`ProjectionHead` (MLP):** `Linear` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Linear`. Hạ chiều đặc trưng xuống không gian 128d/256d và L2-normalize phục vụ tính Cosine Similarity trong Contrastive Learning.
* **`ClassificationHead`**: `Linear(feature_dim, num_classes)`. Tối ưu cho tác vụ hạ nguồn Phân loại hành vi (Action Classification / Linear Probing) thông qua Cross-Entropy Loss.
* **`CentroidScoringHead / AnomalyScoringHead`** :
  * **Supervised Mode (Parametric)**: Mạng MLP / Temporal Conv1D dự đoán trực tiếp xác suất bất thường $S \in [0, 1]$.
  * **Non-parametric Mode ($K$-Centroids)**: Module lưu trữ $K$ tâm mẫu chuẩn $\mathbf{C} \in \mathbb{R}^{K \times D}$ (dưới dạng PyTorch Buffer), tính toán khoảng cách Cosine ngắn nhất tới các tâm chuẩn mà không cần tính đạo hàm (Zero-backprop).

---

## Downstream Tasks & Cơ chế Chấm điểm Bất thường (Anomaly Score Calculation)

Trích xuất vector đặc trưng $\mathbf{z} = f(\mathbf{x}) \in \mathbb{R}^D$ từ Pre-trained Backbone Encoder, hệ thống hỗ trợ 2 nhánh downstream tasks với quy trình chuẩn bị dữ liệu và phương pháp tính Anomaly Score riêng biệt:

### Downstream Task 1: Action Classification

* **Yêu cầu**: Input là clip video $\mathbf{X} \in \mathbb{R}^{B \times C \times T \times H \times W}$.
* **Cấu hình Backbone**: Đóng băng trọng số Encoder `(requires_grad = False)` đối với Linear Probing, hoặc giữ learning rate nhỏ ($10^{-4}$) đối với Full Fine-tuning.
* **Output Head**: Kết nối `ClassificationHead` đưa vector $\mathbf{z} \in \mathbb{R}^D \rightarrow \mathbb{R}^{\text{num\_classes}}$.
* **Hàm Loss**: Tối ưu hóa bằng `CrossEntropyLoss()`.

### Downstream Task 2: Anomaly Scoring (Non-parametric $K$-Centroids)

**Yêu cầu**: Không cần nhãn bất thường; khai thác tối đa không gian biểu diễn của Pre-trained MoCo Encoder thông qua Quy trình 3 Giai đoạn (3-Stage Pipeline) để đảm bảo tính gọn nhẹ và suy luận tức thì.
**Cơ chế**:

* **Stage 1**: Pre-training (SupCon hoặc SSL) (Biểu diễn Đặc trưng)
  * Huấn luyện 3D Backbone Encoder trên tập dữ liệu video không nhãn bằng loss InfoNCE và MoCo Queue ($\mathcal{Q}$)
  * Đầu ra:  Pretrained encoder đã học được tính liên tục và chu kỳ thời gian.
* **Stage 2**: Centroid Mining (0-Epoch / Single Pass)
  * Không qua huấn luyện Gradient Descent (Zero Backpropagation).
  * Đóng băng Pretrained Encoder. Cho toàn bộ **video clip bình thường (Normal Data)** đi qua Encoder đúng 1 lượt để rút ra tập vector đặc trưng $\mathbf{Z}_{\text{normal}} \in \mathbb{R}^{N \times D}$.
  * Chạy thuật toán $K$-Means Clustering ($K \approx 10 \sim 50$) trên không gian đặc trưng đã $L_2$-normalize để trích xuất $K$ tâm đại diện cho các hình thái hành vi chuẩn:
    $$
    \mathbf{C} = \{\mathbf{c}_1, \mathbf{c}_2, \dots, \mathbf{c}_K\} \subset \mathbb{R}^D
    $$
  * Đầu ra: File trọng số nhẹ `centroids.pth`.
* **Stage 3**: Inference & Evaluation (Distance-based Scoring)
  * Khi suy luận, nạp $K$ tâm vào `CentroidScoringHead`. Clip test $\mathbf{x}_{\text{test}}$ qua Encoder thu được $\mathbf{z} = f(\mathbf{x}_{\text{test}})$.
  * Đề xuất Công thức Anomaly Score (Raw Score): Lấy khoảng cách Cosine từ vector test đến tâm chuẩn gần nhất:
    $$
    S_{\text{raw}}(\mathbf{x}) = \min_{k \in \{1 \dots K\}} \left( 1 - \frac{\mathbf{z} \cdot \mathbf{c}_k}{\Vert{}\mathbf{z}\Vert{}_2 \Vert{}\mathbf{c}_k\Vert{}_2} \right)
    $$

**Ý nghĩa**: Clip bất thường sẽ nằm xa tập mẫu chuẩn trong không gian biểu diễn ($$).

### Post-processing ra Anomaly Score cuối của mỗi video test: Tương tự DS-SBD / DSVTN-ASD

---

## Các Module Hỗ trợ (`utils/`)

* **`lr_scheduler.py`:** Triển khai các chiến lược điều chỉnh Learning Rate chuyên biệt cho Contrastive Learning như Cosine Annealing đi kèm Linear Warmup giai đoạn đầu (Warmup Cosine), giúp mô hình hội tụ ổn định.
* **`logger.py`:** Quản lý ghi log đa nền tảng (Weights & Biases, TensorBoard, Console, CSV Log). Tự động ghi lại các chỉ số loss, learning rate, GPU memory usage theo từng step/epoch.
* **`distributed.py`:** Chứa các hàm tiện ích đồng bộ DDP như `all_gather_tensor()`, thiết lập giao thức truyền thông inter-GPU, và hỗ trợ `SyncBatchNorm`.
* **`metrics.py`:** Đóng gói các hàm tính toán chỉ số đánh giá cho Anomaly Detection và Classification:
* **Metrics:** Micro/Macro-AUC-ROC, Equal Error Rate (EER), Far-Ratio.
* (phụ, có thể chưa cần implement ngay) **Classification Metrics:** Top-1 Accuracy, Top-5 Accuracy, Precision/Recall/F1-score.

---

## Quy chuẩn Huấn luyện Multi-GPU (DDP Guidelines) - Ưu tiên tuân theo cơ chế các phương pháp Contrastive Learning thường làm

Khi chạy huấn luyện phân tán với PyTorch Distributed Data Parallel (DDP), cần có các quy tắc kỹ thuật sau:

* **Đồng bộ Memory Queue (`utils/distributed.py`):** Dùng `torch.distributed.all_gather()` để thu gom đầy đủ keys từ tất cả các GPU trước khi thực hiện `dequeue_and_enqueue()` vào Memory Queue của MoCo.
* **Chống rò rỉ thông tin qua BatchNorm:** Chuyển đổi toàn bộ mô hình sang `SyncBatchNorm` bằng `nn.SyncBatchNorm.convert_sync_batchnorm(model)` hoặc tham khảo cơ chế Shuffling Batch Norm mà MoCo đề xuất.
* **Quản lý Momentum Encoder (`model_ema`)**.
* **DataLoader Sampler:** Sử dụng `torch.utils.data.distributed.DistributedSampler` cho tất cả các DataLoader.

---

## Hướng dẫn Sử dụng (Usage Guide)

### 1. Huấn luyện (Training)

Chạy huấn luyện thông qua `train.py` bằng cách chỉ định file config tương ứng:

```bash
# Self-Supervised Learning (MoCo + Temporal Shuffle)
python -m torch.distributed.run --nproc_per_node=4 train.py --config configs/ssl_moco.yaml

# Supervised Contrastive Learning (SupCon)
python -m torch.distributed.run --nproc_per_node=4 train.py --config configs/supcon.yaml
```

### 2. Đánh giá Mô hình (`eval.py`)

File `eval.py` hỗ trợ 2 chế độ đánh giá chính dựa trên cấu hình trong `configs/eval.yaml`:

```bash
# Đánh giá Anomaly Scoring (Chạy Sliding Window tính Frame-level AUC-ROC)
python eval.py --config configs/eval.yaml --mode anomaly_scoring

# Đánh giá Linear Probing (Huấn luyện Linear Head với Frozen Backbone)
python eval.py --config configs/eval.yaml --mode linear_probing
```

---

## Tiềm năng mở rộng

Để phục vụ yêu cầu sau này có thể thay thế/bổ sung:

* Đầu vào khác (keypoints/optical flow)
* Các phương pháp Contrastive khác (thêm các cách Contrastive mới hoặc mở rộng/điều chỉnh các cách Contrastive hiện tại)
* Các lựa chọn augmentation khác

implement code này nên cho phép sự linh hoạt đó.
