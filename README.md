# Engagement Recognition V5

Nhận diện mức độ tham gia (engagement) từ video lớp học, kết hợp:

- **Body / Face**: feature ResNet50 đã cache → Temporal encoder (conv + attention có gate)
- **Skeleton**: 17 khớp COCO → Bi-LSTM + attention pooling, được điều biến bởi FiLM từ feature body
- **K-hop social graph**: K người xung quanh cùng thời điểm → GNN (Fb là 1 node trong graph)
- **Context window**: các clip trước/sau của cùng 1 người
- **Multimodal Fusion Transformer** → 4 lớp engagement
- Loss phụ khi train: Focal loss có trọng số lớp, ordinal (CORAL), SupCon, auxiliary hành vi/cảm xúc

## Cấu trúc thư mục

```
engagement-v5/
├── configs/
│   ├── default.yaml          # toàn bộ siêu tham số
│   ├── kaggle.yaml           # đường dẫn dữ liệu trên Kaggle (kế thừa default.yaml)
│   └── colab.yaml            # đường dẫn dữ liệu trên Google Drive (kế thừa default.yaml)
├── src/engagement/
│   ├── config.py             # đọc YAML, seed, device
│   ├── data/
│   │   ├── manifest.py       # đọc manifest, chia split theo session, nhãn, lọc cache
│   │   ├── geometry.py       # thời gian clip, bbox, đặc trưng quan hệ, hướng cơ thể
│   │   ├── graph_index.py    # xây K-hop neighbors & context window
│   │   └── dataset.py        # Dataset + DataLoader
│   ├── models/
│   │   ├── temporal.py       # CachedTemporalEncoderV3
│   │   ├── skeleton.py       # SkeletonBiLSTMEncoder
│   │   ├── layers.py         # GNN, social graph, context, FiLM, fusion
│   │   └── engagement_model.py  # EngagementModelV5
│   ├── losses.py             # Focal, ordinal, aux, SupCon, LossComputer
│   ├── engine.py             # train / evaluate / benchmark / fit / checkpoint
│   ├── evaluation.py         # tune bias, session-level, confusion matrix, risk-coverage, social
│   └── diagnostics.py        # rò rỉ split, độ ổn định nhãn, kiểm tra cache
├── scripts/
│   ├── train.py              # train + đánh giá test
│   ├── evaluate.py           # đánh giá 1 checkpoint có sẵn
│   ├── diagnose.py           # chẩn đoán dữ liệu
│   └── download_kaggle_data.py  # (Colab) tải dữ liệu từ Kaggle bằng kagglehub
├── notebooks/
│   ├── run_kaggle.ipynb      # notebook chạy trên Kaggle
│   └── run_colab.ipynb       # notebook chạy trên Colab
├── requirements.txt
└── pyproject.toml
```

## 1. Đưa code lên GitHub

```bash
cd engagement-v5
git init
git add .
git commit -m "Engagement V5"
git branch -M main
git remote add origin https://github.com/<USERNAME>/engagement-v5.git
git push -u origin main
```

Sau đó sửa `<USERNAME>` trong `REPO_URL` ở 2 notebook trong `notebooks/`.

> Repo **private**: tạo Personal Access Token trên GitHub (quyền đọc repo) và clone bằng
> `https://<TOKEN>@github.com/<USERNAME>/engagement-v5.git`. Không commit token vào repo.

## 2. Chạy trên Kaggle

1. Tạo notebook mới → **File → Import Notebook** → chọn `notebooks/run_kaggle.ipynb`
   (hoặc copy từng cell).
2. **Settings**: Accelerator = GPU, **Internet = On** (để `git clone`).
3. **Add Input** 4 dataset: `engagement-dataset-npy`, `resnet-engagement`, `skeleton-engagement`, `slip-names`.
   Kiểm tra đường dẫn thật trong panel Input có khớp `configs/kaggle.yaml` không.
4. Chạy lần lượt các cell. Kết quả nằm ở `/kaggle/working/outputs/`.

## 3. Chạy trên Colab

1. Mở `notebooks/run_colab.ipynb` bằng Colab (**File → Open notebook → GitHub**, dán link repo).
2. **Runtime → Change runtime type → GPU**.
3. Chọn cách lấy dữ liệu trong cell 2:
   - **Cách A (mặc định)**: tải thẳng từ Kaggle. Thêm `KAGGLE_USERNAME` và `KAGGLE_KEY`
     (lấy ở kaggle.com → Settings → API → Create New Token) vào **Colab Secrets**.
     Script sẽ tự sinh `configs/colab_auto.yaml`.
   - **Cách B**: dữ liệu đã có trong Google Drive → sửa đường dẫn trong `configs/colab.yaml`.
4. Chạy lần lượt các cell. Nên đặt `CHECKPOINT_DIR` trong Drive để không mất checkpoint khi Colab ngắt.

## 4. Dòng lệnh

```bash
# Chẩn đoán dữ liệu
python scripts/diagnose.py --config configs/kaggle.yaml

# Chạy thử nhanh (2 epoch, 32 sample/split)
python scripts/train.py --config configs/kaggle.yaml --set DRY_RUN=true

# Train đầy đủ + đánh giá test
python scripts/train.py --config configs/kaggle.yaml

# Ghi đè tham số bất kỳ mà không sửa file
python scripts/train.py --config configs/kaggle.yaml --set EPOCHS=40 LR=5e-5 USE_CONTEXT_WINDOW=false

# Đánh giá lại 1 checkpoint
python scripts/evaluate.py --config configs/kaggle.yaml --checkpoint /kaggle/working/outputs/best_model.pt
```

## 5. Kết quả đầu ra (`CHECKPOINT_DIR`)

| File | Nội dung |
|---|---|
| `best_model.pt` | checkpoint có val macro-F1 cao nhất |
| `history.csv` | loss / metric từng epoch |
| `test_results.json` | macro-F1 cấp segment và cấp session, bias đã tune, social diagnostics |
| `session_predictions.csv` | dự đoán majority vote cho từng session |
| `confusion_matrix.png`, `f1_coverage.png` | biểu đồ |
| `risk_coverage.csv` | macro-F1 theo tỷ lệ sample tự tin được giữ lại |

Checkpoint được chọn **chỉ** theo val macro-F1; logit bias chỉ được tune trên validation.
Vì mỗi session chỉ có 1 nhãn, nên báo cáo **session macro-F1** làm chỉ số chính.

## 6. Ablation

Chạy cùng seed/split, chỉ đổi cờ qua `--set`:

| Thí nghiệm | Lệnh thêm |
|---|---|
| Không context window | `--set USE_CONTEXT_WINDOW=false` |
| Thêm relation features | `--set USE_RELATION_FEATURES=true` |
| Không SupCon | `--set USE_SUPCON_LOSS=false` |
| Không ordinal | `--set USE_ORDINAL_AUX_LOSS=false` |
| Không aux hành vi/cảm xúc | `--set USE_BEHAVIOR_EMOTION_AUX=false` |
| Không FiLM | `--set USE_RESNET_TO_SKELETON_FUSION=false` |

Nhớ đổi `CHECKPOINT_DIR` cho mỗi thí nghiệm để không ghi đè, ví dụ
`--set CHECKPOINT_DIR=/kaggle/working/ablation_no_ctx`.
