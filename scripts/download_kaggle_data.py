"""(Colab) Tải 4 dataset từ Kaggle bằng kagglehub và sinh configs/colab_auto.yaml.

Cần đăng nhập Kaggle trước (một trong các cách):
  - Colab Secrets: thêm KAGGLE_USERNAME và KAGGLE_KEY
  - hoặc import kagglehub; kagglehub.login()

Ví dụ:
    python scripts/download_kaggle_data.py
    python scripts/train.py --config configs/colab_auto.yaml
"""
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (key trong config, dataset Kaggle, đường dẫn con bên trong dataset)
DATASETS = [
    ("OUTPUT_DIR", "deadwish1/engagement-dataset-npy", ""),
    ("FEATURE_DIR", "drakhight/resnet-engagement", "resnet_features"),
    ("SKELETON_DIR", "drakhight/skeleton-engagement", "skeleton_features_yolo/skeleton"),
    ("SPLIT_NAMES_JSON_PATH", "deadwish1/slip-names", "split_names.json"),
]


def main():
    import kagglehub

    cfg = {"_base_": "default.yaml", "CHECKPOINT_DIR": "/content/outputs", "NUM_WORKERS": 2}
    for key, handle, sub_path in DATASETS:
        print(f"Đang tải {handle} ...")
        root = kagglehub.dataset_download(handle)
        path = os.path.join(root, sub_path) if sub_path else root
        if not os.path.exists(path):
            print(f"  [!] Không thấy {path} - kiểm tra lại cấu trúc dataset và sửa trong colab_auto.yaml")
        cfg[key] = path
        print(f"  {key} = {path}")

    out_path = os.path.join(ROOT, "configs", "colab_auto.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"\nĐã tạo {out_path}")


if __name__ == "__main__":
    main()
