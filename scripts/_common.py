"""Tiện ích chung cho các script: thêm src/ vào sys.path và parse tham số dòng lệnh."""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")   # chạy script không cần màn hình (Kaggle/Colab subprocess)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "kaggle.yaml"),
                        help="File YAML cấu hình (mặc định: configs/kaggle.yaml)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Ghi đè cấu hình, ví dụ: --set EPOCHS=5 DRY_RUN=true")
    return parser
