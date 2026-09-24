"""Chẩn đoán dữ liệu trước khi train: thư mục cache, rò rỉ split, độ ổn định nhãn.

Ví dụ:
    python scripts/diagnose.py --config configs/kaggle.yaml
"""
from _common import build_parser

from engagement.config import load_config
from engagement.data import load_manifest
from engagement.diagnostics import (
    diagnose_cache_dirs,
    diagnose_label_stability,
    diagnose_track_split_leakage,
)


def main():
    args = build_parser("Chẩn đoán dữ liệu").parse_args()
    cfg = load_config(args.config, args.set)
    df = load_manifest(cfg)

    print("\n===== 1. Thư mục cache =====")
    diagnose_cache_dirs(df, cfg)
    print("\n===== 2. Rò rỉ track giữa các split =====")
    diagnose_track_split_leakage(df)
    print("\n===== 3. Độ ổn định nhãn trong track =====")
    diagnose_label_stability(df)


if __name__ == "__main__":
    main()
