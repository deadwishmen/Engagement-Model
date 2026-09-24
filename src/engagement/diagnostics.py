"""Chẩn đoán dữ liệu: rò rỉ split, độ ổn định nhãn, thư mục cache."""
import os

import pandas as pd

from .data.geometry import TRACK_COLS, sort_group_by_time
from .data.manifest import cache_dirs


def diagnose_track_split_leakage(df):
    """1 track (cùng người) có bị chia vào nhiều split không?"""
    if not all(c in df.columns for c in TRACK_COLS):
        print("[!] Thiếu cột session/camera_id/object_id -> bỏ qua.")
        return None

    splits_per_track = df.groupby(TRACK_COLS)["split"].nunique()
    leaked = splits_per_track[splits_per_track > 1]
    print(f"Tổng số track: {len(splits_per_track)} | Track bị rò rỉ: {len(leaked)}")

    if leaked.empty:
        print("[OK] Mỗi track nằm trọn trong 1 split.")
    else:
        print("[CẢNH BÁO] Có track bị chia vào nhiều split, ví dụ:")
        for key in leaked.head(5).index:
            mask = (df[TRACK_COLS] == pd.Series(key, index=TRACK_COLS)).all(axis=1)
            print(f"  {dict(zip(TRACK_COLS, key))}: {df.loc[mask, 'split'].value_counts().to_dict()}")
    return leaked


def diagnose_label_stability(df, max_gap_seconds=15.0):
    """Tỷ lệ nhãn 'nhảy' giữa các segment liền kề của cùng 1 track (proxy cho nhiễu nhãn)."""
    if not all(c in df.columns for c in TRACK_COLS):
        print("[!] Thiếu cột track -> bỏ qua.")
        return None

    pairs = []
    for _, group in df.groupby(TRACK_COLS, sort=False):
        if len(group) < 2:
            continue
        idxs, starts = sort_group_by_time(group)
        labels = df.loc[idxs, "engagement_label"].tolist()
        for i in range(len(idxs) - 1):
            gap = starts[i + 1] - starts[i]
            if gap <= max_gap_seconds:
                pairs.append({"label_a": labels[i], "label_b": labels[i + 1],
                              "time_gap": gap, "is_same": labels[i] == labels[i + 1]})

    if not pairs:
        print(f"[!] Không có cặp segment nào cách nhau <= {max_gap_seconds}s.")
        return None

    pairs = pd.DataFrame(pairs)
    flip_rate = 1 - pairs["is_same"].mean()
    print(f"Số cặp segment liền kề: {len(pairs)} | Tỷ lệ nhãn khác nhau: {100 * flip_rate:.1f}%")
    print("\nBảng chuyển nhãn (hàng = trước, cột = sau):")
    print(pd.crosstab(pairs["label_a"], pairs["label_b"]))

    print("\nTỷ lệ nhảy nhãn theo khoảng cách thời gian:")
    for max_gap in (3.0, 5.0, 10.0, max_gap_seconds):
        sub = pairs[pairs["time_gap"] <= max_gap]
        if len(sub):
            print(f"  gap <= {max_gap:4.1f}s: {len(sub):5d} cặp, nhảy {100 * (1 - sub['is_same'].mean()):.1f}%")

    if flip_rate > 0.40:
        print("\n[CẢNH BÁO] Nhãn rất không ổn định -> nhiễu nhãn đang giới hạn macro-F1.")
    elif flip_rate > 0.20:
        print("\n[LƯU Ý] Nhiễu nhãn ở mức trung bình.")
    else:
        print("\n[OK] Nhãn tương đối ổn định.")
    return pairs


def diagnose_cache_dirs(df, cfg):
    """Kiểm tra thư mục feature/skeleton có tồn tại và sample_id có khớp tên file không."""
    manifest_ids = set(df["sample_id"])
    print(f"Số sample_id trong manifest: {len(manifest_ids)} (ví dụ: {list(manifest_ids)[:3]})\n")

    for name, folder in cache_dirs(cfg).items():
        if not os.path.isdir(folder):
            print(f"[X] {name}: thư mục không tồn tại -> {folder}\n")
            continue
        files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        ids = {os.path.splitext(f)[0] for f in files}
        print(f"{name}: {len(files)} file .npy (ví dụ: {files[:3]})")
        print(f"  -> khớp với manifest: {len(manifest_ids & ids)}/{len(manifest_ids)}\n")
