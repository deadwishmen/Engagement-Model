"""Filter the manifest to samples that actually have cached features on disk.

Corresponds to notebook section "2c. Loc df theo sample co san ca feature ResNet lan
skeleton cache" (cell 18).
"""
from __future__ import annotations

import os

import pandas as pd


def _feat_body_exists(sid, feature_dir):
    return os.path.exists(os.path.join(feature_dir, "features_body", f"{sid}.npy"))


def _feat_face_exists(sid, feature_dir):
    return os.path.exists(os.path.join(feature_dir, "features_face", f"{sid}.npy"))


def _skel_exists(sid, skeleton_dir):
    return os.path.exists(os.path.join(skeleton_dir, f"{sid}.npy"))


def filter_by_cache_availability(df: pd.DataFrame, neighbor_lists, cfg: dict):
    """Keep only samples with BOTH a body feature and a skeleton cache file (face is
    optional). Re-indexes `neighbor_lists` to match the filtered df and drops any
    neighbor edges pointing at now-removed samples. Also rebuilds label2id/id2label/
    num_classes on the filtered set (some rare classes can lose all their samples).

    Returns: (df, neighbor_lists, label2id, id2label, num_classes)
    """
    n_before = len(df)
    df = df.copy()
    df["feature_body_exists"] = df["sample_id"].apply(lambda sid: _feat_body_exists(sid, cfg["FEATURE_DIR"]))
    df["feature_face_exists"] = df["sample_id"].apply(lambda sid: _feat_face_exists(sid, cfg["FEATURE_DIR"]))
    df["skeleton_exists"] = df["sample_id"].apply(lambda sid: _skel_exists(sid, cfg["SKELETON_DIR"]))

    keep_mask = df["feature_body_exists"] & df["skeleton_exists"]
    df_kept = df[keep_mask].reset_index(drop=False)  # keep old index before filtering
    old_to_new_idx = {old: new for new, old in enumerate(df_kept["index"].tolist())}

    df = df_kept.drop(columns=["index"]).reset_index(drop=True)
    print(f"Loai {n_before - len(df)} sample thieu feature_body ResNet hoac skeleton cache. "
          f"Con lai {len(df)} sample dung duoc.")

    assert len(df) > 0, (
        "KHONG CON SAMPLE NAO sau khi loc theo feature/skeleton cache -- kiem tra lai "
        "duong dan FEATURE_DIR/SKELETON_DIR hoac dinh dang sample_id."
    )

    new_neighbor_lists = [[] for _ in range(len(df))]
    for old_idx, new_idx in old_to_new_idx.items():
        for entry in neighbor_lists[old_idx]:
            cand_old_idx = entry["idx"]
            if cand_old_idx not in old_to_new_idx:
                continue
            new_entry = dict(entry)
            new_entry["idx"] = old_to_new_idx[cand_old_idx]
            new_neighbor_lists[new_idx].append(new_entry)
    neighbor_lists = new_neighbor_lists

    n_with_nb_after = sum(1 for lst in neighbor_lists if len(lst) > 0)
    print(f"Sau dong bo: {n_with_nb_after}/{len(df)} sample con it nhat 1 hang xom hop le "
          f"({100 * n_with_nb_after / max(len(df), 1):.1f}%).")

    train_mask = df["split"] == "train"
    label_names = sorted(df.loc[train_mask, "engagement_label"].unique().tolist())
    label2id = {lbl: i for i, lbl in enumerate(label_names)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    num_classes = len(label2id)
    print(f"So lop engagement (tu TRAIN, SAU khi loc feature/skeleton cache): {num_classes} -> {label2id}")

    for split_name in ("val", "test"):
        split_mask = df["split"] == split_name
        split_labels = set(df.loc[split_mask, "engagement_label"].unique().tolist())
        unseen = split_labels - set(label_names)
        if unseen:
            n_drop = int((split_mask & df["engagement_label"].isin(unseen)).sum())
            print(f"[!] Tap {split_name} co nhan la: {unseen} -- loai {n_drop} sample.")
            df = df[~(split_mask & df["engagement_label"].isin(unseen))].reset_index(drop=True)
            print("  [!] Luu y: neighbor_lists CHUA duoc dong bo lai sau buoc loc nhan la nay "
                  "(truong hop hiem, thuong khong xay ra).")

    print(df["engagement_label"].value_counts())
    return df, neighbor_lists, label2id, id2label, num_classes
