"""Dataset, collate, sampler and DataLoader construction.

Corresponds to notebook section "3. Dataset -- target modalities + neighbor
sequences + interaction relation vectors" (cell 20).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .graph import (
    RELATION_FEATURE_NAMES_BASE,
    _get_clip_time_range,
    build_context_window_index,
    build_neighbor_index,
    print_context_stats,
    print_neighbor_stats,
)


def get_relation_feature_names(cfg: dict):
    names = RELATION_FEATURE_NAMES_BASE + ["neighbor_feature_quality"]
    assert len(names) == cfg["RELATION_DIM"]
    return names


def _load_feature_body(sample_id, feature_dir):
    path = os.path.join(feature_dir, "features_body", f"{sample_id}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path).astype(np.float32)


def _load_feature_face(sample_id, feature_dir):
    path = os.path.join(feature_dir, "features_face", f"{sample_id}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path).astype(np.float32)


def _load_skeleton_npy(sample_id, skeleton_dir, conf_thr):
    path = os.path.join(skeleton_dir, f"{sample_id}.npy")
    raw = np.load(path, allow_pickle=True).item()

    keypoints = np.array(raw["keypoints"], dtype=np.float32)
    keypoint_scores = np.array(raw["keypoint_scores"], dtype=np.float32)
    bbox = np.array(raw["bbox"], dtype=np.float32)
    detected = np.array(raw["detected"], dtype=bool)

    x1 = bbox[:, 0:1]
    y1 = bbox[:, 1:2]
    w = np.clip(bbox[:, 2:3] - bbox[:, 0:1], 1e-3, None)
    h = np.clip(bbox[:, 3:4] - bbox[:, 1:2], 1e-3, None)

    norm_x = (keypoints[:, :, 0] - x1) / w
    norm_y = (keypoints[:, :, 1] - y1) / h
    norm_kpts = np.stack([norm_x, norm_y], axis=-1)
    norm_kpts[~detected] = 0.0

    keypoint_scores = keypoint_scores.copy()
    keypoint_scores[~detected] = 0.0
    keypoint_mask = (keypoint_scores >= conf_thr) & detected[:, None]
    frame_mask = keypoint_mask.any(axis=1) & detected

    return {
        "skeleton_xy": norm_kpts.astype(np.float32),
        "skeleton_conf": keypoint_scores.astype(np.float32),
        "skeleton_kpt_mask": keypoint_mask,
        "skeleton_frame_mask": frame_mask,
    }


class CachedMultiModalDataset(Dataset):
    """Target body/face/skeleton plus K neighbor body sequences (spatial) and context
    window body sequences (temporal, same person before/after) plus edge relations."""

    def __init__(self, sub_df, sub_neighbor_lists, sub_context_lists, cfg, label2id,
                 num_frames, k_neighbors, aux_label2id, neighbor_relation_dim):
        self.df = sub_df.reset_index(drop=True)
        self.neighbor_lists = sub_neighbor_lists
        self.context_lists = sub_context_lists
        self.feature_dir = cfg["FEATURE_DIR"]
        self.skeleton_dir = cfg["SKELETON_DIR"]
        self.conf_thr = cfg["SKELETON_CONF_THR"]
        self.feature_dim = cfg["FEATURE_DIM"]
        self.num_frames = num_frames
        self.k_neighbors = k_neighbors
        self.context_window_size = cfg.get("CONTEXT_WINDOW_SIZE", 2)
        self.context_max_slots = self.context_window_size * 2
        self.label2id = label2id
        self.use_aux = cfg.get("USE_BEHAVIOR_EMOTION_AUX", False)
        self.aux_label_columns = cfg.get("AUX_LABEL_COLUMNS", {})
        self.aux_label2id = aux_label2id
        self.neighbor_relation_dim = neighbor_relation_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]

        # Target body.
        body_feat = _load_feature_body(sample_id, self.feature_dir)
        if body_feat is None:
            body_feat = np.zeros((self.num_frames, self.feature_dim), dtype=np.float32)
            body_frame_mask = np.zeros(self.num_frames, dtype=bool)
        else:
            body_frame_mask = ~np.all(body_feat == 0, axis=-1)

        # Target face may be absent.
        face_feat = _load_feature_face(sample_id, self.feature_dir)
        if face_feat is None:
            face_feat = np.zeros((self.num_frames, self.feature_dim), dtype=np.float32)
            face_frame_mask = np.zeros(self.num_frames, dtype=bool)
            face_present = 0.0
        else:
            face_frame_mask = ~np.all(face_feat == 0, axis=-1)
            face_present = float(face_frame_mask.any())

        # Neighbors.
        neighbor_entries = self.neighbor_lists[idx][: self.k_neighbors]
        neighbor_feat = np.zeros(
            (self.k_neighbors, self.num_frames, self.feature_dim), dtype=np.float32
        )
        neighbor_frame_mask = np.zeros((self.k_neighbors, self.num_frames), dtype=bool)
        neighbor_mask = np.zeros(self.k_neighbors, dtype=np.float32)
        neighbor_relation = np.zeros((self.k_neighbors, self.neighbor_relation_dim), dtype=np.float32)

        for j, entry in enumerate(neighbor_entries):
            n_row = self.df.iloc[entry["idx"]]
            n_sample_id = n_row["sample_id"]
            n_body_feat = _load_feature_body(n_sample_id, self.feature_dir)
            if n_body_feat is None:
                continue

            n_frame_mask = ~np.all(n_body_feat == 0, axis=-1)
            if not n_frame_mask.any():
                continue

            neighbor_feat[j] = n_body_feat
            neighbor_frame_mask[j] = n_frame_mask
            neighbor_mask[j] = 1.0

            relation12 = np.asarray(entry["relation"], dtype=np.float32)
            neighbor_relation[j, :12] = relation12
            neighbor_relation[j, 12] = float(n_frame_mask.mean())

        # ---- Context window (temporal context: clip before/after, same person) ----
        context_entries = self.context_lists[idx][: self.context_max_slots]
        context_feat = np.zeros(
            (self.context_max_slots, self.num_frames, self.feature_dim), dtype=np.float32
        )
        context_frame_mask = np.zeros((self.context_max_slots, self.num_frames), dtype=bool)
        context_mask = np.zeros(self.context_max_slots, dtype=np.float32)
        context_offset = np.zeros(self.context_max_slots, dtype=np.int64)

        for j, entry in enumerate(context_entries):
            c_row = self.df.iloc[entry["idx"]]
            c_sample_id = c_row["sample_id"]
            c_body_feat = _load_feature_body(c_sample_id, self.feature_dir)
            if c_body_feat is None:
                continue

            c_frame_mask = ~np.all(c_body_feat == 0, axis=-1)
            if not c_frame_mask.any():
                continue

            context_feat[j] = c_body_feat
            context_frame_mask[j] = c_frame_mask
            context_mask[j] = 1.0
            context_offset[j] = entry["offset"] + self.context_window_size

        # Skeleton.
        skel = _load_skeleton_npy(sample_id, self.skeleton_dir, self.conf_thr)
        label = self.label2id[row["engagement_label"]]

        # ---- Behavior/emotion auxiliary labels (pose/act/obj/int/emo) ----
        aux_valid = bool(row.get("aux_valid", False)) if self.use_aux else False
        aux_label_ids = {}
        if self.use_aux:
            for task_name in self.aux_label2id.keys():
                col = self.aux_label_columns[task_name]
                if aux_valid:
                    raw_val = str(row[col])
                    aux_label_ids[task_name] = self.aux_label2id[task_name][raw_val]
                else:
                    aux_label_ids[task_name] = 0

        return {
            "body_feat": torch.from_numpy(body_feat),
            "body_frame_mask": torch.from_numpy(body_frame_mask),
            "face_feat": torch.from_numpy(face_feat),
            "face_mask": torch.tensor(face_present, dtype=torch.float32),
            "face_frame_mask": torch.from_numpy(face_frame_mask),
            "neighbor_feat": torch.from_numpy(neighbor_feat),
            "neighbor_mask": torch.from_numpy(neighbor_mask),
            "neighbor_frame_mask": torch.from_numpy(neighbor_frame_mask),
            "neighbor_relation": torch.from_numpy(neighbor_relation),
            "context_feat": torch.from_numpy(context_feat),
            "context_mask": torch.from_numpy(context_mask),
            "context_frame_mask": torch.from_numpy(context_frame_mask),
            "context_offset": torch.from_numpy(context_offset),
            "skeleton_xy": torch.from_numpy(skel["skeleton_xy"]),
            "skeleton_conf": torch.from_numpy(skel["skeleton_conf"]),
            "skeleton_kpt_mask": torch.from_numpy(skel["skeleton_kpt_mask"]),
            "skeleton_frame_mask": torch.from_numpy(skel["skeleton_frame_mask"]),
            "label": torch.tensor(label, dtype=torch.long),
            "aux_valid": torch.tensor(aux_valid, dtype=torch.bool),
            **{
                f"aux_label_{task_name}": torch.tensor(aux_label_ids.get(task_name, 0), dtype=torch.long)
                for task_name in (self.aux_label2id.keys() if self.use_aux else [])
            },
            "sample_id": sample_id,
        }


class TrackSequenceDataset(Dataset):
    """Wraps CachedMultiModalDataset: each sample is now the WHOLE sequence of
    segments of one TRACK (session+camera_id+object_id) instead of a single
    independent segment. K-hop and Context Window still run per-segment internally
    (unchanged) -- this only groups multiple segments into a sequence for a
    Track-level Bi-LSTM on top."""

    def __init__(self, segment_dataset, max_track_len=32):
        self.segment_ds = segment_dataset
        self.max_track_len = max_track_len
        self.tracks = self._build_tracks(segment_dataset.df)

    def _build_tracks(self, df):
        group_cols = [c for c in ["session", "camera_id", "object_id"] if c in df.columns]
        if len(group_cols) < 3:
            print("[!] Thieu cot session/camera_id/object_id -- moi segment se la 1 "
                  "track rieng (do dai 1, khong co ngu canh track).")
            return [[i] for i in range(len(df))]

        tracks = []
        for _, group in df.groupby(group_cols, sort=False):
            time_ranges = group.apply(_get_clip_time_range, axis=1)
            valid_mask = time_ranges.apply(lambda r: r is not None)

            if valid_mask.sum() == 0:
                tracks.append(group.index.tolist())
                continue

            sub = group[valid_mask].copy()
            sub["_t_start"] = time_ranges[valid_mask].apply(lambda r: r[0])
            sub = sub.sort_values("_t_start")
            tracks.append(sub.index.tolist())

            invalid_idx = group.index[~valid_mask].tolist()
            if invalid_idx:
                tracks.append(invalid_idx)

        return tracks

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, track_idx):
        positions = self.tracks[track_idx][: self.max_track_len]
        segment_items = [self.segment_ds[pos] for pos in positions]

        out = {}
        for key in segment_items[0].keys():
            values = [item[key] for item in segment_items]
            if torch.is_tensor(values[0]):
                out[key] = torch.stack(values, dim=0)  # (L, ...)
            else:
                out[key] = values  # e.g. sample_id -> list[str]

        out["seq_len"] = len(positions)
        return out


def track_collate_fn(batch):
    """Pad tracks (variable length L) to a common L_max within the batch. Returns
    'seq_mask' (B, L_max) bool -- True = real segment, False = padding."""
    seq_lens = [item["seq_len"] for item in batch]
    L_max = max(seq_lens)
    B = len(batch)

    tensor_keys = [k for k in batch[0].keys() if torch.is_tensor(batch[0][k])]
    list_keys = [k for k in batch[0].keys() if k not in tensor_keys and k != "seq_len"]

    out = {}
    for key in tensor_keys:
        sample_tensor = batch[0][key]
        pad_shape = sample_tensor.shape[1:]
        padded = sample_tensor.new_zeros((B, L_max, *pad_shape))
        for i, item in enumerate(batch):
            L_i = item[key].shape[0]
            padded[i, :L_i] = item[key]
        out[key] = padded

    for key in list_keys:
        out[key] = [item[key] for item in batch]

    seq_mask = torch.zeros(B, L_max, dtype=torch.bool)
    for i, L_i in enumerate(seq_lens):
        seq_mask[i, :L_i] = True

    out["seq_mask"] = seq_mask
    out["seq_lens"] = torch.tensor(seq_lens, dtype=torch.long)
    return out


def build_sample_weights(labels_series, label2id, mode="sqrt_inverse"):
    counts = labels_series.value_counts()
    class_w = {}
    for lbl, idx in label2id.items():
        c = max(counts.get(lbl, 0), 1)
        class_w[lbl] = (1.0 / c) if mode == "inverse" else (1.0 / math.sqrt(c))
    return labels_series.map(class_w).values.astype("float64")


def make_loader(df, split_name, shuffle, cfg, label2id, aux_label2id, neighbor_relation_dim):
    """Build a DataLoader (+ underlying Dataset) for one split. Neighbor / context
    graphs are (re)built INSIDE the split to avoid any train/val/test leakage."""
    mask = df["split"] == split_name
    if mask.sum() == 0:
        return None, None
    sub_df = df[mask].reset_index(drop=True)

    if cfg.get("DRY_RUN", False):
        n_dry = cfg.get("DRY_RUN_SAMPLES", 32)
        if len(sub_df) > n_dry:
            sub_df = sub_df.sample(n=n_dry, random_state=cfg.get("SEED", 42)).reset_index(drop=True)
        print(f"  [DRY RUN] Split '{split_name}': {len(sub_df)} samples.")

    sub_neighbor_lists = build_neighbor_index(
        sub_df,
        k_neighbors=cfg["K_NEIGHBORS"],
        candidate_multiplier=cfg.get("NEIGHBOR_CANDIDATE_MULTIPLIER", 6),
        require_time_overlap=cfg.get("NEIGHBOR_REQUIRE_TIME_OVERLAP", True),
        skeleton_dir=cfg.get("SKELETON_DIR"),
        skeleton_conf_thr=cfg.get("SKELETON_CONF_THR", 0.05),
        use_orientation_penalty=cfg.get("USE_ORIENTATION_PENALTY", True),
        orientation_angle_threshold_deg=cfg.get("ORIENTATION_ANGLE_THRESHOLD_DEG", 60.0),
        orientation_penalty=cfg.get("ORIENTATION_PENALTY", 1.0),
    )
    print_neighbor_stats(sub_neighbor_lists, sub_df, cfg["K_NEIGHBORS"], prefix=f"  [{split_name}] ")

    sub_context_lists = build_context_window_index(
        sub_df, window_size=cfg["CONTEXT_WINDOW_SIZE"],
        max_time_gap_seconds=cfg.get("CONTEXT_MAX_TIME_GAP_SECONDS", 60.0),
    )
    print_context_stats(sub_context_lists, sub_df, cfg["CONTEXT_WINDOW_SIZE"], prefix=f"  [{split_name}] ")

    segment_ds = CachedMultiModalDataset(
        sub_df, sub_neighbor_lists, sub_context_lists, cfg, label2id,
        num_frames=cfg["NUM_FRAMES"], k_neighbors=cfg["K_NEIGHBORS"],
        aux_label2id=aux_label2id, neighbor_relation_dim=neighbor_relation_dim,
    )

    use_track_level = cfg.get("USE_TRACK_LEVEL_MODEL", False)
    if use_track_level:
        ds = TrackSequenceDataset(segment_ds, max_track_len=cfg.get("MAX_TRACK_LEN", 32))
        track_lengths = [len(t) for t in ds.tracks]
        print(f"  [{split_name}] {len(ds)} track (tu {len(segment_ds)} segment), "
              f"do dai track: mean={np.mean(track_lengths):.1f} "
              f"min={min(track_lengths)} max={max(track_lengths)}")
        collate_fn = track_collate_fn
    else:
        ds = segment_ds
        collate_fn = None

    use_sampler = (
        shuffle and split_name == "train"
        and cfg.get("USE_WEIGHTED_SAMPLER", False)
        and not use_track_level
    )
    sampler = None
    effective_shuffle = shuffle
    if use_sampler:
        sample_weights = build_sample_weights(
            sub_df["engagement_label"], label2id, mode=cfg.get("SAMPLER_MODE", "sqrt_inverse")
        )
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights), replacement=True,
        )
        effective_shuffle = False
        print(f"  [{split_name}] WeightedRandomSampler mode={cfg.get('SAMPLER_MODE')}")
    elif use_track_level and cfg.get("USE_WEIGHTED_SAMPLER", False):
        print(f"  [{split_name}] [!] USE_WEIGHTED_SAMPLER bi bo qua trong che do track-level.")

    loader_kwargs = dict(
        dataset=ds,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=effective_shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=cfg["NUM_WORKERS"],
        pin_memory=cfg["PIN_MEMORY"],
        drop_last=(split_name == "train"),
    )
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    if cfg["NUM_WORKERS"] > 0:
        loader_kwargs["persistent_workers"] = cfg.get("PERSISTENT_WORKERS", True)
        loader_kwargs["prefetch_factor"] = cfg.get("PREFETCH_FACTOR", 2)

    return DataLoader(**loader_kwargs), ds


def build_dataloaders(df, cfg, label2id, aux_label2id):
    """Build train/val/test loaders + datasets. Returns a dict with keys
    'train_loader', 'train_ds', 'val_loader', 'val_ds', 'test_loader', 'test_ds',
    and 'relation_feature_names'."""
    relation_feature_names = get_relation_feature_names(cfg)
    neighbor_relation_dim = len(relation_feature_names)

    train_loader, train_ds = make_loader(df, "train", True, cfg, label2id, aux_label2id, neighbor_relation_dim)
    val_loader, val_ds = make_loader(df, "val", False, cfg, label2id, aux_label2id, neighbor_relation_dim)
    test_loader, test_ds = make_loader(df, "test", False, cfg, label2id, aux_label2id, neighbor_relation_dim)

    print("Relation features:", relation_feature_names)
    print("Train/Val/Test:",
          len(train_ds) if train_ds is not None else 0,
          len(val_ds) if val_ds is not None else 0,
          len(test_ds) if test_ds is not None else 0)

    return {
        "train_loader": train_loader, "train_ds": train_ds,
        "val_loader": val_loader, "val_ds": val_ds,
        "test_loader": test_loader, "test_ds": test_ds,
        "relation_feature_names": relation_feature_names,
    }
