"""Dataset PyTorch và DataLoader cho từng split."""
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .graph_index import build_context_window_index, build_neighbor_index


def load_resnet_feature(feature_dir, kind, sample_id):
    """kind = 'body' | 'face'. Trả về (T, 2048) hoặc None."""
    path = os.path.join(feature_dir, f"features_{kind}", f"{sample_id}.npy")
    return np.load(path).astype(np.float32) if os.path.exists(path) else None


def valid_frame_mask(feat):
    """Frame hợp lệ = frame không toàn số 0."""
    return ~np.all(feat == 0, axis=-1)


def load_skeleton(skeleton_dir, sample_id, conf_thr):
    raw = np.load(os.path.join(skeleton_dir, f"{sample_id}.npy"), allow_pickle=True).item()
    kpts = np.array(raw["keypoints"], dtype=np.float32)            # (T, 17, 2)
    scores = np.array(raw["keypoint_scores"], dtype=np.float32)    # (T, 17)
    bbox = np.array(raw["bbox"], dtype=np.float32)                 # (T, 4)
    detected = np.array(raw["detected"], dtype=bool)               # (T,)

    # Chuẩn hoá toạ độ khớp theo bbox của từng frame
    x1, y1 = bbox[:, 0:1], bbox[:, 1:2]
    w = np.clip(bbox[:, 2:3] - x1, 1e-3, None)
    h = np.clip(bbox[:, 3:4] - y1, 1e-3, None)
    xy = np.stack([(kpts[..., 0] - x1) / w, (kpts[..., 1] - y1) / h], axis=-1)
    xy[~detected] = 0.0

    scores = scores.copy()
    scores[~detected] = 0.0
    kpt_mask = (scores >= conf_thr) & detected[:, None]
    frame_mask = kpt_mask.any(axis=1) & detected

    return {
        "skeleton_xy": xy.astype(np.float32),
        "skeleton_conf": scores,
        "skeleton_kpt_mask": kpt_mask,
        "skeleton_frame_mask": frame_mask,
    }


class EngagementDataset(Dataset):
    """Mỗi sample: body/face/skeleton của target + K hàng xóm + các clip context."""

    def __init__(self, sub_df, neighbor_lists, context_lists, cfg, label2id, aux_label2id):
        self.df = sub_df.reset_index(drop=True)
        self.sample_ids = self.df["sample_id"].tolist()
        self.neighbor_lists = neighbor_lists
        self.context_lists = context_lists
        self.cfg = cfg
        self.label2id = label2id
        self.aux_label2id = aux_label2id
        self.T = cfg["NUM_FRAMES"]
        self.D = cfg["FEATURE_DIM"]
        self.k_neighbors = cfg["K_NEIGHBORS"]
        self.window = cfg["CONTEXT_WINDOW_SIZE"]
        self.n_context = 2 * self.window

    def __len__(self):
        return len(self.df)

    def _load_modality(self, sample_id, kind):
        feat = load_resnet_feature(self.cfg["FEATURE_DIR"], kind, sample_id)
        if feat is None:
            return np.zeros((self.T, self.D), np.float32), np.zeros(self.T, bool)
        return feat, valid_frame_mask(feat)

    def _load_body_slots(self, entries, n_slots):
        """Nạp feature body của các clip khác (hàng xóm / context).
        Trả về feat, frame_mask, slot_mask và list (slot, entry, frame_mask) đã nạp được."""
        feat = np.zeros((n_slots, self.T, self.D), np.float32)
        frame_mask = np.zeros((n_slots, self.T), bool)
        slot_mask = np.zeros(n_slots, np.float32)
        filled = []
        for j, entry in enumerate(entries[:n_slots]):
            f = load_resnet_feature(self.cfg["FEATURE_DIR"], "body", self.sample_ids[entry["idx"]])
            if f is None:
                continue
            fm = valid_frame_mask(f)
            if not fm.any():
                continue
            feat[j], frame_mask[j], slot_mask[j] = f, fm, 1.0
            filled.append((j, entry, fm))
        return feat, frame_mask, slot_mask, filled

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]

        body_feat, body_mask = self._load_modality(sample_id, "body")
        face_feat, face_mask = self._load_modality(sample_id, "face")

        # Hàng xóm (không gian)
        nb_feat, nb_frame_mask, nb_mask, nb_filled = self._load_body_slots(
            self.neighbor_lists[idx], self.k_neighbors)
        nb_relation = np.zeros((self.k_neighbors, self.cfg["RELATION_DIM"]), np.float32)
        for j, entry, fm in nb_filled:
            nb_relation[j, :12] = entry["relation"]
            nb_relation[j, 12] = fm.mean()

        # Context (thời gian) - offset dịch về >= 0 để tra positional embedding
        ctx_feat, ctx_frame_mask, ctx_mask, ctx_filled = self._load_body_slots(
            self.context_lists[idx], self.n_context)
        ctx_offset = np.zeros(self.n_context, np.int64)
        for j, entry, _ in ctx_filled:
            ctx_offset[j] = entry["offset"] + self.window

        skeleton = load_skeleton(self.cfg["SKELETON_DIR"], sample_id, self.cfg["SKELETON_CONF_THR"])

        item = {
            "body_feat": body_feat,
            "body_frame_mask": body_mask,
            "face_feat": face_feat,
            "face_frame_mask": face_mask,
            "neighbor_feat": nb_feat,
            "neighbor_frame_mask": nb_frame_mask,
            "neighbor_mask": nb_mask,
            "neighbor_relation": nb_relation,
            "context_feat": ctx_feat,
            "context_frame_mask": ctx_frame_mask,
            "context_mask": ctx_mask,
            "context_offset": ctx_offset,
            **skeleton,
        }
        item = {key: torch.from_numpy(value) for key, value in item.items()}
        item["label"] = torch.tensor(self.label2id[row["engagement_label"]], dtype=torch.long)
        item["sample_id"] = sample_id

        # Nhãn aux: sample thiếu nhãn dùng id 0 làm placeholder (bị loại khỏi loss)
        aux_valid = bool(row.get("aux_valid", False)) and bool(self.aux_label2id)
        item["aux_valid"] = torch.tensor(aux_valid)
        for task, mapping in self.aux_label2id.items():
            col = self.cfg["AUX_LABEL_COLUMNS"][task]
            aux_id = mapping[str(row[col])] if aux_valid else 0
            item[f"aux_label_{task}"] = torch.tensor(aux_id, dtype=torch.long)
        return item


def build_sampler(labels, label2id, mode):
    """Oversample lớp thiểu số: trọng số sample = 1/count ('inverse') hoặc 1/sqrt(count)."""
    counts = labels.value_counts()

    def class_weight(label):
        c = max(counts.get(label, 0), 1)
        return 1.0 / c if mode == "inverse" else 1.0 / math.sqrt(c)

    weights = labels.map({lbl: class_weight(lbl) for lbl in label2id}).to_numpy(dtype=np.float64)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(weights), replacement=True)


def make_loader(bundle, split_name, cfg):
    """Trả về (DataLoader, Dataset) cho 1 split, hoặc (None, None) nếu split rỗng."""
    sub_df = bundle.df[bundle.df["split"] == split_name].reset_index(drop=True)
    if sub_df.empty:
        return None, None

    if cfg["DRY_RUN"] and len(sub_df) > cfg["DRY_RUN_SAMPLES"]:
        sub_df = sub_df.sample(n=cfg["DRY_RUN_SAMPLES"], random_state=cfg["SEED"]).reset_index(drop=True)

    # Neighbor & context được xây RIÊNG trong từng split để tránh rò rỉ
    neighbor_lists = build_neighbor_index(sub_df, cfg)
    context_lists = build_context_window_index(
        sub_df, cfg["CONTEXT_WINDOW_SIZE"], cfg["CONTEXT_MAX_TIME_GAP_SECONDS"])

    n = len(sub_df)
    n_nb = sum(bool(x) for x in neighbor_lists)
    n_ctx = sum(bool(x) for x in context_lists)
    print(f"[{split_name}] {n} sample | có neighbor: {n_nb} ({100 * n_nb / n:.1f}%) "
          f"| có context: {n_ctx} ({100 * n_ctx / n:.1f}%)")

    dataset = EngagementDataset(sub_df, neighbor_lists, context_lists, cfg,
                                bundle.label2id, bundle.aux_label2id)

    is_train = split_name == "train"
    sampler = None
    if is_train and cfg["USE_WEIGHTED_SAMPLER"]:
        sampler = build_sampler(sub_df["engagement_label"], bundle.label2id, cfg["SAMPLER_MODE"])

    loader_kwargs = dict(
        batch_size=cfg["BATCH_SIZE"],
        shuffle=is_train and sampler is None,
        sampler=sampler,
        num_workers=cfg["NUM_WORKERS"],
        pin_memory=cfg["PIN_MEMORY"] and torch.cuda.is_available(),
        drop_last=is_train,
    )
    if cfg["NUM_WORKERS"] > 0:
        loader_kwargs["persistent_workers"] = cfg["PERSISTENT_WORKERS"]
        loader_kwargs["prefetch_factor"] = cfg["PREFETCH_FACTOR"]

    return DataLoader(dataset, **loader_kwargs), dataset


def make_loaders(bundle, cfg):
    """{'train': loader, 'val': loader, 'test': loader} (loader có thể là None)."""
    return {split: make_loader(bundle, split, cfg)[0] for split in ("train", "val", "test")}
