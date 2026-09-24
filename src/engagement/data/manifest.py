"""Đọc manifest, chia split theo session, tạo nhãn và lọc sample thiếu cache."""
import ast
import json
import os
import re
from dataclasses import dataclass, field

import pandas as pd

KNOWN_SPLIT_PREFIXES = ("train", "val", "test")
BBOX_NUMBER_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")
JSON_COLUMNS = ["bbox_pixel", "bbox_norm", "bbox_time_sec", "frame_paths", "frame_paths_face"]


@dataclass
class DataBundle:
    """Toàn bộ dữ liệu đã chuẩn bị: DataFrame + các bảng ánh xạ nhãn."""
    df: pd.DataFrame
    label2id: dict
    id2label: dict
    aux_label2id: dict = field(default_factory=dict)

    @property
    def num_classes(self):
        return len(self.label2id)

    @property
    def label_names(self):
        return [self.id2label[i] for i in range(self.num_classes)]

    @property
    def aux_task_names(self):
        return list(self.aux_label2id)

    @property
    def aux_num_classes(self):
        return {task: len(mapping) for task, mapping in self.aux_label2id.items()}


# ------------------------------------------------------------------ parse
def parse_maybe_json(value):
    """Chuỗi JSON / Python literal -> object. Không parse được thì giữ nguyên."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def fix_stringified_bbox_list(bbox_list):
    """['[x1, y1, x2, y2]', ...] -> [[x1, y1, x2, y2], ...]"""
    if not isinstance(bbox_list, list):
        return bbox_list
    return [
        [float(x) for x in BBOX_NUMBER_RE.findall(item)] if isinstance(item, str) else item
        for item in bbox_list
    ]


def resolve_clip_abs_path(output_dir, clip_path):
    clip_path = clip_path.replace("\\", "/")
    prefix = clip_path.split("/", 1)[0]
    if "/" in clip_path and prefix in KNOWN_SPLIT_PREFIXES:
        return os.path.join(output_dir, f"clips_{prefix}", clip_path)
    return os.path.join(output_dir, "clips", clip_path)


def clip_exists(clip_path, output_dir):
    if not isinstance(clip_path, str) or not clip_path.strip():
        return False
    return os.path.exists(resolve_clip_abs_path(output_dir, clip_path))


# ------------------------------------------------------------------ đọc manifest
def read_manifest(output_dir):
    """Tìm manifest.csv / .parquet / .jsonl trong manifest/manifest hoặc manifest/."""
    for manifest_dir in (os.path.join(output_dir, "manifest", "manifest"),
                         os.path.join(output_dir, "manifest")):
        for ext in ("csv", "parquet", "jsonl"):
            path = os.path.join(manifest_dir, f"manifest.{ext}")
            if not os.path.exists(path):
                continue
            print(f"Đọc manifest: {path}")
            if ext == "csv":
                return pd.read_csv(path)
            if ext == "parquet":
                return pd.read_parquet(path)
            with open(path, encoding="utf-8") as f:
                return pd.DataFrame([json.loads(line) for line in f if line.strip()])
    raise FileNotFoundError(f"Không tìm thấy manifest trong {output_dir}/manifest")


def apply_session_split(df, json_path):
    """Ghi đè cột 'split' theo file split_names.json (chia ở cấp SESSION)."""
    if not json_path:
        return df
    if not os.path.exists(json_path):
        print(f"[!] Không thấy {json_path} -> dùng cột 'split' có sẵn.")
        return df

    with open(json_path, encoding="utf-8") as f:
        split_names = json.load(f)

    session_to_split = {
        (fname[:-3] if fname.endswith(".pt") else fname): split_name
        for split_name, fnames in split_names.items()
        for fname in fnames
    }

    df = df.copy()
    df["split"] = df["session"].map(session_to_split)
    n_missing = int(df["split"].isna().sum())
    if n_missing:
        print(f"[!] Loại {n_missing} segment có session không nằm trong split_names.json.")
        df = df[df["split"].notna()].reset_index(drop=True)

    print(f"Đã chia split theo session từ {json_path}:")
    print(df["split"].value_counts().to_string())
    return df


def load_manifest(cfg):
    df = read_manifest(cfg["OUTPUT_DIR"])
    print(f"Số dòng manifest: {len(df)}")

    for col in JSON_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(parse_maybe_json)
    for col in ("bbox_pixel", "bbox_norm"):
        if col in df.columns:
            df[col] = df[col].apply(fix_stringified_bbox_list)

    if cfg["EXCLUDE_UNKNOWN"] and "engagement_label" in df.columns:
        df = df[df["engagement_label"] != "unknown"].reset_index(drop=True)

    body_ok = df["clip_path"].apply(lambda p: clip_exists(p, cfg["OUTPUT_DIR"]))
    df = df[body_ok].reset_index(drop=True)

    if "sample_id" not in df.columns:
        df["sample_id"] = df.index
    df["sample_id"] = df["sample_id"].astype(str)

    df = apply_session_split(df, cfg.get("SPLIT_NAMES_JSON_PATH"))
    print(f"Còn {len(df)} sample sau khi lọc unknown / clip không tồn tại.")
    return df


# ------------------------------------------------------------------ nhãn
def build_label_maps(df):
    """label2id chỉ lấy từ tập TRAIN."""
    names = sorted(df.loc[df["split"] == "train", "engagement_label"].unique().tolist())
    label2id = {name: i for i, name in enumerate(names)}
    id2label = {i: name for name, i in label2id.items()}
    return label2id, id2label


def drop_unseen_labels(df, label2id):
    """Bỏ sample val/test có nhãn không xuất hiện trong train."""
    for split_name in ("val", "test"):
        bad = (df["split"] == split_name) & ~df["engagement_label"].isin(list(label2id))
        if bad.any():
            print(f"[!] {split_name}: loại {int(bad.sum())} sample có nhãn lạ.")
            df = df[~bad].reset_index(drop=True)
    return df


def build_aux_label_maps(df, aux_columns):
    """label2id cho từng auxiliary task, chỉ lấy từ TRAIN."""
    train_df = df[df["split"] == "train"]
    aux_label2id = {}
    for task, col in aux_columns.items():
        if col not in df.columns:
            print(f"[!] Thiếu cột '{col}' -> bỏ task '{task}'.")
            continue
        names = sorted(str(v) for v in train_df[col].dropna().unique())
        aux_label2id[task] = {name: i for i, name in enumerate(names)}
        print(f"Aux task '{task}': {len(names)} lớp")
    return aux_label2id


def build_aux_valid_mask(df, aux_columns, aux_label2id):
    """Sample chỉ tính aux loss khi có đủ nhãn cho mọi task."""
    valid = pd.Series(True, index=df.index)
    for task, mapping in aux_label2id.items():
        col = aux_columns[task]
        valid &= df[col].notna() & df[col].astype(str).isin(list(mapping))
    return valid


# ------------------------------------------------------------------ cache
def cache_dirs(cfg):
    return {
        "features_body": os.path.join(cfg["FEATURE_DIR"], "features_body"),
        "features_face": os.path.join(cfg["FEATURE_DIR"], "features_face"),
        "skeleton": cfg["SKELETON_DIR"],
    }


def _npy_exists(folder, sample_id):
    return os.path.exists(os.path.join(folder, f"{sample_id}.npy"))


def filter_by_cache(df, cfg):
    """Giữ sample có feature body + skeleton (face là tuỳ chọn)."""
    dirs = cache_dirs(cfg)
    has_body = df["sample_id"].apply(lambda s: _npy_exists(dirs["features_body"], s))
    has_skeleton = df["sample_id"].apply(lambda s: _npy_exists(dirs["skeleton"], s))
    kept = df[has_body & has_skeleton].reset_index(drop=True)
    print(f"Loại {len(df) - len(kept)} sample thiếu feature body hoặc skeleton. Còn {len(kept)}.")
    if kept.empty:
        raise RuntimeError("Không còn sample nào - kiểm tra FEATURE_DIR / SKELETON_DIR "
                           "(chạy scripts/diagnose.py để xem chi tiết).")
    return kept


# ------------------------------------------------------------------ pipeline
def prepare_data(cfg):
    """manifest -> nhãn aux -> lọc cache -> nhãn engagement. Trả về DataBundle."""
    df = load_manifest(cfg)

    aux_label2id = {}
    if cfg["USE_BEHAVIOR_EMOTION_AUX"]:
        aux_label2id = build_aux_label_maps(df, cfg["AUX_LABEL_COLUMNS"])
        df["aux_valid"] = build_aux_valid_mask(df, cfg["AUX_LABEL_COLUMNS"], aux_label2id)
        print(f"Sample có đủ nhãn aux: {int(df['aux_valid'].sum())}/{len(df)}")
    else:
        df["aux_valid"] = False

    df = filter_by_cache(df, cfg)
    label2id, id2label = build_label_maps(df)
    df = drop_unseen_labels(df, label2id)

    print(f"{len(label2id)} lớp: {label2id}")
    print(df.groupby(["split", "engagement_label"]).size().unstack(fill_value=0))
    return DataBundle(df=df, label2id=label2id, id2label=id2label, aux_label2id=aux_label2id)
