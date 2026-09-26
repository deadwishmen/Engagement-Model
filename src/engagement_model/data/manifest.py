"""Read the dataset manifest and prepare label maps.

Corresponds to notebook section "2. Doc manifest, xay neighbor graph tu manifest"
(cell 8), minus the graph-building part (see `engagement_model.data.graph`).
"""
from __future__ import annotations

import ast
import json
import os
import re

import pandas as pd

_KNOWN_SPLIT_PREFIXES = ("train", "val", "test")
_BBOX_NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def _parse_json_column(series: pd.Series) -> pd.Series:
    def _maybe_parse(v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                pass
            try:
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                return v
        return v
    return series.apply(_maybe_parse)


def _fix_stringified_bbox_list(bbox_list):
    if not isinstance(bbox_list, list):
        return bbox_list
    fixed, changed = [], False
    for item in bbox_list:
        if isinstance(item, str):
            fixed.append([float(x) for x in _BBOX_NUM_RE.findall(item)])
            changed = True
        else:
            fixed.append(item)
    return fixed if changed else bbox_list


def resolve_clip_abs_path(output_dir: str, clip_path: str) -> str:
    clip_path_norm = clip_path.replace("\\", "/")
    parts = clip_path_norm.split("/", 1)
    if len(parts) == 2 and parts[0] in _KNOWN_SPLIT_PREFIXES:
        return os.path.join(output_dir, f"clips_{parts[0]}", clip_path_norm)
    return os.path.join(output_dir, "clips", clip_path_norm)


def load_manifest_and_prepare(cfg: dict):
    """Read manifest.{csv,parquet,jsonl}, clean bbox columns, filter unknown labels
    and missing clips, optionally override the 'split' column from a session-level
    split_names.json, and build label2id/id2label from the TRAIN split.

    Returns: (df, label2id, id2label, num_classes)
    """
    manifest_dir_nested = os.path.join(cfg["OUTPUT_DIR"], "manifest", "manifest")
    manifest_dir_flat = os.path.join(cfg["OUTPUT_DIR"], "manifest")

    def _first_existing_manifest_dir():
        for d in (manifest_dir_nested, manifest_dir_flat):
            if any(os.path.exists(os.path.join(d, f"manifest.{ext}")) for ext in ("csv", "parquet", "jsonl")):
                return d
        return manifest_dir_nested

    manifest_dir = _first_existing_manifest_dir()
    csv_path = os.path.join(manifest_dir, "manifest.csv")
    parquet_path = os.path.join(manifest_dir, "manifest.parquet")
    jsonl_path = os.path.join(manifest_dir, "manifest.jsonl")

    assert os.path.exists(csv_path) or os.path.exists(parquet_path) or os.path.exists(jsonl_path), (
        f"Khong tim thay manifest.csv/parquet/jsonl trong {manifest_dir_nested} hoac {manifest_dir_flat}."
    )
    print(f"Dang doc manifest tu: {manifest_dir}")

    if os.path.exists(csv_path):
        raw_df = pd.read_csv(csv_path)
    elif os.path.exists(parquet_path):
        raw_df = pd.read_parquet(parquet_path)
    else:
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        raw_df = pd.DataFrame(rows)

    print("So dong manifest:", len(raw_df))
    df = raw_df.copy()

    for col in ["bbox_pixel", "bbox_norm", "bbox_time_sec", "frame_paths", "frame_paths_face"]:
        if col in df.columns:
            df[col] = _parse_json_column(df[col])

    for col in ["bbox_pixel", "bbox_norm"]:
        if col not in df.columns:
            continue

        def _row_was_stringified(bbox_list):
            return isinstance(bbox_list, list) and len(bbox_list) > 0 and isinstance(bbox_list[0], str)

        if int(df[col].apply(_row_was_stringified).sum()) > 0:
            df[col] = df[col].apply(_fix_stringified_bbox_list)

    if cfg["EXCLUDE_UNKNOWN"] and "engagement_label" in df.columns:
        n_before = len(df)
        df = df[df["engagement_label"] != "unknown"].reset_index(drop=True)
        print(f"Loai {n_before - len(df)} sample nhan unknown. Con lai {len(df)} sample.")

    def _clip_exists(row, col="clip_path"):
        cp = row.get(col)
        if pd.isna(cp) or (isinstance(cp, str) and cp.strip() == ""):
            return False
        return os.path.exists(resolve_clip_abs_path(cfg["OUTPUT_DIR"], cp))

    df["body_clip_exists"] = df.apply(lambda r: _clip_exists(r, "clip_path"), axis=1)
    if "face_clip_path" in df.columns:
        df["face_clip_exists"] = df.apply(lambda r: _clip_exists(r, "face_clip_path"), axis=1)
    else:
        df["face_clip_exists"] = False
    df = df[df["body_clip_exists"]].reset_index(drop=True)

    if "sample_id" not in df.columns:
        df["sample_id"] = df.index.astype(str)
    df["sample_id"] = df["sample_id"].astype(str)

    n_with_face = int(df["face_clip_exists"].sum())
    print(f"Co face: {n_with_face}/{len(df)} ({100 * n_with_face / max(len(df), 1):.1f}%).")

    # ---- Ghi de cot 'split' bang file split_names.json (chia theo SESSION) neu co ----
    split_json_path = cfg.get("SPLIT_NAMES_JSON_PATH")
    if split_json_path and os.path.exists(split_json_path):
        with open(split_json_path, "r", encoding="utf-8") as f:
            split_names_data = json.load(f)

        session_to_split = {}
        for split_name, filenames in split_names_data.items():
            for fname in filenames:
                session_name = fname[:-3] if fname.endswith(".pt") else fname
                session_to_split[session_name] = split_name

        n_before_split_override = len(df)
        df["split"] = df["session"].map(session_to_split)
        n_unmatched = int(df["split"].isna().sum())
        if n_unmatched > 0:
            unmatched_sessions = sorted(df.loc[df["split"].isna(), "session"].unique().tolist())
            print(f"[!] {n_unmatched} segment thuoc {len(unmatched_sessions)} session KHONG co trong "
                  f"split_names.json -- loai khoi du lieu. Vi du session bi loai: {unmatched_sessions[:5]}")
            df = df[df["split"].notna()].reset_index(drop=True)

        print(f"Da ghi de cot 'split' bang {split_json_path} (chia theo SESSION): "
              f"{n_before_split_override} -> {len(df)} segment con lai.")
        print(df["split"].value_counts())

        json_sessions = set(session_to_split.keys())
        manifest_sessions = set(df["session"].unique().tolist())
        sessions_only_in_json = json_sessions - manifest_sessions
        if sessions_only_in_json:
            print(f"[Thong tin] {len(sessions_only_in_json)} session co trong split_names.json nhung "
                  f"khong xuat hien trong manifest (co the do da bi loc o cac buoc truoc) -- khong sao, "
                  f"bo qua.")
    elif split_json_path:
        print(f"[!] CONFIG['SPLIT_NAMES_JSON_PATH']='{split_json_path}' nhung file khong ton tai -- "
              f"dung cot 'split' co san trong manifest nhu binh thuong.")

    train_mask = df["split"] == "train"
    label_names = sorted(df.loc[train_mask, "engagement_label"].unique().tolist())
    label2id = {lbl: i for i, lbl in enumerate(label_names)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    num_classes = len(label2id)
    print(f"So lop engagement (tu TRAIN, TRUOC khi loc feature/skeleton cache): {num_classes} -> {label2id}")

    for split_name in ("val", "test"):
        split_mask = df["split"] == split_name
        split_labels = set(df.loc[split_mask, "engagement_label"].unique().tolist())
        unseen = split_labels - set(label_names)
        if unseen:
            n_drop = int((split_mask & df["engagement_label"].isin(unseen)).sum())
            print(f"[!] Tap {split_name} co nhan la: {unseen} -- loai {n_drop} sample.")
            df = df[~(split_mask & df["engagement_label"].isin(unseen))].reset_index(drop=True)

    return df, label2id, id2label, num_classes


def build_aux_label_maps(df: pd.DataFrame, aux_label_columns: dict):
    """Build a label2id/id2label dict per auxiliary task (pose/act/obj/int/emo), from
    TRAIN-split values only (mirrors engagement_label handling above)."""
    aux_label2id, aux_id2label = {}, {}
    train_mask = df["split"] == "train"

    for task_name, col in aux_label_columns.items():
        if col not in df.columns:
            print(f"[!] Cot '{col}' (task '{task_name}') khong ton tai trong manifest -- "
                  f"bo qua auxiliary task nay.")
            continue

        train_values = df.loc[train_mask, col].dropna().unique().tolist()
        names = sorted(str(v) for v in train_values)
        aux_label2id[task_name] = {name: i for i, name in enumerate(names)}
        aux_id2label[task_name] = {i: name for name, i in aux_label2id[task_name].items()}
        print(f"Auxiliary task '{task_name}' (cot '{col}'): {len(names)} lop -> "
              f"{aux_label2id[task_name]}")

    return aux_label2id, aux_id2label


def build_aux_valid_mask(df: pd.DataFrame, aux_label_columns: dict, aux_label2id: dict) -> pd.Series:
    """A sample is valid for auxiliary loss only if ALL 5 columns are present and their
    value appeared in TRAIN (i.e. is present in aux_label2id)."""
    valid = pd.Series(True, index=df.index)
    for task_name, col in aux_label_columns.items():
        if task_name not in aux_label2id:
            continue
        col_valid = df[col].notna() & df[col].astype(str).isin(aux_label2id[task_name].keys())
        valid = valid & col_valid
    return valid


def prepare_aux_labels(df: pd.DataFrame, cfg: dict):
    """Wraps build_aux_label_maps + build_aux_valid_mask, mirroring notebook cell 8's
    tail. Adds an 'aux_valid' column to df (mutated + returned) and returns
    (df, aux_label2id, aux_id2label)."""
    aux_label2id, aux_id2label = {}, {}
    if cfg.get("USE_BEHAVIOR_EMOTION_AUX", False):
        aux_label2id, aux_id2label = build_aux_label_maps(df, cfg["AUX_LABEL_COLUMNS"])
        df["aux_valid"] = build_aux_valid_mask(df, cfg["AUX_LABEL_COLUMNS"], aux_label2id)
        n_aux_valid = int(df["aux_valid"].sum())
        print(f"\nSample hop le cho auxiliary loss (co du 5 nhan hanh vi/cam xuc): "
              f"{n_aux_valid}/{len(df)} ({100 * n_aux_valid / max(len(df), 1):.1f}%).")
    else:
        df["aux_valid"] = False
        print("USE_BEHAVIOR_EMOTION_AUX=False -- bo qua auxiliary behavior/emotion task.")
    return df, aux_label2id, aux_id2label
