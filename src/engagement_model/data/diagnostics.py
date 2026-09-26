"""Sanity-check diagnostics to run before trusting any macro-F1 number.

Corresponds to notebook cells 10 (track/split leakage), 14 (label stability),
16 (feature/skeleton cache path sanity check).
"""
from __future__ import annotations

import os

import pandas as pd

from .graph import _get_clip_time_range


def diagnose_track_split_leakage(df: pd.DataFrame):
    """Check whether segments of the SAME track (session+camera_id+object_id) end up
    split across multiple splits -- a serious data leak if so."""
    group_cols = [c for c in ["session", "camera_id", "object_id"] if c in df.columns]
    if len(group_cols) < 3:
        print("[!] Thieu cot session/camera_id/object_id -- khong the kiem tra track leakage.")
        return None

    track_split_counts = df.groupby(group_cols)["split"].nunique()
    leaked_tracks = track_split_counts[track_split_counts > 1]

    n_total_tracks = len(track_split_counts)
    n_leaked_tracks = len(leaked_tracks)
    group_label = "+".join(group_cols)

    print(f"Tong so track (nhom {group_label}): {n_total_tracks}")
    print(f"So track BI CHIA vao nhieu split (RO RI): {n_leaked_tracks} "
          f"({100 * n_leaked_tracks / max(n_total_tracks, 1):.2f}%)")

    if n_leaked_tracks == 0:
        print("\n[OK] KHONG phat hien ro ri: moi track deu nam tron trong 1 split duy nhat.")
        print("Cot split co san trong manifest da duoc chia dung o cap do track/session.")
    else:
        n_leaked_segments = int(df.set_index(group_cols).index.isin(leaked_tracks.index).sum())
        print(f"\n[CANH BAO] {n_leaked_segments} segment ({100 * n_leaked_segments / len(df):.2f}% "
              f"tong so segment) thuoc ve cac track BI RO RI xuyen split.")
        print("\nVi du 5 track bi ro ri (va split cua tung segment trong track do):")
        example_tracks = leaked_tracks.head(5).index.tolist()
        for track_key in example_tracks:
            if len(group_cols) == 1:
                mask = df[group_cols[0]] == track_key
            else:
                mask = pd.Series(True, index=df.index)
                for col, val in zip(group_cols, track_key):
                    mask &= (df[col] == val)
            splits_in_track = df.loc[mask, "split"].value_counts().to_dict()
            track_desc = dict(zip(group_cols, track_key)) if len(group_cols) > 1 else track_key
            print(f"  {track_desc}: {splits_in_track}")

        print("\n[!] KHUYEN NGHI: chia lai split o CAP DO TRACK (session+camera_id+object_id) "
              "TRUOC khi tiep tuc bat ky thu nghiem kien truc nao khac -- moi so sanh macro-F1 "
              "giua cac kien truc da lam TRUOC KHI sua loi nay co the khong dang tin cay.")

    return {
        "n_total_tracks": n_total_tracks,
        "n_leaked_tracks": n_leaked_tracks,
        "leaked_track_keys": leaked_tracks.index.tolist(),
    }


def diagnose_label_stability_within_track(df: pd.DataFrame, max_gap_seconds: float = 15.0):
    """Measure how often the engagement label 'flips' between temporally adjacent
    segments of the same track -- an indirect proxy for label noise."""
    group_cols = [c for c in ["session", "camera_id", "object_id"] if c in df.columns]
    if len(group_cols) < 3:
        print("[!] Thieu cot session/camera_id/object_id -- khong the chan doan.")
        return None

    transitions = []
    n_tracks_checked = 0
    n_pairs_checked = 0

    for _, group in df.groupby(group_cols, sort=False):
        if len(group) < 2:
            continue

        time_ranges = group.apply(_get_clip_time_range, axis=1)
        valid_mask = time_ranges.apply(lambda r: r is not None)
        if valid_mask.sum() < 2:
            continue

        sub = group[valid_mask].copy()
        sub["_t_start"] = time_ranges[valid_mask].apply(lambda r: r[0])
        sub = sub.sort_values("_t_start")

        labels = sub["engagement_label"].tolist()
        starts = sub["_t_start"].tolist()
        n_tracks_checked += 1

        for i in range(len(sub) - 1):
            gap = starts[i + 1] - starts[i]
            if gap <= max_gap_seconds:
                n_pairs_checked += 1
                transitions.append({
                    "label_a": labels[i], "label_b": labels[i + 1],
                    "time_gap": gap, "is_same": labels[i] == labels[i + 1],
                })

    if not transitions:
        print(f"[!] Khong tim duoc cap segment nao cach nhau <= {max_gap_seconds}s trong cung track.")
        return None

    trans_df = pd.DataFrame(transitions)
    n_total = len(trans_df)
    n_flip = int((~trans_df["is_same"]).sum())
    flip_rate = n_flip / n_total

    print(f"So track duoc kiem tra: {n_tracks_checked}")
    print(f"So cap segment lien ke (gap <= {max_gap_seconds}s): {n_total}")
    print(f"So cap nhan KHAC NHAU (nhan 'nhay'): {n_flip} ({100 * flip_rate:.1f}%)")
    print(f"So cap nhan GIONG NHAU (on dinh): {n_total - n_flip} ({100 * (1 - flip_rate):.1f}%)")

    print("\nBang cheo chuyen doi nhan (hang = nhan segment truoc, cot = nhan segment sau):")
    crosstab = pd.crosstab(trans_df["label_a"], trans_df["label_b"])
    print(crosstab)

    print("\nTy le 'nhay nhan' theo khoang cach thoi gian:")
    for max_g in [3.0, 5.0, 10.0, max_gap_seconds]:
        sub_trans = trans_df[trans_df["time_gap"] <= max_g]
        if len(sub_trans) > 0:
            sub_flip_rate = (~sub_trans["is_same"]).mean()
            print(f"  gap <= {max_g:>5.1f}s: {len(sub_trans):>5d} cap, ty le nhay = {100 * sub_flip_rate:.1f}%")

    if flip_rate > 0.40:
        print(f"\n[CANH BAO] Ty le nhay nhan RAT CAO ({100 * flip_rate:.1f}%) -- dau hieu manh cua "
              f"TRAN NHIEU NHAN gioi han macro-F1 co the dat duoc.")
    elif flip_rate > 0.20:
        print(f"\n[LUU Y] Ty le nhay nhan o muc trung binh ({100 * flip_rate:.1f}%).")
    else:
        print(f"\n[OK] Ty le nhay nhan tuong doi thap ({100 * flip_rate:.1f}%).")

    return {
        "n_tracks_checked": n_tracks_checked,
        "n_pairs_checked": n_total,
        "flip_rate": flip_rate,
        "crosstab": crosstab,
        "transitions_df": trans_df,
    }


def _diagnose_cache_dir(name: str, cache_dir: str, expected_ext: str):
    print(f"--- {name}: {cache_dir} ---")
    if not os.path.isdir(cache_dir):
        print(f"  [X] Thu muc KHONG TON TAI. Kiem tra lai duong dan trong CONFIG['{name}'].")
        kaggle_input = "/kaggle/input"
        if os.path.isdir(kaggle_input):
            print(f"  Cac dataset hien co trong {kaggle_input}: {os.listdir(kaggle_input)}")
        return None

    all_files = os.listdir(cache_dir)
    matching_ext = [f for f in all_files if f.endswith(expected_ext)]
    print(f"  Thu muc TON TAI. Tong so file: {len(all_files)} | So file duoi '{expected_ext}': {len(matching_ext)}")
    if len(matching_ext) == 0 and len(all_files) > 0:
        print(f"  [!] Co file trong thu muc nhung KHONG file nao duoi '{expected_ext}'. Vi du ten file thuc te: "
              f"{all_files[:5]}")
        subdirs = [f for f in all_files if os.path.isdir(os.path.join(cache_dir, f))]
        if subdirs:
            print(f"  Phat hien thu muc con: {subdirs[:10]} -- co the ban can tro FEATURE_DIR/SKELETON_DIR "
                  f"sau vao ben trong thu muc con nay.")
    elif matching_ext:
        print(f"  Vi du 5 ten file thuc te: {matching_ext[:5]}")
    return set(os.path.splitext(f)[0] for f in matching_ext)


def diagnose_cache_dirs(df: pd.DataFrame, cfg: dict):
    """Sanity-check FEATURE_DIR/SKELETON_DIR against sample_id's actually present in
    the manifest -- catches wrong paths / sample_id format mismatches early."""
    if len(df) == 0 or "sample_id" not in df.columns:
        print("[!] Khong the chan doan vi df rong hoac thieu cot sample_id.")
        return

    sample_ids_manifest = set(df["sample_id"].astype(str).tolist())
    print(f"So sample_id duy nhat trong manifest (sau loc unknown/body_clip_exists): {len(sample_ids_manifest)}")
    print(f"Vi du 5 sample_id trong manifest: {list(sample_ids_manifest)[:5]}")
    print()

    feature_body_dir = os.path.join(cfg["FEATURE_DIR"], "features_body")
    feature_face_dir = os.path.join(cfg["FEATURE_DIR"], "features_face")

    print(f"--- FEATURE_DIR: {cfg['FEATURE_DIR']} ---")
    if not os.path.isdir(cfg["FEATURE_DIR"]):
        print(f"  [X] Thu muc KHONG TON TAI.")
        if os.path.isdir("/kaggle/input"):
            print(f"  Cac dataset hien co trong /kaggle/input: {os.listdir('/kaggle/input')}")
    else:
        print(f"  Noi dung ben trong: {os.listdir(cfg['FEATURE_DIR'])}")
    print()

    body_ids = _diagnose_cache_dir("FEATURE_DIR/features_body", feature_body_dir, ".npy")
    print()
    face_ids = _diagnose_cache_dir("FEATURE_DIR/features_face", feature_face_dir, ".npy")
    print()
    skeleton_ids = _diagnose_cache_dir("SKELETON_DIR", cfg["SKELETON_DIR"], ".npy")
    print()

    if body_ids is not None:
        overlap_body = sample_ids_manifest & body_ids
        print(f"So sample_id KHOP giua manifest va features_body: {len(overlap_body)}/{len(sample_ids_manifest)}")
        if len(overlap_body) == 0 and len(body_ids) > 0:
            print(f"  [!] KHONG sample_id nao khop, du features_body co {len(body_ids)} file. "
                  f"Vi du sample_id file thuc te: {list(body_ids)[:5]}")

    if face_ids is not None:
        overlap_face = sample_ids_manifest & face_ids
        print(f"So sample_id KHOP giua manifest va features_face: {len(overlap_face)}/{len(sample_ids_manifest)}")

    if skeleton_ids is not None:
        overlap_skel = sample_ids_manifest & skeleton_ids
        print(f"So sample_id KHOP giua manifest va SKELETON_DIR: {len(overlap_skel)}/{len(sample_ids_manifest)}")
        if len(overlap_skel) == 0 and len(skeleton_ids) > 0:
            print(f"  [!] KHONG sample_id nao khop, du SKELETON_DIR co {len(skeleton_ids)} file. "
                  f"Vi du sample_id file thuc te: {list(skeleton_ids)[:5]}")
