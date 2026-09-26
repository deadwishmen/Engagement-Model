"""K-hop spatial neighbor graph + temporal context-window construction.

Corresponds to notebook section "2a. Target-centric neighbor graph + explicit
spatio-temporal relation features" (cell 12).
"""
from __future__ import annotations

import math
import os

import numpy as np
from scipy.spatial import cKDTree

# COCO-17 keypoint index: 5 = left_shoulder, 6 = right_shoulder (matches YOLO-Pose/COCO
# ordering used by the skeleton extraction pipeline).
_LEFT_SHOULDER_IDX = 5
_RIGHT_SHOULDER_IDX = 6

# Cache facing-direction estimates by sample_id (a sample may be evaluated as a
# neighbor candidate many times across different targets in the same session/camera).
_skeleton_orientation_cache: dict = {}

RELATION_FEATURE_NAMES_BASE = [
    "dx_mean",
    "dy_mean",
    "distance_mean",
    "distance_min",
    "relative_vx",
    "relative_vy",
    "distance_change",
    "temporal_iou",
    "log_area_ratio",
    "target_bbox_motion",
    "neighbor_bbox_motion",
    "relative_scale_change",
]


def _estimate_facing_direction(sample_id, skeleton_dir, conf_thr):
    """Estimate a person's plausible facing axis (unit vector) from shoulder
    keypoints, averaged over valid frames of their 16-frame skeleton clip. Returns
    None if there isn't enough data. Result is cached by sample_id."""
    if sample_id in _skeleton_orientation_cache:
        return _skeleton_orientation_cache[sample_id]

    path = os.path.join(skeleton_dir, f"{sample_id}.npy")
    if not os.path.exists(path):
        _skeleton_orientation_cache[sample_id] = None
        return None

    try:
        raw = np.load(path, allow_pickle=True).item()
        keypoints = np.asarray(raw["keypoints"], dtype=np.float64)          # (T,17,2)
        keypoint_scores = np.asarray(raw["keypoint_scores"], dtype=np.float64)  # (T,17)
        detected = np.asarray(raw["detected"], dtype=bool)                  # (T,)
    except Exception:
        _skeleton_orientation_cache[sample_id] = None
        return None

    left_sh = keypoints[:, _LEFT_SHOULDER_IDX, :]
    right_sh = keypoints[:, _RIGHT_SHOULDER_IDX, :]
    left_conf = keypoint_scores[:, _LEFT_SHOULDER_IDX]
    right_conf = keypoint_scores[:, _RIGHT_SHOULDER_IDX]

    frame_valid = detected & (left_conf >= conf_thr) & (right_conf >= conf_thr)
    if not frame_valid.any():
        _skeleton_orientation_cache[sample_id] = None
        return None

    shoulder_vec = right_sh[frame_valid] - left_sh[frame_valid]  # (N_valid, 2)
    norms = np.linalg.norm(shoulder_vec, axis=1, keepdims=True)
    good = (norms[:, 0] > 1e-6)
    if not good.any():
        _skeleton_orientation_cache[sample_id] = None
        return None
    shoulder_vec = shoulder_vec[good] / norms[good]

    normal_vec = np.stack([-shoulder_vec[:, 1], shoulder_vec[:, 0]], axis=1)

    ref = normal_vec[0]
    signs = np.sign((normal_vec * ref).sum(axis=1))
    signs[signs == 0] = 1.0
    aligned = normal_vec * signs[:, None]

    facing = aligned.mean(axis=0)
    facing_norm = np.linalg.norm(facing)
    if facing_norm < 1e-6:
        _skeleton_orientation_cache[sample_id] = None
        return None
    facing = facing / facing_norm

    _skeleton_orientation_cache[sample_id] = facing
    return facing


def _orientation_angle_deg(facing_dir, to_target_vec):
    """Angle (deg) between the plausible facing axis and the vector to the target.
    Since facing_dir is an axis (direction unknown), take the smaller of the two
    possible angles."""
    target_norm = np.linalg.norm(to_target_vec)
    if target_norm < 1e-6:
        return 0.0
    to_target_unit = to_target_vec / target_norm

    cos_a = float(np.clip(np.dot(facing_dir, to_target_unit), -1.0, 1.0))
    angle_a = math.degrees(math.acos(abs(cos_a)))
    return angle_a


def _bbox_array(row):
    """Return an Nx4 bbox trajectory. Prefer normalized boxes when available."""
    for col in ("bbox_norm", "bbox_pixel"):
        values = row.get(col)
        if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0:
            try:
                arr = np.asarray(values, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] >= 4:
                    arr = arr[:, :4]
                    good = np.isfinite(arr).all(axis=1)
                    arr = arr[good]
                    if len(arr) > 0:
                        return arr
            except Exception:
                pass
    return None


def _bbox_stats_whole_track(row):
    arr = _bbox_array(row)
    if arr is None:
        return None
    cx = ((arr[:, 0] + arr[:, 2]) / 2.0).mean()
    cy = ((arr[:, 1] + arr[:, 3]) / 2.0).mean()
    w = np.clip(arr[:, 2] - arr[:, 0], 1e-6, None).mean()
    h = np.clip(arr[:, 3] - arr[:, 1], 1e-6, None).mean()
    if np.isfinite([cx, cy, w, h]).all():
        return float(cx), float(cy), float(w), float(h)
    return None


def _bbox_center_whole_track(row):
    stats = _bbox_stats_whole_track(row)
    return (stats[0], stats[1]) if stats is not None else (np.nan, np.nan)


def _track_time_range(row):
    sequence_cols = (
        "bbox_time_sec", "time_sec", "frame_time_sec", "timestamps",
        "bbox_timestamp", "frame_timestamps",
    )
    for col in sequence_cols:
        values = row.get(col)
        if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0:
            try:
                arr = np.asarray(values, dtype=np.float64).reshape(-1)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    return float(arr.min()), float(arr.max())
            except Exception:
                pass

    for start_col, end_col in (
        ("start_time", "end_time"),
        ("start_sec", "end_sec"),
        ("track_start_sec", "track_end_sec"),
    ):
        if start_col in row.index and end_col in row.index:
            try:
                t0, t1 = float(row[start_col]), float(row[end_col])
                if np.isfinite([t0, t1]).all():
                    return min(t0, t1), max(t0, t1)
            except Exception:
                pass
    return None


def _temporal_overlap_ratio(row_a, row_b):
    """Temporal IoU in [0,1]."""
    ra = _track_time_range(row_a)
    rb = _track_time_range(row_b)
    if ra is None or rb is None:
        return 0.0
    start = max(ra[0], rb[0])
    end = min(ra[1], rb[1])
    inter = max(0.0, end - start)
    union = max(max(ra[1], rb[1]) - min(ra[0], rb[0]), 1e-8)
    return float(inter / union)


def _resample_vector(values, length=16):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.zeros(length, dtype=np.float64)
    if len(values) == 1:
        return np.repeat(values, length)
    old_x = np.linspace(0.0, 1.0, len(values))
    new_x = np.linspace(0.0, 1.0, length)
    return np.interp(new_x, old_x, values)


def _trajectory_info(row, resample_len=16):
    arr = _bbox_array(row)
    if arr is None:
        return None

    cx = (arr[:, 0] + arr[:, 2]) / 2.0
    cy = (arr[:, 1] + arr[:, 3]) / 2.0
    w = np.clip(arr[:, 2] - arr[:, 0], 1e-6, None)
    h = np.clip(arr[:, 3] - arr[:, 1], 1e-6, None)
    area = np.clip(w * h, 1e-8, None)

    return {
        "cx": _resample_vector(cx, resample_len),
        "cy": _resample_vector(cy, resample_len),
        "area": _resample_vector(area, resample_len),
        "mean_area": float(area.mean()),
    }


def _compute_relation_features(main_row, neighbor_row):
    """12 explicit relation features for the target->neighbor edge. Spatial
    quantities are normalized by sqrt(target mean bbox area)."""
    tm = _trajectory_info(main_row)
    tn = _trajectory_info(neighbor_row)
    if tm is None or tn is None:
        return None

    scale = max(math.sqrt(tm["mean_area"]), 1e-6)
    dx = (tn["cx"] - tm["cx"]) / scale
    dy = (tn["cy"] - tm["cy"]) / scale
    dist = np.sqrt(dx * dx + dy * dy)

    target_vx = (tm["cx"][-1] - tm["cx"][0]) / scale
    target_vy = (tm["cy"][-1] - tm["cy"][0]) / scale
    neigh_vx = (tn["cx"][-1] - tn["cx"][0]) / scale
    neigh_vy = (tn["cy"][-1] - tn["cy"][0]) / scale

    target_step = np.sqrt(np.diff(tm["cx"]) ** 2 + np.diff(tm["cy"]) ** 2) / scale
    neigh_step = np.sqrt(np.diff(tn["cx"]) ** 2 + np.diff(tn["cy"]) ** 2) / scale

    area_ratio = max(tn["mean_area"] / max(tm["mean_area"], 1e-8), 1e-8)
    log_area_ratio = float(np.clip(np.log(area_ratio), -4.0, 4.0))

    t_scale_growth = np.log(max(tm["area"][-1], 1e-8) / max(tm["area"][0], 1e-8))
    n_scale_growth = np.log(max(tn["area"][-1], 1e-8) / max(tn["area"][0], 1e-8))

    relation = np.array([
        dx.mean(),
        dy.mean(),
        dist.mean(),
        dist.min(),
        neigh_vx - target_vx,
        neigh_vy - target_vy,
        dist[-1] - dist[0],
        _temporal_overlap_ratio(main_row, neighbor_row),
        log_area_ratio,
        target_step.mean() if len(target_step) else 0.0,
        neigh_step.mean() if len(neigh_step) else 0.0,
        n_scale_growth - t_scale_growth,
    ], dtype=np.float32)

    return np.nan_to_num(relation, nan=0.0, posinf=10.0, neginf=-10.0).clip(-10.0, 10.0)


def build_neighbor_index(
    df,
    k_neighbors=4,
    candidate_multiplier=6,
    require_time_overlap=True,
    skeleton_dir=None,
    skeleton_conf_thr=0.05,
    use_orientation_penalty=True,
    orientation_angle_threshold_deg=60.0,
    orientation_penalty=1.0,
):
    """Build target-centric K-nearest-neighbor lists inside each session/camera group.

    Top-K is NOT purely Euclidean: when `use_orientation_penalty=True` and a
    skeleton_dir is given, each candidate's plausible facing axis (from shoulder
    keypoints) is compared against the direction to the target; if the angle exceeds
    `orientation_angle_threshold_deg`, `orientation_penalty` is added to the distance
    before ranking (someone facing away is de-prioritized even if geometrically close).

    IMPORTANT: call this separately for each split (train/val/test) to avoid
    cross-split leakage -- see `data.dataset.make_loader`.
    """
    n = len(df)
    neighbor_lists = [[] for _ in range(n)]

    if "bbox_center" not in df.columns:
        df["bbox_center"] = df.apply(_bbox_center_whole_track, axis=1)

    group_cols = [c for c in ["session", "camera_id"] if c in df.columns]
    if not group_cols:
        print("[!] No session/camera_id columns: social graph will be empty.")
        return neighbor_lists

    centers_all = np.array(df["bbox_center"].tolist(), dtype=np.float64)
    obj_ids_all = df["object_id"].to_numpy() if "object_id" in df.columns else np.arange(n)

    for _, group in df.groupby(group_cols, sort=False):
        idxs = group.index.to_numpy(dtype=np.int64)
        centers = centers_all[idxs]
        obj_ids = obj_ids_all[idxs]
        valid_center = np.isfinite(centers).all(axis=1)
        if valid_center.sum() < 2:
            continue

        pool_local_idx = np.where(valid_center)[0]
        pool_centers = centers[pool_local_idx]
        tree = cKDTree(pool_centers)
        k_query = min(len(pool_local_idx), max(2, k_neighbors * candidate_multiplier + 1))

        for local_i in range(len(group)):
            if not valid_center[local_i]:
                continue

            _, cand_pool_pos = tree.query(centers[local_i], k=k_query)
            cand_pool_pos = np.atleast_1d(cand_pool_pos)
            main_global_idx = int(idxs[local_i])
            main_row = df.iloc[main_global_idx]
            candidates = []

            for pool_pos in cand_pool_pos:
                cand_local = int(pool_local_idx[int(pool_pos)])
                if cand_local == local_i:
                    continue
                if obj_ids[cand_local] == obj_ids[local_i]:
                    continue

                cand_global_idx = int(idxs[cand_local])
                cand_row = df.iloc[cand_global_idx]
                overlap = _temporal_overlap_ratio(main_row, cand_row)
                if require_time_overlap and overlap <= 0.0:
                    continue

                relation = _compute_relation_features(main_row, cand_row)
                if relation is None:
                    continue

                euclidean_dist = float(relation[2])
                ranking_score = euclidean_dist
                orientation_angle_deg = None

                if use_orientation_penalty and skeleton_dir is not None:
                    cand_facing = _estimate_facing_direction(
                        cand_row["sample_id"], skeleton_dir, skeleton_conf_thr
                    )
                    if cand_facing is not None:
                        main_stats = _bbox_stats_whole_track(main_row)
                        cand_stats = _bbox_stats_whole_track(cand_row)
                        if main_stats is not None and cand_stats is not None:
                            to_target_vec = np.array(
                                [main_stats[0] - cand_stats[0], main_stats[1] - cand_stats[1]]
                            )
                            orientation_angle_deg = _orientation_angle_deg(cand_facing, to_target_vec)
                            if orientation_angle_deg > orientation_angle_threshold_deg:
                                ranking_score = euclidean_dist + orientation_penalty

                candidates.append({
                    "idx": cand_global_idx,
                    "dist": euclidean_dist,
                    "ranking_score": ranking_score,
                    "orientation_angle_deg": orientation_angle_deg,
                    "relation": relation,
                    "_object_id": obj_ids[cand_local],
                })

            candidates.sort(key=lambda c: c["ranking_score"])
            selected, seen_object_ids = [], set()
            for cand in candidates:
                cand = dict(cand)
                oid = str(cand.pop("_object_id", ""))
                if oid in seen_object_ids:
                    continue
                seen_object_ids.add(oid)
                selected.append(cand)
                if len(selected) >= k_neighbors:
                    break
            neighbor_lists[main_global_idx] = selected

    return neighbor_lists


def print_neighbor_stats(neighbor_lists, df, k_neighbors, prefix=""):
    n_with_neighbors = sum(1 for lst in neighbor_lists if len(lst) > 0)
    avg_neighbors = np.mean([len(lst) for lst in neighbor_lists]) if len(neighbor_lists) else 0.0
    print(f"{prefix}Samples with >=1 neighbor: {n_with_neighbors}/{len(df)} "
          f"({100 * n_with_neighbors / max(len(df), 1):.1f}%)")
    print(f"{prefix}Average neighbors/sample: {avg_neighbors:.2f} / max {k_neighbors}")


def _get_clip_time_range(row):
    """Prefer seg_start/seg_end (absolute time of the clip in the source video,
    directly available in the manifest) to order clips of the same person by time.
    Falls back to _track_time_range."""
    if "seg_start" in row.index and "seg_end" in row.index:
        try:
            t0, t1 = float(row["seg_start"]), float(row["seg_end"])
            if np.isfinite([t0, t1]).all():
                return min(t0, t1), max(t0, t1)
        except Exception:
            pass
    return _track_time_range(row)


def build_context_window_index(df, window_size=2, max_time_gap_seconds=60.0):
    """Build the TEMPORAL context: for each clip, find up to `window_size` clips
    before and `window_size` clips after in the time sequence of the SAME person
    (same session+camera_id+object_id), ordered by clip start time. This is the
    TIME axis, parallel to build_neighbor_index's SPACE axis (different person, same
    instant). If the time gap to an adjacent clip in the table is too large (person
    left frame and came back), it is not treated as continuous context.

    Returns list[list[dict]]: each dict has 'idx' (position in df), 'offset' (relative
    position vs target: -2,-1,+1,+2,... in this fixed order so the positional
    embedding learns a stable meaning), 'time_gap' (seconds).

    Call separately per split (train/val/test) in make_loader to avoid leakage,
    exactly like build_neighbor_index.
    """
    n = len(df)
    context_lists = [[] for _ in range(n)]

    group_cols = [c for c in ["session", "camera_id", "object_id"] if c in df.columns]
    if len(group_cols) < 3:
        print("[!] Thieu cot session/camera_id/object_id -- khong the xay ngu canh thoi gian "
              "(context window se rong cho tat ca sample).")
        return context_lists

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
        idxs = sub.index.to_numpy(dtype=np.int64)
        starts = sub["_t_start"].to_numpy(dtype=np.float64)
        n_g = len(idxs)

        for pos in range(n_g):
            entries = []
            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue
                j = pos + offset
                if j < 0 or j >= n_g:
                    continue
                time_gap = abs(float(starts[j] - starts[pos]))
                if max_time_gap_seconds is not None and time_gap > max_time_gap_seconds:
                    continue
                entries.append({
                    "idx": int(idxs[j]), "offset": int(offset), "time_gap": time_gap,
                })
            context_lists[int(idxs[pos])] = entries

    return context_lists


def print_context_stats(context_lists, df, window_size, prefix=""):
    n_with_context = sum(1 for lst in context_lists if len(lst) > 0)
    avg_context = np.mean([len(lst) for lst in context_lists]) if len(context_lists) else 0.0
    print(f"{prefix}Samples with >=1 context clip (before/after, same person): "
          f"{n_with_context}/{len(df)} ({100 * n_with_context / max(len(df), 1):.1f}%)")
    print(f"{prefix}Average context clips/sample: {avg_context:.2f} / max {window_size * 2}")
