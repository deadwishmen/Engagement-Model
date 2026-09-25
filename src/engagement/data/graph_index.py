"""Xây danh sách K-hop neighbors (không gian) và context window (thời gian) cho mỗi sample."""
import numpy as np
from scipy.spatial import cKDTree

from .geometry import (
    TRACK_COLS,
    bbox_mean_stats,
    compute_relation_features,
    estimate_facing_direction,
    orientation_angle_deg,
    sort_group_by_time,
    temporal_iou,
)


def _score_candidate(relation, target_row, cand_row, stats_all, target_idx, cand_idx, cfg):
    """Điểm càng THẤP càng được ưu tiên chọn làm hàng xóm. relation là mảng 12 chiều đã tính sẵn
    (compute_relation_features), nên các chế độ bên dưới không cần đọc/tính thêm gì."""
    dist = float(relation[2])          # distance_mean - đã chuẩn hoá theo sqrt(diện tích bbox target)
    mode = cfg.get("NEIGHBOR_RANKING", "distance_motion")

    if mode == "distance_only":
        return dist

    if mode == "distance_motion":
        # Chỉ dùng bbox, KHÔNG cần skeleton:
        #   - temporal_iou cao   -> 2 người đồng hiện diện lâu hơn -> ưu tiên hơn (trừ điểm)
        #   - distance_change âm -> khoảng cách đang giảm dần theo thời gian (đang tới gần nhau)
        #                           -> tín hiệu tương tác mạnh hơn tư thế tĩnh -> ưu tiên hơn (cộng điểm âm)
        overlap = temporal_iou(target_row, cand_row)
        distance_change = float(relation[6])
        return (dist - cfg["NEIGHBOR_TEMPORAL_IOU_WEIGHT"] * overlap
                    + cfg["NEIGHBOR_CONVERGENCE_WEIGHT"] * distance_change)

    if mode == "distance_orientation":
        # (kiểu cũ) dựa vào skeleton để đoán trục cơ thể - CHỈ biết trục, không biết mặt/lưng.
        score = dist
        if cfg["USE_ORIENTATION_PENALTY"] and cfg.get("SKELETON_DIR"):
            facing = estimate_facing_direction(
                cand_row["sample_id"], cfg["SKELETON_DIR"], cfg["SKELETON_CONF_THR"])
            t_stats, c_stats = stats_all[target_idx], stats_all[cand_idx]
            if facing is not None and t_stats and c_stats:
                to_target = np.array([t_stats[0] - c_stats[0], t_stats[1] - c_stats[1]])
                if orientation_angle_deg(facing, to_target) > cfg["ORIENTATION_ANGLE_THRESHOLD_DEG"]:
                    score += cfg["ORIENTATION_PENALTY"]
        return score

    raise ValueError(f"NEIGHBOR_RANKING không hợp lệ: {mode!r} "
                     "(chỉ nhận 'distance_only', 'distance_motion' hoặc 'distance_orientation')")


def build_neighbor_index(df, cfg):
    """Với mỗi sample, chọn tối đa K hàng xóm (khác object_id) trong cùng session/camera.
    Xếp hạng theo cfg["NEIGHBOR_RANKING"] (xem _score_candidate).

    `df` phải có index 0..N-1 (đã reset_index). Gọi RIÊNG cho từng split để tránh rò rỉ.
    """
    k = cfg["K_NEIGHBORS"]
    neighbor_lists = [[] for _ in range(len(df))]

    group_cols = [c for c in ("session", "camera_id") if c in df.columns]
    if not group_cols:
        print("[!] Không có cột session/camera_id -> graph rỗng.")
        return neighbor_lists

    rows = [row for _, row in df.iterrows()]
    stats_all = [bbox_mean_stats(row) for row in rows]
    centers_all = np.array([(s[0], s[1]) if s else (np.nan, np.nan) for s in stats_all])
    object_ids = df["object_id"].to_numpy() if "object_id" in df.columns else np.arange(len(df))

    for _, group in df.groupby(group_cols, sort=False):
        idxs = group.index.to_numpy(dtype=np.int64)
        pool = idxs[np.isfinite(centers_all[idxs]).all(axis=1)]
        if len(pool) < 2:
            continue

        tree = cKDTree(centers_all[pool])
        k_query = min(len(pool), max(2, k * cfg["NEIGHBOR_CANDIDATE_MULTIPLIER"] + 1))

        for target_idx in pool:
            target_row = rows[target_idx]
            _, nearest = tree.query(centers_all[target_idx], k=k_query)
            candidates = []

            for cand_idx in pool[np.atleast_1d(nearest)]:
                if cand_idx == target_idx or object_ids[cand_idx] == object_ids[target_idx]:
                    continue
                cand_row = rows[cand_idx]
                if cfg["NEIGHBOR_REQUIRE_TIME_OVERLAP"] and temporal_iou(target_row, cand_row) <= 0:
                    continue
                relation = compute_relation_features(target_row, cand_row)
                if relation is None:
                    continue

                dist = float(relation[2])
                score = _score_candidate(relation, target_row, cand_row, stats_all, target_idx, cand_idx, cfg)

                candidates.append((score, int(cand_idx), dist, relation))

            candidates.sort(key=lambda c: c[0])
            selected, seen_objects = [], set()
            for _, cand_idx, dist, relation in candidates:
                oid = str(object_ids[cand_idx])
                if oid in seen_objects:
                    continue
                seen_objects.add(oid)
                selected.append({"idx": cand_idx, "dist": dist, "relation": relation})
                if len(selected) >= k:
                    break
            neighbor_lists[target_idx] = selected

    return neighbor_lists


def build_context_window_index(df, window_size, max_time_gap):
    """Với mỗi clip, lấy tối đa `window_size` clip trước/sau của CÙNG 1 người (cùng track),
    bỏ qua clip cách quá `max_time_gap` giây. offset = -2, -1, +1, +2..."""
    context_lists = [[] for _ in range(len(df))]
    if not all(c in df.columns for c in TRACK_COLS):
        print("[!] Thiếu cột session/camera_id/object_id -> context window rỗng.")
        return context_lists

    for _, group in df.groupby(TRACK_COLS, sort=False):
        if len(group) < 2:
            continue
        idxs, starts = sort_group_by_time(group)
        for pos in range(len(idxs)):
            entries = []
            for offset in range(-window_size, window_size + 1):
                j = pos + offset
                if offset == 0 or not 0 <= j < len(idxs):
                    continue
                gap = abs(starts[j] - starts[pos])
                if max_time_gap is not None and gap > max_time_gap:
                    continue
                entries.append({"idx": int(idxs[j]), "offset": offset, "time_gap": float(gap)})
            context_lists[int(idxs[pos])] = entries

    return context_lists