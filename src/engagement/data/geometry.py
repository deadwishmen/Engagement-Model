"""Thời gian clip, quỹ đạo bbox, đặc trưng quan hệ và hướng cơ thể."""
import math
import os

import numpy as np

TRACK_COLS = ["session", "camera_id", "object_id"]   # 1 track = 1 người trong 1 session/camera

TIME_SEQUENCE_COLS = ("bbox_time_sec", "time_sec", "frame_time_sec", "timestamps",
                      "bbox_timestamp", "frame_timestamps")
TIME_RANGE_COLS = (("start_time", "end_time"), ("start_sec", "end_sec"),
                   ("track_start_sec", "track_end_sec"))

RELATION_FEATURE_NAMES_BASE = [
    "dx_mean", "dy_mean", "distance_mean", "distance_min",
    "relative_vx", "relative_vy", "distance_change", "temporal_iou",
    "log_area_ratio", "target_bbox_motion", "neighbor_bbox_motion", "relative_scale_change",
]
RELATION_FEATURE_NAMES = RELATION_FEATURE_NAMES_BASE + ["neighbor_feature_quality"]

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6   # chỉ số COCO-17


# ------------------------------------------------------------------ thời gian
def _finite_range(t0, t1):
    t0, t1 = float(t0), float(t1)
    return (min(t0, t1), max(t0, t1)) if np.isfinite([t0, t1]).all() else None


def track_time_range(row):
    """(t_min, t_max) của clip từ chuỗi timestamp hoặc cặp cột start/end."""
    for col in TIME_SEQUENCE_COLS:
        values = row.get(col)
        if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0:
            try:
                arr = np.asarray(values, dtype=np.float64).ravel()
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    return float(arr.min()), float(arr.max())
            except (TypeError, ValueError):
                pass
    for start_col, end_col in TIME_RANGE_COLS:
        if start_col in row.index and end_col in row.index:
            try:
                result = _finite_range(row[start_col], row[end_col])
                if result:
                    return result
            except (TypeError, ValueError):
                pass
    return None


def clip_time_range(row):
    """Ưu tiên seg_start/seg_end (thời gian tuyệt đối của clip), fallback track_time_range."""
    if "seg_start" in row.index and "seg_end" in row.index:
        try:
            result = _finite_range(row["seg_start"], row["seg_end"])
            if result:
                return result
        except (TypeError, ValueError):
            pass
    return track_time_range(row)


def sort_group_by_time(group):
    """(index đã sắp theo thời gian bắt đầu, mảng thời gian bắt đầu). Bỏ clip không rõ thời gian."""
    starts = group.apply(lambda r: (clip_time_range(r) or (np.nan,))[0], axis=1)
    starts = starts.dropna().sort_values()
    return starts.index.to_numpy(dtype=np.int64), starts.to_numpy(dtype=np.float64)


def temporal_iou(row_a, row_b):
    ra, rb = track_time_range(row_a), track_time_range(row_b)
    if ra is None or rb is None:
        return 0.0
    inter = max(0.0, min(ra[1], rb[1]) - max(ra[0], rb[0]))
    union = max(max(ra[1], rb[1]) - min(ra[0], rb[0]), 1e-8)
    return float(inter / union)


# ------------------------------------------------------------------ bbox
def bbox_array(row):
    """Quỹ đạo bbox (N, 4), ưu tiên bbox_norm."""
    for col in ("bbox_norm", "bbox_pixel"):
        values = row.get(col)
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values) == 0:
            continue
        try:
            arr = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 2 and arr.shape[1] >= 4:
            arr = arr[:, :4]
            arr = arr[np.isfinite(arr).all(axis=1)]
            if len(arr):
                return arr
    return None


def bbox_mean_stats(row):
    """(cx, cy, w, h) trung bình trên cả clip."""
    arr = bbox_array(row)
    if arr is None:
        return None
    cx = ((arr[:, 0] + arr[:, 2]) / 2).mean()
    cy = ((arr[:, 1] + arr[:, 3]) / 2).mean()
    w = np.clip(arr[:, 2] - arr[:, 0], 1e-6, None).mean()
    h = np.clip(arr[:, 3] - arr[:, 1], 1e-6, None).mean()
    stats = (float(cx), float(cy), float(w), float(h))
    return stats if np.isfinite(stats).all() else None


def _resample(values, length=16):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.zeros(length)
    if len(values) == 1:
        return np.repeat(values, length)
    return np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(values)), values)


def _trajectory(row, length=16):
    arr = bbox_array(row)
    if arr is None:
        return None
    w = np.clip(arr[:, 2] - arr[:, 0], 1e-6, None)
    h = np.clip(arr[:, 3] - arr[:, 1], 1e-6, None)
    area = np.clip(w * h, 1e-8, None)
    return {
        "cx": _resample((arr[:, 0] + arr[:, 2]) / 2, length),
        "cy": _resample((arr[:, 1] + arr[:, 3]) / 2, length),
        "area": _resample(area, length),
        "mean_area": float(area.mean()),
    }


def compute_relation_features(target_row, neighbor_row):
    """12 đặc trưng cạnh target -> neighbor, chuẩn hoá theo sqrt(diện tích bbox target)."""
    tt, tn = _trajectory(target_row), _trajectory(neighbor_row)
    if tt is None or tn is None:
        return None

    scale = max(math.sqrt(tt["mean_area"]), 1e-6)
    dx = (tn["cx"] - tt["cx"]) / scale
    dy = (tn["cy"] - tt["cy"]) / scale
    dist = np.sqrt(dx ** 2 + dy ** 2)

    def velocity(t):
        return (t["cx"][-1] - t["cx"][0]) / scale, (t["cy"][-1] - t["cy"][0]) / scale

    def mean_step(t):
        steps = np.sqrt(np.diff(t["cx"]) ** 2 + np.diff(t["cy"]) ** 2) / scale
        return steps.mean() if len(steps) else 0.0

    def scale_growth(t):
        return np.log(max(t["area"][-1], 1e-8) / max(t["area"][0], 1e-8))

    t_vx, t_vy = velocity(tt)
    n_vx, n_vy = velocity(tn)
    area_ratio = max(tn["mean_area"] / max(tt["mean_area"], 1e-8), 1e-8)

    relation = np.array([
        dx.mean(), dy.mean(), dist.mean(), dist.min(),
        n_vx - t_vx, n_vy - t_vy, dist[-1] - dist[0],
        temporal_iou(target_row, neighbor_row),
        np.clip(np.log(area_ratio), -4.0, 4.0),
        mean_step(tt), mean_step(tn),
        scale_growth(tn) - scale_growth(tt),
    ], dtype=np.float32)
    return np.nan_to_num(relation, nan=0.0, posinf=10.0, neginf=-10.0).clip(-10.0, 10.0)


# ------------------------------------------------------------------ hướng cơ thể
_facing_cache = {}


def estimate_facing_direction(sample_id, skeleton_dir, conf_thr):
    """Vector đơn vị vuông góc với trục vai (trung bình cả clip) = trục nhìn khả dĩ.
    Không biết chiều (mặt hay lưng) nên khi dùng sẽ xét cả 2 chiều."""
    key = (skeleton_dir, sample_id)
    if key in _facing_cache:
        return _facing_cache[key]

    facing = None
    path = os.path.join(skeleton_dir, f"{sample_id}.npy")
    if os.path.exists(path):
        try:
            raw = np.load(path, allow_pickle=True).item()
            facing = _facing_from_shoulders(
                np.asarray(raw["keypoints"], dtype=np.float64),
                np.asarray(raw["keypoint_scores"], dtype=np.float64),
                np.asarray(raw["detected"], dtype=bool),
                conf_thr,
            )
        except Exception:
            facing = None

    _facing_cache[key] = facing
    return facing


def _facing_from_shoulders(kpts, scores, detected, conf_thr):
    valid = (detected
             & (scores[:, LEFT_SHOULDER] >= conf_thr)
             & (scores[:, RIGHT_SHOULDER] >= conf_thr))
    if not valid.any():
        return None

    shoulder = kpts[valid, RIGHT_SHOULDER] - kpts[valid, LEFT_SHOULDER]
    norms = np.linalg.norm(shoulder, axis=1, keepdims=True)
    good = norms[:, 0] > 1e-6
    if not good.any():
        return None
    shoulder = shoulder[good] / norms[good]

    normal = np.stack([-shoulder[:, 1], shoulder[:, 0]], axis=1)
    # Căn theo frame đầu để các frame không triệt tiêu nhau khi lấy trung bình
    signs = np.sign(normal @ normal[0])
    signs[signs == 0] = 1.0
    facing = (normal * signs[:, None]).mean(axis=0)

    norm = np.linalg.norm(facing)
    return facing / norm if norm >= 1e-6 else None


def orientation_angle_deg(facing_dir, to_target):
    """Góc nhỏ nhất (độ) giữa trục nhìn và hướng tới target."""
    length = np.linalg.norm(to_target)
    if length < 1e-6:
        return 0.0
    cos = float(np.clip(np.dot(facing_dir, to_target / length), -1.0, 1.0))
    return math.degrees(math.acos(abs(cos)))
