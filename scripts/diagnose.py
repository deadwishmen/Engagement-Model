"""So sánh 2 chế độ NEIGHBOR_RANKING xem chúng có thực sự chọn ra TẬP hàng xóm khác nhau không,
hay chỉ đổi thứ tự (vô nghĩa vì BuildGraphAttention xử lý Fk bằng self-attention, không phân
biệt vị trí slot).

Ví dụ:
    python scripts/diagnose_neighbor_ranking.py --config configs/kaggle.yaml
    python scripts/diagnose_neighbor_ranking.py --config configs/kaggle.yaml \
        --mode-a distance_only --mode-b distance_orientation
"""
from _common import build_parser

from engagement.config import load_config
from engagement.data import prepare_data
from engagement.data.geometry import estimate_facing_direction
from engagement.data.graph_index import build_neighbor_index


def main():
    parser = build_parser("So sánh tác động của 2 chế độ NEIGHBOR_RANKING")
    parser.add_argument("--mode-a", default="distance_orientation", help="Chế độ cũ (baseline)")
    parser.add_argument("--mode-b", default="distance_motion", help="Chế độ mới muốn so sánh")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    bundle = prepare_data(cfg)
    df = bundle.df.reset_index(drop=True)

    cfg_a = {**cfg, "NEIGHBOR_RANKING": args.mode_a, "USE_ORIENTATION_PENALTY": args.mode_a == "distance_orientation"}
    cfg_b = {**cfg, "NEIGHBOR_RANKING": args.mode_b, "USE_ORIENTATION_PENALTY": args.mode_b == "distance_orientation"}

    print(f"Đang xây neighbor index với NEIGHBOR_RANKING={args.mode_a!r} ...")
    nb_a = build_neighbor_index(df, cfg_a)
    print(f"Đang xây neighbor index với NEIGHBOR_RANKING={args.mode_b!r} ...")
    nb_b = build_neighbor_index(df, cfg_b)

    n_with_a = sum(bool(x) for x in nb_a)
    n_with_b = sum(bool(x) for x in nb_b)
    n_diff_set = sum({e["idx"] for e in a} != {e["idx"] for e in b} for a, b in zip(nb_a, nb_b))
    n_hit_full_k = sum(len(a) >= cfg["K_NEIGHBORS"] or len(b) >= cfg["K_NEIGHBORS"] for a, b in zip(nb_a, nb_b))
    n_with = max(n_with_a, n_with_b, 1)

    print(f"\n{args.mode_a!r}: {n_with_a} sample có hàng xóm | {args.mode_b!r}: {n_with_b} sample có hàng xóm")
    print(f"Sample chọn ĐỦ K={cfg['K_NEIGHBORS']} hàng xóm (điều kiện CẦN để cách xếp hạng có tác động): "
          f"{n_hit_full_k} ({100 * n_hit_full_k / n_with:.1f}%)")
    print(f"Sample mà 2 cách xếp hạng chọn ra TẬP hàng xóm KHÁC NHAU: "
          f"{n_diff_set} ({100 * n_diff_set / n_with:.1f}%)")

    if n_diff_set == 0:
        print(f"\n-> 2 chế độ cho kết quả GIỐNG HỆT NHAU trên dữ liệu này. Dùng {args.mode_b!r} "
              "an toàn (không đổi input của model) và không cần đọc skeleton khi build neighbor.")
    else:
        print(f"\n-> Có {n_diff_set} sample bị ảnh hưởng. Nên train 2 model rồi so bằng "
              "bootstrap_compare.py để biết cách nào thực sự tốt hơn, thay vì chỉ dựa vào số này.")

    if "distance_orientation" in (args.mode_a, args.mode_b):
        sample_ids = df["sample_id"].tolist()
        limit = min(len(sample_ids), 3000)
        n_none = sum(estimate_facing_direction(sid, cfg["SKELETON_DIR"], cfg["SKELETON_CONF_THR"]) is None
                    for sid in sample_ids[:limit])
        print(f"\n[Chỉ liên quan tới distance_orientation] Tỷ lệ facing=None trên {limit} sample: "
              f"{100 * n_none / limit:.1f}% (các trường hợp này bị bỏ qua phạt hướng, chỉ còn xếp theo khoảng cách).")


if __name__ == "__main__":
    main()