"""Đánh giá 1 checkpoint đã train trên tập test.

Ví dụ:
    python scripts/evaluate.py --config configs/kaggle.yaml --checkpoint /kaggle/working/outputs/best_model.pt
"""
import os

from _common import build_parser

from engagement.config import get_device, load_config, seed_everything
from engagement.data import make_loaders, prepare_data
from engagement.engine import Engine, load_checkpoint
from engagement.evaluation import run_test_evaluation
from engagement.models import build_model, unwrap_model


def main():
    parser = build_parser("Evaluate engagement model V5")
    parser.add_argument("--checkpoint", default=None, help="Mặc định: CHECKPOINT_DIR/CHECKPOINT_NAME")
    parser.add_argument("--out-dir", default=None, help="Thư mục lưu kết quả (mặc định: CHECKPOINT_DIR/eval)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    seed_everything(cfg["SEED"])
    device = get_device()
    ckpt_path = args.checkpoint or os.path.join(cfg["CHECKPOINT_DIR"], cfg["CHECKPOINT_NAME"])

    bundle = prepare_data(cfg)
    loaders = make_loaders(bundle, cfg)
    engine = Engine(cfg, bundle, device)

    ckpt = load_checkpoint(ckpt_path, device)
    if ckpt.get("label2id") and ckpt["label2id"] != bundle.label2id:
        raise RuntimeError(f"label2id của checkpoint {ckpt['label2id']} khác dữ liệu hiện tại {bundle.label2id}")
    model = build_model(cfg, bundle, device)
    unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
    print(f"Checkpoint: epoch={ckpt['epoch']} | val_macro_f1={ckpt['val_macro_f1']:.4f}")

    run_test_evaluation(engine, model, loaders, args.out_dir or os.path.join(cfg["CHECKPOINT_DIR"], "eval"))


if __name__ == "__main__":
    main()
