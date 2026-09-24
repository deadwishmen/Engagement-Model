"""Huấn luyện EngagementModelV5 rồi đánh giá checkpoint tốt nhất trên test.

Ví dụ:
    python scripts/train.py --config configs/kaggle.yaml
    python scripts/train.py --config configs/kaggle.yaml --set DRY_RUN=true
"""
import os

from _common import build_parser

from engagement.config import get_device, load_config, print_environment, seed_everything
from engagement.data import make_loaders, prepare_data
from engagement.engine import Engine, load_checkpoint
from engagement.evaluation import run_test_evaluation
from engagement.models import build_model, unwrap_model


def main():
    parser = build_parser("Train engagement model V5")
    parser.add_argument("--skip-test", action="store_true", help="Không đánh giá test sau khi train")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    print_environment()
    seed_everything(cfg["SEED"])
    device = get_device()
    os.makedirs(cfg["CHECKPOINT_DIR"], exist_ok=True)

    bundle = prepare_data(cfg)
    loaders = make_loaders(bundle, cfg)
    if loaders["train"] is None or loaders["val"] is None:
        raise RuntimeError("Thiếu tập train hoặc val.")

    engine = Engine(cfg, bundle, device)
    model = build_model(cfg, bundle, device)
    print(f"Tham số model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | "
          f"AMP: {engine.use_amp} ({engine.amp_dtype})")

    if cfg["RUN_SPEED_BENCHMARK"]:
        engine.benchmark(model, loaders["train"], cfg["BENCHMARK_BATCHES"])

    ckpt_path = os.path.join(cfg["CHECKPOINT_DIR"], cfg["CHECKPOINT_NAME"])
    engine.fit(model, loaders["train"], loaders["val"], ckpt_path)

    if not args.skip_test:
        ckpt = load_checkpoint(ckpt_path, device)
        print(f"\nNạp checkpoint tốt nhất: epoch={ckpt['epoch']} | val_macro_f1={ckpt['val_macro_f1']:.4f}")
        unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        run_test_evaluation(engine, model, loaders, cfg["CHECKPOINT_DIR"])


if __name__ == "__main__":
    main()
