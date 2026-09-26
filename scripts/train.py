#!/usr/bin/env python3
"""CLI entry point: run the full engagement-model training pipeline.

Usage:
    python scripts/train.py --config configs/default_paths.json
    python scripts/train.py --output-dir /path/to/data --feature-dir /path/to/features \
        --skeleton-dir /path/to/skeleton --checkpoint-dir /path/to/checkpoints \
        --epochs 25 --batch-size 16

Any key from `engagement_model.config.DEFAULT_CONFIG` can be supplied either via a
JSON config file (--config) or as an individual --set KEY=VALUE override (repeatable).
Command-line --output-dir/--feature-dir/... flags are shorthands for the most common
path overrides.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running this script directly from the repo without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engagement_model.pipeline import run_full_pipeline  # noqa: E402


def _parse_value(raw: str):
    """Best-effort type coercion for --set KEY=VALUE overrides."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None,
                    help="Path to a JSON file with CONFIG overrides.")
    p.add_argument("--output-dir", type=str, default=None, help="Shorthand for OUTPUT_DIR.")
    p.add_argument("--feature-dir", type=str, default=None, help="Shorthand for FEATURE_DIR.")
    p.add_argument("--skeleton-dir", type=str, default=None, help="Shorthand for SKELETON_DIR.")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Shorthand for CHECKPOINT_DIR.")
    p.add_argument("--split-names-json", type=str, default=None, help="Shorthand for SPLIT_NAMES_JSON_PATH.")
    p.add_argument("--epochs", type=int, default=None, help="Shorthand for EPOCHS.")
    p.add_argument("--batch-size", type=int, default=None, help="Shorthand for BATCH_SIZE.")
    p.add_argument("--dry-run", action="store_true", help="Shorthand for DRY_RUN=True (quick smoke test).")
    p.add_argument("--no-diagnostics", action="store_true",
                    help="Skip the track-leakage / label-stability / cache-path diagnostics.")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="Arbitrary CONFIG override, e.g. --set LR=5e-5. Repeatable.")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    overrides: dict = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            file_overrides = json.load(f)
        file_overrides.pop("_comment", None)
        overrides.update(file_overrides)

    shorthand_map = {
        "output_dir": "OUTPUT_DIR",
        "feature_dir": "FEATURE_DIR",
        "skeleton_dir": "SKELETON_DIR",
        "checkpoint_dir": "CHECKPOINT_DIR",
        "split_names_json": "SPLIT_NAMES_JSON_PATH",
        "epochs": "EPOCHS",
        "batch_size": "BATCH_SIZE",
    }
    for attr, key in shorthand_map.items():
        value = getattr(args, attr)
        if value is not None:
            overrides[key] = value

    if args.dry_run:
        overrides["DRY_RUN"] = True

    for kv in args.set:
        if "=" not in kv:
            raise SystemExit(f"--set expects KEY=VALUE, got: {kv!r}")
        key, raw_value = kv.split("=", 1)
        overrides[key] = _parse_value(raw_value)

    result = run_full_pipeline(config_overrides=overrides, run_diagnostics=not args.no_diagnostics)

    best_ckpt = result["train_result"]["best_ckpt_path"]
    best_f1 = result["train_result"]["best_val_macro_f1"]
    print(f"\nDone. Best checkpoint: {best_ckpt} (val macro_f1={best_f1:.4f})")
    if result["test_result"] is not None:
        print(f"Test macro_f1: {result['test_result']['test_metrics']['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
