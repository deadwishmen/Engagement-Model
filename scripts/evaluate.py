#!/usr/bin/env python3
"""CLI entry point: evaluate an already-trained checkpoint on the test split without
retraining.

Usage:
    python scripts/evaluate.py --config configs/default_paths.json \
        --checkpoint /path/to/best_model_v4_social_interaction.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engagement_model.config import get_config, print_config, setup_device  # noqa: E402
from engagement_model.data.dataset import build_dataloaders  # noqa: E402
from engagement_model.data.filtering import filter_by_cache_availability  # noqa: E402
from engagement_model.data.graph import build_context_window_index, build_neighbor_index  # noqa: E402
from engagement_model.data.manifest import load_manifest_and_prepare, prepare_aux_labels  # noqa: E402
from engagement_model.evaluation.test_eval import run_test_evaluation  # noqa: E402
from engagement_model.losses import build_criterion  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None, help="Path to a JSON file with CONFIG overrides.")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to the checkpoint .pt file. Defaults to "
                         "<CHECKPOINT_DIR>/best_model_v4_social_interaction.pt")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="Arbitrary CONFIG override, e.g. --set BATCH_SIZE=8. Repeatable.")
    return p


def _parse_value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    overrides: dict = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            file_overrides = json.load(f)
        file_overrides.pop("_comment", None)
        overrides.update(file_overrides)
    for kv in args.set:
        key, raw_value = kv.split("=", 1)
        overrides[key] = _parse_value(raw_value)

    cfg = get_config(overrides)
    print_config(cfg)
    device, n_gpus = setup_device()

    df, label2id, id2label, num_classes = load_manifest_and_prepare(cfg)
    df, aux_label2id, aux_id2label = prepare_aux_labels(df, cfg)

    neighbor_lists = build_neighbor_index(
        df, k_neighbors=cfg["K_NEIGHBORS"],
        candidate_multiplier=cfg.get("NEIGHBOR_CANDIDATE_MULTIPLIER", 6),
        require_time_overlap=cfg.get("NEIGHBOR_REQUIRE_TIME_OVERLAP", True),
        skeleton_dir=cfg.get("SKELETON_DIR"), skeleton_conf_thr=cfg.get("SKELETON_CONF_THR", 0.05),
        use_orientation_penalty=cfg.get("USE_ORIENTATION_PENALTY", True),
        orientation_angle_threshold_deg=cfg.get("ORIENTATION_ANGLE_THRESHOLD_DEG", 60.0),
        orientation_penalty=cfg.get("ORIENTATION_PENALTY", 1.0),
    )
    build_context_window_index(df, window_size=cfg["CONTEXT_WINDOW_SIZE"],
                                max_time_gap_seconds=cfg.get("CONTEXT_MAX_TIME_GAP_SECONDS", 60.0))

    df, neighbor_lists, label2id, id2label, num_classes = filter_by_cache_availability(df, neighbor_lists, cfg)

    aux_task_names = list(cfg.get("AUX_LABEL_COLUMNS", {}).keys()) if cfg.get("USE_BEHAVIOR_EMOTION_AUX", False) else []
    cfg["_AUX_NUM_CLASSES"] = {task: len(aux_label2id.get(task, {})) for task in aux_task_names}

    loaders = build_dataloaders(df, cfg, label2id, aux_label2id)
    criterion = build_criterion(cfg, df, label2id).to(device)

    label_names_ordered = [id2label[i] for i in range(num_classes)]
    ckpt_path = args.checkpoint or f"{cfg['CHECKPOINT_DIR']}/best_model_v4_social_interaction.pt"

    result = run_test_evaluation(
        cfg, device, n_gpus, num_classes, ckpt_path,
        loaders["test_loader"], loaders["val_loader"], criterion,
        label2id, id2label, label_names_ordered, aux_task_names, df,
    )
    if result is None:
        return 1
    print(f"\nDone. Test macro_f1: {result['test_metrics']['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
