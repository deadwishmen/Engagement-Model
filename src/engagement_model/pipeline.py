"""End-to-end pipeline: load data -> build graphs/loaders -> build model -> train ->
evaluate on test.

This is the programmatic equivalent of running the whole original notebook top to
bottom. Each step is also usable standalone (see the submodules), which is handy for
interactive work in a notebook.
"""
from __future__ import annotations

from .config import get_config, print_config, setup_device
from .data.diagnostics import diagnose_cache_dirs, diagnose_label_stability_within_track, \
    diagnose_track_split_leakage
from .data.dataset import build_dataloaders
from .data.filtering import filter_by_cache_availability
from .data.graph import build_context_window_index, build_neighbor_index, print_context_stats, \
    print_neighbor_stats
from .data.manifest import load_manifest_and_prepare, prepare_aux_labels
from .evaluation.test_eval import run_test_evaluation
from .losses import build_criterion
from .training.engine import benchmark_pipeline
from .training.loop import run_training
from .training.setup import build_model, build_optimizer_and_scheduler, build_scaler


def run_full_pipeline(config_overrides: dict | None = None, run_diagnostics: bool = True):
    """Run the complete V5 engagement-model pipeline and return a dict with every
    intermediate/final artifact (config, dataframes, loaders, model, history,
    test evaluation, ...).
    """
    cfg = get_config(config_overrides)
    print_config(cfg)
    device, n_gpus = setup_device()

    # ---- 1. Manifest ----
    df, label2id, id2label, num_classes = load_manifest_and_prepare(cfg)
    print(df["engagement_label"].value_counts())

    if run_diagnostics:
        diagnose_track_split_leakage(df)
        diagnose_label_stability_within_track(df, max_gap_seconds=15.0)

    df, aux_label2id, aux_id2label = prepare_aux_labels(df, cfg)

    if run_diagnostics:
        diagnose_cache_dirs(df, cfg)

    # ---- 2. K-hop neighbor graph + temporal context window (whole-df preview; the
    # loaders below rebuild these PER SPLIT to avoid leakage) ----
    neighbor_lists = build_neighbor_index(
        df,
        k_neighbors=cfg["K_NEIGHBORS"],
        candidate_multiplier=cfg.get("NEIGHBOR_CANDIDATE_MULTIPLIER", 6),
        require_time_overlap=cfg.get("NEIGHBOR_REQUIRE_TIME_OVERLAP", True),
        skeleton_dir=cfg.get("SKELETON_DIR"),
        skeleton_conf_thr=cfg.get("SKELETON_CONF_THR", 0.05),
        use_orientation_penalty=cfg.get("USE_ORIENTATION_PENALTY", True),
        orientation_angle_threshold_deg=cfg.get("ORIENTATION_ANGLE_THRESHOLD_DEG", 60.0),
        orientation_penalty=cfg.get("ORIENTATION_PENALTY", 1.0),
    )
    print_neighbor_stats(neighbor_lists, df, cfg["K_NEIGHBORS"])

    context_lists = build_context_window_index(
        df, window_size=cfg["CONTEXT_WINDOW_SIZE"],
        max_time_gap_seconds=cfg.get("CONTEXT_MAX_TIME_GAP_SECONDS", 60.0),
    )
    print_context_stats(context_lists, df, cfg["CONTEXT_WINDOW_SIZE"])

    # ---- 3. Filter to samples with cached features/skeleton ----
    df, neighbor_lists, label2id, id2label, num_classes = filter_by_cache_availability(
        df, neighbor_lists, cfg
    )

    aux_task_names = list(cfg.get("AUX_LABEL_COLUMNS", {}).keys()) if cfg.get("USE_BEHAVIOR_EMOTION_AUX", False) else []
    aux_num_classes = {task: len(aux_label2id.get(task, {})) for task in aux_task_names}
    cfg["_AUX_NUM_CLASSES"] = aux_num_classes if cfg.get("USE_BEHAVIOR_EMOTION_AUX", False) else {}

    # ---- 4. DataLoaders ----
    loaders = build_dataloaders(df, cfg, label2id, aux_label2id)

    # ---- 5. Loss, model, optimizer, scheduler, AMP ----
    criterion = build_criterion(cfg, df, label2id).to(device)
    model = build_model(cfg, num_classes, device, n_gpus)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, loaders["train_loader"])
    scaler, amp_dtype, use_grad_scaler = build_scaler(cfg, device)

    if cfg.get("RUN_SPEED_BENCHMARK", True):
        benchmark_pipeline(model, loaders["train_loader"], cfg, device,
                            n_batches=int(cfg.get("BENCHMARK_BATCHES", 5)))

    # ---- 6. Train ----
    train_result = run_training(
        model, loaders["train_loader"], loaders["val_loader"],
        optimizer, scheduler, criterion, scaler, cfg, device,
        label2id, id2label, num_classes, aux_task_names,
    )

    # ---- 7. Test evaluation using the best checkpoint ----
    test_result = run_test_evaluation(
        cfg, device, n_gpus, num_classes,
        train_result["best_ckpt_path"],
        loaders["test_loader"], loaders["val_loader"], criterion,
        label2id, id2label, train_result["label_names_ordered"],
        aux_task_names, df,
        ordinal_rank_by_id=train_result["ordinal_rank_by_id"],
    )

    return {
        "config": cfg,
        "device": device,
        "n_gpus": n_gpus,
        "df": df,
        "label2id": label2id,
        "id2label": id2label,
        "num_classes": num_classes,
        "aux_label2id": aux_label2id,
        "aux_id2label": aux_id2label,
        "aux_task_names": aux_task_names,
        "loaders": loaders,
        "criterion": criterion,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "train_result": train_result,
        "test_result": test_result,
    }
