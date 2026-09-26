"""Global CONFIG dict + reproducibility / device helpers.

Corresponds to notebook section "1. Cai dat & cau hinh" (cells 5-6).
"""
from __future__ import annotations

import copy
import os
import random
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")


DEFAULT_CONFIG = {
    "DRY_RUN": False,
    "DRY_RUN_SAMPLES": 32,
    "DRY_RUN_EPOCHS": 2,

    # ---- Paths ----
    "OUTPUT_DIR": "/kaggle/input/datasets/deadwish1/engagement-dataset-npy",
    "FEATURE_DIR": "/kaggle/input/datasets/drakhight/resnet-engagement/resnet_features",
    "SKELETON_DIR": "/kaggle/input/datasets/drakhight/skeleton-engagement/skeleton_features_yolo/skeleton",
    "CHECKPOINT_DIR": "/kaggle/working/checkpoints_v4_social",
    "EXCLUDE_UNKNOWN": True,

    # Split train/val/test theo SESSION (file split_names.json, dang {"train":[...],
    # "val":[...], "test":[...]} voi ten file "<session>.pt"). Neu dat duong dan hop
    # le, GHI DE cot 'split' co san trong manifest -- dam bao MOI segment cua CUNG 1
    # session luon nam TRON trong 1 split (tranh ro ri xuyen split o cap session).
    # De None/duong dan sai -- dung lai cot 'split' co san trong manifest nhu truoc.
    "SPLIT_NAMES_JSON_PATH": "/kaggle/input/datasets/deadwish1/slip-names/split_names.json",

    # ---- Cached data ----
    "NUM_FRAMES": 16,
    "NUM_KEYPOINTS": 17,
    "FEATURE_DIM": 2048,

    # ---- Spatial / social graph ----
    "K_NEIGHBORS": 4,
    "NEIGHBOR_CANDIDATE_MULTIPLIER": 6,
    "NEIGHBOR_REQUIRE_TIME_OVERLAP": True,

    "USE_ORIENTATION_PENALTY": True,
    "ORIENTATION_ANGLE_THRESHOLD_DEG": 60.0,
    "ORIENTATION_PENALTY": 1.0,

    "RELATION_DIM": 13,
    "USE_RELATION_FEATURES": False,
    "SOCIAL_DROPOUT": 0.30,

    # ---- Context Window (ngu canh THOI GIAN) ----
    "USE_CONTEXT_WINDOW": True,
    "CONTEXT_WINDOW_SIZE": 2,
    "CONTEXT_MAX_TIME_GAP_SECONDS": 60.0,
    "CONTEXT_GNN_LAYERS": 2,
    "CONTEXT_GNN_HEADS": 4,
    "CONTEXT_DROPOUT": 0.20,
    "USE_CONTEXT_RESIDUAL_GATE": True,

    # ---- GNN (Build Graph tren cac node Fk) ----
    "GNN_LAYERS": 2,
    "GNN_HEADS": 4,
    "USE_SOCIAL_RESIDUAL_GATE": True,

    # ---- Time Attention cho Body/Face/Neighbor ----
    "USE_ADAPTIVE_ATTENTION_GATE": True,

    # ---- Multimodal Self-Attention Fusion ----
    "FUSION_LAYERS": 2,
    "FUSION_HEADS": 4,

    # ---- Behavior/emotion auxiliary multi-task supervision ----
    "USE_BEHAVIOR_EMOTION_AUX": True,
    "AUX_LABEL_COLUMNS": {
        "pose": "pose_label",
        "act": "act_label",
        "obj": "obj_label",
        "int": "int_label",
        "emo": "emo_label",
    },
    "AUX_LOSS_WEIGHT": 0.08,
    "AUX_WARMUP_EPOCHS": 2,
    "AUX_STOP_GRADIENT": True,
    "AUX_HEAD_DROPOUT": 0.30,

    # ---- Target encoders ----
    "EMBED_DIM": 192,
    "DROPOUT": 0.40,
    "SKELETON_CONF_THR": 0.05,

    # ---- ResNet -> Skeleton FiLM conditioning ----
    "USE_RESNET_TO_SKELETON_FUSION": True,
    "FILM_DROPOUT": 0.10,

    # ---- Supervised Contrastive Learning ----
    "USE_SUPCON_LOSS": True,
    "SUPCON_PROJECTION_DIM": 128,
    "SUPCON_TEMPERATURE": 0.10,
    "SUPCON_LOSS_WEIGHT": 0.30,
    "SUPCON_WARMUP_EPOCHS": 3,

    # ---- Classification loss ----
    "USE_FOCAL_LOSS": True,
    "FOCAL_GAMMA": 1.25,
    "FOCAL_ALPHA_SMOOTHING": 0.35,
    "CLASS_WEIGHT_MODE": "effective_number",
    "EFFECTIVE_NUMBER_BETA": 0.999,
    "LABEL_SMOOTHING": 0.10,

    "USE_ORDINAL_AUX_LOSS": True,
    "ORDINAL_LABEL_ORDER": [
        "eng_disengaged",
        "eng_normal",
        "eng_engaged",
        "eng_very_engaged",
    ],
    "ORDINAL_LOSS_WEIGHT": 0.15,
    "ORDINAL_WARMUP_EPOCHS": 3,
    "ORDINAL_HEAD_DROPOUT": 0.10,

    # ---- Sampling ----
    "USE_WEIGHTED_SAMPLER": True,
    "SAMPLER_MODE": "inverse",

    # ---- Track-level evaluation ----
    "USE_TRACK_LEVEL_MODEL": False,
    "MAX_TRACK_LEN": 32,
    "TRACK_LSTM_LAYERS": 2,
    "TRACK_LSTM_DROPOUT": 0.30,

    # ---- Training ----
    "BATCH_SIZE": 16,
    "GRAD_ACCUM_STEPS": 1,
    "EPOCHS": 25,
    "LR": 1.0e-4,
    "WEIGHT_DECAY": 0.10,
    "WARMUP_EPOCHS": 1,
    "NUM_WORKERS": 4,
    "PERSISTENT_WORKERS": True,
    "PREFETCH_FACTOR": 4,
    "USE_AMP": True,
    "AMP_DTYPE": "fp16",
    "PIN_MEMORY": True,
    "GRAD_CLIP_NORM": 1.0,
    "EARLY_STOPPING_PATIENCE": 4,

    # Test-time bias tuning
    "TUNE_LOGIT_BIAS": True,
    "LOGIT_BIAS_MIN": -0.6,
    "LOGIT_BIAS_MAX": 0.6,
    "LOGIT_BIAS_STEP": 0.1,

    "RUN_SPEED_BENCHMARK": True,
    "BENCHMARK_BATCHES": 5,
    "SEED": 42,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_config(overrides: dict | None = None) -> dict:
    """Return a fresh CONFIG dict (deep copy of DEFAULT_CONFIG merged with overrides),
    create the checkpoint directory, seed all RNGs and set fast-math CUDA flags."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if overrides:
        cfg.update(overrides)

    os.makedirs(cfg["CHECKPOINT_DIR"], exist_ok=True)
    seed_everything(cfg["SEED"])

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    return cfg


def setup_device(verbose: bool = True) -> tuple[torch.device, int]:
    """Detect CUDA devices and return (device, n_gpus). Mirrors notebook cell 5."""
    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | So GPU: {n_gpus}")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} -- {props.total_memory / (1024**3):.1f} GB")
    return device, n_gpus


def print_config(cfg: dict) -> None:
    print("\nV5 configuration ready.")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
