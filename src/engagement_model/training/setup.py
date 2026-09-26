"""Model instantiation, optimizer/scheduler construction, AMP setup.

Corresponds to notebook section "8. Khoi tao EngagementModelV5, optimizer, cosine
scheduler, AMP" (cell 30).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..models.engagement_model import EngagementModelV5, TrackEngagementModel


def unwrap_model(model):
    """Strip the DataParallel wrapper (if any) to get the underlying module."""
    return model.module if hasattr(model, "module") else model


def build_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_model(cfg: dict, num_classes: int, device, n_gpus: int):
    """Instantiate EngagementModelV5 or TrackEngagementModel (per
    cfg['USE_TRACK_LEVEL_MODEL']), move to device, wrap in DataParallel if n_gpus>1."""
    use_track_level = cfg.get("USE_TRACK_LEVEL_MODEL", False)
    if use_track_level:
        model = TrackEngagementModel(cfg, num_classes, cfg["FEATURE_DIM"]).to(device)
        print("Dang dung TrackEngagementModel (danh gia theo TRACK).")
    else:
        model = EngagementModelV5(cfg, num_classes, cfg["FEATURE_DIM"]).to(device)
        print("Dang dung EngagementModelV5 (danh gia theo SEGMENT doc lap).")

    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Da boc model bang nn.DataParallel (so GPU: {n_gpus}).")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Tong tham so model: {n_params / 1e6:.2f}M")
    return model


def build_optimizer_and_scheduler(model, cfg: dict, train_loader):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"],
    )

    steps_per_epoch = math.ceil(len(train_loader) / cfg["GRAD_ACCUM_STEPS"])
    total_steps = steps_per_epoch * cfg["EPOCHS"]
    warmup_steps = steps_per_epoch * cfg["WARMUP_EPOCHS"]
    scheduler = build_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)

    print(f"So buoc/epoch (sau grad-accum): {steps_per_epoch} | Tong so buoc: {total_steps}")
    return optimizer, scheduler


def resolve_amp_dtype(cfg: dict, device):
    if device.type != "cuda" or not cfg.get("USE_AMP", True):
        return torch.float32
    requested = str(cfg.get("AMP_DTYPE", "fp16")).lower()
    if requested in {"bf16", "bfloat16"} and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16  # e.g. T4 does not support bf16 -> always fp16


def build_scaler(cfg: dict, device):
    """Resolve AMP dtype (stored back into cfg['_AMP_DTYPE']) and build the
    GradScaler. Returns (scaler, amp_dtype, use_grad_scaler)."""
    amp_dtype = resolve_amp_dtype(cfg, device)
    cfg["_AMP_DTYPE"] = amp_dtype
    use_grad_scaler = device.type == "cuda" and cfg.get("USE_AMP", True) and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
    print(f"AMP dtype: {amp_dtype} | GradScaler enabled: {use_grad_scaler}")
    return scaler, amp_dtype, use_grad_scaler
