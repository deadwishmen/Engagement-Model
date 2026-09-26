"""Training loop: epochs, validation, early stopping, checkpointing.

Corresponds to notebook section "10. Huan luyen V4 -- early stopping theo validation
macro-F1" (cell 34).
"""
from __future__ import annotations

import os

import pandas as pd
import torch

from ..losses import build_ordinal_rank_by_class_id
from .engine import evaluate, train_one_epoch
from .setup import unwrap_model


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    scaler,
    cfg: dict,
    device,
    label2id: dict,
    id2label: dict,
    num_classes: int,
    aux_task_names: list,
):
    """Run the full train/validate/early-stop loop.

    Returns a dict with keys: 'history_df', 'best_ckpt_path', 'best_val_macro_f1',
    'ordinal_rank_by_id', 'label_names_ordered'.
    """
    label_names_ordered = [id2label[i] for i in range(num_classes)]

    ordinal_rank_by_id = None
    if cfg.get("USE_ORDINAL_AUX_LOSS", False):
        ordinal_rank_by_id = build_ordinal_rank_by_class_id(
            label2id, cfg["ORDINAL_LABEL_ORDER"],
        ).to(device)

        print("Ordinal semantic order:")
        for class_id, class_name in enumerate(label_names_ordered):
            print(
                f"  class_id={class_id} | {class_name:>18s} "
                f"-> ordinal_rank={int(ordinal_rank_by_id[class_id].item())}"
            )

    best_val_metric = -1.0
    best_ckpt_path = os.path.join(cfg["CHECKPOINT_DIR"], "best_model_v4_social_interaction.pt")
    history = []
    epochs_since_improve = 0

    n_epochs_to_run = cfg.get("DRY_RUN_EPOCHS", 2) if cfg.get("DRY_RUN", False) else cfg["EPOCHS"]
    if cfg.get("DRY_RUN", False):
        print(f"[DRY RUN] Chi chay {n_epochs_to_run} epoch de kiem tra code.")

    for epoch in range(1, n_epochs_to_run + 1):
        print(f"\n===== Epoch {epoch}/{n_epochs_to_run} =====")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            scaler,
            cfg,
            epoch,
            device,
            aux_task_names=aux_task_names,
            ordinal_rank_by_id=ordinal_rank_by_id,
        )
        train_loss = train_metrics["loss"]
        print(
            f"[Epoch {epoch}] train_loss={train_metrics['loss']:.4f} "
            f"cls_loss={train_metrics['classification_loss']:.4f} "
            f"aux_behavior_emo={train_metrics.get('aux_behavior_emo_loss', 0.0):.4f} "
            f"lambda_aux={train_metrics.get('aux_loss_weight', 0.0):.3f} "
            f"supcon={train_metrics.get('supcon_loss', 0.0):.4f} "
            f"lambda_supcon={train_metrics.get('supcon_weight', 0.0):.3f}"
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            cfg,
            device,
            aux_task_names=aux_task_names,
            desc=f"val (epoch {epoch})",
            label_names=label_names_ordered,
            ordinal_rank_by_id=ordinal_rank_by_id,
        )
        print(
            f"[{val_metrics['desc']}] loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['acc']:.4f} macro_f1={val_metrics['macro_f1']:.4f}"
            + (
                f" | ord_mae={val_metrics['ordinal_mae']:.3f} "
                f"ord_acc={val_metrics['ordinal_rank_acc']:.3f}"
                if val_metrics["ordinal_mae"] is not None else ""
            )
        )
        if val_metrics.get("aux_accuracy"):
            aux_acc_str = " | ".join(
                f"{task}={acc:.3f}" if acc is not None else f"{task}=n/a"
                for task, acc in val_metrics["aux_accuracy"].items()
            )
            print(f"  Auxiliary task accuracy (val, diagnostic only): {aux_acc_str}")

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_classification_loss": train_metrics["classification_loss"],
            "train_ordinal_loss": train_metrics["ordinal_loss"],
            "ordinal_weight": train_metrics["ordinal_weight"],
            "train_aux_behavior_emo_loss": train_metrics.get("aux_behavior_emo_loss"),
            "aux_loss_weight": train_metrics.get("aux_loss_weight"),
            "train_supcon_loss": train_metrics.get("supcon_loss"),
            "supcon_weight": train_metrics.get("supcon_weight"),
            "val_macro_f1": val_metrics["macro_f1"],
            "val_acc": val_metrics["acc"],
            "val_ordinal_mae": val_metrics["ordinal_mae"],
            "val_ordinal_rank_acc": val_metrics["ordinal_rank_acc"],
            "val_aux_accuracy": val_metrics.get("aux_accuracy"),
        }
        history.append(epoch_record)

        improved = val_metrics["macro_f1"] > best_val_metric
        if improved:
            best_val_metric = val_metrics["macro_f1"]
            epochs_since_improve = 0

            base_model = unwrap_model(model)

            torch.save({
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": cfg, "label2id": label2id, "id2label": id2label,
                "epoch": epoch, "val_macro_f1": best_val_metric,
                "history": history,
            }, best_ckpt_path)
            print(f"  -> Checkpoint moi tot nhat (val macro_f1={best_val_metric:.4f}), da luu.")

            report = val_metrics["per_class_report"]
            if report is not None:
                print("  Per-class F1 (val, epoch tot nhat):")
                for lbl in label_names_ordered:
                    r = report.get(lbl, {})
                    print(f"    {lbl}: precision={r.get('precision', 0):.3f} "
                          f"recall={r.get('recall', 0):.3f} f1={r.get('f1-score', 0):.3f}")
        else:
            epochs_since_improve += 1
            print(f"  (khong cai thien val macro_f1, {epochs_since_improve}/{cfg['EARLY_STOPPING_PATIENCE']})")

        if epochs_since_improve >= cfg["EARLY_STOPPING_PATIENCE"]:
            print(f"\nDung som (early stopping) sau epoch {epoch}.")
            break

    history_df = pd.DataFrame(history)
    print("\nLich su huan luyen:")
    print(history_df)

    return {
        "history_df": history_df,
        "best_ckpt_path": best_ckpt_path,
        "best_val_macro_f1": best_val_metric,
        "ordinal_rank_by_id": ordinal_rank_by_id,
        "label_names_ordered": label_names_ordered,
    }
