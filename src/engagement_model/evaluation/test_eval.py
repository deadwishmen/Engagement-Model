"""Final evaluation on the (held-out) test set using the best validation checkpoint.

Corresponds to notebook section "11. Danh gia test bang checkpoint V4 tot nhat + social
diagnostics" (cell 38).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from ..losses import build_ordinal_rank_by_class_id
from ..models.engagement_model import EngagementModelV5, TrackEngagementModel
from ..training.engine import (
    aggregate_session_level_evaluation,
    apply_logit_bias,
    compute_uncertainty_metrics,
    evaluate,
    tune_logit_bias,
)
from .social_diagnostics import analyze_social_interaction


def run_test_evaluation(
    cfg: dict,
    device,
    n_gpus: int,
    num_classes: int,
    best_ckpt_path: str,
    test_loader,
    val_loader,
    criterion,
    label2id: dict,
    id2label: dict,
    label_names_ordered: list,
    aux_task_names: list,
    df: pd.DataFrame,
    ordinal_rank_by_id=None,
    make_plots: bool = True,
):
    """Load the best checkpoint, evaluate on test (or val if no test split), tune the
    per-class logit bias on validation, aggregate to session level, and run social
    diagnostics. Returns a dict of all computed artifacts, or None if no checkpoint
    was found."""
    if not os.path.exists(best_ckpt_path):
        print(f"[!] Checkpoint not found: {best_ckpt_path}")
        return None

    ckpt = torch.load(best_ckpt_path, map_location=device)
    print(
        f"Selected checkpoint: epoch={ckpt['epoch']} | "
        f"val_macro_f1={ckpt['val_macro_f1']:.4f}"
    )

    if ordinal_rank_by_id is None and cfg.get("USE_ORDINAL_AUX_LOSS", False):
        ordinal_rank_by_id = build_ordinal_rank_by_class_id(
            label2id, cfg["ORDINAL_LABEL_ORDER"]
        ).to(device)

    # IMPORTANT: instantiate exactly the same architecture used during training.
    if cfg.get("USE_TRACK_LEVEL_MODEL", False):
        final_model = TrackEngagementModel(cfg, num_classes, cfg["FEATURE_DIM"]).to(device)
    else:
        final_model = EngagementModelV5(cfg, num_classes, cfg["FEATURE_DIM"]).to(device)
    final_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if n_gpus > 1:
        final_model = nn.DataParallel(final_model)

    target_loader = test_loader if test_loader is not None else val_loader
    if test_loader is None:
        print("[!] No separate test split; reporting validation metrics instead.")

    test_metrics = evaluate(
        final_model,
        target_loader,
        criterion,
        cfg,
        device,
        aux_task_names=aux_task_names,
        desc="TEST",
        label_names=label_names_ordered,
        ordinal_rank_by_id=ordinal_rank_by_id,
    )

    print(
        f"[{test_metrics['desc']}] loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['acc']:.4f} macro_f1={test_metrics['macro_f1']:.4f} (bias=0, truoc khi tune)"
    )

    # ---- Test-time logit bias tuning (coordinate descent on VALIDATION) ----
    tuned_bias = None
    if cfg.get("TUNE_LOGIT_BIAS", False) and val_loader is not None:
        val_metrics_for_bias = evaluate(
            final_model, val_loader, criterion, cfg, device,
            aux_task_names=aux_task_names,
            desc="val-for-bias-tuning", label_names=label_names_ordered,
            ordinal_rank_by_id=ordinal_rank_by_id,
        )
        val_probs = val_metrics_for_bias["probs"]
        val_labels_arr = val_metrics_for_bias["labels"]

        tuned_bias, tuned_val_score = tune_logit_bias(
            val_probs, val_labels_arr, num_classes,
            bias_min=cfg.get("LOGIT_BIAS_MIN", -0.6),
            bias_max=cfg.get("LOGIT_BIAS_MAX", 0.6),
            bias_step=cfg.get("LOGIT_BIAS_STEP", 0.1),
        )
        print(f"\nTuned logit bias (chon tren validation, val macro_f1 sau tune={tuned_val_score:.4f}): "
              + ", ".join(f"{lbl}={tuned_bias[i]:+.2f}" for i, lbl in enumerate(label_names_ordered)))

        test_probs = test_metrics["probs"]
        test_labels_arr = test_metrics["labels"]

        biased_preds = apply_logit_bias(test_probs, tuned_bias)
        biased_acc = accuracy_score(test_labels_arr, biased_preds)
        biased_macro_f1 = f1_score(test_labels_arr, biased_preds, average="macro", zero_division=0)

        print(f"[TEST] acc={biased_acc:.4f} macro_f1={biased_macro_f1:.4f} "
              f"(SAU khi tune bias tren validation)")

        test_metrics["preds"] = biased_preds
        test_metrics["acc"] = biased_acc
        test_metrics["macro_f1"] = biased_macro_f1
    else:
        print("[!] TUNE_LOGIT_BIAS=False hoac khong co validation loader -- bo qua calibration, "
              "dung nguyen preds argmax (bias=0).")

    if test_metrics.get("aux_accuracy"):
        print("\nAuxiliary behavior/emotion task accuracy (diagnostic only):")
        for task_name, acc_val in test_metrics["aux_accuracy"].items():
            acc_str = f"{acc_val:.4f}" if acc_val is not None else "n/a (khong co sample hop le)"
            print(f"  {task_name}: accuracy={acc_str}")

    # ---- Session-level evaluation (majority vote) ----
    session_eval = aggregate_session_level_evaluation(test_metrics, df, label2id, id2label)
    if session_eval is not None:
        print(f"\n=== Danh gia CAP DO SESSION (majority vote {session_eval['n_sessions']} session) ===")
        print(f"session_acc={session_eval['session_acc']:.4f} "
              f"session_macro_f1={session_eval['session_macro_f1']:.4f}")
        print("\nSo sanh: segment-level macro_f1="
              f"{test_metrics['macro_f1']:.4f} vs session-level macro_f1={session_eval['session_macro_f1']:.4f}")
        print("\nPer-class F1 (cap do session):")
        for lbl in label_names_ordered:
            r = session_eval["session_report"].get(lbl, {})
            print(f"  {lbl}: precision={r.get('precision', 0):.3f} "
                  f"recall={r.get('recall', 0):.3f} f1={r.get('f1-score', 0):.3f} "
                  f"support={r.get('support', 0):.0f}")

    print("\nClassification report (segment-level):")
    print(classification_report(
        test_metrics["labels"],
        test_metrics["preds"],
        labels=list(range(num_classes)),
        target_names=label_names_ordered,
        zero_division=0,
    ))

    cm = confusion_matrix(
        test_metrics["labels"], test_metrics["preds"], labels=list(range(num_classes))
    )
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{x}" for x in label_names_ordered],
        columns=[f"pred_{x}" for x in label_names_ordered],
    )
    print("\nConfusion matrix:")
    print(cm_df)

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)
    print("\nRow-normalized confusion matrix:")
    print(pd.DataFrame(
        cm_norm,
        index=[f"true_{x}" for x in label_names_ordered],
        columns=[f"pred_{x}" for x in label_names_ordered],
    ).round(3))

    cm_percent = cm_norm * 100
    cm_percent_df = pd.DataFrame(
        cm_percent,
        index=[f"true_{x}" for x in label_names_ordered],
        columns=[f"pred_{x}" for x in label_names_ordered],
    )
    print("\nConfusion matrix (%):")
    print(cm_percent_df.round(1).astype(str) + "%")

    if make_plots:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(1.6 * num_classes + 2, 1.4 * num_classes + 2))
            im = ax.imshow(cm_percent, cmap="Blues", vmin=0, vmax=100)

            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))
            ax.set_xticklabels(label_names_ordered, rotation=45, ha="right")
            ax.set_yticklabels(label_names_ordered)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix (%) — V5 test set")

            for i in range(num_classes):
                for j in range(num_classes):
                    value = cm_percent[i, j]
                    text_color = "white" if value > 50 else "black"
                    ax.text(j, i, f"{value:.1f}%", ha="center", va="center",
                            color=text_color, fontsize=10)

            fig.colorbar(im, ax=ax, label="% (theo hang)")
            plt.tight_layout()
            plt.savefig(os.path.join(cfg["CHECKPOINT_DIR"], "confusion_matrix_percent_v5.png"), dpi=120)
            plt.close(fig)
        except Exception as e:
            print(f"[!] Khong ve duoc confusion matrix heatmap: {e}")

    # Confidence / coverage diagnostic.
    unc = compute_uncertainty_metrics(
        test_metrics, method="entropy", coverage_target=0.8, num_classes=num_classes
    )
    print("\nRisk/coverage table:")
    print(unc["risk_coverage_table"].round(4).to_string(index=False))

    if make_plots:
        try:
            import matplotlib.pyplot as plt
            rc = unc["risk_coverage_table"]
            fig = plt.figure(figsize=(7, 4.5))
            plt.plot(rc["coverage"], rc["macro_f1"], marker="o")
            plt.xlabel("Coverage (fraction of most-confident samples kept)")
            plt.ylabel("Macro-F1")
            plt.gca().invert_xaxis()
            plt.title("F1-Coverage curve — V5 engagement model")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(cfg["CHECKPOINT_DIR"], "f1_coverage_v5.png"), dpi=120)
            plt.close(fig)
        except Exception as e:
            print(f"[!] Could not draw coverage curve: {e}")

    # Social-context diagnostics.
    social = analyze_social_interaction(final_model, target_loader, device, cfg)
    print("\n=== Social interaction diagnostics ===")
    print("Normalized attention entropy (Fb -> Fk):", social["mean_attn_entropy_normalized"])
    print("\nWith vs without neighbors:")
    for name, metrics in social["group_compare"].items():
        print(name, metrics)
    print("\nMetrics by number of neighbors:")
    print(social["by_neighbor_count"].round(4).to_string(index=False))

    return {
        "final_model": final_model,
        "test_metrics": test_metrics,
        "tuned_bias": tuned_bias,
        "session_eval": session_eval,
        "confusion_matrix": cm_df,
        "uncertainty": unc,
        "social_diagnostics": social,
    }
