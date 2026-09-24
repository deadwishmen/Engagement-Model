"""Đánh giá test: tune logit bias, cấp session, confusion matrix, risk-coverage, social diagnostics."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .engine import print_metrics, print_per_class


# ------------------------------------------------------------------ logit bias
def apply_logit_bias(probs, bias):
    return (np.log(np.clip(probs, 1e-12, 1.0)) + np.asarray(bias)[None, :]).argmax(axis=1)


def tune_logit_bias(probs, labels, num_classes, bias_min=-0.6, bias_max=0.6, step=0.1, reference_class=1):
    """Coordinate descent tìm bias mỗi lớp để tối đa macro-F1. CHỈ dùng trên validation."""
    grid = np.arange(bias_min, bias_max + step / 2, step)
    class_ids = list(range(num_classes))

    def score(b):
        return f1_score(labels, apply_logit_bias(probs, b), labels=class_ids, average="macro", zero_division=0)

    bias = np.zeros(num_classes)
    best = score(bias)
    for _ in range(3):
        improved = False
        for c in class_ids:
            if c == reference_class:
                continue
            for v in grid:
                candidate = bias.copy()
                candidate[c] = v
                s = score(candidate)
                if s > best + 1e-12:
                    best, bias, improved = s, candidate, True
        if not improved:
            break
    return bias.astype(np.float32), float(best)


# ------------------------------------------------------------------ session-level
def session_level_evaluation(metrics, df, num_classes, label_names):
    """Mỗi session chỉ có 1 nhãn -> majority vote các segment -> macro-F1 trên session."""
    pred_df = pd.DataFrame({"sample_id": metrics["sample_ids"],
                            "pred": metrics["preds"], "true": metrics["labels"]})
    pred_df["session"] = pred_df["sample_id"].map(dict(zip(df["sample_id"], df["session"])))
    pred_df = pred_df.dropna(subset=["session"])

    rows = []
    for session, g in pred_df.groupby("session"):
        if g["true"].nunique() > 1:
            print(f"  [!] Session {session} có nhiều nhãn thật khác nhau -> dùng nhãn phổ biến nhất.")
        votes = g["pred"].value_counts()
        rows.append({"session": session, "n_segments": len(g),
                     "pred": int(votes.idxmax()), "true": int(g["true"].value_counts().idxmax()),
                     "vote_confidence": votes.max() / len(g)})
    session_df = pd.DataFrame(rows)

    class_ids = list(range(num_classes))
    return {
        "session_df": session_df,
        "acc": accuracy_score(session_df["true"], session_df["pred"]),
        "macro_f1": f1_score(session_df["true"], session_df["pred"], labels=class_ids,
                             average="macro", zero_division=0),
        "report": classification_report(session_df["true"], session_df["pred"], labels=class_ids,
                                        target_names=label_names, output_dict=True, zero_division=0),
    }


# ------------------------------------------------------------------ uncertainty
def risk_coverage_table(metrics, num_classes):
    """Giữ lại X% sample model tự tin nhất (entropy thấp nhất) thì macro-F1 là bao nhiêu."""
    probs, preds, labels = metrics["probs"], metrics["preds"], metrics["labels"]
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1) / np.log(probs.shape[1])
    order = np.argsort(entropy)
    rows = []
    for coverage in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        idx = order[:max(1, int(round(coverage * len(order))))]
        rows.append({
            "coverage": coverage,
            "n_samples": len(idx),
            "acc": accuracy_score(labels[idx], preds[idx]),
            "macro_f1": f1_score(labels[idx], preds[idx], labels=list(range(num_classes)),
                                 average="macro", zero_division=0),
            "entropy_cutoff": float(entropy[idx[-1]]),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ social
@torch.no_grad()
def analyze_social_interaction(engine, model, loader):
    """So sánh chất lượng dự đoán theo số hàng xóm + entropy attention Fb -> Fk."""
    model.eval()
    n_nb, labels, preds, entropies = [], [], [], []
    for batch in loader:
        batch = engine.to_device(batch)
        with engine.autocast():
            out = model(batch)
        counts = batch["neighbor_mask"].bool().sum(dim=1)
        n_nb.append(counts.cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        preds.append(out["logits"].argmax(dim=-1).cpu().numpy())

        attn = out["neighbor_attn_weights"].float().cpu().numpy()
        for row, count in zip(attn, counts.cpu().numpy()):
            p = row[row > 0]
            if count > 1 and len(p) > 1:
                p = p / p.sum()
                entropies.append(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))

    n_nb, labels, preds = map(np.concatenate, (n_nb, labels, preds))

    def summarize(mask):
        return {"n_samples": int(mask.sum()),
                "acc": float(accuracy_score(labels[mask], preds[mask])),
                "macro_f1": float(f1_score(labels[mask], preds[mask], average="macro", zero_division=0))}

    by_count = pd.DataFrame([{"n_neighbors": int(k), **summarize(n_nb == k)} for k in np.unique(n_nb)])
    groups = {name: summarize(m) if m.any() else None
              for name, m in (("with_neighbors", n_nb > 0), ("without_neighbors", n_nb == 0))}
    return {"by_neighbor_count": by_count, "group_compare": groups,
            "mean_attn_entropy": float(np.mean(entropies)) if entropies else None}


# ------------------------------------------------------------------ biểu đồ
def plot_confusion_matrix(cm_percent, label_names, path):
    n = len(label_names)
    fig, ax = plt.subplots(figsize=(1.6 * n + 2, 1.4 * n + 2))
    im = ax.imshow(cm_percent, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(n))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (%)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm_percent[i, j]:.1f}%", ha="center", va="center",
                    color="white" if cm_percent[i, j] > 50 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_coverage(rc_table, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rc_table["coverage"], rc_table["macro_f1"], marker="o")
    ax.invert_xaxis()
    ax.set_xlabel("Coverage (tỷ lệ sample tự tin nhất được giữ)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("F1 - Coverage")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ pipeline
def run_test_evaluation(engine, model, loaders, out_dir):
    """Đánh giá đầy đủ trên test (hoặc val nếu không có test). Lưu kết quả vào out_dir."""
    cfg, bundle = engine.cfg, engine.bundle
    os.makedirs(out_dir, exist_ok=True)
    eval_loader = loaders["test"] if loaders["test"] is not None else loaders["val"]

    metrics = engine.evaluate(model, eval_loader, desc="TEST (bias=0)")
    print_metrics(metrics)
    summary = {"segment_macro_f1_no_bias": metrics["macro_f1"], "segment_acc_no_bias": metrics["acc"]}

    # Tune bias trên validation rồi áp dụng cố định lên test
    if cfg["TUNE_LOGIT_BIAS"] and loaders["val"] is not None:
        val = engine.evaluate(model, loaders["val"], desc="val (tune bias)")
        bias, val_f1 = tune_logit_bias(val["probs"], val["labels"], bundle.num_classes,
                                       cfg["LOGIT_BIAS_MIN"], cfg["LOGIT_BIAS_MAX"], cfg["LOGIT_BIAS_STEP"])
        print(f"\nBias (val macro_f1={val_f1:.4f}): "
              + ", ".join(f"{lbl}={b:+.2f}" for lbl, b in zip(bundle.label_names, bias)))
        metrics["preds"] = apply_logit_bias(metrics["probs"], bias)
        metrics["acc"] = accuracy_score(metrics["labels"], metrics["preds"])
        metrics["macro_f1"] = f1_score(metrics["labels"], metrics["preds"], average="macro", zero_division=0)
        print(f"[TEST sau tune bias] acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f}")
        summary["logit_bias"] = dict(zip(bundle.label_names, map(float, bias)))
    summary.update(segment_macro_f1=metrics["macro_f1"], segment_acc=metrics["acc"])

    # Cấp session (chỉ số chính)
    session = session_level_evaluation(metrics, bundle.df, bundle.num_classes, bundle.label_names)
    print(f"\n=== Cấp SESSION ({len(session['session_df'])} session) ===")
    print(f"session acc={session['acc']:.4f} | session macro_f1={session['macro_f1']:.4f}")
    print_per_class(session["report"], bundle.label_names)
    session["session_df"].to_csv(os.path.join(out_dir, "session_predictions.csv"), index=False)
    summary.update(session_macro_f1=session["macro_f1"], session_acc=session["acc"])

    print("\nClassification report (cấp segment):")
    print(classification_report(metrics["labels"], metrics["preds"], labels=list(range(bundle.num_classes)),
                                target_names=bundle.label_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(metrics["labels"], metrics["preds"], labels=list(range(bundle.num_classes)))
    cm_percent = 100 * cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    print(pd.DataFrame(cm_percent, index=[f"true_{x}" for x in bundle.label_names],
                       columns=[f"pred_{x}" for x in bundle.label_names]).round(1))
    plot_confusion_matrix(cm_percent, bundle.label_names, os.path.join(out_dir, "confusion_matrix.png"))

    # Risk / coverage
    rc = risk_coverage_table(metrics, bundle.num_classes)
    print("\nRisk / coverage:")
    print(rc.round(4).to_string(index=False))
    rc.to_csv(os.path.join(out_dir, "risk_coverage.csv"), index=False)
    plot_coverage(rc, os.path.join(out_dir, "f1_coverage.png"))

    # Social diagnostics
    social = analyze_social_interaction(engine, model, eval_loader)
    print("\n=== Social diagnostics ===")
    print("Entropy attention (Fb -> Fk):", social["mean_attn_entropy"])
    for name, stats in social["group_compare"].items():
        print(f"  {name}: {stats}")
    print(social["by_neighbor_count"].round(4).to_string(index=False))
    summary["social"] = {"mean_attn_entropy": social["mean_attn_entropy"], **social["group_compare"]}

    with open(os.path.join(out_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nĐã lưu kết quả vào {out_dir}")
    return summary
