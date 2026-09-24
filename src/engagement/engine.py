"""Vòng lặp train / evaluate / benchmark."""
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .losses import LossComputer, get_loss_weights, ordinal_predicted_rank
from .models import unwrap_model


class Engine:
    """Giữ cấu hình, thiết bị, AMP và hàm loss dùng chung cho train/eval."""

    def __init__(self, cfg, bundle, device):
        self.cfg = cfg
        self.bundle = bundle
        self.device = device
        self.loss_fn = LossComputer(cfg, bundle, device)

        self.use_amp = device.type == "cuda" and cfg["USE_AMP"]
        want_bf16 = str(cfg["AMP_DTYPE"]).lower() in ("bf16", "bfloat16")
        self.amp_dtype = (torch.bfloat16 if self.use_amp and want_bf16 and torch.cuda.is_bf16_supported()
                          else torch.float16)     # T4 không hỗ trợ bf16

    # ------------------------------------------------------------ tiện ích
    def to_device(self, batch):
        return {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()}

    def autocast(self):
        return torch.autocast("cuda", dtype=self.amp_dtype) if self.use_amp else nullcontext()

    def make_scaler(self):
        return torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

    def make_optimizer_and_scheduler(self, model, steps_per_epoch):
        cfg = self.cfg
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
        warmup_steps = steps_per_epoch * cfg["WARMUP_EPOCHS"]
        total_steps = steps_per_epoch * cfg["EPOCHS"]

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------ train
    def train_one_epoch(self, model, loader, optimizer, scheduler, scaler, epoch):
        model.train()
        weights = get_loss_weights(self.cfg, epoch)
        accum = self.cfg["GRAD_ACCUM_STEPS"]
        n_batches = len(loader)
        running = {k: 0.0 for k in ("total", "cls", "ordinal", "aux", "supcon")}

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            batch = self.to_device(batch)
            group_start = (step // accum) * accum
            group_size = min(accum, n_batches - group_start)

            with self.autocast():
                out = model(batch)
                losses = self.loss_fn(out, batch, weights)
            scaler.scale(losses["total"] / group_size).backward()

            if step + 1 == group_start + group_size:          # hết 1 nhóm grad-accum
                if self.cfg["GRAD_CLIP_NORM"]:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg["GRAD_CLIP_NORM"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            for k in running:
                running[k] += float(losses[k].detach())

            if (step + 1) % max(1, n_batches // 5) == 0:
                avg = {k: v / (step + 1) for k, v in running.items()}
                print(f"  [Epoch {epoch}] {step + 1}/{n_batches} loss={avg['total']:.4f} cls={avg['cls']:.4f} "
                      f"aux={avg['aux']:.4f} supcon={avg['supcon']:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # ------------------------------------------------------------ evaluate
    @torch.no_grad()
    def evaluate(self, model, loader, desc="eval"):
        model.eval()
        bundle = self.bundle
        rank_by_id = self.loss_fn.rank_by_id
        total_loss = 0.0
        preds, labels, probs, sample_ids = [], [], [], []
        ord_pred, ord_true = [], []
        aux_correct = {t: 0 for t in bundle.aux_task_names}
        aux_total = {t: 0 for t in bundle.aux_task_names}

        for batch in loader:
            batch = self.to_device(batch)
            with self.autocast():
                out = model(batch)
                total_loss += self.loss_fn.criterion(out["logits"], batch["label"]).item()

            p = F.softmax(out["logits"].float(), dim=-1)
            probs.append(p.cpu())
            preds.append(p.argmax(dim=-1).cpu())
            labels.append(batch["label"].cpu())
            sample_ids.extend(batch["sample_id"])

            valid = batch["aux_valid"]
            for task, task_logits in out["aux_logits"].items():
                target = batch[f"aux_label_{task}"][valid]
                aux_correct[task] += int((task_logits[valid].argmax(dim=-1) == target).sum())
                aux_total[task] += int(target.numel())

            if out["ordinal_logits"] is not None:
                ord_pred.append(ordinal_predicted_rank(out["ordinal_logits"]).cpu())
                ord_true.append(rank_by_id[batch["label"]].cpu())

        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()
        probs = torch.cat(probs).numpy()
        class_ids = list(range(bundle.num_classes))

        result = {
            "desc": desc,
            "loss": total_loss / max(len(loader), 1),
            "acc": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, labels=class_ids, average="macro", zero_division=0),
            "preds": preds,
            "labels": labels,
            "probs": probs,
            "sample_ids": sample_ids,
            "per_class_report": classification_report(
                labels, preds, labels=class_ids, target_names=bundle.label_names,
                output_dict=True, zero_division=0),
            "aux_accuracy": {t: aux_correct[t] / aux_total[t] if aux_total[t] else None
                             for t in bundle.aux_task_names},
            "ordinal_mae": None,
            "ordinal_rank_acc": None,
        }
        if ord_pred:
            op, ot = torch.cat(ord_pred).numpy(), torch.cat(ord_true).numpy()
            result["ordinal_mae"] = float(np.abs(op - ot).mean())
            result["ordinal_rank_acc"] = float((op == ot).mean())
        return result

    # ------------------------------------------------------------ benchmark
    @torch.no_grad()
    def benchmark(self, model, loader, n_batches=5):
        """Đo thời gian đọc dữ liệu vs forward để biết đâu là nút thắt."""
        model.eval()
        it = iter(loader)
        timings = []
        for i in range(min(len(loader), n_batches + 1)):
            t0 = time.perf_counter()
            batch = self.to_device(next(it))
            t1 = time.perf_counter()
            with self.autocast():
                model(batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            if i > 0:                                  # bỏ batch đầu (warm-up)
                timings.append((len(batch["label"]), t1 - t0, t2 - t1))

        if timings:
            n_samples = sum(t[0] for t in timings)
            data_ms = 1000 * np.mean([t[1] for t in timings])
            fwd_ms = 1000 * np.mean([t[2] for t in timings])
            wall = sum(t[1] + t[2] for t in timings)
            print(f"[Benchmark] data: {data_ms:.1f} ms/batch | forward: {fwd_ms:.1f} ms/batch "
                  f"| {n_samples / wall:.1f} sample/s")
            print("  -> I/O chậm hơn: tăng NUM_WORKERS/PREFETCH_FACTOR." if data_ms > fwd_ms
                  else "  -> Model là nút thắt chính.")

    # ------------------------------------------------------------ fit
    def fit(self, model, train_loader, val_loader, ckpt_path):
        """Train với early stopping theo val macro-F1, lưu checkpoint tốt nhất. Trả về history DataFrame."""
        cfg = self.cfg
        n_epochs = cfg["DRY_RUN_EPOCHS"] if cfg["DRY_RUN"] else cfg["EPOCHS"]
        steps_per_epoch = math.ceil(len(train_loader) / cfg["GRAD_ACCUM_STEPS"])
        optimizer, scheduler = self.make_optimizer_and_scheduler(model, steps_per_epoch)
        scaler = self.make_scaler()

        best_f1, no_improve, history = -1.0, 0, []
        for epoch in range(1, n_epochs + 1):
            print(f"\n===== Epoch {epoch}/{n_epochs} =====")
            train_stats = self.train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)
            val = self.evaluate(model, val_loader, desc=f"val epoch {epoch}")
            print(f"[train] loss={train_stats['total']:.4f} cls={train_stats['cls']:.4f} "
                  f"aux={train_stats['aux']:.4f} supcon={train_stats['supcon']:.4f}")
            print_metrics(val)

            history.append({
                "epoch": epoch,
                **{f"train_{k}_loss": v for k, v in train_stats.items()},
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "val_macro_f1": val["macro_f1"],
                "val_ordinal_mae": val["ordinal_mae"],
            })

            if val["macro_f1"] > best_f1:
                best_f1, no_improve = val["macro_f1"], 0
                save_checkpoint(ckpt_path, model, self, epoch, best_f1, history,
                                optimizer=optimizer, scheduler=scheduler, scaler=scaler)
                print(f"  -> Lưu checkpoint tốt nhất (val macro_f1={best_f1:.4f}) vào {ckpt_path}")
                print_per_class(val["per_class_report"], self.bundle.label_names)
            else:
                no_improve += 1
                print(f"  (không cải thiện {no_improve}/{cfg['EARLY_STOPPING_PATIENCE']})")
                if no_improve >= cfg["EARLY_STOPPING_PATIENCE"]:
                    print(f"\nDừng sớm sau epoch {epoch}.")
                    break

        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(os.path.dirname(ckpt_path), "history.csv"), index=False)
        return history_df


# ------------------------------------------------------------ checkpoint & in ấn
def save_checkpoint(path, model, engine, epoch, val_macro_f1, history,
                    optimizer=None, scheduler=None, scaler=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "config": engine.cfg,
        "label2id": engine.bundle.label2id,
        "id2label": engine.bundle.id2label,
        "aux_label2id": engine.bundle.aux_label2id,
        "epoch": epoch,
        "val_macro_f1": val_macro_f1,
        "history": history,
    }, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def print_metrics(m):
    line = f"[{m['desc']}] loss={m['loss']:.4f} acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f}"
    if m["ordinal_mae"] is not None:
        line += f" | ord_mae={m['ordinal_mae']:.3f} ord_acc={m['ordinal_rank_acc']:.3f}"
    print(line)
    if m["aux_accuracy"]:
        print("  aux acc: " + " | ".join(
            f"{t}={a:.3f}" if a is not None else f"{t}=n/a" for t, a in m["aux_accuracy"].items()))


def print_per_class(report, label_names):
    for lbl in label_names:
        r = report.get(lbl, {})
        print(f"    {lbl:>20s}: P={r.get('precision', 0):.3f} R={r.get('recall', 0):.3f} "
              f"F1={r.get('f1-score', 0):.3f} n={r.get('support', 0):.0f}")
