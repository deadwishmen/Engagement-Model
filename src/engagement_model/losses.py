"""Classification loss (Focal / weighted CE), ordinal auxiliary loss, behavior/emotion
auxiliary loss and Supervised Contrastive loss.

Corresponds to notebook section "7. Loss V3 -- inverse-sqrt class weighting + Focal
Loss" (cell 28). The notebook-level `criterion = build_criterion(...)` call becomes
an explicit call site in `engagement_model.pipeline`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        if self.label_smoothing > 0:
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                self.label_smoothing / num_classes
        pt = (probs * targets_one_hot).sum(dim=-1)
        logpt = (log_probs * targets_one_hot).sum(dim=-1)
        focal_term = (1 - pt).clamp(min=1e-6) ** self.gamma
        loss = -focal_term * logpt
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss
        return loss.mean()


def compute_class_weights(labels_series, label2id, mode="inverse_sqrt", beta=0.999):
    counts = labels_series.value_counts()
    weights = torch.ones(len(label2id), dtype=torch.float32)
    for lbl, idx in label2id.items():
        c = max(counts.get(lbl, 0), 1)
        if mode == "inverse":
            weights[idx] = 1.0 / c
        elif mode == "inverse_sqrt":
            weights[idx] = 1.0 / math.sqrt(c)
        elif mode == "effective_number":
            effective_num = 1.0 - beta ** c
            weights[idx] = (1.0 - beta) / max(effective_num, 1e-8)
        else:
            raise ValueError(f"CLASS_WEIGHT_MODE khong hop le: {mode}")
    weights = weights / weights.mean()
    return weights


def _soften_alpha(class_weights, smoothing):
    uniform = torch.ones_like(class_weights)
    return class_weights * (1 - smoothing) + uniform * smoothing



def build_ordinal_rank_by_class_id(label2id, ordinal_label_order):
    """Tao tensor rank theo class-id.

    Vi du neu label2id theo alphabet:
      disengaged=0, engaged=1, normal=2, very_engaged=3
    thi ordinal rank dung phai la:
      id0->0, id1->2, id2->1, id3->3.
    """
    if ordinal_label_order is None:
        raise ValueError("ORDINAL_LABEL_ORDER khong duoc None.")

    ordinal_label_order = list(ordinal_label_order)
    expected = set(label2id.keys())
    supplied = set(ordinal_label_order)

    missing = expected - supplied
    extra = supplied - expected
    if missing or extra or len(ordinal_label_order) != len(label2id):
        raise ValueError(
            "ORDINAL_LABEL_ORDER phai chua dung moi engagement label mot lan. "
            f"missing={sorted(missing)}, extra={sorted(extra)}, "
            f"order={ordinal_label_order}"
        )

    rank_by_id = torch.empty(len(label2id), dtype=torch.long)
    for rank, label_name in enumerate(ordinal_label_order):
        rank_by_id[label2id[label_name]] = rank
    return rank_by_id


def build_cumulative_ordinal_targets(labels, rank_by_class_id, num_thresholds):
    """Chuyen nhan 4 muc thanh cumulative binary targets.

    rank 0 -> [0,0,0]
    rank 1 -> [1,0,0]
    rank 2 -> [1,1,0]
    rank 3 -> [1,1,1]
    """
    if num_thresholds <= 0:
        return labels.new_zeros((labels.shape[0], 0), dtype=torch.float32)

    rank_by_class_id = rank_by_class_id.to(labels.device)
    true_rank = rank_by_class_id[labels]
    thresholds = torch.arange(
        num_thresholds,
        device=labels.device,
        dtype=true_rank.dtype,
    )
    targets = true_rank.unsqueeze(1) > thresholds.unsqueeze(0)
    return targets.float()


def compute_ordinal_aux_loss(model_out, labels, rank_by_class_id):
    ordinal_logits = model_out.get("ordinal_logits", None)
    if ordinal_logits is None:
        return model_out["logits"].new_zeros(())

    targets = build_cumulative_ordinal_targets(
        labels,
        rank_by_class_id,
        num_thresholds=ordinal_logits.shape[1],
    )
    return F.binary_cross_entropy_with_logits(
        ordinal_logits,
        targets,
        reduction="mean",
    )


def get_ordinal_loss_weight(cfg, epoch):
    """Warm-up ordinal supervision de classifier chinh on dinh truoc."""
    if not cfg.get("USE_ORDINAL_AUX_LOSS", False):
        return 0.0

    target_weight = float(cfg.get("ORDINAL_LOSS_WEIGHT", 0.15))
    warmup_epochs = int(cfg.get("ORDINAL_WARMUP_EPOCHS", 0))

    if warmup_epochs <= 0:
        return target_weight

    progress = min(max(float(epoch) / warmup_epochs, 0.0), 1.0)
    return target_weight * progress


def build_criterion(cfg, df, label2id):
    class_weights = compute_class_weights(
        df[df["split"] == "train"]["engagement_label"], label2id,
        mode=cfg["CLASS_WEIGHT_MODE"], beta=cfg.get("EFFECTIVE_NUMBER_BETA", 0.999),
    )
    print(f"Trong so lop (che do '{cfg['CLASS_WEIGHT_MODE']}', tu tap train):")
    for lbl, idx in label2id.items():
        print(f"  {lbl}: {class_weights[idx]:.3f}")

    if cfg["USE_FOCAL_LOSS"]:
        alpha_smoothing = cfg.get("FOCAL_ALPHA_SMOOTHING", 0.0)
        focal_alpha = _soften_alpha(class_weights, alpha_smoothing)
        criterion = FocalLoss(alpha=focal_alpha, gamma=cfg["FOCAL_GAMMA"],
                               label_smoothing=cfg.get("LABEL_SMOOTHING", 0.0))
        print(f"Dung Focal Loss (gamma={cfg['FOCAL_GAMMA']}, alpha_smoothing={alpha_smoothing}).")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights,
                                         label_smoothing=cfg.get("LABEL_SMOOTHING", 0.0))
        print(f"Dung CrossEntropyLoss co trong so lop (label_smoothing={cfg.get('LABEL_SMOOTHING', 0.0)}).")
    return criterion




def get_aux_loss_weight(cfg, epoch):
    """Warm-up cho auxiliary behavior/emotion loss, giong cach lam voi ordinal/social_aux
    de classifier engagement chinh on dinh truoc khi cac task phu bat dau dong gop gradient
    manh."""
    if not cfg.get("USE_BEHAVIOR_EMOTION_AUX", False):
        return 0.0
    target_weight = float(cfg.get("AUX_LOSS_WEIGHT", 0.15))
    warmup_epochs = int(cfg.get("AUX_WARMUP_EPOCHS", 0))
    if warmup_epochs <= 0:
        return target_weight
    progress = min(max(float(epoch) / warmup_epochs, 0.0), 1.0)
    return target_weight * progress


def compute_aux_task_loss(model_out, batch, aux_task_names):
    """Tinh CrossEntropy RIENG cho tung auxiliary task (pose/act/obj/int/emo) roi lay
    TRUNG BINH cong cac task lai (khong trong so khac nhau giua cac task, vi moi task
    dong vai tro regularizer ngang nhau). CHI tinh loss tren sample co aux_valid=True
    (loai hoan toan sample thieu nhan khoi auxiliary loss, dung theo yeu cau).

    Tra ve: (total_aux_loss, per_task_loss_dict) -- per_task_loss_dict dung de logging/debug,
    total_aux_loss la scalar tensor dung de cong vao combined_loss.
    """
    aux_logits = model_out.get("aux_logits", None)
    if not aux_logits or not aux_task_names:
        zero = model_out["logits"].new_zeros(())
        return zero, {}

    aux_valid = batch.get("aux_valid", None)
    if aux_valid is None or not aux_valid.any():
        zero = model_out["logits"].new_zeros(())
        return zero, {task: 0.0 for task in aux_task_names}

    per_task_loss = {}
    valid_idx = torch.where(aux_valid)[0]

    total_loss = model_out["logits"].new_zeros(())
    n_tasks_with_loss = 0
    for task_name in aux_task_names:
        if task_name not in aux_logits:
            continue
        task_logits = aux_logits[task_name][valid_idx]
        task_labels = batch[f"aux_label_{task_name}"][valid_idx]
        if task_logits.shape[0] == 0:
            per_task_loss[task_name] = 0.0
            continue
        task_loss = F.cross_entropy(task_logits, task_labels)
        per_task_loss[task_name] = float(task_loss.detach().item())
        total_loss = total_loss + task_loss
        n_tasks_with_loss += 1

    if n_tasks_with_loss > 0:
        total_loss = total_loss / n_tasks_with_loss

    return total_loss, per_task_loss


def get_supcon_loss_weight(cfg, epoch):
    """Warm-up cho SupCon loss, cung tinh than voi ordinal/aux -- de classifier chinh
    on dinh mot chut truoc khi SupCon bat dau dinh hinh manh khong gian embedding."""
    if not cfg.get("USE_SUPCON_LOSS", False):
        return 0.0
    target_weight = float(cfg.get("SUPCON_LOSS_WEIGHT", 0.30))
    warmup_epochs = int(cfg.get("SUPCON_WARMUP_EPOCHS", 0))
    if warmup_epochs <= 0:
        return target_weight
    progress = min(max(float(epoch) / warmup_epochs, 0.0), 1.0)
    return target_weight * progress


def supervised_contrastive_loss(embeddings, labels, temperature=0.10):
    """Supervised Contrastive Loss (Khosla et al. 2020).

    embeddings: (B,D) DA duoc L2-normalize (xem model.forward: contrastive_embedding).
    labels: (B,) long -- nhan engagement that.

    Voi moi anchor i trong batch: positive = cac j!=i CUNG nhan, negative = phan con
    lai. Loss keo embedding cua positive lai GAN anchor, day negative RA XA, trong
    khong gian cosine similarity (chia cho temperature). Anchor nao KHONG co positive
    nao trong batch (vd nhan hiem, chi xuat hien 1 lan) duoc bo qua (dong gop 0 loss)
    thay vi gay loi chia cho 0.

    Luon tinh trong float32 (embeddings.float()) de on dinh so hoc, giong cach cac loss
    khac trong notebook nay xu ly gia tri xac suat/logit duoi AMP fp16.
    """
    embeddings = embeddings.float()
    B = embeddings.size(0)
    if B <= 1:
        return embeddings.new_zeros(())

    sim = torch.matmul(embeddings, embeddings.t()) / temperature  # (B,B) cosine sim / tau
    sim_max = sim.max(dim=1, keepdim=True).values
    sim = sim - sim_max.detach()  # on dinh so hoc, khong doi gia tri loss (bat bien theo hang)

    self_mask = torch.eye(B, dtype=torch.bool, device=embeddings.device)
    valid_pair_mask = ~self_mask  # loai tru chinh no khoi ca positive lan negative/denominator

    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.t()) & valid_pair_mask  # (B,B) True neu cung nhan, khac chinh no

    exp_sim = torch.exp(sim) * valid_pair_mask.float()
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))
    log_prob = sim - log_denom  # log-probability cua tung cap (i,j)

    pos_count = positive_mask.float().sum(dim=1)  # so positive cua tung anchor
    safe_pos_count = pos_count.clamp_min(1.0)
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / safe_pos_count

    loss_per_anchor = -mean_log_prob_pos
    has_positive = pos_count > 0
    if has_positive.sum() == 0:
        return embeddings.new_zeros(())

    return loss_per_anchor[has_positive].mean()