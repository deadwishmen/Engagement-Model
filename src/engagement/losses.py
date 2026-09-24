"""Các hàm loss: Focal/CE có trọng số lớp, ordinal (CORAL), auxiliary, SupCon."""
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
        C = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        target_dist = F.one_hot(targets, C).float()
        if self.label_smoothing > 0:
            target_dist = target_dist * (1 - self.label_smoothing) + self.label_smoothing / C

        pt = (log_probs.exp() * target_dist).sum(dim=-1)
        log_pt = (log_probs * target_dist).sum(dim=-1)
        loss = -((1 - pt).clamp(min=1e-6) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = self.alpha.to(logits.device)[targets] * loss
        return loss.mean()


def compute_class_weights(labels, label2id, mode, beta=0.999):
    counts = labels.value_counts()
    weights = torch.ones(len(label2id))
    for lbl, idx in label2id.items():
        c = max(counts.get(lbl, 0), 1)
        if mode == "inverse":
            weights[idx] = 1.0 / c
        elif mode == "inverse_sqrt":
            weights[idx] = 1.0 / math.sqrt(c)
        elif mode == "effective_number":
            weights[idx] = (1.0 - beta) / max(1.0 - beta ** c, 1e-8)
        else:
            raise ValueError(f"CLASS_WEIGHT_MODE không hợp lệ: {mode}")
    return weights / weights.mean()


def build_criterion(cfg, bundle):
    df = bundle.df
    weights = compute_class_weights(
        df.loc[df["split"] == "train", "engagement_label"], bundle.label2id,
        cfg["CLASS_WEIGHT_MODE"], cfg["EFFECTIVE_NUMBER_BETA"])
    print(f"Trọng số lớp ({cfg['CLASS_WEIGHT_MODE']}):",
          {lbl: round(float(weights[i]), 3) for lbl, i in bundle.label2id.items()})

    if cfg["USE_FOCAL_LOSS"]:
        s = cfg["FOCAL_ALPHA_SMOOTHING"]
        alpha = weights * (1 - s) + s          # làm mềm về phía trọng số đều
        return FocalLoss(alpha, cfg["FOCAL_GAMMA"], cfg["LABEL_SMOOTHING"])
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg["LABEL_SMOOTHING"])


# ------------------------------------------------------------------ ordinal (CORAL)
def build_ordinal_rank_by_class_id(label2id, ordinal_order):
    """rank_by_id[class_id] = thứ hạng của lớp đó trong ORDINAL_LABEL_ORDER."""
    if sorted(ordinal_order) != sorted(label2id):
        raise ValueError(f"ORDINAL_LABEL_ORDER {ordinal_order} không khớp với nhãn {list(label2id)}")
    rank_by_id = torch.empty(len(label2id), dtype=torch.long)
    for rank, name in enumerate(ordinal_order):
        rank_by_id[label2id[name]] = rank
    return rank_by_id


def ordinal_loss(ordinal_logits, labels, rank_by_id):
    """rank r -> target [1]*r + [0]*(C-1-r), BCE trên từng ngưỡng."""
    ranks = rank_by_id.to(labels.device)[labels]
    thresholds = torch.arange(ordinal_logits.size(1), device=labels.device)
    targets = (ranks.unsqueeze(1) > thresholds.unsqueeze(0)).float()
    return F.binary_cross_entropy_with_logits(ordinal_logits, targets)


def ordinal_predicted_rank(ordinal_logits):
    return (torch.sigmoid(ordinal_logits.float()) >= 0.5).sum(dim=1)


# ------------------------------------------------------------------ auxiliary
def aux_task_loss(aux_logits, batch):
    """Trung bình CE của các task, chỉ trên sample có đủ nhãn. None nếu không tính được."""
    valid = batch["aux_valid"]
    if not aux_logits or not valid.any():
        return None
    losses = [F.cross_entropy(aux_logits[task][valid], batch[f"aux_label_{task}"][valid])
              for task in aux_logits]
    return torch.stack(losses).mean()


# ------------------------------------------------------------------ SupCon
def supervised_contrastive_loss(embeddings, labels, temperature=0.10):
    """Supervised Contrastive Loss (Khosla et al. 2020). embeddings đã L2-normalize."""
    emb = embeddings.float()
    B = emb.size(0)
    if B <= 1:
        return emb.new_zeros(())

    sim = emb @ emb.t() / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    not_self = ~torch.eye(B, dtype=torch.bool, device=emb.device)
    positive = (labels.view(-1, 1) == labels.view(1, -1)) & not_self

    log_denom = torch.log((torch.exp(sim) * not_self).sum(dim=1, keepdim=True).clamp_min(1e-12))
    log_prob = sim - log_denom

    n_pos = positive.sum(dim=1)
    has_pos = n_pos > 0
    if not has_pos.any():
        return emb.new_zeros(())
    mean_log_prob_pos = (positive * log_prob).sum(dim=1) / n_pos.clamp_min(1)
    return -mean_log_prob_pos[has_pos].mean()


# ------------------------------------------------------------------ tổng hợp
LOSS_WEIGHT_KEYS = {
    "ordinal": ("USE_ORDINAL_AUX_LOSS", "ORDINAL_LOSS_WEIGHT", "ORDINAL_WARMUP_EPOCHS"),
    "aux": ("USE_BEHAVIOR_EMOTION_AUX", "AUX_LOSS_WEIGHT", "AUX_WARMUP_EPOCHS"),
    "supcon": ("USE_SUPCON_LOSS", "SUPCON_LOSS_WEIGHT", "SUPCON_WARMUP_EPOCHS"),
}


def get_loss_weights(cfg, epoch):
    """Trọng số các loss phụ, tăng tuyến tính trong giai đoạn warm-up."""
    weights = {}
    for name, (enable_key, weight_key, warmup_key) in LOSS_WEIGHT_KEYS.items():
        if not cfg[enable_key]:
            weights[name] = 0.0
            continue
        warmup = cfg[warmup_key]
        progress = 1.0 if warmup <= 0 else min(max(epoch / warmup, 0.0), 1.0)
        weights[name] = cfg[weight_key] * progress
    return weights


class LossComputer:
    """Gom loss chính + các loss phụ. Gọi: losses = loss_fn(out, batch, weights)."""

    def __init__(self, cfg, bundle, device):
        self.criterion = build_criterion(cfg, bundle).to(device)
        self.temperature = cfg["SUPCON_TEMPERATURE"]
        self.rank_by_id = None
        if cfg["USE_ORDINAL_AUX_LOSS"]:
            self.rank_by_id = build_ordinal_rank_by_class_id(
                bundle.label2id, cfg["ORDINAL_LABEL_ORDER"]).to(device)

    def __call__(self, out, batch, weights):
        labels = batch["label"]
        losses = {"cls": self.criterion(out["logits"], labels)}
        zero = losses["cls"].new_zeros(())

        losses["ordinal"] = (ordinal_loss(out["ordinal_logits"], labels, self.rank_by_id)
                             if out["ordinal_logits"] is not None else zero)
        aux = aux_task_loss(out["aux_logits"], batch)
        losses["aux"] = zero if aux is None else aux
        losses["supcon"] = (supervised_contrastive_loss(out["contrastive_embedding"], labels, self.temperature)
                            if out["contrastive_embedding"] is not None else zero)

        losses["total"] = losses["cls"] + sum(weights[k] * losses[k] for k in ("ordinal", "aux", "supcon"))
        return losses
