"""Social-interaction diagnostics: does the model actually use surrounding people?

Corresponds to notebook section "10b. Social-interaction diagnostics: gate,
attention, neighbor count" (cell 36).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from ..training.engine import _autocast_context, _move_batch_to_device


def analyze_social_interaction(model, loader, device, cfg):
    """Phan tich xem model co thuc su su dung nguoi xung quanh (social context) hay
    khong. Kien truc moi khong con 'social gate' (sigmoid gate cua V4 cu) -- thay vao
    do, phan tich dua tren trong so attention (neighbor_attn_weights) ma Fb dung de
    'hoi' cac node Fk trong BuildGraphAttention.target_pool."""
    was_training = model.training
    model.eval()

    n_neighbors_list = []
    attn_rows = []
    labels_list = []
    preds_list = []

    use_track = cfg.get("USE_TRACK_LEVEL_MODEL", False)

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            with _autocast_context(cfg, device):
                out = model(batch)

            is_track_batch = use_track and "seq_mask" in out
            if is_track_batch:
                seq_mask = out["seq_mask"]
                valid_idx = seq_mask.reshape(-1)

                labels = batch["label"].reshape(-1)[valid_idx]
                logits_flat = out["logits"].reshape(-1, out["logits"].shape[-1])[valid_idx]
                preds = logits_flat.argmax(dim=-1)

                neighbor_mask_flat = batch["neighbor_mask"].reshape(-1, batch["neighbor_mask"].shape[-1])[valid_idx]
                valid = neighbor_mask_flat.bool()

                attn = out.get("neighbor_attn_weights")
                if attn is not None:
                    attn = attn.reshape(-1, attn.shape[-1])[valid_idx]
            else:
                labels = batch["label"]
                preds = out["logits"].argmax(dim=-1)
                valid = batch["neighbor_mask"].bool()
                attn = out.get("neighbor_attn_weights")

            n_neighbors_list.append(valid.sum(dim=1).cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            preds_list.append(preds.cpu().numpy())

            if attn is not None:
                attn_rows.append(attn.float().cpu().numpy())

    if was_training:
        model.train()

    n_neighbors = np.concatenate(n_neighbors_list) if n_neighbors_list else np.zeros(0)
    labels_arr = np.concatenate(labels_list) if labels_list else np.zeros(0, dtype=np.int64)
    preds_arr = np.concatenate(preds_list) if preds_list else np.zeros(0, dtype=np.int64)
    attn_all = np.concatenate(attn_rows, axis=0) if attn_rows else None

    rows = []
    for k in sorted(np.unique(n_neighbors).astype(int).tolist()):
        m = n_neighbors == k
        rows.append({
            "n_neighbors": int(k),
            "n_samples": int(m.sum()),
            "acc": accuracy_score(labels_arr[m], preds_arr[m]),
            "macro_f1": f1_score(labels_arr[m], preds_arr[m], average="macro", zero_division=0),
        })
    by_neighbor_count = pd.DataFrame(rows)

    group_compare = {}
    for name, m in [("with_neighbors", n_neighbors > 0), ("without_neighbors", n_neighbors == 0)]:
        if m.sum() > 0:
            group_compare[name] = {
                "n_samples": int(m.sum()),
                "acc": accuracy_score(labels_arr[m], preds_arr[m]),
                "macro_f1": f1_score(labels_arr[m], preds_arr[m], average="macro", zero_division=0),
            }
        else:
            group_compare[name] = None

    # Entropy chuan hoa cua attention weight (Fb -> cac Fk): entropy thap = model chi
    # tap trung vao 1-2 hang xom cu the; entropy cao = model chia deu su chu y.
    mean_attn_entropy = None
    if attn_all is not None and len(attn_all) > 0:
        entropies = []
        for i, row in enumerate(attn_all):
            valid_n = int(n_neighbors[i])
            if valid_n <= 1:
                continue
            p = row[row > 0]
            if len(p) <= 1:
                continue
            p = p / np.clip(p.sum(), 1e-12, None)
            ent = -(p * np.log(p + 1e-12)).sum() / np.log(len(p))
            entropies.append(ent)
        if entropies:
            mean_attn_entropy = float(np.mean(entropies))

    return {
        "by_neighbor_count": by_neighbor_count,
        "group_compare": group_compare,
        "mean_attn_entropy_normalized": mean_attn_entropy,
        "n_total_samples": int(len(labels_arr)),
    }
