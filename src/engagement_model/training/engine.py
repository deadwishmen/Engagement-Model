"""Training loop internals: forward/backward step, evaluation, logit-bias tuning,
session-level aggregation, uncertainty/risk-coverage analysis, and a small speed
benchmark.

Corresponds to notebook section "9. Train / eval -- classification loss + masked
social reconstruction auxiliary loss" (cell 32).

Unlike the notebook (which read module-level globals like AUX_TASK_NAMES, df,
label2id), every function here takes its dependencies as explicit arguments.
"""
from __future__ import annotations

import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score

from ..losses import compute_aux_task_loss, compute_ordinal_aux_loss, get_aux_loss_weight, \
    get_ordinal_loss_weight, get_supcon_loss_weight, supervised_contrastive_loss


def _move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _autocast_context(cfg, device):
    if device.type == "cuda" and cfg.get("USE_AMP", True):
        return torch.autocast(device_type="cuda", dtype=cfg["_AMP_DTYPE"])
    return nullcontext()


def flatten_valid_positions(out, batch):
    """Trich xuat cac VI TRI HOP LE (khong phai padding) tu output cua TrackEngagementModel
    (che do 'nhieu-den-nhieu': logits/label co them chieu L la do dai track), lam phang
    ve (N_valid, ...) de TAI SU DUNG NGUYEN VEN cac ham loss/eval da viet cho che do
    segment-doc-lap (criterion, compute_aux_task_loss, supervised_contrastive_loss) ma
    KHONG can sua gi ben trong chung -- chi can goi voi du lieu da flatten."""
    seq_mask = out["seq_mask"]  # (B,L) bool
    valid_idx = seq_mask.reshape(-1)  # (B*L,)

    logits = out["logits"]
    logits_flat = logits.reshape(-1, logits.shape[-1])[valid_idx]
    labels_flat = batch["label"].reshape(-1)[valid_idx]

    aux_logits_flat = {}
    for task_name, task_logits in (out.get("aux_logits") or {}).items():
        aux_logits_flat[task_name] = task_logits.reshape(-1, task_logits.shape[-1])[valid_idx]

    aux_valid_flat = None
    if "aux_valid" in batch:
        aux_valid_flat = batch["aux_valid"].reshape(-1)[valid_idx]

    aux_label_flat = {}
    for key in batch.keys():
        if key.startswith("aux_label_"):
            aux_label_flat[key] = batch[key].reshape(-1)[valid_idx]

    contrastive_flat = None
    if out.get("contrastive_embedding") is not None:
        ce = out["contrastive_embedding"]
        contrastive_flat = ce.reshape(-1, ce.shape[-1])[valid_idx]

    return {
        "logits": logits_flat,
        "labels": labels_flat,
        "aux_logits": aux_logits_flat,
        "aux_valid": aux_valid_flat,
        "aux_label": aux_label_flat,  # dict {"aux_label_<task>": tensor (N_valid,)}
        "contrastive_embedding": contrastive_flat,
    }



def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    scaler,
    cfg,
    epoch,
    device,
    aux_task_names=None,
    ordinal_rank_by_id=None,
):
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_ordinal_loss = 0.0
    total_aux_loss = 0.0
    total_supcon_loss = 0.0
    n_micro = 0

    L = len(loader)
    A = cfg["GRAD_ACCUM_STEPS"]
    optimizer.zero_grad(set_to_none=True)

    use_ordinal = cfg.get("USE_ORDINAL_AUX_LOSS", False) and ordinal_rank_by_id is not None
    ordinal_weight = get_ordinal_loss_weight(cfg, epoch) if use_ordinal else 0.0
    use_aux = cfg.get("USE_BEHAVIOR_EMOTION_AUX", False)
    aux_weight = get_aux_loss_weight(cfg, epoch) if use_aux else 0.0
    aux_task_names = (aux_task_names or []) if use_aux else []
    use_supcon = cfg.get("USE_SUPCON_LOSS", False)
    supcon_weight = get_supcon_loss_weight(cfg, epoch) if use_supcon else 0.0
    supcon_temperature = cfg.get("SUPCON_TEMPERATURE", 0.10)

    use_track = cfg.get("USE_TRACK_LEVEL_MODEL", False)

    for step, batch in enumerate(loader):
        batch = _move_batch_to_device(batch, device)

        group_start = (step // A) * A
        group_size = min(A, L - group_start)
        is_last_in_group = (step + 1 - group_start) == group_size

        with _autocast_context(cfg, device):
            out = model(batch)

            if use_track and "seq_mask" in out:
                # Che do track-level: flatten ve cac VI TRI HOP LE (khong phai padding)
                # roi tai su dung nguyen ven criterion/compute_aux_task_loss/
                # supervised_contrastive_loss (khong doi ben trong cac ham do).
                flat = flatten_valid_positions(out, batch)
                labels = flat["labels"]
                classification_loss = criterion(flat["logits"], labels)

                # Ordinal aux loss CHUA ho tro che do track-level -- bo qua (ordinal
                # dang tat mac dinh trong CONFIG nen khong anh huong thi nghiem hien tai).
                ordinal_loss = classification_loss.new_zeros(())

                if use_aux:
                    flat_model_out = {"logits": flat["logits"], "aux_logits": flat["aux_logits"]}
                    flat_batch_for_aux = {"aux_valid": flat["aux_valid"], **flat["aux_label"]}
                    aux_loss, _ = compute_aux_task_loss(flat_model_out, flat_batch_for_aux, aux_task_names)
                else:
                    aux_loss = classification_loss.new_zeros(())

                if use_supcon and flat.get("contrastive_embedding") is not None:
                    supcon_loss = supervised_contrastive_loss(
                        flat["contrastive_embedding"], labels, temperature=supcon_temperature
                    )
                else:
                    supcon_loss = classification_loss.new_zeros(())
            else:
                labels = batch["label"]
                classification_loss = criterion(out["logits"], labels)

                if use_ordinal:
                    ordinal_loss = compute_ordinal_aux_loss(out, labels, ordinal_rank_by_id)
                else:
                    ordinal_loss = classification_loss.new_zeros(())

                if use_aux:
                    aux_loss, _ = compute_aux_task_loss(out, batch, aux_task_names)
                else:
                    aux_loss = classification_loss.new_zeros(())

                if use_supcon and out.get("contrastive_embedding") is not None:
                    supcon_loss = supervised_contrastive_loss(
                        out["contrastive_embedding"], labels, temperature=supcon_temperature
                    )
                else:
                    supcon_loss = classification_loss.new_zeros(())

            combined_loss = (
                classification_loss
                + ordinal_weight * ordinal_loss
                + aux_weight * aux_loss
                + supcon_weight * supcon_loss
            )
            loss_for_backward = combined_loss / group_size

        scaler.scale(loss_for_backward).backward()

        if is_last_in_group:
            if cfg["GRAD_CLIP_NORM"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["GRAD_CLIP_NORM"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += float(combined_loss.detach().item())
        total_cls_loss += float(classification_loss.detach().item())
        total_ordinal_loss += float(ordinal_loss.detach().item())
        total_aux_loss += float(aux_loss.detach().item())
        total_supcon_loss += float(supcon_loss.detach().item())
        n_micro += 1

        if (step + 1) % max(1, L // 5) == 0:
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"  [Epoch {epoch}] batch {step + 1}/{L} "
                f"loss={total_loss / n_micro:.4f} "
                f"cls={total_cls_loss / n_micro:.4f} "
                f"aux_behavior_emo={total_aux_loss / n_micro:.4f} "
                f"supcon={total_supcon_loss / n_micro:.4f} "
                f"lambda_aux={aux_weight:.3f} "
                f"lambda_supcon={supcon_weight:.3f} "
                f"lr={cur_lr:.2e}"
            )

    denom = max(n_micro, 1)
    return {
        "loss": total_loss / denom,
        "classification_loss": total_cls_loss / denom,
        "ordinal_loss": total_ordinal_loss / denom,
        "ordinal_weight": ordinal_weight,
        "aux_behavior_emo_loss": total_aux_loss / denom,
        "aux_loss_weight": aux_weight,
        "supcon_loss": total_supcon_loss / denom,
        "supcon_weight": supcon_weight,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    cfg,
    device,
    aux_task_names=None,
    desc="eval",
    label_names=None,
    ordinal_rank_by_id=None,
):
    """Danh gia classifier chinh + diagnostic cho ordinal auxiliary head.

    loss/acc/macro_f1 van dua tren classifier 4 lop.
    ordinal_mae/ordinal_rank_acc chi la diagnostic, KHONG dung de chon checkpoint.
    """
    was_training = model.training
    model.eval()

    total_loss, n_batches = 0.0, 0
    total_ordinal_loss, n_ordinal_batches = 0.0, 0

    all_preds, all_labels, all_probs = [], [], []
    all_sample_ids = []
    all_ordinal_pred_rank = []
    all_ordinal_true_rank = []

    use_aux = cfg.get("USE_BEHAVIOR_EMOTION_AUX", False)
    aux_task_names = (aux_task_names or []) if use_aux else []
    aux_correct = {task: 0 for task in aux_task_names}
    aux_total = {task: 0 for task in aux_task_names}

    use_ordinal = (
        cfg.get("USE_ORDINAL_AUX_LOSS", False)
        and ordinal_rank_by_id is not None
    )

    use_track = cfg.get("USE_TRACK_LEVEL_MODEL", False)

    for batch in loader:
        batch = _move_batch_to_device(batch, device)

        with _autocast_context(cfg, device):
            out = model(batch)
            is_track_batch = use_track and "seq_mask" in out

            if is_track_batch:
                flat = flatten_valid_positions(out, batch)
                labels = flat["labels"]
                logits_for_eval = flat["logits"]
            else:
                labels = batch["label"]
                logits_for_eval = out["logits"]

            loss = criterion(logits_for_eval, labels)

        total_loss += loss.item()
        n_batches += 1

        if is_track_batch:
            aux_valid_batch = flat.get("aux_valid", None)
            aux_logits_src = flat.get("aux_logits", {})
            if use_aux and aux_valid_batch is not None and aux_valid_batch.any():
                valid_idx = torch.where(aux_valid_batch)[0]
                for task_name in aux_task_names:
                    if task_name not in aux_logits_src:
                        continue
                    task_logits = aux_logits_src[task_name][valid_idx]
                    task_labels = flat["aux_label"][f"aux_label_{task_name}"][valid_idx]
                    task_preds = task_logits.argmax(dim=-1)
                    aux_correct[task_name] += int((task_preds == task_labels).sum().item())
                    aux_total[task_name] += int(task_labels.shape[0])
        elif use_aux and out.get("aux_logits"):
            aux_valid_batch = batch.get("aux_valid", None)
            if aux_valid_batch is not None and aux_valid_batch.any():
                valid_idx = torch.where(aux_valid_batch)[0]
                for task_name in aux_task_names:
                    if task_name not in out["aux_logits"]:
                        continue
                    task_logits = out["aux_logits"][task_name][valid_idx]
                    task_labels = batch[f"aux_label_{task_name}"][valid_idx]
                    task_preds = task_logits.argmax(dim=-1)
                    aux_correct[task_name] += int((task_preds == task_labels).sum().item())
                    aux_total[task_name] += int(task_labels.shape[0])

        probs = F.softmax(logits_for_eval.float(), dim=-1)
        preds = probs.argmax(dim=-1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

        # sample_id de doi chieu nguoc ve session/track khi can danh gia THEM o cap do
        # session (xem aggregate_session_level_evaluation() o muc 11). Chi thu thap o
        # che do segment-doc-lap (batch["sample_id"] la list[str] dung khop voi labels).
        if (not is_track_batch) and "sample_id" in batch:
            all_sample_ids.extend(batch["sample_id"])

        # Ordinal diagnostics CHUA ho tro che do track-level -- bo qua neu is_track_batch.
        if (not is_track_batch) and use_ordinal and out.get("ordinal_logits", None) is not None:
            ord_loss = compute_ordinal_aux_loss(
                out,
                labels,
                ordinal_rank_by_id,
            )
            total_ordinal_loss += float(ord_loss.item())
            n_ordinal_batches += 1

            # Vì logits được xây từ ordered cutpoints, đếm số threshold có P>0.5
            # chính là ordinal rank dự đoán (0..C-1).
            ord_pred_rank = (
                torch.sigmoid(out["ordinal_logits"].float()) >= 0.5
            ).sum(dim=1)

            rank_map = ordinal_rank_by_id.to(labels.device)
            ord_true_rank = rank_map[labels]

            all_ordinal_pred_rank.append(ord_pred_rank.cpu())
            all_ordinal_true_rank.append(ord_true_rank.cpu())

    if was_training:
        model.train()

    all_preds = (
        torch.cat(all_preds).numpy()
        if all_preds else np.zeros(0, dtype=np.int64)
    )
    all_labels = (
        torch.cat(all_labels).numpy()
        if all_labels else np.zeros(0, dtype=np.int64)
    )
    all_probs = (
        torch.cat(all_probs).numpy()
        if all_probs else np.zeros((0, 0), dtype=np.float32)
    )

    avg_loss = total_loss / max(n_batches, 1)
    acc = accuracy_score(all_labels, all_preds)

    all_class_ids = (
        list(range(all_probs.shape[1]))
        if all_probs.ndim == 2 else None
    )
    macro_f1 = f1_score(
        all_labels,
        all_preds,
        labels=all_class_ids,
        average="macro",
        zero_division=0,
    )

    per_class_report = None
    if label_names is not None:
        per_class_report = classification_report(
            all_labels,
            all_preds,
            labels=list(range(len(label_names))),
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

    ordinal_loss = None
    ordinal_mae = None
    ordinal_rank_acc = None

    if all_ordinal_pred_rank:
        ord_pred = torch.cat(all_ordinal_pred_rank).numpy()
        ord_true = torch.cat(all_ordinal_true_rank).numpy()
        ordinal_loss = total_ordinal_loss / max(n_ordinal_batches, 1)
        ordinal_mae = float(np.mean(np.abs(ord_pred - ord_true)))
        ordinal_rank_acc = float(np.mean(ord_pred == ord_true))

    aux_accuracy = None
    if use_aux and aux_task_names:
        aux_accuracy = {
            task: (aux_correct[task] / aux_total[task] if aux_total[task] > 0 else None)
            for task in aux_task_names
        }

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
        "per_class_report": per_class_report,
        "desc": desc,
        "ordinal_loss": ordinal_loss,
        "ordinal_mae": ordinal_mae,
        "ordinal_rank_acc": ordinal_rank_acc,
        # Diagnostic ONLY -- accuracy cua tung auxiliary task (pose/act/obj/int/emo)
        # tren cac sample co du nhan. KHONG duoc dung de chon checkpoint (chi macro_f1
        # cua engagement classifier chinh moi dung de chon checkpoint, xem muc 10).
        "aux_accuracy": aux_accuracy,
        # sample_id tuong ung 1-1 voi preds/labels/probs (chi co o che do segment-doc-lap)
        # -- dung de doi chieu nguoc ve session cho aggregate_session_level_evaluation().
        "sample_ids": all_sample_ids if all_sample_ids else None,
    }


def apply_logit_bias(probs, bias):
    """Ap dung class-specific bias tren log-probability."""
    probs = np.asarray(probs, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    return (log_probs + bias[None, :]).argmax(axis=1)


def tune_logit_bias(
    probs,
    labels,
    num_classes,
    bias_min=-0.6,
    bias_max=0.6,
    bias_step=0.1,
    reference_class=1,
):
    """Coordinate descent tren validation de toi uu macro-F1.

    Khong search tren test. Mot class duoc giu bias=0 lam reference de loai bo
    bat dinh cong cung mot hang so vao tat ca logits.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    class_ids = list(range(num_classes))

    grid = np.arange(
        bias_min,
        bias_max + bias_step * 0.5,
        bias_step,
        dtype=np.float64,
    )
    bias = np.zeros(num_classes, dtype=np.float64)

    def score(b):
        pred = apply_logit_bias(probs, b)
        return f1_score(
            labels,
            pred,
            labels=class_ids,
            average="macro",
            zero_division=0,
        )

    best_score = score(bias)

    # Coordinate descent 3 passes: re nhanh hon exhaustive 13^(C-1),
    # va du de tim calibration tot cho 4 classes.
    for _ in range(3):
        improved = False
        for c in class_ids:
            if c == reference_class:
                continue
            local_best_score = best_score
            local_best_value = bias[c]
            for v in grid:
                candidate = bias.copy()
                candidate[c] = float(v)
                sc = score(candidate)
                if sc > local_best_score + 1e-12:
                    local_best_score = sc
                    local_best_value = float(v)
            if local_best_score > best_score + 1e-12:
                bias[c] = local_best_value
                best_score = local_best_score
                improved = True
        if not improved:
            break

    return bias.astype(np.float32), float(best_score)


def aggregate_session_level_evaluation(eval_result, df, label2id, id2label,
                                        sample_id_col="sample_id", session_col="session"):
    """Danh gia THEM o cap do SESSION -- dung dan hon macro-F1 tinh tren tung segment
    khi nhan that su chi co 1 GIA TRI DUY NHAT cho ca session (lap lai cho moi segment
    ben trong). Tinh macro-F1 tren tung segment trong truong hop nay se de phien dien
    (session dai, nhieu segment se anh huong den diem so nhieu hon session ngan mot
    cach khong cong bang -- day chinh la nguon goc cua van de 'intra' da thao luan).

    Cach lam: voi moi session, gop TAT CA du doan segment thuoc session do lai bang
    MAJORITY VOTE (bo phieu da so) de ra 1 DU DOAN DUY NHAT cho ca session, roi tinh
    macro-F1 tren cac SESSION (khong phai tren tung segment). Nhan that (true label)
    cua ca session lay tu segment dau tien (vi tat ca segment deu chung 1 nhan).

    Yeu cau eval_result phai co 'sample_ids' (tra ve tu evaluate(), chi co o che do
    segment-doc-lap -- xem CONFIG['USE_TRACK_LEVEL_MODEL']).
    """
    sample_ids = eval_result.get("sample_ids")
    if not sample_ids:
        print("[!] Khong co sample_ids trong eval_result (co the dang o che do track-level) "
              "-- bo qua danh gia cap session.")
        return None

    preds = eval_result["preds"]
    labels = eval_result["labels"]
    n = len(sample_ids)
    assert len(preds) == n and len(labels) == n, (
        f"So luong sample_ids ({n}) khong khop voi preds/labels ({len(preds)}) -- "
        f"co the do thu tu batch bi xao tron (shuffle=True) luc eval, kiem tra lai "
        f"loader dung cho evaluate() co dang shuffle=False khong."
    )

    pred_df = pd.DataFrame({
        "sample_id": sample_ids,
        "pred": preds,
        "true_label": labels,
    })

    # Doi chieu sample_id -> session tu manifest df (df phai la bien toan cuc chua
    # manifest day du, dung sample_id lam khoa tra cuu).
    sample_to_session = df.set_index(sample_id_col)[session_col].to_dict()
    pred_df["session"] = pred_df["sample_id"].map(sample_to_session)

    n_missing_session = int(pred_df["session"].isna().sum())
    if n_missing_session > 0:
        print(f"[!] {n_missing_session}/{n} segment khong tim thay session tuong ung trong "
              f"manifest -- loai khoi danh gia cap session.")
        pred_df = pred_df.dropna(subset=["session"])

    session_rows = []
    for session_id, group in pred_df.groupby("session"):
        # Majority vote tren du doan cua tat ca segment thuoc session nay.
        vote_counts = group["pred"].value_counts()
        session_pred = int(vote_counts.idxmax())

        # Nhan that: lay tu segment dau tien (tat ca segment trong 1 session PHAI
        # chung 1 nhan theo gia dinh da xac nhan -- kiem tra luon o day de phat hien
        # neu gia dinh nay sai o mot vai session hiem).
        true_labels_in_session = group["true_label"].unique()
        if len(true_labels_in_session) > 1:
            print(f"  [!] Session '{session_id}' co NHIEU nhan that khac nhau trong cac "
                  f"segment ({true_labels_in_session}) -- gia dinh '1 session = 1 nhan' "
                  f"KHONG dung cho session nay. Dung nhan xuat hien nhieu nhat.")
            session_true = int(group["true_label"].value_counts().idxmax())
        else:
            session_true = int(true_labels_in_session[0])

        session_rows.append({
            "session": session_id, "n_segments": len(group),
            "pred": session_pred, "true_label": session_true,
            "vote_confidence": float(vote_counts.max() / len(group)),
        })

    session_df = pd.DataFrame(session_rows)

    session_acc = accuracy_score(session_df["true_label"], session_df["pred"])
    label_names_ordered_local = [id2label[i] for i in range(len(label2id))]
    session_macro_f1 = f1_score(
        session_df["true_label"], session_df["pred"],
        labels=list(range(len(label2id))), average="macro", zero_division=0,
    )
    session_report = classification_report(
        session_df["true_label"], session_df["pred"],
        labels=list(range(len(label2id))), target_names=label_names_ordered_local,
        output_dict=True, zero_division=0,
    )

    return {
        "n_sessions": len(session_df),
        "session_acc": session_acc,
        "session_macro_f1": session_macro_f1,
        "session_report": session_report,
        "session_df": session_df,
    }


def compute_uncertainty_metrics(eval_result, threshold=None, coverage_target=None,
                                 method="entropy", num_classes=None):
    """Danh gia dua tren do KHONG CHAC CHAN cua model (uncertainty-based evaluation).

    Thay vi tinh F1 chung tren toan bo test set, ham nay dung xac suat softmax ('probs'
    tra ve tu evaluate()) de:
      1. Tinh diem UNCERTAINTY cho tung sample (entropy chuan hoa hoac 1 - max_prob).
      2. Chia test set thanh 2 nhom theo nguong: 'confident' (model tu tin) va
         'uncertain' (model khong chac chan).
      3. Tinh F1/accuracy RIENG cho tung nhom -- F1 tren nhom confident phan anh dung
         hon nang luc that cua model khi no tu tin, thay vi bi 'pha loang' boi cac mau
         model doan mo ho.
      4. Tra ve bang risk-coverage (neu chi giu lai X% mau tu tin nhat thi F1 la bao nhieu)
         de ban chon nguong phu hop voi bai toan cua minh.

    Tham so:
      eval_result: dict tra ve tu evaluate() (phai co 'probs', 'preds', 'labels').
      method: 'entropy' (mac dinh, entropy chuan hoa ve [0,1]) hoac 'max_prob'
              (uncertainty = 1 - max softmax probability -- don gian, de hieu hon).
      threshold: nguong uncertainty CO DINH (0..1) de tach confident/uncertain. Neu None
                 va coverage_target duoc cho, se tu dong chon nguong sao cho giu lai
                 dung coverage_target % mau tu tin nhat. Neu ca 2 deu None, dung median
                 lam nguong mac dinh (50% coverage) chi de tham khao.
      coverage_target: ti le mau muon giu lai o nhom 'confident' (vd 0.8 = giu 80% mau
                 tu tin nhat, 20% con lai la 'uncertain'). Bo qua neu threshold da duoc cho.

    Tra ve dict:
      'uncertainty': mang (N,) diem uncertainty tung sample.
      'threshold_used': nguong thuc te da dung.
      'confident_mask' / 'uncertain_mask': mang bool (N,).
      'confident_metrics' / 'uncertain_metrics': dict {n_samples, acc, macro_f1}.
      'risk_coverage_table': DataFrame the hien F1 thay doi the nao theo coverage
                 (100%, 90%, ..., 10%) -- dung de chon nguong hoac ve bieu do.
    """
    probs = eval_result["probs"]
    preds = eval_result["preds"]
    labels = eval_result["labels"]
    n = len(labels)
    if n == 0 or probs.size == 0:
        return {"uncertainty": np.zeros(0), "threshold_used": None,
                "confident_mask": np.zeros(0, dtype=bool), "uncertain_mask": np.zeros(0, dtype=bool),
                "confident_metrics": None, "uncertain_metrics": None,
                "risk_coverage_table": pd.DataFrame()}

    C = probs.shape[1] if num_classes is None else num_classes

    if method == "entropy":
        eps = 1e-12
        raw_entropy = -(probs * np.log(probs + eps)).sum(axis=1)
        max_entropy = np.log(C) if C > 1 else 1.0
        uncertainty = raw_entropy / max_entropy  # chuan hoa ve [0, 1]
    elif method == "max_prob":
        uncertainty = 1.0 - probs.max(axis=1)
    else:
        raise ValueError(f"method khong hop le: {method}")

    # ---- Xac dinh nguong ----
    if threshold is not None:
        threshold_used = threshold
    elif coverage_target is not None:
        # Nguong sao cho dung coverage_target% mau co uncertainty THAP NHAT duoc giu lai.
        threshold_used = float(np.quantile(uncertainty, coverage_target))
    else:
        threshold_used = float(np.median(uncertainty))

    confident_mask = uncertainty <= threshold_used
    uncertain_mask = ~confident_mask

    def _sub_metrics(mask):
        n_sub = int(mask.sum())
        if n_sub == 0:
            return {"n_samples": 0, "acc": None, "macro_f1": None}
        sub_labels = labels[mask]
        sub_preds = preds[mask]
        return {
            "n_samples": n_sub,
            "acc": accuracy_score(sub_labels, sub_preds),
            "macro_f1": f1_score(
                sub_labels, sub_preds, labels=list(range(C)),
                average="macro", zero_division=0,
            ),
        }

    confident_metrics = _sub_metrics(confident_mask)
    uncertain_metrics = _sub_metrics(uncertain_mask)

    # ---- Bang risk-coverage: F1 thay doi the nao neu chi giu X% mau tu tin nhat ----
    coverage_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    rows = []
    order = np.argsort(uncertainty)  # tu chac chan nhat -> khong chac chan nhat
    for cov in coverage_levels:
        k = max(1, int(round(cov * n)))
        idx = order[:k]
        rows.append({
            "coverage": cov,
            "n_samples": k,
            "acc": accuracy_score(labels[idx], preds[idx]),
            "macro_f1": f1_score(
                labels[idx], preds[idx], labels=list(range(C)),
                average="macro", zero_division=0,
            ),
            "uncertainty_cutoff": float(uncertainty[order[k - 1]]),
        })
    risk_coverage_table = pd.DataFrame(rows)

    return {
        "uncertainty": uncertainty,
        "threshold_used": threshold_used,
        "confident_mask": confident_mask,
        "uncertain_mask": uncertain_mask,
        "confident_metrics": confident_metrics,
        "uncertain_metrics": uncertain_metrics,
        "risk_coverage_table": risk_coverage_table,
    }


@torch.no_grad()
def benchmark_pipeline(model, loader, cfg, device, n_batches=5):
    if loader is None or len(loader) == 0:
        return
    was_training = model.training
    model.eval()
    it = iter(loader)
    rows = []
    total_to_run = min(len(loader), max(2, n_batches + 1))

    for bi in range(total_to_run):
        t0 = time.perf_counter()
        batch = next(it)
        t1 = time.perf_counter()

        batch = _move_batch_to_device(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        with _autocast_context(cfg, device):
            _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        if bi > 0:
            # Trong che do track-level, batch["label"].shape[0] la SO TRACK, khong phai
            # so segment thuc -- dung seq_mask (neu co) de dem dung so segment THUC su
            # duoc xu ly (khong tinh padding).
            if "seq_mask" in batch:
                bs = int(batch["seq_mask"].sum().item())
            else:
                bs = int(batch["label"].shape[0])
            rows.append((bs, t1 - t0, t2 - t1, t3 - t2))

    if rows:
        n = len(rows)
        samples = sum(r[0] for r in rows)
        data_s = sum(r[1] for r in rows) / n
        h2d_s = sum(r[2] for r in rows) / n
        fwd_s = sum(r[3] for r in rows) / n
        wall = sum(r[1] + r[2] + r[3] for r in rows)
        print(f"\n[Benchmark pipeline]")
        print(f"  batches measured : {n}")
        print(f"  avg DataLoader wait: {data_s*1000:.1f} ms/batch")
        print(f"  avg H2D transfer : {h2d_s*1000:.1f} ms/batch")
        print(f"  avg forward      : {fwd_s*1000:.1f} ms/batch")
        print(f"  samples/s: {samples/max(wall,1e-9):.1f}")
        if data_s > fwd_s:
            print("  -> DataLoader/I/O dang cham hon forward: tang NUM_WORKERS/PREFETCH_FACTOR.")
        else:
            print("  -> Forward dang chiem nhieu thoi gian hon I/O: model la nut that chinh (binh thuong "
                  "voi model nho nay -- co the tang BATCH_SIZE de tan dung GPU tot hon).")

    if was_training:
        model.train()

