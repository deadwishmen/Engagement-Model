"""EngagementModelV5 - model chính."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    BuildGraphAttention,
    ContextWindowAttention,
    FiLMConditioner,
    ModalityFusionTransformer,
    make_scalar_gate,
)
from .skeleton import SkeletonBiLSTMEncoder
from .temporal import CachedTemporalEncoderV3


class EngagementModelV5(nn.Module):
    """
    Body     -> TemporalEncoder -> Fb ─┬──────────────────────────┐
    Skeleton -> (FiLM từ body) Bi-LSTM ┴─ concat -> F_target      │
    K neighbors -> TemporalEncoder -> Fk ── Graph(Fb + Fk) -> F_social (có gate)
    Context clips -> body encoder -> ContextAttention -> F_context (có gate)
    Face     -> TemporalEncoder -> Ff
    [F_social, F_context, F_target, Ff] -> Fusion Transformer -> logits
    Head phụ (chỉ dùng khi train): aux hành vi/cảm xúc, SupCon, ordinal.
    """

    def __init__(self, cfg, num_classes, aux_num_classes=None):
        super().__init__()
        D = cfg["EMBED_DIM"]
        dropout = cfg["DROPOUT"]
        feat_dim = cfg["FEATURE_DIM"]
        use_attn_gate = cfg["USE_ADAPTIVE_ATTENTION_GATE"]

        # Encoder cho từng modality
        self.body_encoder = CachedTemporalEncoderV3(feat_dim, D, dropout, use_attn_gate)
        self.face_encoder = CachedTemporalEncoderV3(feat_dim, D, dropout, use_attn_gate)
        self.neighbor_encoder = CachedTemporalEncoderV3(feat_dim, D, dropout, use_attn_gate)
        self.skeleton_encoder = SkeletonBiLSTMEncoder(input_proj_dim=D, hidden_dim=D // 2, dropout=dropout)

        self.use_film = cfg["USE_RESNET_TO_SKELETON_FUSION"]
        if self.use_film:
            self.film_conditioner = FiLMConditioner(D, D, dropout=cfg["FILM_DROPOUT"])

        self.target_concat_proj = nn.Sequential(
            nn.Linear(D + self.skeleton_encoder.out_dim, D), nn.GELU(), nn.LayerNorm(D), nn.Dropout(dropout))

        # Graph xã hội (K-hop)
        self.use_relation_features = cfg["USE_RELATION_FEATURES"]
        self.social_graph = BuildGraphAttention(
            D, cfg["GNN_LAYERS"], cfg["GNN_HEADS"], cfg["SOCIAL_DROPOUT"],
            relation_dim=cfg["RELATION_DIM"] if self.use_relation_features else None)

        # Context window (dùng chung body_encoder vì là cùng 1 người)
        self.use_context_window = cfg["USE_CONTEXT_WINDOW"]
        self.use_context_gate = self.use_context_window and cfg["USE_CONTEXT_RESIDUAL_GATE"]
        if self.use_context_window:
            self.context_window = ContextWindowAttention(
                D, cfg["CONTEXT_WINDOW_SIZE"], cfg["CONTEXT_GNN_LAYERS"],
                cfg["CONTEXT_GNN_HEADS"], cfg["CONTEXT_DROPOUT"])
        if self.use_context_gate:
            self.context_gate = make_scalar_gate(D, init_bias=-1.0)

        self.fusion = ModalityFusionTransformer(
            D, max_modalities=4, num_heads=cfg["FUSION_HEADS"], num_layers=cfg["FUSION_LAYERS"],
            dropout=dropout, num_classes=num_classes)

        self.use_social_gate = cfg["USE_SOCIAL_RESIDUAL_GATE"]
        if self.use_social_gate:
            self.social_gate = make_scalar_gate(D, init_bias=-2.0)

        # Head phụ
        self.aux_stop_gradient = cfg["AUX_STOP_GRADIENT"]
        self.aux_heads = nn.ModuleDict({
            task: nn.Sequential(nn.Linear(D, D // 2), nn.GELU(), nn.Dropout(cfg["AUX_HEAD_DROPOUT"]),
                                nn.Linear(D // 2, n_cls))
            for task, n_cls in (aux_num_classes or {}).items() if n_cls > 0
        })

        self.use_supcon = cfg["USE_SUPCON_LOSS"]
        if self.use_supcon:
            self.contrastive_head = nn.Sequential(
                nn.Linear(D, D), nn.GELU(), nn.Linear(D, cfg["SUPCON_PROJECTION_DIM"]))

        self.use_ordinal = cfg["USE_ORDINAL_AUX_LOSS"]
        if self.use_ordinal:
            self.ordinal_head = nn.Sequential(
                nn.Linear(D, D // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(D // 2, num_classes - 1))

    @staticmethod
    def _encode_slots(encoder, feat, frame_mask, slot_mask):
        """Mã hoá (B, S, T, F) bằng 1 encoder -> (B, S, D) + mask slot hợp lệ."""
        B, S, T, Fdim = feat.shape
        frame_mask = frame_mask.bool()
        emb = encoder(feat.reshape(B * S, T, Fdim), frame_mask.reshape(B * S, T)).reshape(B, S, -1)
        valid = slot_mask.bool() & frame_mask.any(dim=-1)
        return emb * valid.float().unsqueeze(-1), valid

    @staticmethod
    def _apply_gate(gate, feature, F_target):
        value = torch.sigmoid(gate(torch.cat([feature, F_target], dim=-1)))
        return feature * value, value

    def forward(self, batch):
        body_mask = batch["body_frame_mask"].bool()
        face_mask = batch["face_frame_mask"].bool()

        # Body (+ chuỗi từng frame cho FiLM)
        if self.use_film:
            Fb, body_seq = self.body_encoder(batch["body_feat"], body_mask, return_sequence=True)
            film_gamma, film_beta = self.film_conditioner(body_seq, body_mask)
        else:
            Fb = self.body_encoder(batch["body_feat"], body_mask)
            film_gamma = film_beta = None

        # Face
        face_valid = face_mask.any(dim=1)
        Ff = self.face_encoder(batch["face_feat"], face_mask) * face_valid.float().unsqueeze(-1)

        # Skeleton & target token
        Fskel = self.skeleton_encoder(
            batch["skeleton_xy"], batch["skeleton_conf"], batch["skeleton_kpt_mask"],
            batch["skeleton_frame_mask"], film_gamma, film_beta)
        F_target = self.target_concat_proj(torch.cat([Fb, Fskel], dim=-1))

        # Social
        Fk, Fk_valid = self._encode_slots(
            self.neighbor_encoder, batch["neighbor_feat"], batch["neighbor_frame_mask"], batch["neighbor_mask"])
        relation = batch["neighbor_relation"].float() if self.use_relation_features else None
        F_social, neighbor_attn, has_neighbor = self.social_graph(Fk, Fk_valid, Fb, relation)
        social_gate_value = None
        if self.use_social_gate:
            F_social, social_gate_value = self._apply_gate(self.social_gate, F_social, F_target)

        # Tokens cho fusion: social, [context], target, face
        tokens, token_mask = [F_social], [has_neighbor]

        context_gate_value = None
        if self.use_context_window:
            Fctx, ctx_valid = self._encode_slots(
                self.body_encoder, batch["context_feat"], batch["context_frame_mask"], batch["context_mask"])
            F_context, has_context = self.context_window(Fctx, ctx_valid, batch["context_offset"], Fb)
            if self.use_context_gate:
                F_context, context_gate_value = self._apply_gate(self.context_gate, F_context, F_target)
            tokens.append(F_context)
            token_mask.append(has_context)

        tokens += [F_target, Ff]
        token_mask += [torch.ones_like(has_neighbor), face_valid]
        logits, fused = self.fusion(torch.stack(tokens, dim=1), torch.stack(token_mask, dim=1))

        # Head phụ
        aux_input = fused.detach() if self.aux_stop_gradient else fused
        aux_logits = {task: head(aux_input) for task, head in self.aux_heads.items()}
        contrastive = F.normalize(self.contrastive_head(fused), dim=-1) if self.use_supcon else None
        ordinal_logits = self.ordinal_head(fused) if self.use_ordinal else None

        return {
            "logits": logits,
            "fused": fused,
            "aux_logits": aux_logits,
            "contrastive_embedding": contrastive,
            "ordinal_logits": ordinal_logits,
            "neighbor_attn_weights": neighbor_attn,
            "has_neighbor": has_neighbor,
            "social_gate_value": social_gate_value,
            "context_gate_value": context_gate_value,
        }


def build_model(cfg, bundle, device):
    """Tạo model, tự bọc DataParallel nếu có nhiều GPU."""
    model = EngagementModelV5(cfg, bundle.num_classes, bundle.aux_num_classes).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model
