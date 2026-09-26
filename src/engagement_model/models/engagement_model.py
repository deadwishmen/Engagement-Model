r"""EngagementModelV5: K-hop Graph + Backbone/Time-Attention + Bi-LSTM Skeleton +
Multimodal Self-Attention Fusion, plus the Track-level wrapper.

Corresponds to notebook section "6. V5 -- Kien truc theo so do" (cell 26).

    K-hop neighbors --Backbone--> Time Attention --> Fk (nhieu node)  ----\
                                                                            |
    Body (16 frame) --Backbone--> Time Attention --> Fb -----------  Build Graph
                                                          |     \    (Fk<->Fk<->Fb)
                                                          |      \        |
                                                          |       \      GNN
                                                          |        \      |
                                                          |         \-----+---> Fb_refined = F_social
    Skeleton (16 frame) --Bi-LSTM+Attn----------\         |
                                                  Concat(Fb, skeleton) = Ftarget
                                                        |
    Face (16 frame) --Backbone--> Time Attention --> Ff |
                                                        | |
                                          F_social, Ftarget, Ff --> Multimodal Self-Attention Fusion
                                                                              |
                                                                             MLP -> logits

Additions beyond the original diagram:
  - FiLM: per-frame ResNet features (from Body) modulate the per-frame Skeleton
    vector before the Bi-LSTM -- injects visual context into the skeleton branch.
  - SupCon: a projection head on 'fused' (post-Fusion) for Supervised Contrastive
    Loss, to separate embeddings of visually-close classes.
  - Context Window: TEMPORAL context (clips before/after, same person), parallel to
    K-hop (SPATIAL context, different people at the same instant).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .skeleton_encoder import SkeletonBiLSTMEncoder
from .temporal_encoder import CachedTemporalEncoderV3


class GraphAttentionLayer(nn.Module):
    """1 lop Graph-Attention tong quat: MultiheadAttention full-connect giua cac node
    hop le (key_padding_mask dong vai tro adjacency "full-connect + mask"), tiep theo
    la feed-forward + residual + LayerNorm -- day chinh la "GNN" trong so do, ap dung
    cho cac node Fk trong khung 'Build Graph'."""

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, node_mask=None):
        key_padding_mask = (~node_mask) if node_mask is not None else None
        attn_out, attn_w = self.attn(nodes, nodes, nodes, key_padding_mask=key_padding_mask)
        x = self.norm1(nodes + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_w


class BuildGraphAttention(nn.Module):
    """'Build Graph' + GNN DUNG THEO SO DO: Fb (target) la 1 NODE THUC SU trong graph,
    cung voi cac Fk (hang xom) -- KHONG con dung ngoai "hoi" graph nhu ban truoc. Tat
    ca cac node (Fb + Fk) duoc noi VOI NHAU (full-connect trong pham vi hop le, dung
    mask cho hang xom khong ton tai) va CUNG duoc tinh chinh qua Graph Attention (hoc
    trong so canh, khac GCN co dinh). Sau khi GNN xu ly xong, lay LAI chinh node Fb
    (gio da duoc tinh chinh boi ngu canh xa hoi xung quanh no) lam F_social -- dung y
    het mui ten 2 chieu Fk<->Fk<->Fb trong khung 'Build Graph' cua so do.
    """

    def __init__(self, dim=192, num_gnn_layers=2, num_heads=4, dropout=0.20):
        super().__init__()
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(dim, num_heads, dropout) for _ in range(num_gnn_layers)
        ])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, Fk_nodes, Fk_mask, Fb):
        """Fk_nodes: (B,K,D), Fk_mask: (B,K) bool -- True = hang xom hop le.
        Fb: (B,D) -- target embedding, gio la 1 NODE THUC SU trong graph (node dau tien)."""
        B, K, D = Fk_nodes.shape
        device = Fk_nodes.device

        main_node = Fb.unsqueeze(1)  # (B,1,D) -- Fb la node DAU TIEN trong graph
        nodes = torch.cat([main_node, Fk_nodes], dim=1)  # (B,1+K,D)
        node_mask = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=device), Fk_mask.bool()], dim=1
        )  # Fb LUON hop le (node dau tien luon True); Fk theo Fk_mask

        attn_weights = None
        for layer in self.gnn_layers:
            nodes, attn_weights = layer(nodes, node_mask=node_mask)

        Fb_refined = nodes[:, 0, :]  # node Fb SAU KHI da duoc GNN tinh chinh boi Fk
        social_context = self.out_norm(Fb_refined)

        has_neighbor = Fk_mask.any(dim=1)
        # Khong co hang xom nao -> giu nguyen quy uoc cu: F_social = 0 (mask het), du
        # ve ly thuyet Fb_refined van con residual/FF cua chinh no khi graph chi co 1
        # node -- zero-out de nhat quan voi thiet ke Fusion (modality_mask=False khi
        # khong co social context that su, giong Ff khi vang face).
        social_context = torch.where(
            has_neighbor.unsqueeze(-1), social_context, torch.zeros_like(social_context)
        )

        # attn_weights tu lop GNN CUOI CUNG la ma tran full-attention (B, 1+K, 1+K).
        # Lay hang 0 (Fb la query) cot 1: (cac Fk) -- Fb dang chu y bao nhieu toi tung
        # hang xom -- de tuong thich voi social diagnostics da co (do entropy/uu tien).
        if attn_weights is not None:
            neighbor_attn = attn_weights[:, 0, 1:]  # (B, K)
        else:
            neighbor_attn = torch.zeros(B, K, device=device)

        return social_context, neighbor_attn, has_neighbor


class RelationAwareBuildGraphAttention(nn.Module):
    """Bien the CO relation features (dx,dy,distance,velocity,...) tren tung canh --
    giu lai nhu 1 TUY CHON bo sung (CONFIG['USE_RELATION_FEATURES']=True), khong phai
    kien truc mac dinh theo so do (so do chi dung Fk tho). Khi bat, moi canh
    target->neighbor duoc "boi tram" bang relation embedding truoc khi vao GNN. Cung
    dua Fb la 1 NODE THUC SU trong graph, giong BuildGraphAttention o tren."""

    def __init__(self, dim=192, relation_dim=13, num_gnn_layers=2, num_heads=4, dropout=0.20):
        super().__init__()
        self.relation_encoder = nn.Sequential(
            nn.Linear(relation_dim, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, dim), nn.GELU(), nn.LayerNorm(dim),
        )
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(dim, num_heads, dropout) for _ in range(num_gnn_layers)
        ])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, Fk_nodes, Fk_mask, Fb, relation_feat):
        relation_feat = torch.nan_to_num(relation_feat, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        relation_emb = self.relation_encoder(relation_feat)
        Fk_nodes_boosted = Fk_nodes + relation_emb  # boi tram dac trung hang xom bang quan he hinh hoc

        B, K, D = Fk_nodes.shape
        device = Fk_nodes.device

        main_node = Fb.unsqueeze(1)
        nodes = torch.cat([main_node, Fk_nodes_boosted], dim=1)
        node_mask = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=device), Fk_mask.bool()], dim=1
        )

        attn_weights = None
        for layer in self.gnn_layers:
            nodes, attn_weights = layer(nodes, node_mask=node_mask)

        Fb_refined = nodes[:, 0, :]
        social_context = self.out_norm(Fb_refined)

        has_neighbor = Fk_mask.any(dim=1)
        social_context = torch.where(
            has_neighbor.unsqueeze(-1), social_context, torch.zeros_like(social_context)
        )

        if attn_weights is not None:
            neighbor_attn = attn_weights[:, 0, 1:]
        else:
            neighbor_attn = torch.zeros(B, K, device=device)

        return social_context, neighbor_attn, has_neighbor


class FiLMConditioner(nn.Module):
    """Sinh (gamma, beta) tu chuoi dac trung ResNet TUNG FRAME (khong phai vector da
    pool ca clip) de dieu bien (FiLM: Feature-wise Linear Modulation, Perez et al. 2018)
    truc tiep vector skeleton TUNG FRAME truoc khi vao Bi-LSTM: frame_vec_moi = frame_vec
    * (1 + gamma) + beta. Y tuong: ResNet 'noi cho' skeleton biet boi canh hinh anh (mau
    sac, do sang, vat the/tay cam xung quanh...) de giup phan biet cac tu the MO HO ve
    mat hinh hoc thuan tuy (vd tay dat gan mat: che mieng ngap (buon chan) hay dang cam
    do an/dien thoai (dang tham gia)? -- toa do khop khong the biet, nhung anh ResNet co
    the co dau hieu)."""

    def __init__(self, appearance_dim, skeleton_dim, hidden_dim=None, dropout=0.10):
        super().__init__()
        hidden_dim = hidden_dim or skeleton_dim
        self.net = nn.Sequential(
            nn.Linear(appearance_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, skeleton_dim * 2),
        )
        # Khoi tao lop cuoi ve 0 de FiLM BAT DAU tu identity transform (gamma=0, beta=0)
        # -- luc dau train, skeleton hoat dong y het nhu KHONG co FiLM, tranh pha vo
        # nhung gi Bi-LSTM da hoc duoc tu du lieu hinh hoc thuan tuy. FiLM chi dan dan
        # "them" thong tin ngu canh hinh anh khi no THUC SU giup ich (qua gradient).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, appearance_seq, appearance_frame_mask):
        """appearance_seq: (B,T,D_app) dac trung ResNet TUNG FRAME (truoc khi pool).
        appearance_frame_mask: (B,T) bool -- frame nao cua NHANH BODY hop le.
        Tra ve gamma, beta: (B,T,D_skel), da duoc ep ve 0 (= identity, khong dieu bien
        gi) tai cac frame ma chinh du lieu body/ResNet khong hop le cho frame do."""
        params = self.net(appearance_seq)  # (B,T,2*D_skel)
        gamma, beta = params.chunk(2, dim=-1)
        mask_f = appearance_frame_mask.float().unsqueeze(-1)
        gamma = gamma * mask_f
        beta = beta * mask_f
        return gamma, beta


class ContextWindowAttention(nn.Module):
    """Tuong tu BuildGraphAttention (K-hop KHONG GIAN: khac nguoi, cung thoi diem)
    nhung doc theo TRUC THOI GIAN: cac clip lien ke (truoc/sau) CUNG 1 NGUOI trong
    cung session duoc dua qua Graph Attention (tu chu y lan nhau, co gan positional
    embedding theo vi tri tuong doi -2,-1,+1,+2...), roi Fb (target) 'hoi' ngu canh da
    tinh chinh de lay ra F_context -- dai dien 'xu huong tham gia gan day' cua chinh
    nguoi do, thay vi chi xet duy nhat 1 khoanh khac 16-frame hien tai."""

    def __init__(self, dim, max_window=2, num_layers=2, num_heads=4, dropout=0.20):
        super().__init__()
        num_slots = max_window * 2 + 1  # +1 du phong cho offset=0 (khong bao gio dung toi)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_slots, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.target_pool = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, Fctx_nodes, ctx_mask, ctx_offset_idx, Fb):
        """Fctx_nodes: (B,W,D), ctx_mask: (B,W) bool, ctx_offset_idx: (B,W) long (chi
        so vao pos_embed, da +max_window de khong am, xem Dataset), Fb: (B,D)."""
        B, W, D = Fctx_nodes.shape

        pos = torch.gather(
            self.pos_embed.expand(B, -1, -1), dim=1,
            index=ctx_offset_idx.unsqueeze(-1).expand(-1, -1, D),
        )
        nodes = Fctx_nodes + pos * ctx_mask.float().unsqueeze(-1)

        safe_mask = ctx_mask.clone()
        all_invalid = ~safe_mask.any(dim=1)
        if all_invalid.any():
            safe_mask[all_invalid, 0] = True

        x = nodes
        for layer in self.gnn_layers:
            x, _ = layer(x, node_mask=safe_mask)

        query = Fb.unsqueeze(1)
        key_padding_mask = ~safe_mask
        context_out, attn_w = self.target_pool(query, x, x, key_padding_mask=key_padding_mask)
        context_out = context_out.squeeze(1)

        has_context = ctx_mask.any(dim=1)
        context_out = torch.where(
            has_context.unsqueeze(-1), context_out, torch.zeros_like(context_out)
        )
        return self.out_norm(context_out), attn_w.squeeze(1), has_context


class ModalityFusionTransformer(nn.Module):
    """'Multimodal Self-Attention Fusion' trong so do: cac modality token (social,
    target, face) + 1 CLS token hoc duoc -> Transformer encoder -> lay CLS lam vector
    dai dien cuoi cung -> MLP phan loai."""

    def __init__(self, dim, max_modalities, num_heads=4, num_layers=2, dropout=0.2, num_classes=4):
        super().__init__()
        self.modality_embed = nn.Parameter(torch.zeros(1, max_modalities, dim))
        nn.init.trunc_normal_(self.modality_embed, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, num_classes)
        )

    def forward(self, modality_tokens, modality_mask=None):
        B, M, D = modality_tokens.shape
        tokens = modality_tokens + self.modality_embed[:, :M, :]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        if modality_mask is not None:
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=modality_mask.device)
            full_mask = torch.cat([cls_mask, modality_mask], dim=1)
            key_padding_mask = ~full_mask
        else:
            key_padding_mask = None

        out = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        cls_out = self.norm(out[:, 0])
        logits = self.head(cls_out)
        return logits, cls_out


class EngagementModelV5(nn.Module):
    r"""Kien truc dung theo so do (Fb LA 1 NODE THUC SU trong Build Graph, dung mui ten
    2 chieu Fk<->Fk<->Fb trong hinh -- khac ban truoc coi Fb dung ngoai "hoi" graph):

      K-hop neighbors --Backbone--> Time Attention --> Fk (nhieu node)  ----\
                                                                              |
      Body (16 frame) --Backbone--> Time Attention --> Fb -----------  Build Graph
                                                            |     \    (Fk<->Fk<->Fb)
                                                            |      \        |
                                                            |       \      GNN
                                                            |        \      |
                                                            |         \-----+---> Fb_refined = F_social
      Skeleton (16 frame) --Bi-LSTM+Attn----------\         |
                                                    Concat(Fb, skeleton) = Ftarget
                                                          |
      Face (16 frame) --Backbone--> Time Attention --> Ff |
                                                          | |
                                            F_social, Ftarget, Ff --> Multimodal Self-Attention Fusion
                                                                                |
                                                                               MLP -> logits

    Bo sung (khong co trong so do goc):
      - FiLM: dac trung ResNet TUNG FRAME (tu Body) dieu bien vector Skeleton tung
        frame TRUOC khi vao Bi-LSTM -- "tiem" ngu canh hinh anh vao nhanh skeleton.
      - SupCon: 1 projection head tren 'fused' tinh Supervised Contrastive Loss, giup
        tach biet embedding cac lop gan nhau (disengaged/engaged, normal/very_engaged).
      - Context Window: ngu canh THOI GIAN (clip truoc/sau CUNG 1 nguoi), song song
        voi K-hop (ngu canh KHONG GIAN, khac nguoi cung thoi diem).
    """

    def __init__(self, cfg, num_classes, feature_dim=2048):
        super().__init__()
        self.cfg = dict(cfg)
        D = cfg.get("EMBED_DIM", 192)
        dropout = cfg.get("DROPOUT", 0.30)
        social_dropout = cfg.get("SOCIAL_DROPOUT", 0.20)

        # ---- Backbone + Time Attention (dung chung kien truc CachedTemporalEncoderV3
        # cho ca Body, Face va tung Neighbor -- "Backbone" o day la ResNet50 DA duoc
        # trich xuat va cache san tu truoc, "Time Attention" la lop temporal-conv +
        # masked-pool ben trong CachedTemporalEncoderV3). ----
        use_attn_gate = cfg.get("USE_ADAPTIVE_ATTENTION_GATE", True)
        self.body_encoder = CachedTemporalEncoderV3(
            feature_dim, D, dropout, use_adaptive_attention_gate=use_attn_gate)
        self.face_encoder = CachedTemporalEncoderV3(
            feature_dim, D, dropout, use_adaptive_attention_gate=use_attn_gate)
        self.neighbor_encoder = CachedTemporalEncoderV3(
            feature_dim, D, dropout, use_adaptive_attention_gate=use_attn_gate)  # trong so rieng

        # ---- Skeleton: Bi-LSTM + attention pooling (theo dung so do) ----
        self.skeleton_encoder = SkeletonBiLSTMEncoder(
            num_keypoints=cfg["NUM_KEYPOINTS"], input_proj_dim=D, hidden_dim=D // 2, dropout=dropout
        )
        skeleton_out_dim = self.skeleton_encoder.out_dim  # = D (hidden_dim*2 = D)

        # ---- ResNet -> Skeleton FiLM conditioning ----
        # FiLM can dac trung ResNet TUNG FRAME (khong phai Fb da pool) -- lay tu chinh
        # body_encoder qua return_sequence=True (xem CachedTemporalEncoderV3.forward).
        # skeleton_dim = D vi joint_proj (trong SkeletonBiLSTMEncoder) chieu vao dung D.
        self.use_resnet_to_skeleton = cfg.get("USE_RESNET_TO_SKELETON_FUSION", True)
        if self.use_resnet_to_skeleton:
            self.film_conditioner = FiLMConditioner(
                appearance_dim=D, skeleton_dim=D, dropout=cfg.get("FILM_DROPOUT", 0.10)
            )

        # ---- Concat(Fb, skeleton) -> target token ----
        self.target_concat_proj = nn.Sequential(
            nn.Linear(D + skeleton_out_dim, D), nn.GELU(), nn.LayerNorm(D), nn.Dropout(dropout)
        )

        # ---- Build Graph + GNN tren cac Fk (mac dinh: Fk tho, khong relation features) ----
        self.use_relation_features = cfg.get("USE_RELATION_FEATURES", False)
        if self.use_relation_features:
            self.social_graph = RelationAwareBuildGraphAttention(
                dim=D, relation_dim=cfg.get("RELATION_DIM", 13),
                num_gnn_layers=cfg.get("GNN_LAYERS", 2), num_heads=cfg.get("GNN_HEADS", 4),
                dropout=social_dropout,
            )
        else:
            self.social_graph = BuildGraphAttention(
                dim=D, num_gnn_layers=cfg.get("GNN_LAYERS", 2), num_heads=cfg.get("GNN_HEADS", 4),
                dropout=social_dropout,
            )

        # ---- Context Window (ngu canh THOI GIAN: clip truoc/sau CUNG 1 nguoi) ----
        # Dung LAI trong so cua body_encoder (KHONG tao encoder rieng nhu neighbor_encoder)
        # vi cac clip ngu canh la CUNG 1 NGUOI, chi khac thoi diem -- ve mat ngu nghia
        # hop ly hon la chia se bieu dien voi chinh body_encoder cua target, thay vi coi
        # nhu 1 phan phoi hoan toan khac (nhu neighbor -- khac nguoi).
        self.use_context_window = cfg.get("USE_CONTEXT_WINDOW", True)
        if self.use_context_window:
            self.context_window = ContextWindowAttention(
                dim=D, max_window=cfg.get("CONTEXT_WINDOW_SIZE", 2),
                num_layers=cfg.get("CONTEXT_GNN_LAYERS", 2), num_heads=cfg.get("CONTEXT_GNN_HEADS", 4),
                dropout=cfg.get("CONTEXT_DROPOUT", 0.20),
            )
            # Gate cho F_context, giong social gate -- ngu canh thoi gian (cung 1 nguoi)
            # a priori DE huu ich hon social (khac nguoi) nen khoi tao it e de hon (bias
            # -1.0 thay vi -2.0 cua social), nhung van than trong, khong tin tuyet doi
            # tu dau vi du lieu ngu canh cung co the thua/nhieu.
            self.use_context_gate = cfg.get("USE_CONTEXT_RESIDUAL_GATE", True)
            if self.use_context_gate:
                self.context_gate = nn.Sequential(
                    nn.Linear(D * 2, D // 2), nn.GELU(), nn.Linear(D // 2, 1),
                )
                nn.init.zeros_(self.context_gate[-1].weight)
                nn.init.constant_(self.context_gate[-1].bias, -1.0)

        # ---- Multimodal Self-Attention Fusion (4 token: social, context, target, face) ----
        self.fusion = ModalityFusionTransformer(
            D, max_modalities=4, num_heads=cfg.get("FUSION_HEADS", 4),
            num_layers=cfg.get("FUSION_LAYERS", 2), dropout=dropout, num_classes=num_classes,
        )

        # ---- Social residual gate ----
        # F_social duoc quan sat la LAM HAI hieu suat khi neighbor qua thua (chi ~11%
        # sample co neighbor trong du lieu thuc te), khien GNN de hoc theo nhieu/vai
        # truong hop ca biet trong tap train roi khong tong quat hoa duoc. Thay vi luon
        # dua F_social vao Fusion voi "trong so ngam dinh = 1", them 1 GATE SCALAR hoc
        # duoc theo tung sample (dua tren chinh F_social va Ftarget) de model TU QUYET
        # DINH can tin F_social bao nhieu. Khoi tao gan 0 (bias am) de model BAT DAU
        # tu trang thai "gan nhu bo qua social", chi tang dan len neu du lieu that su
        # cho thay no huu ich -- an toan hon la ep model phai dung no ngay tu dau.
        self.use_social_gate = cfg.get("USE_SOCIAL_RESIDUAL_GATE", True)
        if self.use_social_gate:
            self.social_gate = nn.Sequential(
                nn.Linear(D * 2, D // 2), nn.GELU(), nn.Linear(D // 2, 1),
            )
            # Khoi tao bias am de sigmoid(gate) ~ 0.1-0.2 luc bat dau train, tranh
            # F_social nhieu lan at gradient cua Ftarget/Ff ngay tu epoch dau.
            nn.init.zeros_(self.social_gate[-1].weight)
            nn.init.constant_(self.social_gate[-1].bias, -2.0)

        # ---- Auxiliary behavior/emotion heads (giu tu ban patch truoc, khong lien quan
        # den kien truc chinh trong so do -- van la MLP nho tren Ffinal). ----
        # AUX_STOP_GRADIENT=True (mac dinh): aux heads doc tu ban sao DETACH cua 'fused'
        # (sau Fusion), nen KHONG con gradient tu 5 auxiliary loss nay lan nguoc ve
        # body/face/skeleton/social encoder hay ve chinh Fusion -- chi ban than aux_heads
        # duoc cap nhat boi auxiliary loss. Dieu nay ngan aux task (dang hoc rat tot,
        # vd pose~99%) "canh tranh" gradient/capacity voi nhiem vu engagement chinh.
        self.use_aux = cfg.get("USE_BEHAVIOR_EMOTION_AUX", False)
        self.aux_stop_gradient = cfg.get("AUX_STOP_GRADIENT", True)
        self.aux_heads = nn.ModuleDict()
        if self.use_aux:
            aux_num_classes = cfg.get("_AUX_NUM_CLASSES", {})
            aux_dropout = cfg.get("AUX_HEAD_DROPOUT", 0.20)
            for task_name, n_cls in aux_num_classes.items():
                if n_cls <= 0:
                    continue
                self.aux_heads[task_name] = nn.Sequential(
                    nn.Linear(D, D // 2), nn.GELU(), nn.Dropout(aux_dropout), nn.Linear(D // 2, n_cls),
                )

        # ---- Supervised Contrastive projection head ----
        # Chieu 'fused' (embedding sau Fusion, cung la dau vao cua classifier chinh)
        # sang 1 khong gian rieng de tinh SupCon loss. KHONG stop-gradient (khac aux
        # heads) vi muc tieu cua SupCon (tach biet embedding theo lop) dong huong truc
        # tiep voi nhiem vu engagement chinh -- gradient tu day duoc phep lan nguoc ve
        # toan bo encoder, giup dinh hinh hinh hoc cua chinh 'fused'. Projection head
        # nay CHI dung khi train, bi bo qua luc inference (giong thuc hanh chuan cua
        # SupCon/SimCLR: discard projection head, giu lai encoder).
        self.use_supcon = cfg.get("USE_SUPCON_LOSS", False)
        if self.use_supcon:
            proj_dim = cfg.get("SUPCON_PROJECTION_DIM", 128)
            self.contrastive_head = nn.Sequential(
                nn.Linear(D, D), nn.GELU(), nn.Linear(D, proj_dim),
            )

        # ---- Ordinal auxiliary head (CORAL-style cumulative thresholds) ----
        # Engagement co THU TU tu nhien (disengaged < normal < engaged < very_engaged).
        # CrossEntropy/Focal coi MOI cap nham lan la nhu nhau -- nham very_engaged
        # thanh disengaged (cach xa 3 bac) bi phat NHE NHU nham thanh engaged (cach 1
        # bac). Ordinal head du doan (num_classes-1) NGUONG NHI PHAN tich luy: logit
        # thu i = "rank > i hay khong" (i=0..num_classes-2). Diem so cuoi cung van la
        # 'logits' cua classifier chinh (khong doi cach quyet dinh) -- ordinal CHI la
        # 1 loss PHU giup embedding 'fused' hoc duoc cau truc thu tu tot hon, gan tiep
        # giup phan biet cac lop LIEN KE nhau (vd engaged vs very_engaged) ro rang hon.
        self.use_ordinal = cfg.get("USE_ORDINAL_AUX_LOSS", False)
        if self.use_ordinal:
            self.ordinal_head = nn.Sequential(
                nn.Linear(D, D // 2), nn.GELU(), nn.Dropout(cfg.get("DROPOUT", 0.30)),
                nn.Linear(D // 2, num_classes - 1),
            )

    def _encode_neighbors(self, batch):
        nfeat = batch["neighbor_feat"]
        nframe_mask = batch["neighbor_frame_mask"].bool()
        B, K, T, Fdim = nfeat.shape
        flat_feat = nfeat.reshape(B * K, T, Fdim)
        flat_mask = nframe_mask.reshape(B * K, T)
        Fk = self.neighbor_encoder(flat_feat, flat_mask).reshape(B, K, -1)
        valid = batch["neighbor_mask"].bool() & nframe_mask.any(dim=-1)
        Fk = Fk * valid.float().unsqueeze(-1)
        return Fk, valid

    def _encode_context(self, batch):
        """Ma hoa cac clip ngu canh (truoc/sau, CUNG 1 nguoi) bang CHINH body_encoder
        (chia se trong so voi target -- xem ly do trong __init__)."""
        cfeat = batch["context_feat"]
        cframe_mask = batch["context_frame_mask"].bool()
        B, W, T, Fdim = cfeat.shape
        flat_feat = cfeat.reshape(B * W, T, Fdim)
        flat_mask = cframe_mask.reshape(B * W, T)
        Fctx = self.body_encoder(flat_feat, flat_mask).reshape(B, W, -1)
        valid = batch["context_mask"].bool() & cframe_mask.any(dim=-1)
        Fctx = Fctx * valid.float().unsqueeze(-1)
        return Fctx, valid

    def forward(self, batch):
        body_mask = batch["body_frame_mask"].bool()
        face_mask = batch["face_frame_mask"].bool()
        skel_frame_mask = batch["skeleton_frame_mask"].bool()

        # ---- Body ----
        # return_sequence=True khi FiLM bat: can ca vector da pool (Fb, dung cho GNN va
        # Concat nhu truoc) LAN chuoi dac trung TUNG FRAME (body_frame_seq, dung rieng
        # cho FiLM dieu bien skeleton -- KHONG anh huong gi den Fb/luong xu ly con lai).
        if self.use_resnet_to_skeleton:
            Fb, body_frame_seq = self.body_encoder(batch["body_feat"], body_mask, return_sequence=True)
        else:
            Fb = self.body_encoder(batch["body_feat"], body_mask)
            body_frame_seq = None
        body_valid = body_mask.any(dim=1)

        # ---- Face ----
        Ff = self.face_encoder(batch["face_feat"], face_mask)
        face_valid = face_mask.any(dim=1)
        Ff = Ff * face_valid.float().unsqueeze(-1)

        # ---- ResNet -> Skeleton FiLM: dac trung ResNet tung frame dieu bien vector
        # skeleton tung frame TRUOC khi vao Bi-LSTM (gia dinh frame i cua body va frame
        # i cua skeleton tuong ung cung 1 thoi diem trong clip, vi ca 2 deu lay
        # NUM_FRAMES frame tu CUNG 1 clip goc). ----
        film_gamma, film_beta = None, None
        if self.use_resnet_to_skeleton:
            film_gamma, film_beta = self.film_conditioner(body_frame_seq, body_mask)

        # ---- Skeleton (Bi-LSTM + attention pooling, co the duoc FiLM dieu bien) ----
        Fskel = self.skeleton_encoder(
            batch["skeleton_xy"], batch["skeleton_conf"], batch["skeleton_kpt_mask"], skel_frame_mask,
            appearance_gamma=film_gamma, appearance_beta=film_beta,
        )
        skel_valid = skel_frame_mask.any(dim=1)

        # ---- Concat(Fb, skeleton) -> target token ----
        Ftarget = self.target_concat_proj(torch.cat([Fb, Fskel], dim=-1))

        # ---- K-hop neighbors -> Fk -> Build Graph -> GNN -> social context ----
        Fk_nodes, Fk_valid = self._encode_neighbors(batch)
        if self.use_relation_features:
            relation = batch["neighbor_relation"].float()
            Fsocial_raw, neighbor_attn, has_neighbor = self.social_graph(Fk_nodes, Fk_valid, Fb, relation)
        else:
            Fsocial_raw, neighbor_attn, has_neighbor = self.social_graph(Fk_nodes, Fk_valid, Fb)

        # ---- Social residual gate: model tu quyet dinh muc do tin F_social ----
        social_gate_value = None
        if self.use_social_gate:
            gate_input = torch.cat([Fsocial_raw, Ftarget], dim=-1)
            social_gate_value = torch.sigmoid(self.social_gate(gate_input))  # (B,1), khoi tao ~0.1-0.2
            Fsocial = Fsocial_raw * social_gate_value
        else:
            Fsocial = Fsocial_raw

        B = Fb.size(0)
        device = Fb.device

        # ---- Context window -> F_context (ngu canh thoi gian: clip truoc/sau cung nguoi) ----
        Fcontext, context_gate_value, has_context = None, None, None
        if self.use_context_window:
            Fctx_nodes, Fctx_valid = self._encode_context(batch)
            context_offset_idx = batch["context_offset"]
            Fcontext_raw, context_attn, has_context = self.context_window(
                Fctx_nodes, Fctx_valid, context_offset_idx, Fb
            )
            if self.use_context_gate:
                ctx_gate_input = torch.cat([Fcontext_raw, Ftarget], dim=-1)
                context_gate_value = torch.sigmoid(self.context_gate(ctx_gate_input))
                Fcontext = Fcontext_raw * context_gate_value
            else:
                Fcontext = Fcontext_raw
        else:
            Fctx_valid, context_attn = None, None

        # ---- Multimodal Self-Attention Fusion: social, [context], target, face ----
        if self.use_context_window:
            modality_tokens = torch.stack([Fsocial, Fcontext, Ftarget, Ff], dim=1)
            modality_mask = torch.stack(
                [has_neighbor, has_context, torch.ones(B, dtype=torch.bool, device=device), face_valid], dim=1
            )
        else:
            modality_tokens = torch.stack([Fsocial, Ftarget, Ff], dim=1)
            modality_mask = torch.stack(
                [has_neighbor, torch.ones(B, dtype=torch.bool, device=device), face_valid], dim=1
            )

        logits, fused = self.fusion(modality_tokens, modality_mask)

        aux_logits = {}
        if self.use_aux:
            # AUX_STOP_GRADIENT: aux heads doc tu ban sao DETACH cua 'fused', nen 5
            # auxiliary loss se KHONG lan gradient nguoc ve body/face/skeleton/social
            # encoder hay ve Fusion -- chi cap nhat rieng aux_heads. Ngan aux (hoc rat
            # tot, vd pose~99%) canh tranh gradient/capacity voi nhiem vu engagement chinh.
            aux_input = fused.detach() if self.aux_stop_gradient else fused
            for task_name, head in self.aux_heads.items():
                aux_logits[task_name] = head(aux_input)

        # ---- Supervised Contrastive embedding ----
        # KHONG detach 'fused' o day (khac aux_heads): SupCon can gradient lan nguoc ve
        # toan bo encoder de THUC SU dinh hinh lai hinh hoc cua embedding, giup tach
        # biet cac lop gan nhau (disengaged/engaged, normal/very_engaged) tot hon.
        contrastive_embedding = None
        if self.use_supcon:
            contrastive_embedding = F.normalize(self.contrastive_head(fused), dim=-1)

        ordinal_logits = None
        if self.use_ordinal:
            ordinal_logits = self.ordinal_head(fused)  # (B, num_classes-1)

        return {
            "logits": logits,
            "fused": fused,
            "Fb": Fb,
            "Ff": Ff,
            "Fskel": Fskel,
            "Ftarget": Ftarget,
            "Fsocial": Fsocial,
            "Fsocial_raw": Fsocial_raw,
            "social_gate_value": social_gate_value,
            "aux_logits": aux_logits,
            "contrastive_embedding": contrastive_embedding,
            "neighbor_attn_weights": neighbor_attn,
            "neighbor_valid_mask": Fk_valid,
            "has_neighbor": has_neighbor,
            "Fcontext": Fcontext,
            "context_gate_value": context_gate_value,
            "context_attn_weights": context_attn,
            "context_valid_mask": Fctx_valid,
            "has_context": has_context,
            "ordinal_logits": ordinal_logits,  # (B, num_classes-1) neu USE_ORDINAL_AUX_LOSS=True, khong thi None
            # Compatibility voi generic evaluation utilities (khong dung trong kien truc nay).
            "ordinal_score": None,
            "ordinal_cutpoints": None,
            "social_aux_loss": logits.new_zeros(()),
        }


class TrackTemporalLSTM(nn.Module):
    """Bidirectional LSTM tren TOAN BO track: dau vao la chuoi 'fused' embedding (da
    qua toan bo segment-level pipeline: Body/Face/Skeleton/K-hop/Context/Fusion) cua
    MOI segment trong track, dau ra la embedding da duoc TINH CHINH LAI bang ngu canh
    toan track (nhin duoc ca qua khu va tuong lai cua chinh track do), giong tinh than
    'TemporalLSTM' trong notebook tham khao ResNet+LSTM."""

    def __init__(self, dim, num_layers=2, dropout=0.30):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dim, hidden_size=dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, fused_seq, seq_mask):
        """fused_seq: (B,L,D), seq_mask: (B,L) bool -- True = segment thuc, False = padding."""
        x = fused_seq * seq_mask.float().unsqueeze(-1)

        # Giong SkeletonBiLSTMEncoder: torch.inference_mode(False) de tranh loi
        # "Inplace update to inference tensor" khi nn.LSTM.flatten_parameters() chay
        # ben trong torch.inference_mode() (vd benchmark_pipeline) hoac voi DataParallel.
        with torch.inference_mode(False):
            out, _ = self.lstm(x)

        out = self.drop(self.out_proj(out))
        out = self.norm(out)
        return out * seq_mask.float().unsqueeze(-1)


class TrackEngagementModel(nn.Module):
    """Boc EngagementModelV5 (segment-level, khong doi) + Track-level Bi-LSTM phia
    tren, giong tinh than notebook tham khao ResNet+LSTM: van la bai toan 'nhieu-den-
    nhieu' (moi segment co 1 logits/loss RIENG, tinh qua flatten_valid_positions o
    muc train/eval), nhung moi segment gio co THEM ngu canh tu TOAN BO track thay vi
    bi xu ly hoan toan doc lap nhu EngagementModelV5 don thuan.

    K-hop (hang xom khong gian) va Context Window (+-N segment) VAN chay o BEN TRONG
    EngagementModelV5 theo TUNG SEGMENT doc lap nhu truoc -- khong doi. Track-level
    Bi-LSTM la 1 lop THEM VAO NGOAI CUNG, tinh chinh lai embedding 'fused' bang ngu
    canh TOAN BO track (manh hon Context Window vi khong gioi han cua so nho +-2).
    """

    def __init__(self, cfg, num_classes, feature_dim=2048):
        super().__init__()
        D = cfg.get("EMBED_DIM", 192)

        # Segment encoder: TAT contrastive rieng cua no (USE_SUPCON_LOSS=False noi bo)
        # vi contrastive gio duoc tinh tren embedding DA qua Track-level LSTM (refined),
        # phu hop hon voi chinh embedding thuc su dung de phan loai cuoi cung.
        segment_cfg = dict(cfg)
        segment_cfg["USE_SUPCON_LOSS"] = False
        self.segment_encoder = EngagementModelV5(segment_cfg, num_classes, feature_dim)

        self.track_lstm = TrackTemporalLSTM(
            dim=D, num_layers=cfg.get("TRACK_LSTM_LAYERS", 2),
            dropout=cfg.get("TRACK_LSTM_DROPOUT", 0.30),
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Dropout(cfg.get("DROPOUT", 0.30)), nn.Linear(D, num_classes)
        )

        self.use_supcon = cfg.get("USE_SUPCON_LOSS", False)
        if self.use_supcon:
            proj_dim = cfg.get("SUPCON_PROJECTION_DIM", 128)
            self.contrastive_head = nn.Sequential(
                nn.Linear(D, D), nn.GELU(), nn.Linear(D, proj_dim),
            )

    def forward(self, batch):
        """batch: moi field per-segment co THEM 1 chieu L (do dai track) o dau, vd
        body_feat (B,L,T,Feat), neighbor_feat (B,L,K,T,Feat), context_feat (B,L,W,T,Feat),
        skeleton_xy (B,L,T,K,2), label (B,L), aux_valid (B,L), aux_label_<task> (B,L),
        cong them seq_mask (B,L) bool -- tat ca do track_collate_fn tao ra."""
        seq_mask = batch["seq_mask"]
        B, L = seq_mask.shape

        # Flatten (B,L) -> (B*L) cho MOI field per-segment, goi segment_encoder 1 LAN
        # duy nhat cho toan bo B*L segment (hieu qua, khong doi logic ben trong no).
        flat_batch = {}
        for key, value in batch.items():
            if key in ("seq_mask", "seq_lens", "sample_id", "session", "roi_id"):
                continue
            if torch.is_tensor(value) and value.dim() >= 2 and value.shape[0] == B and value.shape[1] == L:
                flat_batch[key] = value.reshape(B * L, *value.shape[2:])
            elif torch.is_tensor(value):
                flat_batch[key] = value

        seg_out = self.segment_encoder(flat_batch)

        fused_flat = seg_out["fused"]  # (B*L, D)
        D = fused_flat.shape[-1]
        fused_seq = fused_flat.reshape(B, L, D)

        refined = self.track_lstm(fused_seq, seq_mask)  # (B,L,D), da mask padding

        logits = self.classifier_head(refined)  # (B,L,num_classes)

        aux_logits_seq = {}
        for task_name, task_logits in (seg_out.get("aux_logits") or {}).items():
            n_cls = task_logits.shape[-1]
            aux_logits_seq[task_name] = task_logits.reshape(B, L, n_cls)

        contrastive_embedding = None
        if self.use_supcon:
            contrastive_embedding = F.normalize(self.contrastive_head(refined), dim=-1)  # (B,L,proj_dim)

        # Neighbor/context diagnostics: reshape lai tu (B*L, K) ve (B,L,K) de
        # analyze_social_interaction van dung duoc qua flatten_valid_positions.
        neighbor_attn_bl = None
        if seg_out.get("neighbor_attn_weights") is not None:
            K_dim = seg_out["neighbor_attn_weights"].shape[-1]
            neighbor_attn_bl = seg_out["neighbor_attn_weights"].reshape(B, L, K_dim)

        neighbor_valid_bl = None
        if seg_out.get("neighbor_valid_mask") is not None:
            K_dim = seg_out["neighbor_valid_mask"].shape[-1]
            neighbor_valid_bl = seg_out["neighbor_valid_mask"].reshape(B, L, K_dim)

        has_neighbor_bl = None
        if seg_out.get("has_neighbor") is not None:
            has_neighbor_bl = seg_out["has_neighbor"].reshape(B, L)

        has_context_bl = None
        if seg_out.get("has_context") is not None:
            has_context_bl = seg_out["has_context"].reshape(B, L)

        return {
            "logits": logits,              # (B,L,num_classes)
            "seq_mask": seq_mask,          # (B,L)
            "fused": refined,              # (B,L,D) -- embedding SAU track LSTM (dung de phan loai)
            "fused_pre_track": fused_seq,  # (B,L,D) -- embedding TRUOC track LSTM (segment-only), de doi chieu/debug
            "aux_logits": aux_logits_seq,
            "contrastive_embedding": contrastive_embedding,
            "has_neighbor": has_neighbor_bl,
            "has_context": has_context_bl,
            "neighbor_attn_weights": neighbor_attn_bl,
            "neighbor_valid_mask": neighbor_valid_bl,
        }
