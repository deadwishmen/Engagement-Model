"""Các khối: GNN, social graph, context window, FiLM, multimodal fusion, gate."""
import torch
import torch.nn as nn


class GraphAttentionLayer(nn.Module):
    """1 lớp GNN: attention full-connect giữa các node hợp lệ + feed-forward."""

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, node_mask, attn_bias=None):
        """attn_bias (tuỳ chọn): (B * num_heads, N, N), giá trị thực cộng thẳng vào attention
        score trước softmax (không phải mask 0/1). Dùng để "thiên vị" một số cặp node theo
        thông tin bên ngoài (ví dụ đặc trưng quan hệ hình học), tách biệt với key_padding_mask."""
        attn_out, attn_w = self.attn(nodes, nodes, nodes, key_padding_mask=~node_mask, attn_mask=attn_bias)
        x = self.norm1(nodes + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, attn_w


class RelationAttentionBias(nn.Module):
    """Đặc trưng quan hệ Fb->Fk (khoảng cách, vận tốc tương đối, ...) -> 1 số thực bias
    cho MỖI head attention. Bias này được cộng thẳng vào attention score (trước softmax),
    tách biệt với nội dung ngoại hình của Fk - tức "nên chú ý bao nhiêu" (do hình học quyết
    định) và "nội dung gì được truyền đi" (do embedding ngoại hình quyết định) không bị trộn
    lẫn vào cùng 1 vector như cách cộng thẳng relation vào node."""

    def __init__(self, relation_dim, num_heads, hidden_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.net = nn.Sequential(nn.Linear(relation_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_heads))
        nn.init.zeros_(self.net[-1].weight)     # khởi tạo bias = 0 => ban đầu không đổi so với attention thường
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, relation, Fk_mask):
        """relation: (B, K, relation_dim), Fk_mask: (B, K) -> attn_mask (B*num_heads, 1+K, 1+K)."""
        B, K, _ = relation.shape
        N = K + 1
        relation = torch.nan_to_num(relation, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)

        edge_bias = self.net(relation) * Fk_mask.float().unsqueeze(-1)   # (B, K, num_heads)
        edge_bias = edge_bias.permute(0, 2, 1)                          # (B, num_heads, K)

        full = edge_bias.new_zeros(B, self.num_heads, N, N)
        full[:, :, 0, 1:] = edge_bias      # Fb (node 0) chú ý tới từng Fk
        full[:, :, 1:, 0] = edge_bias      # dùng chung giá trị cho chiều Fk chú ý ngược lại Fb
        return full.reshape(B * self.num_heads, N, N)


class BuildGraphAttention(nn.Module):
    """Graph gồm Fb (node 0) + K node Fk. Sau GNN lấy lại node Fb làm F_social.

    relation_dim != None -> dùng đặc trưng quan hệ theo 1 trong 2 cách (relation_encoding):
      - "node": (kiểu cũ) cộng thẳng embedding quan hệ vào từng Fk trước khi vào GNN.
                Đơn giản, nhưng trộn lẫn "nên chú ý bao nhiêu" và "nội dung gì" vào 1 vector.
      - "bias": encode quan hệ thành 1 số thực mỗi head, cộng thẳng vào attention SCORE
                (trước softmax). Quan hệ hình học quyết định mức độ chú ý, còn Fk vẫn giữ
                nguyên nội dung ngoại hình - không bị pha trộn.
    """

    def __init__(self, dim, num_gnn_layers=2, num_heads=4, dropout=0.20,
                relation_dim=None, relation_encoding="node"):
        super().__init__()
        if relation_dim is not None and relation_encoding not in ("node", "bias"):
            raise ValueError(f"RELATION_ENCODING không hợp lệ: {relation_encoding!r} (chỉ nhận 'node' hoặc 'bias')")

        self.num_heads = num_heads
        self.relation_encoding = relation_encoding
        self.relation_encoder = None        # chế độ "node" - giữ tên cũ để tương thích checkpoint cũ
        self.relation_bias_encoder = None   # chế độ "bias" - mới

        if relation_dim is not None and relation_encoding == "node":
            self.relation_encoder = nn.Sequential(
                nn.Linear(relation_dim, 64), nn.GELU(), nn.LayerNorm(64),
                nn.Linear(64, dim), nn.GELU(), nn.LayerNorm(dim),
            )
        elif relation_dim is not None and relation_encoding == "bias":
            self.relation_bias_encoder = RelationAttentionBias(relation_dim, num_heads)

        self.gnn_layers = nn.ModuleList(
            [GraphAttentionLayer(dim, num_heads, dropout) for _ in range(num_gnn_layers)])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, Fk, Fk_mask, Fb, relation=None):
        attn_bias = None
        if self.relation_encoder is not None:
            relation_clamped = torch.nan_to_num(relation, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
            Fk = Fk + self.relation_encoder(relation_clamped)
        elif self.relation_bias_encoder is not None:
            attn_bias = self.relation_bias_encoder(relation, Fk_mask)

        B = Fk.size(0)
        nodes = torch.cat([Fb.unsqueeze(1), Fk], dim=1)                     # (B, 1+K, D)
        node_mask = torch.cat([Fk_mask.new_ones(B, 1), Fk_mask], dim=1)     # Fb luôn hợp lệ

        attn_w = None
        for layer in self.gnn_layers:
            nodes, attn_w = layer(nodes, node_mask, attn_bias=attn_bias)

        has_neighbor = Fk_mask.any(dim=1)
        social = self.out_norm(nodes[:, 0]) * has_neighbor.float().unsqueeze(-1)   # không có hàng xóm -> 0
        neighbor_attn = attn_w[:, 0, 1:]                                           # Fb chú ý tới từng Fk
        return social, neighbor_attn, has_neighbor


class ContextWindowAttention(nn.Module):
    """Các clip trước/sau (cùng người) + positional embedding theo offset -> GNN -> Fb query -> F_context."""

    def __init__(self, dim, max_window=2, num_layers=2, num_heads=4, dropout=0.20):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 * max_window + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.gnn_layers = nn.ModuleList(
            [GraphAttentionLayer(dim, num_heads, dropout) for _ in range(num_layers)])
        self.target_pool = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, Fctx, ctx_mask, ctx_offset, Fb):
        pos = self.pos_embed[0][ctx_offset]                                 # (B, W, D)
        x = Fctx + pos * ctx_mask.float().unsqueeze(-1)

        has_context = ctx_mask.any(dim=1)
        safe_mask = ctx_mask.clone()
        safe_mask[~has_context, 0] = True

        for layer in self.gnn_layers:
            x, _ = layer(x, safe_mask)

        out, _ = self.target_pool(Fb.unsqueeze(1), x, x, key_padding_mask=~safe_mask)
        out = out.squeeze(1) * has_context.float().unsqueeze(-1)
        return self.out_norm(out), has_context


class FiLMConditioner(nn.Module):
    """Đặc trưng body từng frame -> (gamma, beta) điều biến skeleton từng frame.
    Lớp cuối khởi tạo 0 => ban đầu là phép đồng nhất."""

    def __init__(self, appearance_dim, skeleton_dim, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(appearance_dim, skeleton_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(skeleton_dim, skeleton_dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, appearance_seq, frame_mask):
        gamma, beta = self.net(appearance_seq).chunk(2, dim=-1)
        mf = frame_mask.float().unsqueeze(-1)
        return gamma * mf, beta * mf


class ModalityFusionTransformer(nn.Module):
    """[CLS] + các token modality -> Transformer encoder -> CLS -> MLP -> logits."""

    def __init__(self, dim, max_modalities, num_heads=4, num_layers=2, dropout=0.2, num_classes=4):
        super().__init__()
        self.modality_embed = nn.Parameter(torch.zeros(1, max_modalities, dim))
        nn.init.trunc_normal_(self.modality_embed, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(dim, num_heads, dim * 4, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, num_classes))

    def forward(self, tokens, token_mask):
        B, M, _ = tokens.shape
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens + self.modality_embed[:, :M]], dim=1)
        full_mask = torch.cat([token_mask.new_ones(B, 1), token_mask], dim=1)

        out = self.encoder(tokens, src_key_padding_mask=~full_mask)
        cls = self.norm(out[:, 0])
        return self.head(cls), cls


def make_scalar_gate(dim, init_bias):
    """Gate vô hướng từ concat(F_x, F_target). Khởi tạo = sigmoid(init_bias)."""
    gate = nn.Sequential(nn.Linear(dim * 2, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
    nn.init.zeros_(gate[-1].weight)
    nn.init.constant_(gate[-1].bias, init_bias)
    return gate