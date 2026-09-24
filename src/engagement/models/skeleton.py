"""Skeleton encoder: Bi-LSTM + attention pooling (tuỳ chọn điều biến FiLM)."""
import torch
import torch.nn as nn


class SkeletonBiLSTMEncoder(nn.Module):
    """17 khớp (x, y, conf) mỗi frame -> 1 vector/frame -> Bi-LSTM -> attention pooling."""

    def __init__(self, input_proj_dim=192, hidden_dim=96, dropout=0.30):
        super().__init__()
        self.joint_proj = nn.Sequential(
            nn.Linear(3, input_proj_dim), nn.GELU(), nn.LayerNorm(input_proj_dim), nn.Dropout(dropout))
        self.lstm = nn.LSTM(input_proj_dim, hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.out_dim = hidden_dim * 2

        self.attn_query = nn.Parameter(torch.zeros(1, 1, self.out_dim))
        nn.init.trunc_normal_(self.attn_query, std=0.02)
        self.attn_pool = nn.MultiheadAttention(self.out_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(self.out_dim)

    def forward(self, xy, conf, kpt_mask, frame_mask, film_gamma=None, film_beta=None):
        frame_mask = frame_mask.bool()
        B = xy.size(0)

        # Pool 17 khớp -> 1 vector/frame, trọng số = confidence
        conf = conf * kpt_mask.float()                                          # (B, T, K)
        joints = self.joint_proj(torch.cat([xy, conf.unsqueeze(-1)], dim=-1))   # (B, T, K, D)
        weight = conf.unsqueeze(-1)
        frame_vec = (joints * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1e-6)

        if film_gamma is not None:
            frame_vec = frame_vec * (1.0 + film_gamma) + film_beta
        frame_vec = frame_vec * frame_mask.float().unsqueeze(-1)

        # inference_mode(False): tránh lỗi flatten_parameters() của LSTM khi gọi trong inference_mode
        with torch.inference_mode(False):
            lstm_out, _ = self.lstm(frame_vec)

        has_valid = frame_mask.any(dim=1)
        safe_mask = frame_mask.clone()
        safe_mask[~has_valid, 0] = True
        query = self.attn_query.expand(B, -1, -1)
        pooled, _ = self.attn_pool(query, lstm_out, lstm_out, key_padding_mask=~safe_mask)
        pooled = pooled.squeeze(1) * has_valid.float().unsqueeze(-1)

        return self.out_norm(pooled)
