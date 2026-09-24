"""Temporal encoder cho feature ResNet đã cache (Body / Face / Neighbor / Context)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedTemporalStats(nn.Module):
    """Pool theo thời gian có mask: [mean, std, max, mean |delta|] -> (B, 4D)."""

    def forward(self, x, mask):
        mask = mask.bool()
        mf = mask.float().unsqueeze(-1)
        count = mf.sum(dim=1).clamp_min(1.0)

        mean = (x * mf).sum(dim=1) / count
        std = torch.sqrt((((x - mean.unsqueeze(1)) ** 2) * mf).sum(dim=1) / count + 1e-5)

        maxv = x.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        maxv = torch.where(mask.any(dim=1, keepdim=True), maxv, torch.zeros_like(maxv))

        if x.size(1) > 1:
            dmf = (mask[:, 1:] & mask[:, :-1]).float().unsqueeze(-1)
            delta = ((x[:, 1:] - x[:, :-1]).abs() * dmf).sum(dim=1) / dmf.sum(dim=1).clamp_min(1.0)
        else:
            delta = torch.zeros_like(mean)

        return torch.cat([mean, std, maxv, delta], dim=-1)


class TemporalGatedAttention(nn.Module):
    """Self-attention theo thời gian, cộng residual qua 1 gate học được (khởi tạo ~0.12)."""

    def __init__(self, dim, num_heads=4, dropout=0.30):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.gate_proj = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        nn.init.constant_(self.gate_proj[-1].bias, -2.0)

    def forward(self, x, mask):
        all_invalid = ~mask.any(dim=1)
        padding_mask = ~mask
        padding_mask[all_invalid] = False     # tránh NaN khi không có frame nào hợp lệ

        attn_out, _ = self.attn(x, x, x, key_padding_mask=padding_mask, need_weights=False)
        attn_out = attn_out.masked_fill(all_invalid[:, None, None], 0.0)

        mf = mask.float().unsqueeze(-1)
        pooled = (x * mf).sum(dim=1) / mf.sum(dim=1).clamp_min(1.0)
        gate = torch.sigmoid(self.gate_proj(pooled)).unsqueeze(1)    # (B, 1, 1)

        return self.norm(x + gate * self.dropout(attn_out)) * mf


class CachedTemporalEncoderV3(nn.Module):
    """Feature (B, T, 2048) -> proj -> 2 conv residual -> [attention gate] -> pool -> (B, D)."""

    def __init__(self, input_dim=2048, dim=192, dropout=0.30, use_adaptive_attention_gate=True):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(nn.Linear(input_dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.use_adaptive_attention_gate = use_adaptive_attention_gate
        if use_adaptive_attention_gate:
            self.temporal_attn_gate = TemporalGatedAttention(dim, dropout=dropout)

        self.pool = MaskedTemporalStats()
        self.out = nn.Sequential(
            nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim),
        )

    def _residual_conv(self, x, conv, norm, mf):
        y = conv(x.transpose(1, 2)).transpose(1, 2)
        return norm(x + self.dropout(F.gelu(y))) * mf

    def forward(self, feat, mask, return_sequence=False):
        """return_sequence=True -> trả thêm chuỗi đặc trưng từng frame (B, T, D) cho FiLM."""
        mask = mask.bool()
        mf = mask.float().unsqueeze(-1)

        x = self.proj(self.input_norm(feat)) * mf
        x = self._residual_conv(x, self.conv1, self.norm1, mf)
        x = self._residual_conv(x, self.conv2, self.norm2, mf)
        if self.use_adaptive_attention_gate:
            x = self.temporal_attn_gate(x, mask)

        pooled = self.out(self.pool(x, mask))
        return (pooled, x) if return_sequence else pooled
