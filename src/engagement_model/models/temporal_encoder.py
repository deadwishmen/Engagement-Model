"""Backbone + Time Attention encoder shared by Body/Face/Neighbor branches.

Corresponds to notebook section "4. V3 Body/Face Temporal Encoder" (cell 22).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedTemporalStats(nn.Module):
    """Masked temporal pooling: mean, std, max, mean absolute frame-to-frame delta."""

    def forward(self, x, mask):
        mask = mask.bool()
        mf = mask.float().unsqueeze(-1)
        denom = mf.sum(dim=1).clamp_min(1.0)

        mean = (x * mf).sum(dim=1) / denom
        var = (((x - mean.unsqueeze(1)) ** 2) * mf).sum(dim=1) / denom
        std = torch.sqrt(var + 1e-5)

        x_max = x.masked_fill(~mask.unsqueeze(-1), -1e4)
        maxv = x_max.max(dim=1).values
        valid = mask.any(dim=1)
        maxv = torch.where(valid.unsqueeze(-1), maxv, torch.zeros_like(maxv))

        if x.size(1) > 1:
            delta = x[:, 1:] - x[:, :-1]
            dmask = mask[:, 1:] & mask[:, :-1]
            dmf = dmask.float().unsqueeze(-1)
            dden = dmf.sum(dim=1).clamp_min(1.0)
            delta_mean = (delta.abs() * dmf).sum(dim=1) / dden
        else:
            delta_mean = torch.zeros_like(mean)

        return torch.cat([mean, std, maxv, delta_mean], dim=-1)


class TemporalGatedAttention(nn.Module):
    """Adaptive self-attention over time + a per-sample learned gate.

    Unlike residual conv (fixed prior: "nearby frames are related"), this block lets
    the model LEARN which frame to attend to, then uses a per-sample scalar gate
    (derived from the pooled features) to decide how much to trust the attention
    output before adding it as a residual -- same spirit as the social residual gate
    used elsewhere in this model.

    The gate is initialized near 0 (sigmoid(-2) ~ 0.12) so this block starts out
    almost a no-op (safe warm start), then the model can increase it if it actually
    helps.
    """

    def __init__(self, dim, num_heads=4, dropout=0.30):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.gate_proj = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )
        nn.init.constant_(self.gate_proj[-1].bias, -2.0)  # sigmoid(-2) ~ 0.12

    def forward(self, x, mask):
        """x: (B,T,D) already normalized/projected. mask: (B,T) bool, True = valid frame."""
        key_padding_mask = ~mask  # MultiheadAttention: True = position to IGNORE
        all_invalid = (~mask).all(dim=1)
        safe_kpm = key_padding_mask.clone()
        safe_kpm[all_invalid] = False

        attn_out, _ = self.attn(x, x, x, key_padding_mask=safe_kpm, need_weights=False)
        attn_out = torch.where(
            all_invalid.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(attn_out), attn_out
        )

        mf = mask.float().unsqueeze(-1)
        denom = mf.sum(dim=1).clamp_min(1.0)
        pooled_for_gate = (x * mf).sum(dim=1) / denom
        gate = torch.sigmoid(self.gate_proj(pooled_for_gate)).unsqueeze(1)  # (B,1,1)

        x = self.norm(x + gate * self.dropout(attn_out)) * mf
        return x, gate.squeeze(-1).squeeze(-1)  # also return gate (B,) for diagnostics


class CachedTemporalEncoderV3(nn.Module):
    def __init__(self, input_dim=2048, dim=192, dropout=0.30,
                 use_adaptive_attention_gate=True, attn_num_heads=4):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.use_adaptive_attention_gate = use_adaptive_attention_gate
        if use_adaptive_attention_gate:
            self.temporal_attn_gate = TemporalGatedAttention(
                dim, num_heads=attn_num_heads, dropout=dropout
            )

        self.pool = MaskedTemporalStats()
        self.out = nn.Sequential(
            nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim),
        )

    def forward(self, feat, mask, return_sequence=False, return_gate=False):
        """return_sequence=False (default, fully backward compatible): returns a
        single pooled (B,D) vector, used for Body/Face/Neighbor.
        return_sequence=True: ALSO returns the per-frame feature sequence (B,T,D)
        before pooling -- used to condition (FiLM) the skeleton branch with per-frame
        ResNet context.
        return_gate=True: ALSO returns the adaptive-attention gate value (B,) for
        logging/diagnostics."""
        mask = mask.bool()
        mf = mask.float().unsqueeze(-1)
        x = self.proj(self.input_norm(feat)) * mf

        y = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm1(x + self.dropout(F.gelu(y))) * mf

        y = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm2(x + self.dropout(F.gelu(y))) * mf

        gate_value = None
        if self.use_adaptive_attention_gate:
            x, gate_value = self.temporal_attn_gate(x, mask)

        pooled = self.out(self.pool(x, mask))

        if return_sequence and return_gate:
            return pooled, x, gate_value
        if return_sequence:
            return pooled, x
        if return_gate:
            return pooled, gate_value
        return pooled
