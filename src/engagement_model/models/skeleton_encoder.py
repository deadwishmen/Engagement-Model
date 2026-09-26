"""Skeleton encoder: Bi-LSTM + attention pooling.

Corresponds to notebook section "5. Skeleton Encoder -- Bi-LSTM + Attention Pooling"
(cell 24).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SkeletonBiLSTMEncoder(nn.Module):
    """Skeleton encoder theo dung hinh ve: 16-frame skeleton -> LSTM -- nang cap
    thanh Bi-LSTM (2 chieu, bat ca chuyen dong tien va lui trong clip) + attention
    pooling co hoc duoc (thay vi chi lay hidden state cuoi cung, attention pooling
    cho phep model tu quyet dinh frame nao quan trong nhat cho engagement).

    Dau vao moi frame: toa do (x,y) da chuan hoa theo bbox + do tin cay (confidence)
    cua 17 khop COCO, duoc gop (mean-pool co trong so theo confidence) thanh 1 vector
    truoc khi dua vao LSTM -- giu don gian dung tinh than "LSTM" trong hinh, khac voi
    GCN+Transformer phuc tap cua V3/V4 cu.

    Tuy chon: co the nhan them FiLM tu dac trung ResNet TUNG FRAME (xem FiLMConditioner
    trong model chinh) de dieu bien vector pose tung frame bang ngu canh hinh anh, giup
    phan biet cac tu the mo ho ve mat hinh hoc thuan tuy.
    """

    def __init__(self, num_keypoints=17, input_proj_dim=128, hidden_dim=96, dropout=0.30):
        super().__init__()
        self.num_keypoints = num_keypoints

        # Nhung (x,y,conf) cua tung khop thanh 1 vector nho, roi pool co trong so theo
        # confidence tren 17 khop -> 1 vector dai dien pose CUA TUNG FRAME.
        self.joint_proj = nn.Sequential(
            nn.Linear(3, input_proj_dim), nn.GELU(), nn.LayerNorm(input_proj_dim), nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=input_proj_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        lstm_out_dim = hidden_dim * 2  # bidirectional -> nhan doi

        # Attention pooling: 1 query hoc duoc chon frame nao (trong toan bo output LSTM
        # qua thoi gian) quan trong nhat, thay vi chi lay hidden state cuoi.
        self.attn_query = nn.Parameter(torch.zeros(1, 1, lstm_out_dim))
        nn.init.trunc_normal_(self.attn_query, std=0.02)
        self.attn_pool = nn.MultiheadAttention(lstm_out_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(lstm_out_dim)
        self.out_dim = lstm_out_dim

    def forward(self, xy, conf, kpt_mask, frame_mask, appearance_gamma=None, appearance_beta=None):
        """xy: (B,T,K,2), conf: (B,T,K), kpt_mask: (B,T,K) bool, frame_mask: (B,T) bool.
        appearance_gamma, appearance_beta: (B,T,D) hoac None -- FiLM tu dac trung ResNet
        TUNG FRAME (xem FiLMConditioner), dieu bien vector skeleton tung frame TRUOC khi
        vao Bi-LSTM: frame_vec = frame_vec * (1 + gamma) + beta. Neu None (FiLM tat, hoac
        khong co du lieu ResNet), skeleton hoat dong y het nhu truoc (chi hinh hoc thuan tuy)."""
        kpt_mask = kpt_mask.bool()
        frame_mask = frame_mask.bool()
        B, T, K, _ = xy.shape

        conf_valid = (conf * kpt_mask.float())  # (B,T,K)
        joint_in = torch.cat([xy, conf_valid.unsqueeze(-1)], dim=-1)  # (B,T,K,3)
        joint_h = self.joint_proj(joint_in)  # (B,T,K,D)

        w = conf_valid.unsqueeze(-1)  # (B,T,K,1)
        w_sum = w.sum(dim=2).clamp_min(1e-6)  # (B,T,1) -- KHONG keepdim de khop chieu voi (B,T,D) sau khi sum(dim=2)
        frame_vec = (joint_h * w).sum(dim=2) / w_sum  # (B,T,D) -- pool 17 khop -> 1 vector/frame

        if appearance_gamma is not None and appearance_beta is not None:
            frame_vec = frame_vec * (1.0 + appearance_gamma) + appearance_beta

        frame_vec = frame_vec * frame_mask.float().unsqueeze(-1)

        # nn.LSTM goi flatten_parameters() moi lan forward, ham nay SUA TAI CHO (inplace)
        # cac weight tensor de gop chung lai cho cuDNN. Neu ham nay chay ben trong
        # torch.inference_mode() (vd trong benchmark_pipeline/analyze_social_interaction,
        # hoac code inference nguoi dung tu viet sau nay), PyTorch se bao loi "Inplace
        # update to inference tensor outside InferenceMode is not allowed" -- dac biet
        # de gap hon khi dung chung voi nn.DataParallel (replica model tren tung GPU
        # duoc tao lai moi lan forward). torch.inference_mode(False) "mo khoa" tam thoi
        # ngay tai day, bat ke ngu canh ben ngoai dang bat inference_mode hay khong,
        # nen luon an toan du goi tu dau (train, torch.no_grad(), hay torch.inference_mode()).
        with torch.inference_mode(False):
            lstm_out, _ = self.lstm(frame_vec)  # (B,T,2*hidden_dim)

        # Attention pooling co mask: frame khong hop le (frame_mask=False) khong duoc chu y toi.
        all_invalid = ~frame_mask.any(dim=1)
        safe_frame_mask = frame_mask.clone()
        if all_invalid.any():
            safe_frame_mask[all_invalid, 0] = True
        key_padding_mask = ~safe_frame_mask

        query = self.attn_query.expand(B, -1, -1)
        pooled, _ = self.attn_pool(query, lstm_out, lstm_out, key_padding_mask=key_padding_mask)
        pooled = pooled.squeeze(1)

        # Sample khong co frame skeleton hop le nao -> vector 0 (giong cach body/face
        # xu ly modality vang mat trong cac notebook truoc).
        has_valid = frame_mask.any(dim=1).float().unsqueeze(-1)
        pooled = pooled * has_valid

        return self.out_norm(pooled)
