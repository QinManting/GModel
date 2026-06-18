#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file: model_v2_1.py
# @time: 2026/05/26
# @author: qin manting
# @version: 2.1
# @desc: v2_1 - 增强版噪声预测网络

"""
模型原理：【时间序列+条件+timestep的条件transformer】，输出噪声ε̂

v2_1 改进点：
1.

"""

import math
import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """增强版时间步编码：sinusoidal + MLP"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 方式一：直接输出sinusoidal编码（不学习）
        # 方式二：加MLP增强非线性（推荐）
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        
        # 初始化：让MLP接近恒等映射，避免训练初期不稳定
        self.mlp[-1].weight.data.zero_()
        self.mlp[-1].bias.data.zero_()
    
    def forward(self, t):
        """
        t: (B,)
        return: (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device) *
            (-math.log(10000) / (half_dim - 1))
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # 经过MLP增强表达力
        emb = self.mlp(emb)
        
        return emb


class TimeStepEncoding(nn.Module):
    """时间步t的FiLM条件注入（scale + shift）"""
    
    def __init__(self, d_model):
        super().__init__()
        
        # 生成scale和shift的MLP
        self.scale_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.shift_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # 初始化：scale=0, shift=0，初始时不影响特征
        self.scale_net[-1].weight.data.zero_()
        self.scale_net[-1].bias.data.zero_()
        self.shift_net[-1].weight.data.zero_()
        self.shift_net[-1].bias.data.zero_()
    
    def forward(self, x, t_emb):
        """
        x: (B, N, d_model) 其中 N = num_vars * seq_len
        t_emb: (B, d_model)
        return: (B, N, d_model) 经scale+shift后的特征
        """
        scale = self.scale_net(t_emb)[:, None, :]   # (B, 1, d_model)
        shift = self.shift_net(t_emb)[:, None, :]   # (B, 1, d_model)
        
        # FiLM风格：x = x * (1 + scale) + shift
        return x * (1 + scale) + shift


# SeasonConditionEncoding 已替换为 Season Token（见 NoisePredictorV2 内联实现）


class NoisePredictorV2(nn.Module):
    """增强版噪声预测网络 v2"""
    
    def __init__(
        self,
        num_vars=2,
        seq_len=24,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.num_vars = num_vars
        self.seq_len = seq_len
        self.d_model = d_model

        # ========= 1. 输入投影：将2个变量特征映射到 d_model =========
        # 输入 (B, 24, 2) -> Linear(2, d_model) -> (B, 24, d_model)
        self.input_linear = nn.Sequential(
            nn.Linear(num_vars, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

        # ========= 3. 时间位置编码（sinusoidal，可外推）=========
        # 为24个时间步生成位置编码
        self.pos_encoding = SinusoidalTimeEmbedding(d_model)
        
        # ========= 4. 条件编码 =========
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        
        # FiLM风格的条件注入（仅用于time step）
        self.time_condition = TimeStepEncoding(d_model)
        # Season Token: 可学习的季节嵌入，作为特殊token拼接到序列中
        # 嵌入维度 = num_seasons + 1（+1 用于 CFG 无条件的 null token）
        self.season_token = nn.Embedding(5, d_model)
        # ========= 5. Transformer Encoder =========
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,  # 增加到4倍，与标准一致
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # ========= 6. 跳跃连接投影 =========
        self.skip_proj = nn.Linear(d_model * 2, d_model)

        # ========= 7. 输出投影 =========
        self.output_proj = nn.Linear(d_model, num_vars)
        
        # ========= 8. LayerNorm（可选） =========
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x_t, t, season):
        """
        x_t      : (B, num_vars, seq_len)  即 (B, 2, 24)
        t        : (B,)
        season   : (B,)

        return:
        eps_pred : (B, num_vars, seq_len)  即 (B, 2, 24)
        """
        B, C, L = x_t.shape
        assert C == self.num_vars and L == self.seq_len

        device = x_t.device

        # ========= 1. 转置输入：(B, 2, 24) -> (B, 24, 2) =========
        x_t_seq = x_t.permute(0, 2, 1)  # (B, 24, 2)

        # ========= 2. 保存跳跃连接的原始输入 =========
        skip_input = x_t_seq  # (B, 24, 2)

        # ========= 3. 输入投影：(B, 24, 2) -> (B, 24, d_model) =========
        tokens = self.input_linear(x_t_seq)  # (B, 24, d_model)

        # ========= 4. 加时间位置编码 =========
        pos_emb = self.pos_encoding(torch.arange(self.seq_len, device=device).float())
        # (seq_len, d_model) -> (1, seq_len, d_model)
        tokens = tokens + pos_emb[None, :, :]

        # ========= 5. 加入 Season Token =========
        season_emb = self.season_token(season)[:, None, :]  # (B, 1, d_model)
        tokens = torch.cat([season_emb, tokens], dim=1)  # (B, 25, d_model)

        # ========= 6. 时间步条件注入（FiLM） =========
        t_emb = self.time_emb(t)  # (B, d_model)
        tokens = self.time_condition(tokens, t_emb)

        # ========= 7. Transformer =========
        tokens = self.transformer(tokens)  # (B, 25, d_model)

        # ========= 8. 移除 Season Token =========
        tokens = tokens[:, 1:, :]  # (B, 24, d_model)

        # ========= 9. 跳跃连接 =========
        skip_feat = self.input_linear(skip_input)  # (B, 24, d_model)
        combined = torch.cat([skip_feat, tokens], dim=-1)  # (B, 24, 2*d_model)
        tokens = self.skip_proj(combined)  # (B, 24, d_model)

        # ========= 10. 输出投影 =========
        tokens = self.final_norm(tokens)
        out = self.output_proj(tokens)  # (B, 24, num_vars)

        # ========= 11. 转回原始格式：(B, 24, 2) -> (B, 2, 24) =========
        noises_pred = out.permute(0, 2, 1)  # (B, 2, 24)

        return noises_pred


# 为了保持与v1接口兼容
NoisePredictor = NoisePredictorV2