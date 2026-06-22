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


class SeasonConditionEncoding(nn.Module):
    """季节条件的FiLM注入"""
    
    def __init__(self, num_seasons=4, d_model=128):
        super().__init__()
        self.season_emb = nn.Embedding(num_seasons, d_model)
        
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
        
        # 初始化
        self.scale_net[-1].weight.data.zero_()
        self.scale_net[-1].bias.data.zero_()
        self.shift_net[-1].weight.data.zero_()
        self.shift_net[-1].bias.data.zero_()
    
    def forward(self, x, season):
        """
        x: (B, N, d_model)
        season: (B,)
        return: (B, N, d_model)
        """
        season_emb = self.season_emb(season)  # (B, d_model)
        scale = self.scale_net(season_emb)[:, None, :]
        shift = self.shift_net(season_emb)[:, None, :]
        
        return x * (1 + scale) + shift


class NoisePredictorV2(nn.Module):
    """增强版噪声预测网络 v2"""
    
    def __init__(
        self,
        num_vars=2,
        seq_len=24,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.num_vars = num_vars
        self.seq_len = seq_len
        self.d_model = d_model

        # ========= 1. 每个变量的特征提取（Cross-variable Encoder）=========
        # price_token = Price(24) -> Linear(24, d_model) -> (B, d_model)
        # gen_token   = Generation(24) -> Linear(24, d_model) -> (B, d_model)
        # 每个变量整体作为一个 token，而非24个时间步各为 token
        self.var_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_vars)
        ])

        # ========= 2. 变量 embedding（区分 price / generation）=========
        self.var_emb = nn.Embedding(num_vars, d_model)
        
        # ========= 4. 条件编码 =========
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        
        # FiLM风格的条件注入
        self.time_condition = TimeStepEncoding(d_model)
        self.season_condition = SeasonConditionEncoding(4, d_model)
        # Season Token: 可学习的季节嵌入，作为特殊token拼接到序列中
        self.season_token = nn.Embedding(4, d_model)
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

        # ========= 7. 输出投影：每个 token 还原为 24 步序列 =========
        self.output_proj = nn.Linear(d_model, seq_len)
        
        # ========= 8. LayerNorm（可选） =========
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x_t, t, season):
        """
        x_t      : (B, num_vars, seq_len)  即 (B, 2, 24)
        t        : (B,)
        season   : (B,)

        return:
        eps_pred : (B, num_vars, seq_len)  即 (B, 2, 24)

        Cross-variable Encoder:
        - 每个变量(24步)整体作为一个 token
        - Transformer 在 2 个变量 token 间做注意力，学习 P(price|generation)、P(generation|price)
        """
        B, C, L = x_t.shape
        assert C == self.num_vars and L == self.seq_len

        device = x_t.device

        # ========= 1. 每个变量独立编码为单个 token =========
        # price_token: (B, 24) -> (B, d_model)
        # gen_token  : (B, 24) -> (B, d_model)
        var_tokens = []
        for i in range(self.num_vars):
            xi = x_t[:, i, :]                  # (B, 24)
            hi = self.var_proj[i](xi)          # (B, d_model)
            var_tokens.append(hi)

        # (B, num_vars, d_model) 即 (B, 2, d_model)
        tokens = torch.stack(var_tokens, dim=1)
        skip_input = tokens  # 保存用于跳跃连接

        # ========= 2. 加变量 embedding（区分 price / generation）=========
        var_ids = torch.arange(self.num_vars, device=device)
        var_emb = self.var_emb(var_ids)[None, :, :]  # (1, num_vars, d_model)
        tokens = tokens + var_emb

        # ========= 3. 加入 Season Token =========
        season_emb = self.season_token(season)[:, None, :]  # (B, 1, d_model)
        tokens = torch.cat([season_emb, tokens], dim=1)  # (B, 3, d_model)

        # ========= 4. 时间步条件注入（FiLM）=========
        t_emb = self.time_emb(t)  # (B, d_model)
        tokens = self.time_condition(tokens, t_emb)
        tokens = self.season_condition(tokens, season)

        # ========= 5. Transformer（cross-variable attention）=========
        tokens = self.transformer(tokens)  # (B, 3, d_model)

        # ========= 6. 移除 Season Token =========
        tokens = tokens[:, 1:, :]  # (B, 2, d_model)

        # ========= 7. 跳跃连接 =========
        combined = torch.cat([skip_input, tokens], dim=-1)  # (B, 2, 2*d_model)
        tokens = self.skip_proj(combined)  # (B, 2, d_model)

        # ========= 8. 输出投影：每个 token 还原为 24 步 =========
        tokens = self.final_norm(tokens)
        out = self.output_proj(tokens)  # (B, 2, 24)

        return out


# 为了保持与v1接口兼容
NoisePredictor = NoisePredictorV2