#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file: model_v2.py
# @time: 2026/04/27
# @author: qin manting
# @version: 2.0
# @desc: v2 - 增强版噪声预测网络

"""
模型原理：【时间序列+条件+timestep的条件transformer】，输出噪声ε̂

v2 改进点：
1. 时间步编码增强：sinusoidal + MLP非线性映射，增强对t的响应
2. 条件注入方式改进：使用scale+shift（FiLM风格）替代简单相加
3. 增加跳跃连接：输入与输出拼接后投影，缓解梯度消失
4. 时间位置编码改用sinusoidal（更适合外推）
5. 可选：增加dropout正则化
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
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
        use_skip_connection=True,      # 是否使用跳跃连接
        use_film_condition=True,        # 是否使用FiLM条件注入
    ):
        super().__init__()

        self.num_vars = num_vars
        self.seq_len = seq_len
        self.d_model = d_model

        # ========= 1. 每个变量的特征提取 =========
        # 改为对每个时间点做逐点投影，保留时序信息
        # (B, L, 1) -> (B, L, d_model)
        self.var_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_vars)
        ])

        # ========= 2. 变量 embedding =========
        self.var_emb = nn.Embedding(num_vars, d_model)

        # ========= 3. 时间位置编码（sinusoidal，可外推）=========
        self.pos_encoding = SinusoidalTimeEmbedding(d_model)  # 复用sinusoidal实现
        
        # ========= 4. 条件编码 =========
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        
        # FiLM风格的条件注入
        self.time_condition = TimeStepEncoding(d_model)
        self.season_condition = SeasonConditionEncoding(4, d_model)
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
        # 将输入特征与Transformer输出拼接后投影
        self.input_proj = nn.Linear(1, d_model)
        self.skip_proj = nn.Linear(d_model * 2, d_model)
        
        # ========= 7. 输出投影 =========
        self.output_proj = nn.Linear(d_model, 1)
        
        # ========= 8. LayerNorm（可选） =========
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x_t, t, season):
        """
        x_t      : (B, num_vars, seq_len)
        t        : (B,)
        season   : (B,)

        return:
        eps_pred : (B, num_vars, seq_len)
        """
        B, C, L = x_t.shape
        assert C == self.num_vars and L == self.seq_len

        device = x_t.device

        # ========= 1. 保存跳跃连接的输入 =========
        # 将输入展平为 (B, N, 1) 用于后续拼接
        x_t_flat = x_t.reshape(B, C * L, 1)  # (B, N, 1)

        # ========= 2. 每个变量独立特征提取 =========
        var_tokens = []
        for i in range(self.num_vars):
            xi = x_t[:, i, :].unsqueeze(-1)      # (B, L, 1)
            hi = self.var_proj[i](xi)            # (B, L, d_model)
            var_tokens.append(hi)

        # (B, num_vars, seq_len, d_model)
        var_tokens = torch.stack(var_tokens, dim=1)

        # ========= 4. 加变量 embedding =========
        var_ids = torch.arange(self.num_vars, device=device)
        var_emb = self.var_emb(var_ids)[None, :, None, :]  # (1, num_vars, 1, d_model)
        var_tokens = var_tokens + var_emb

        # ========= 5. 加时间位置编码（sinusoidal）=========
        # 为每个时间步生成位置编码
        pos_emb = self.pos_encoding(torch.arange(self.seq_len, device=device).float())
        # (seq_len, d_model) -> (1, 1, seq_len, d_model)
        pos_emb = pos_emb[None, None, :, :]
        var_tokens = var_tokens + pos_emb

        # ========= 6. reshape 成 Transformer tokens =========
        # (B, num_vars * seq_len, d_model)
        tokens = var_tokens.reshape(B, self.num_vars * self.seq_len, self.d_model)

        # ========= 7. 条件注入 =========
        t_emb = self.time_emb(t)  # (B, d_model)
        
        # FiLM风格：scale + shift
        tokens = self.time_condition(tokens, t_emb)
        tokens = self.season_condition(tokens, season)

        # ========= 8. Transformer =========
        tokens = self.transformer(tokens)

        # ========= 9. 跳跃连接 =========
        # 提取原始输入的投影（对齐维度）
        # 将 x_t_flat 通过投影对齐到 d_model
        input_feat = self.input_proj(x_t_flat)  # (B, N, d_model)
        
        # 拼接并投影
        combined = torch.cat([input_feat, tokens], dim=-1)  # (B, N, 2*d_model)
        tokens = self.skip_proj(combined)
            
        # ========= 10. 输出投影 =========
        tokens = self.final_norm(tokens)
        
        out = self.output_proj(tokens)  # (B, N, 1)

        # 还原为 (B, num_vars, seq_len)
        noises_pred = out.view(B, self.num_vars, self.seq_len)

        return noises_pred


# 为了保持与v1接口兼容
NoisePredictor = NoisePredictorV2
