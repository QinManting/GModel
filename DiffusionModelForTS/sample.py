# /usr/bin/env python
# -*- coding: utf-8 -*-

# @file: sample.py
# @time: 2025/12/25 17:10:36
# @author: lemonlover
# @version: 1.0
# @eamil: 1920425406@qq.com
# @desc: Sample the data based on a trained model

import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from model.model_v3 import NoisePredictor

from evaluate import Evaluator
from utils import to_json_serializable

from forward import ForwardProcess
from dataset import MultiVarTimeSeriesDataset

from datetime import datetime, timedelta


class Sampler:
    """"""
    def __init__(self, arg_dict, model, fp) -> None:
        
        self.arg_dict = arg_dict
        self.model = model
        self.fp = fp

        # betas parameters of forward process
        self.betas = fp.betas
        self.sqrt_one_minus_alphas_cumprod = fp.sqrt_one_minus_alphas_cumprod
        self.sqrt_recip_alphas = fp.sqrt_recip_alphas
        self.sqrt_alphas_cumprod = fp.sqrt_alphas_cumprod
        # p[x(t-1)|x(t)]下的方差
        self.posterior_variance = fp.posterior_variance

        # stepsize
        self.stepsize = 30


    @torch.no_grad()
    def sample(self):
        B = self.arg_dict['num']
        C = 2
        L = 24
        T = self.arg_dict['T']

        # 初始化噪声
        xt = torch.randn(size=(B, C, L))
        
        # 条件输入
        season = torch.full((B,), self.arg_dict['season'], dtype=torch.long)

        for t in tqdm(reversed(range(T)), desc='Sampling'):
            t_batch = torch.full((B,), t, dtype=torch.long)

            # 预测当前噪声
            # noise_pred = self.model(xt, t_batch, season)
            noise_cond = self.model(xt, t_batch, season)
            null_season = torch.full_like(season, 4)
            noise_uncond = self.model(xt, t_batch, null_season)
            
            guidance_scale = self.arg_dict.get("guidance_scale", 3.0)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            

            # 估计 x0
            sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            x0_pred = (xt - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t

            if t > 0:
                # 后验均值 μ_t
                alpha_t = self.fp.alphas[t]
                alpha_cumprod_prev = self.fp.alphas_cumprod[t-1]
                beta_t = self.fp.betas[t]

                mu_t = (torch.sqrt(alpha_cumprod_prev) * beta_t / (1 - self.fp.alphas_cumprod[t])) * x0_pred \
                    + (torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev) / (1 - self.fp.alphas_cumprod[t])) * xt

                # 后验方差 σ_t
                sigma_t = torch.sqrt(self.fp.posterior_variance[t])

                # 采样 x_{t-1}
                xt = mu_t + sigma_t * torch.randn_like(xt)
            else:
                xt = x0_pred

        return xt



def get_timeseries_by_condition(
    dataset,
    season=None,
    denorm=True,
):
    """
    从 MultiVarTimeSeriesDataset 中：
    - 按条件筛选
    - 拼接 2 个变量
    - 可选反归一化

    return:
        X: (N, 2, 24)
        season: (N,)
    """
    X_list, season_list = [], []

    for i in range(len(dataset)):
        s = dataset.season[i]

        if season is not None and s != season:
            continue
        
        X_price, X_generation, season = dataset[i]
        X = torch.cat([X_price, X_generation], dim=0)  # (2, 24)

        X_list.append(X)
        season_list.append(torch.tensor(s))

    if len(X_list) == 0:
        raise ValueError("No samples match the given condition.")

    X = torch.stack(X_list)
    season = torch.stack(season_list)

    # ========= 反归一化 =========
    if denorm:
        stats = dataset.stats
        X[:, 0] = X[:, 0] * stats['std']['Price'] + stats['mean']['Price']
        X[:, 1] = X[:, 1] * stats['std']['Generation'] + stats['mean']['Generation']

    return X

def denormalize_timeseries(x, stats):
    """
    x: torch.Tensor, (B, 2, L), normalized
    stats: dataset.stats
    """
    x = x.clone()

    x[:, 0] = x[:, 0] * stats['std']['Price'] + stats['mean']['Price']
    x[:, 1] = x[:, 1] * stats['std']['Generation'] + stats['mean']['Generation']

    return x

def save_generated_by_condition(
    x_fake,
    season,
    save_root="./results/generated",
    save_format="csv",  # "npy" or "csv"
):
    """
    x_fake: torch.Tensor, (B, 2, 24), 已反归一化
    season: int
    """

    assert x_fake.ndim == 3 and x_fake.shape[1] == 2

    var_names = ["Price", "Generation"]

    # ===== 保存路径 =====
    save_dir = os.path.join(
        save_root,
        f"season_{season}"
    )
    os.makedirs(save_dir, exist_ok=True)

    x_fake = x_fake.cpu().numpy()  # (B, 2, 24)

    for i, var in enumerate(var_names):
        data = x_fake[:, i, :]  # (B, 24)

        file_path = os.path.join(save_dir, f"{var}.{save_format}")

        if save_format == "npy":
            np.save(file_path, data)

        elif save_format == "csv":
            # 每一行是一条样本（24小时）
            np.savetxt(
                file_path,
                data,
                delimiter=",",
                fmt="%.6f"
            )

        else:
            raise ValueError(f"Unsupported save format: {save_format}")

    print(f"Generated data saved to: {save_dir}")


def plot_generated_timeseries_single_season(
    x_fake,
    x_real=None,
    fake_indices=None,
    real_indices=None,
    season=None,
    results_root="./results",
   ):
    """
    x_fake: torch.Tensor, (Bf, 2, L)
    x_real: torch.Tensor, (Br, 2, L) or None
    fake_indices: list[int]
    real_indices: list[int]
    season: int or None
    """

    if fake_indices is None:
        fake_indices = list(range(min(100, x_fake.shape[0])))

    if x_real is not None and real_indices is None:
        real_indices = list(range(min(100, x_real.shape[0])))

    x_fake = x_fake.cpu().numpy()
    if x_real is not None:
        x_real = x_real.cpu().numpy()

    channels = ['Price', 'Generation']

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    
    for c in range(2):

        # ===== fake 数据 =====
        for idx in fake_indices:
            axes[c].plot(
                x_fake[idx, c],
                color='tab:blue',
                alpha=0.6,
                linewidth=2,
                label='Sample data' if idx == fake_indices[0] else None
            )

        # ===== real 数据 =====
        if x_real is not None:
            for idx in real_indices:
                axes[c].plot(
                    x_real[idx, c],
                    color='tab:orange',
                    alpha=0.6,
                    linewidth=2,
                    linestyle='--',
                    label='Real data' if idx == real_indices[0] else None
                )

        axes[c].set_title(channels[c])
        axes[c].set_ylabel(channels[c])
        axes[c].grid(True)
        axes[c].legend()

    axes[1].set_xlabel('Time step')
    # axes[2].set_xlabel('Time step')

    plt.tight_layout()
    plt.savefig(f'{results_root}/generated/season_{season}_fake.png', format="png", bbox_inches='tight', dpi=300)
    
# 在一张图上绘制season0-3的数据
def plot_generated_timeseries_all_seasons(
    x_fake,
    x_real=None,
    fake_indices=None,
    real_indices=None,
    results_root="./results",
   ):
    """
    x_fake: torch.Tensor, (Bf, 2, L)
    x_real: torch.Tensor, (Br, 2, L) or None
    fake_indices: list[int]
    real_indices: list[int]
    results_root: str
    """
    # 支持两种输入：
    # 1) list[Tensor]，长度为4，每个元素形状(B,2,L)
    # 2) Tensor，形状(B,2,L)，将复用于4个season（兼容旧调用）
    if isinstance(x_fake, list):
        x_fake_list = [xf.cpu().numpy() if torch.is_tensor(xf) else np.asarray(xf) for xf in x_fake]
    else:
        xf = x_fake.cpu().numpy() if torch.is_tensor(x_fake) else np.asarray(x_fake)
        x_fake_list = [xf for _ in range(4)]

    if x_real is None:
        x_real_list = None
    elif isinstance(x_real, list):
        x_real_list = [xr.cpu().numpy() if torch.is_tensor(xr) else np.asarray(xr) for xr in x_real]
    else:
        xr = x_real.cpu().numpy() if torch.is_tensor(x_real) else np.asarray(x_real)
        x_real_list = [xr for _ in range(4)]

    if fake_indices is None:
        min_bf = min(arr.shape[0] for arr in x_fake_list)
        fake_indices = list(range(min(100, min_bf)))

    if x_real_list is not None and real_indices is None:
        min_br = min(arr.shape[0] for arr in x_real_list)
        real_indices = list(range(min(100, min_br)))

    channels = ['Price', 'Generation']

    fig, axes = plt.subplots(2, 4, figsize=(32, 8), sharex=True)
    axes = axes.flatten()
    for c in range(2):

        for season in range(4):
            fake_arr = x_fake_list[season]
            # ===== fake 数据 =====
            for idx in fake_indices:
                axes[c*4 + season].plot(
                    fake_arr[idx, c],
                    color='tab:blue',
                    alpha=0.6,
                    linewidth=2,
                    label='Sample data' if idx == fake_indices[0] else None
                )

            # ===== real 数据 =====
            if x_real_list is not None:
                real_arr = x_real_list[season]
                for idx in real_indices:
                    axes[c*4 + season].plot(
                        real_arr[idx, c],
                        color='tab:orange',
                        alpha=0.6,
                        linewidth=2,
                        linestyle='--',
                        label='Real data' if idx == real_indices[0] else None
                    )

            axes[c*4 + season].set_title(f"{channels[c]} - Season {season}")
            axes[c*4 + season].set_ylabel(channels[c])
            axes[c*4 + season].grid(True)
            axes[c*4 + season].legend()
    axes[1].set_xlabel('Time step')

    plt.tight_layout()
    plt.savefig(f'{results_root}/generated/all_seasons_fake.png', format="png", bbox_inches='tight', dpi=300)

def main(arg_dict):
    """"""
    checkpoints = torch.load(arg_dict['checkpoints'], map_location='cpu', weights_only=True)
    model = NoisePredictor()
    model.load_state_dict(checkpoints)
    model.eval()
    dataset = MultiVarTimeSeriesDataset()
    
    # 从训练的模型中采样数据
    # forward process
    fp = ForwardProcess(arg_dict['T'])

    # sampler
    sampler = Sampler(arg_dict, model, fp)
    
    curtime = datetime.now() + timedelta(hours=0)
    results_root = f"./results/{curtime.strftime('%Y-%m-%d_%H-%M-%S')}" + f"/evaluation_results"


    if arg_dict['single_season']:
        x_fake = sampler.sample()
        # ===== 反归一化 =====
        x_fake1 = denormalize_timeseries(x_fake, dataset.stats)
        
        # 电价信息处理
        # 小于40的电价设为40，避免过低的电价导致评估指标失真
        x_fake1[:, 0] = np.clip(x_fake1[:, 0], a_min=40, a_max=None)  
        
        # 由模型得到的数据与真实数据进行比较
        # 评估
        evaluator = Evaluator()
        rea_norm = get_timeseries_by_condition(dataset, season=arg_dict["season"], denorm=False)
        results = evaluator.evaluate(rea_norm, x_fake)
        results_json = to_json_serializable(results)
        os.makedirs(results_root, exist_ok=True)
        with open(f"{results_root}/season_{arg_dict['season']}_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)

        # 保存数据
        save_generated_by_condition(
            x_fake1,
            season=arg_dict["season"],
            save_root=f"{results_root}/generated",
            save_format="csv",  # 或 "npy"
        )
        
        # plot
        # real data 按照特定条件筛选真实数据
        real = get_timeseries_by_condition(dataset, season=arg_dict['season'])
        plot_generated_timeseries_single_season(x_fake1, real, season=arg_dict["season"], results_root=results_root)
    else:
        # 同时生成4个季节的数据
        x_fake_seasons = []
        x_real_seasons = []
        for season in range(4):
            arg_dict['season'] = season
            x_fake = sampler.sample()
            x_fake1 = denormalize_timeseries(x_fake, dataset.stats)
            x_fake1[:, 0] = np.clip(x_fake1[:, 0], a_min=40, a_max=None)  
            x_fake_seasons.append(x_fake1)

            # 由模型得到的数据与真实数据进行比较
            # 评估
            evaluator = Evaluator()
            rea_norm = get_timeseries_by_condition(dataset, season=arg_dict["season"], denorm=False)
            results = evaluator.evaluate(rea_norm, x_fake)
            results_json = to_json_serializable(results)
            os.makedirs(results_root, exist_ok=True)
            with open(f"{results_root}/season_{arg_dict['season']}_evaluation.json", "w", encoding="utf-8") as f:
                json.dump(results_json, f, indent=2, ensure_ascii=False)
                
            save_generated_by_condition(
                x_fake1,
                season=arg_dict["season"],
                save_root=f"{results_root}/generated",
                save_format="csv",  # 或 "npy"
            )
            
            x_real = get_timeseries_by_condition(dataset, season=season)
            x_real_seasons.append(x_real)
            

        plot_generated_timeseries_all_seasons(x_fake_seasons, x_real_seasons, results_root=results_root)



if __name__ == '__main__': 

    arg_dict = {
        # 当前采用 linear 编码器，使用的训练数据是按照每天的顺序连接起来的
        "checkpoints": './logs/[05-28]14.19.27/model_1999.tar', 
        "num": 200,  # the number of data you want to generate
        "T": 200,
        "single_season": False,  # False: 同时生成4个季节的数据；True: 只生成一个季节的数据
        "season": 0,
        "guidance_scale": 3.0,  # classifier-free guidance的scale，越大越重条件，过大会失真
    }

    main(arg_dict)