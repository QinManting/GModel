# /usr/bin/env python
# -*- coding: utf-8 -*-

# @file: main.py
# @time: 2025/12/25 14:57:43
# @author: lemonlover
# @version: 1.0
# @eamil: 1920425406@qq.com
# @desc: the train script


import os
from tqdm import tqdm
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.model_v2_2 import NoisePredictor
from forward import ForwardProcess
from dataset import MultiVarTimeSeriesDataset
from utils import bulid_log_dir


def pearson_corr(x, y):
    """
    计算 x, y 沿最后一维的 Pearson 相关系数
    x, y: (B, L)
    return: (B,) 每个样本的相关系数
    """
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    x_c = x - x_mean
    y_c = y - y_mean
    denom = torch.sqrt((x_c ** 2).sum(dim=-1) * (y_c ** 2).sum(dim=-1) + 1e-8)
    return (x_c * y_c).sum(dim=-1) / denom


class Trainer:
    """"""
    def __init__(self, hyper_params, fp, model, optimizer, train_loader, device):
        """"""
        self.hyper_params = hyper_params
        self.train_loader = train_loader
        self.device = device
        self.fp = fp
        self.model = model
        self.optimizer = optimizer
        # tensorboard writer
        self.writer = SummaryWriter(self.hyper_params['log_dir'])
        # loss func 均方误差损失函数
        self.loss_func = F.mse_loss

    def _train_for_epoch(self, epoch):
        """"""
        loop = tqdm(self.train_loader, total=len(self.train_loader), leave=False)
        loop.set_description(f'epoch {epoch}')
        loss_list = []

        # 从数据加载器中获取
        for (X_price, X_generation, season) in loop:
            X_price = X_price.to(self.device)
            X_generation = X_generation.to(self.device)
            season = season.to(self.device)

            # forward process
            # generate the random t
            """
            我们在训练时，所用的t是随机采样的，这样做的目的是：让模型在所有噪声强度上都学会去噪
            为什么t不是一步一步增加？
            从概率模型的角度，训练不是在模拟轨迹，而是在学习条件分布
            从优化目标的角度来看，随机采样的t才是无偏估计
            """
            # 将4-variate的时间序列数据进行拼接，作为模型的输入
            # 最终模型学习到的是4-variate的联合概率分布，即学习了时间相关性与变量相关性
            # concat
            X = torch.concat([X_price, X_generation], dim=1)
            t = torch.randint(0, self.hyper_params["T"], size=(X.shape[0], )).to(self.device)
            # forward process 前向加噪
            noisy_X, noises = self.fp(X, t)

            # backward process 使用模型预测噪声
            noise_pred = self.model(noisy_X, t, season)

            # ---- loss_noise: 噪声预测 MSE ----
            loss_noise = self.loss_func(noises, noise_pred)

            # ---- loss_corr: 联合分布约束 ----
            # 从噪声预测恢复 X_0_hat
            sqrt_alpha_bar_t = self.fp.get_index_from_list(
                self.fp.sqrt_alphas_cumprod, t, noisy_X.shape
            )
            sqrt_one_minus_alpha_bar_t = self.fp.get_index_from_list(
                self.fp.sqrt_one_minus_alphas_cumprod, t, noisy_X.shape
            )
            X_0_hat = (noisy_X - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t

            # 真实数据的 Price-Generation 相关性
            corr_real = pearson_corr(X[:, 0, :], X[:, 1, :]).mean()
            # 模型预测的 Price-Generation 相关性
            corr_fake = pearson_corr(X_0_hat[:, 0, :], X_0_hat[:, 1, :]).mean()
            loss_corr = (corr_real - corr_fake) ** 2

            # ---- 总损失 ----
            loss = loss_noise + self.hyper_params['corr_weight'] * loss_corr

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.detach().cpu().item())
        
            loss_noise_val = loss_noise.detach().cpu().item()
            loss_corr_val = loss_corr.detach().cpu().item()
            corr_real_val = corr_real.detach().cpu().item()
            corr_fake_val = corr_fake.detach().cpu().item()

        return float(sum(loss_list) / len(loss_list)), loss_noise_val, loss_corr_val, corr_real_val, corr_fake_val



    def train(self):
        """Train the model."""
        for epoch in range(self.hyper_params['num_epochs']):
            # train for epoch
            loss, loss_noise_val, loss_corr_val, corr_real_val, corr_fake_val = self._train_for_epoch(epoch)
            if (epoch+1) % 500 == 0:
                self._save_model(epoch, min=True)
            # 记录日志
            self.writer.add_scalar(tag='loss', scalar_value=loss, global_step=epoch)
            self.writer.add_scalar(tag='loss_noise', scalar_value=loss_noise_val, global_step=epoch)
            self.writer.add_scalar(tag='loss_corr', scalar_value=loss_corr_val, global_step=epoch)
            self.writer.add_scalar(tag='corr_real', scalar_value=corr_real_val, global_step=epoch)
            self.writer.add_scalar(tag='corr_fake', scalar_value=corr_fake_val, global_step=epoch)


    def _save_model(self, epoch, min=False):
        """Save the model."""
        if not os.path.exists(self.hyper_params['log_dir']):
            os.makedirs(self.hyper_params['log_dir'])
        checkpoints = self.model.state_dict()
        path = self.hyper_params['log_dir'] + f'/model_{epoch}.tar'
        print(f'==> Saving checkpoints: {epoch}')
        torch.save(checkpoints, path)


# 基于真实数据进行训练
# hyper parameter
hyper_params = {
    "T": 200,   # time steps
    "batch_size": 16,
    "lr": 2e-5,
    "num_epochs": 5000,
    "weight_decay": 1e-4,
    "corr_weight": 0.2,
    "log_dir": bulid_log_dir(dir='./logs')
}


# create the dataset
dataset = MultiVarTimeSeriesDataset()
train_loader = DataLoader(dataset, batch_size=hyper_params['batch_size'], shuffle=False)
# gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# forward process model
fp = ForwardProcess(hyper_params["T"]).to(device)

# noise predictor
model = NoisePredictor().to(device)
# optimizer
optimizer = optim.AdamW(model.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay'])

# trainer = Trainer(hyper_params, fp, model, optimizer, train_loader, val_loader, device)
trainer = Trainer(hyper_params, fp, model, optimizer, train_loader, device) 
trainer.train()

print(sum(p.numel() for p in model.parameters() if p.requires_grad))