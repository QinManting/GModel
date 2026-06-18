# /usr/bin/env python
# -*- coding: utf-8 -*-

# @file: main_v2_1.py
# @time: 2025/06/17
# @author: qin manting
# @version: 2.1
# @desc: the train script


import os
from tqdm import tqdm
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.model_v2_1 import NoisePredictor
from forward import ForwardProcess
from dataset import MultiVarTimeSeriesDataset
from utils import bulid_log_dir


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

            # ========= CFG: Classifier-Free Guidance 训练 =========
            # 以 p_uncond 概率随机丢弃季节条件（替换为 null token id=4）
            p_uncond = 0.05
            drop_mask = torch.rand(season.shape[0], device=self.device) < p_uncond
            season_cfg = season.clone()
            season_cfg[drop_mask] = 4  # null token id

            # backward process 使用模型预测噪声！（重点）
            noise_pred = self.model(noisy_X, t, season_cfg)

            loss = self.loss_func(noises, noise_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录损失
            loss_list.append(loss.detach().cpu().item())
        
        return float(sum(loss_list) / len(loss_list))


    def train(self):
        """Train the model."""
        for epoch in range(self.hyper_params['num_epochs']):
            # train for epoch
            loss = self._train_for_epoch(epoch)
            if (epoch+1) % 500 == 0:
                self._save_model(epoch, min=True)
            # 记录日志
            self.writer.add_scalar(tag='loss', scalar_value=loss, global_step=epoch)


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