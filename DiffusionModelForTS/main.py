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

from model.model_v2 import NoisePredictor
from forward import ForwardProcess
from dataset import MultiVarTimeSeriesDataset
from utils import bulid_log_dir




class Trainer:
    """"""
    def __init__(self, hyper_params, fp, model, optimizer, train_loader, val_loader, device):
        """"""
        self.hyper_params = hyper_params
        self.train_loader = train_loader
        self.val_loader = val_loader
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

            # backward process 使用模型预测噪声！（重点）
            noise_pred = self.model(noisy_X, t, season)

            loss = self.loss_func(noises, noise_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录损失
            loss_list.append(loss.detach().cpu().item())
        
        return float(sum(loss_list) / len(loss_list))

    @torch.no_grad()
    def _validate_for_epoch(self, epoch):
        """Compute validation loss for current epoch."""
        self.model.eval()
        loss_list = []
        for (X_price, X_generation, season) in self.val_loader:
            X_price = X_price.to(self.device)
            X_generation = X_generation.to(self.device)
            season = season.to(self.device)

            X = torch.concat([X_price, X_generation], dim=1)
            t = torch.randint(0, self.hyper_params["T"], size=(X.shape[0], )).to(self.device)
            noisy_X, noises = self.fp(X, t)
            noise_pred = self.model(noisy_X, t, season)
            loss = self.loss_func(noises, noise_pred)
            loss_list.append(loss.detach().cpu().item())

        self.model.train()
        return float(sum(loss_list) / len(loss_list))


    def train(self):
        """"""
        best_val = float('inf')
        patience = 0
        early_stop_patience = self.hyper_params.get('early_stop_patience', 50)

        for epoch in range(self.hyper_params['num_epochs']):

            # train for epoch
            train_loss = self._train_for_epoch(epoch)

            # validate
            val_loss = self._validate_for_epoch(epoch)

            # 记录日志到 tensorboard
            self.writer.add_scalar('loss/train', train_loss, epoch)
            self.writer.add_scalar('loss/val', val_loss, epoch)

            # 保存最优模型并实现早停
            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                self._save_model(epoch, min=True)
            else:
                patience += 1

            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}, best_val={best_val:.6f}")
                break

    def _save_model(self, epoch, min=False):
        """
        """
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
    "log_dir": bulid_log_dir(dir='./logs'),
    "val_ratio": 0.15,
    "early_stop_patience": 5000
}


# create the dataset
dataset = MultiVarTimeSeriesDataset()
# 在此处将dataset按时间序列切分为训练/验证集，设置batch_size和shuffle等参数
N = len(dataset)
val_ratio = hyper_params.get('val_ratio')
val_size = int(N * val_ratio)
train_size = N - val_size
# 时间序列切分：前 train_size 为训练，后 val_size 为验证
train_indices = list(range(0, train_size))
val_indices = list(range(train_size, N))
from torch.utils.data import Subset
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=hyper_params['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hyper_params['batch_size'], shuffle=False)
# gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# forward process model
fp = ForwardProcess(hyper_params["T"]).to(device)

# noise predictor
model = NoisePredictor().to(device)
# optimizer
optimizer = optim.AdamW(model.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay'])

trainer = Trainer(hyper_params, fp, model, optimizer, train_loader, val_loader, device)
trainer.train()