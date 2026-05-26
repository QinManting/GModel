# /usr/bin/env python
# -*- coding: utf-8 -*-

# @file: dataset.py
# @time: 2025/12/25 09:34:13
# @author: lemonlover
# @version: 1.0
# @eamil: 1920425406@qq.com
# @desc: data processing script, including load, normalization, etc


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

def Renewable_energy_with_price():
    df = pd.read_csv('./data/GS_2024_sorted.csv',usecols=['price', 'generation', 'season'])
    price = np.array(df['price'])
    generation = np.array(df['generation'])
    season = np.array(df['season'])

    # plt.figure(figsize=(12, 4))
    # plt.plot(pv, linewidth=1)
    # plt.savefig('./results/real_pv.svg', format="svg", bbox_inches='tight', dpi=300)

    # restruct (8760, ) --> (365, 24)
    price = price.reshape(-1,24)
    generation = generation.reshape(-1,24)
    # season_daily = np.array([season[i * 24] for i in range(366)])
    season_daily = season[::24]  # 每隔24行取一个元素，得到每天的季节标签

    return price, generation, season_daily

class MultiVarTimeSeriesDataset(Dataset):
    """
    Dataset for multivariatr time series with conditional labels.
    each sample is one day of 2-variate time series(Price, Generation), with conditions:
    season(0-3) and day_type(0=workday, 1=non-workday)
    """
    def __init__(self):
        # load data
        self.price, self.generation, self.season = Renewable_energy_with_price() # (365, 24) / # (365, 24) /# (365, )

        self.stats = {}
        self.stats['mean'] = {
            'Price': self.price.mean(),
            'Generation': self.generation.mean()
        }
        self.stats['std'] = {
            'Price': self.price.std(),
            'Generation': self.generation.std()
        }

        # normalize
        self.price = (self.price - self.stats['mean']['Price']) / self.stats['std']['Price']
        self.generation = (self.generation - self.stats['mean']['Generation']) / self.stats['std']['Generation']

    def __len__(self):
        return self.price.shape[0]  


    def __getitem__(self, idx):
        X_price = torch.tensor(self.price[idx], dtype=torch.float32).unsqueeze(0)
        X_generation = torch.tensor(self.generation[idx], dtype=torch.float32).unsqueeze(0)
        season = torch.tensor(self.season[idx], dtype=torch.long)
        return X_price, X_generation, season

if __name__ == "__main__":

    price, generation, season = Renewable_energy_with_price()
