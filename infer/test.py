from __future__ import print_function, division

import os
import torch
import numpy as np

from inferDataset import inferDataset
from torch.utils.data import Dataset, DataLoader
from model import INFER
from utils import heatmap_accuracy

train_dataset = inferDataset(mat_file="../datasets/v3/TrainSet.mat", t_h=20, t_f=40, d_s=2)

for i in range(len(train_dataset)):
    ego_occ_grid, veh_occ_grid, fut_occ_grid, lat_enc, lon_enc = train_dataset[i]
    if i == 1:
        print(ego_occ_grid.shape, veh_occ_grid.shape, fut_occ_grid.shape)
        break

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

model = INFER(activation='relu', init_type='default', num_channels=2, image_height=240, image_width=360)

for i, data in enumerate(train_loader):
    ego_occ_grid, veh_occ_grid, fut_occ_grid, lat_enc, lon_enc = data
    ego_occ_grid = ego_occ_grid.unsqueeze(2).float()
    veh_occ_grid = veh_occ_grid.unsqueeze(2).float()
    fut_occ_grid = fut_occ_grid.unsqueeze(1).float()
    res = model(ego_occ_grid, veh_occ_grid)
    print("res.size(): {}, fut_occ_grid.size(): {}".format(res.size(), fut_occ_grid.size()))
    # heatmap_accuracy(res.squeeze(1), fut_occ_grid.squeeze(1))
    # res = torch.zeros(*fut_occ_grid.shape)
    # print("res.shape: {}".format(res.shape))
    heatmap_accuracy(res, fut_occ_grid)
    break
