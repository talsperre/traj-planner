import os
import csv
import sys
import math
import time
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
import matplotlib.pyplot as plt

from model import INFER
from args import arguments
from inferDataset import inferDataset
from torch.utils.data import DataLoader
from utils import heatmap_accuracy, disp_tensor, disp_occ_grid

# Make Directory Structure to Save the Models:
base_dir = os.path.dirname(os.path.realpath(__file__))
exp_dir = os.path.join(base_dir, 'cache', time.strftime("%d_%m_%Y_%H_%M"))
loss_dir = os.path.join(exp_dir, 'loss')
plot_dir = os.path.join(exp_dir, 'plots')

os.makedirs(exp_dir, exist_ok=True)
os.makedirs(loss_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Save the command line arguments
with open(os.path.join(exp_dir, 'args.txt'), 'w') as args_file:
    for arg in vars(arguments):
        args_file.write(arg + ' ' + str(getattr(arguments, arg)) + '\n')

# Default CUDA tensor
# CUDA / CPU Device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
# torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Model
model = INFER(
    activation=arguments.activation,
    init_type=arguments.init_type,
    num_channels=arguments.num_channels,
    image_height=arguments.image_height,
    image_width=arguments.image_width,
)
model.init_weights()
model = model.to(device)

# Optimizer
optimizer = None
if arguments.opt_method == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=arguments.lr, betas=(arguments.beta1, arguments.beta2), weight_decay=arguments.weight_decay)
elif arguments.opt_method == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=arguments.lr, momentum=arguments.momentum, weight_decay=arguments.weight_decay, nesterov=False)
elif arguments.opt_method == 'amsgrad':
    optimizer = optim.Adam(model.parameters(), lr=arguments.lr, betas=(arguments.beta1, arguments.beta2), weight_decay=arguments.weight_decay, amsgrad=True)

# Loss Function
criterion = nn.MSELoss(reduction='sum')

# Datasets
train_dataset = inferDataset(mat_file=arguments.train_path, t_h=arguments.time_hist, t_f=arguments.time_fut, d_s=arguments.sampling_rate)
# train_loader = DataLoader(train_dataset, batch_size=arguments.batch_size, shuffle=True, num_workers=8)
train_loader = DataLoader(train_dataset, batch_size=arguments.batch_size, shuffle=False, num_workers=8)

test_dataset = inferDataset(mat_file=arguments.test_path, t_h=arguments.time_hist, t_f=arguments.time_fut, d_s=arguments.sampling_rate)
test_loader = DataLoader(test_dataset, batch_size=arguments.batch_size, num_workers=8)

num_train = len(train_dataset)
num_test = len(test_dataset)

# Loss function
def loss_fun(out, fut_occ_grid):
    weight_mat = torch.zeros_like(fut_occ_grid)
    weight_mat[fut_occ_grid != 0] = 1
    weight_mat = weight_mat.to(device)
    loss = torch.sum(weight_mat * (out - fut_occ_grid) ** 2)
    return loss

def train(model, dataloader, epoch, print_every=20):
    model.train()
    train_loss = 0
    train_fde = 0
    for i, data in enumerate(dataloader):
        ego_occ_grid, veh_occ_grid, fut_occ_grid, _, _ = data
        ego_occ_grid = ego_occ_grid.unsqueeze(2).float().to(device)
        veh_occ_grid = veh_occ_grid.unsqueeze(2).float().to(device)
        fut_occ_grid = fut_occ_grid.unsqueeze(1).float().to(device)

        out = model(ego_occ_grid, veh_occ_grid)
        print(out.size(), fut_occ_grid.size())

        optimizer.zero_grad()
        # loss = loss_fun(out, fut_occ_grid) + 0.005 * torch.sum(torch.abs(out))
        # loss = criterion(out, fut_occ_grid) # MSE between our prediction and gt grid
        # print("out.size(): {}, fut_occ_grid.size(): {}".format(out.size(), fut_occ_grid.size()))
        loss = criterion(out, fut_occ_grid) + 0.001 * torch.sum(torch.abs(out))
        fde = heatmap_accuracy(out, fut_occ_grid, device)
        
        loss.backward()
        train_loss += loss.item()
        train_fde += torch.sum(fde).item()
        optimizer.step()
        
        if i % print_every == 0:
            print("Num elements: {}".format(ego_occ_grid.size()[0]))
            print("Epoch: {}, Batch: {}, Training Loss: {} Training FDE: {}".format(epoch, i, loss.item(), torch.sum(fde).item() / ego_occ_grid.size()[0]))
            disp_occ_grid(out[2].cpu(), os.path.join(plot_dir, 'pred_{}_{}.png'.format(epoch, i)))
            disp_occ_grid(fut_occ_grid[2].cpu(), os.path.join(plot_dir, 'gt_{}_{}.png'.format(epoch, i)))
            if i >= 100:
                break
    return train_loss, train_fde

def val_test(model, dataloader, epoch, print_every=20):
    model.eval()
    test_fde = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            ego_occ_grid, veh_occ_grid, fut_occ_grid, _, _ = data
            ego_occ_grid = ego_occ_grid.unsqueeze(2).float().to(device)
            veh_occ_grid = veh_occ_grid.unsqueeze(2).float().to(device)
            fut_occ_grid = fut_occ_grid.unsqueeze(1).float().to(device)

            out = model(ego_occ_grid, veh_occ_grid)

            fde = heatmap_accuracy(out, fut_occ_grid, device)
            test_fde += torch.sum(fde).item()

            if i % print_every == 0:
                print("Epoch: {}, Batch: {}, Testing FDE: {}".format(epoch, i, torch.sum(fde).item() / ego_occ_grid.size()[0]))
    return test_fde

def run(model, train_loader, test_loader, num_epochs):
    epoch_train = [["Epoch", "Train Loss", "Train FDE"]]
    epoch_test = [["Epoch", "Test FDE"]]
    best_epoch_loss = 1e8
    for epoch in range(num_epochs):
        print("-"*100)
        train_loss, train_fde = train(model, train_loader, epoch, 20)
        # break
        # test_fde = val_test(model, test_loader, epoch)
        # test_fde = -1
        # print("-"*50)
        # print("Epoch: {}, Train Loss: {}, Train FDE: {}".format(epoch, train_loss / num_train, train_fde / num_train))
        # print("Epoch: {}, Test FDE: {}".format(epoch, test_fde / num_test))

        # epoch_train.append([epoch, train_loss / num_train, train_fde / num_train])
        # epoch_test.append([epoch, test_fde / num_test])

        # test_fde_avg = test_fde / num_test
        # if test_fde_avg < best_epoch_loss:
        #     best_epoch_loss = test_fde_avg
        #     model_path = os.path.join(exp_dir, 'model_{}.pth'.format(epoch))
        #     torch.save({
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict()
        #     }, model_path)
    return epoch_train, epoch_test

epoch_train, epoch_test = run(model, train_loader, test_loader, arguments.num_epochs)

with open(os.path.join(loss_dir, "train.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerows(epoch_train)

with open(os.path.join(loss_dir, "test.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerows(epoch_test)
