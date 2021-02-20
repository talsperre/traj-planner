import os
import csv
import json
import yaml
import time
import warnings
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from models.PEC_CSP import PEC_CSP_Net
from models.CSPEncoder import CSPEncoder
from utils.ngsim_dataloader import ngsimDataset


# CUDA / CPU Device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# Initialize training, val and test dataset
batch_size = 128
train_dataset = ngsimDataset(mat_file="./datasets/v3/TrainSet.mat", t_h=32, t_f=48, d_s=4)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=train_dataset.collate_fn)

val_dataset = ngsimDataset(mat_file="./datasets/v3/ValSet.mat", t_h=32, t_f=48, d_s=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, collate_fn=val_dataset.collate_fn)

test_dataset = ngsimDataset(mat_file="./datasets/v3/TestSet.mat", t_h=32, t_f=48, d_s=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, collate_fn=test_dataset.collate_fn)

# Loss function
def calculate_loss(x, reconstructed_x, mean, log_var):
    # Weights
    w1 = torch.tensor([10, 1]).to(device)
    
    # reconstruction loss
    RCL_dest = torch.mean(w1 * (x - reconstructed_x) ** 2)

    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return RCL_dest, KLD

# Load hyperparameters
def load_hyper_parameters(file_name='optimal.yaml'):
    with open(file_name, 'r') as file:
        hyper_params = yaml.load(file)
    
    return hyper_params

# Network parameters
hyper_params = load_hyper_parameters()
hyper_params['encoder_size'] = 64
hyper_params['grid_size'] = (13, 3)
hyper_params['soc_conv_depth'] = 64
hyper_params['conv_3x1_depth'] = 16
hyper_params['dyn_embedding_size'] = 32
hyper_params['input_embedding_size'] = 32
hyper_params['num_epochs'] = 20
hyper_params['lr'] = 0.001
# hyper_params['lr'] = 0.0005
hyper_params['weight_decay'] = 0.01

# Make Directory Structure to Save the Models and plots:
base_dir = os.path.dirname(os.path.realpath(__file__))
exp_dir = os.path.join(base_dir, 'cache', time.strftime("%d_%m_%Y_%H_%M"))
loss_dir = os.path.join(exp_dir, 'loss')
metrics_dir = os.path.join(exp_dir, 'metrics')
os.makedirs(exp_dir, exist_ok=True)
os.makedirs(loss_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# Save the command line arguments
with open(os.path.join(exp_dir, 'args.json'), 'w') as fp:
    json.dump(hyper_params, fp, indent=4)

# Initialize network
model = PEC_CSP_Net(
    hyper_params["enc_dest_size"],
    hyper_params["enc_latent_size"],
    hyper_params["dec_size"],
    hyper_params["fdim"],
    hyper_params["zdim"],
    hyper_params["sigma"],
    hyper_params,
    device,
    verbose=True
)

model = model.double().to(device)
optimizer = optim.Adam(model.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay'])

def train(print_every=250):
    model.train()
    train_loss = 0
    total_rcl, total_kld = 0, 0
    num_datapoints = 0
    for i, data in enumerate(train_loader):
        hist, nbrs, mask, _, _, fut, _ = data

        hist = hist.double().to(device)
        fut = fut.double().to(device)
        nbrs = nbrs.double().to(device)
        mask = mask.to(device)
        dest = fut[-1, :, :]

        dest_recon, mu, var = model.forward(hist, nbrs, mask, dest, device)

        optimizer.zero_grad()
        rcl, kld = calculate_loss(dest, dest_recon, mu, var)
        loss = rcl + kld * hyper_params["kld_reg"]
        loss.backward()

        train_loss += loss.item()
        total_rcl += rcl.item()
        total_kld += kld.item()
        num_datapoints += hist.size(1)
        optimizer.step()
        if i % print_every == 0:
            print("Batch: {}, train_loss: {}, total_rcl: {}, total_kld: {}".format(i, loss.item(), rcl.item(), kld.item()))
    return train_loss, total_rcl, total_kld, num_datapoints

def val_test(dataloader, best_of_n=20, print_every=50):
    model.eval()
    l2error_avg_dest_list = []
    l2error_dest_list = []
    best_fde, mean_fde = [], []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            hist, nbrs, mask, _, _, fut, _ = data
            if nbrs.size()[1] == 0:
                continue

            hist = hist.double().to(device)
            fut = fut.double().to(device)
            nbrs = nbrs.double().to(device)
            mask = mask.to(device)
            dest = fut[-1, :, :]
            dest = dest.cpu().numpy()

            all_l2_errors_dest = []
            all_guesses = []
            for _ in range(best_of_n):
                dest_recon = model.forward(hist, nbrs, mask, dest=None, device=device)
                dest_recon = dest_recon.cpu().numpy()
                all_guesses.append(dest_recon)
                l2error_sample = np.linalg.norm(dest_recon - dest, axis = 1)
                all_l2_errors_dest.append(l2error_sample)

            all_l2_errors_dest = np.array(all_l2_errors_dest)
            all_guesses = np.array(all_guesses)
            # average error
            l2error_avg_dest = np.mean(all_l2_errors_dest)
            l2error_avg_dest_list.append(l2error_avg_dest)
            # taking the minimum error out of all guess
            l2error_dest = np.mean(np.min(all_l2_errors_dest, axis = 0))
            l2error_dest_list.append(l2error_dest)

            best_fde.append(np.min(all_l2_errors_dest, axis = 0))
            mean_fde.append(np.mean(all_l2_errors_dest, axis = 0))

            if i % print_every == 0:
                print('Batch: {}, Test time error in destination best: {:0.3f} and mean: {:0.3f}'.format(i, l2error_dest, l2error_avg_dest))
        best_fde = np.concatenate(best_fde)
        mean_fde = np.concatenate(mean_fde)
        print("Best destination error (mean): {}, Mean desitation error : {}".format(np.mean(best_fde), np.mean(mean_fde)))
        return np.mean(best_fde), np.mean(mean_fde)

train_data_csv = os.path.join(exp_dir, 'train_loss.csv')
test_data_csv = os.path.join(exp_dir, 'test_loss.csv')

best_test_loss = 1e5
epoch_train_loss = []
epoch_test_loss = []

with open(train_data_csv, 'w') as f1, open(test_data_csv, 'w') as f2:
    wr1 = csv.writer(f1)
    wr2 = csv.writer(f2)
    wr1.writerow(['Epoch', 'Train Loss', 'Total RCL Loss', 'Total KLD Loss', 'Num Data Points'])
    wr2.writerow(['Epoch', 'L2 Error Best', 'L2 Error Mean'])
    for epoch in range(hyper_params['num_epochs']):
        print("-"*100)
        train_loss, total_rcl, total_kld, num_datapoints = train()
        print("-"*50)
        test_l2error_best, test_l2error_mean = val_test(test_loader, best_of_n=20)
        # Saving models
        if test_l2error_best < best_test_loss:
            best_test_loss = test_l2error_best
            model_path = os.path.join(exp_dir, 'model_{}.pth'.format(epoch))
            torch.save({
                'hyper_params': hyper_params,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, model_path)
        
        print("-"*50)
        print("Epoch: {}, train_loss: {}, total_rcl: {}, total_kld: {}".format(epoch, train_loss / num_datapoints, total_rcl / num_datapoints, total_kld / num_datapoints))
        print("Epoch: {}, test_l2error_best: {}, test_l2error_mean: {}".format(epoch, test_l2error_best, test_l2error_mean))
        print("Best Test Loss: {}".format(best_test_loss))

        # Log the losses into files
        epoch_train_loss.append([epoch, train_loss, total_rcl, total_kld, num_datapoints])
        epoch_test_loss.append([epoch, test_l2error_best, test_l2error_mean])
        wr1.writerow(epoch_train_loss[-1])
        wr2.writerow(epoch_test_loss[-1])

        # Plotting
        if epoch % 5 == 0:
            plot_path = os.path.join(loss_dir, 'loss_{}.png'.format(epoch))
            fig, ax = plt.subplots(1)
            ax.plot(list(range(epoch+1)), np.log([a[1] for a in epoch_train_loss]), 'r')
            plt.ylabel('Train Loss (Log Scale)')
            plt.xlabel('Epochs')
            fig.savefig(plot_path)
            plt.close()

            plot_path = os.path.join(metrics_dir, 'FDE_{}.png'.format(epoch))
            fig, ax = plt.subplots(1)
            ax.plot(list(range(epoch+1)), [a[2] / a[4] for a in epoch_train_loss], 'r', label='Train RCL Loss')
            ax.plot(list(range(epoch+1)), [a[1] for a in epoch_test_loss], 'b', label='Test FDE')
            ax.legend()
            plt.ylabel('FDE')
            plt.xlabel('Epochs')
            fig.savefig(plot_path)
            plt.close()
