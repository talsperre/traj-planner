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
import torch.multiprocessing

from torch.utils.data import DataLoader

from models.PEC_CSP import PEC_CSP_Net
from models.CSPEncoder import CSPEncoder
# from utils.ngsim_dataloader import ngsimDataset
from utils.pecnet_dataloader import ngsimDataset

torch.multiprocessing.set_sharing_strategy('file_system')

# CUDA / CPU Device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# Initialize training, val and test dataset
# batch_size = 128
# train_dataset = ngsimDataset(mat_file="./datasets/v3/TrainSet.mat", t_h=32, t_f=48, d_s=4)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=train_dataset.collate_fn)

# val_dataset = ngsimDataset(mat_file="./datasets/v3/ValSet.mat", t_h=32, t_f=48, d_s=4)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, collate_fn=val_dataset.collate_fn)
batch_size = 128
test_dataset = ngsimDataset(mat_file="./datasets/v3/TestSet.mat", t_h=32, t_f=48, d_s=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, collate_fn=test_dataset.collate_fn)
print("Num Test: {}".format(len(test_dataset)))

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
hyper_params['num_epochs'] = 5
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

# Load pretrained model and checkpoint
checkpoint_dir = "/home/talsperre/IROS21/traj-planner/cache/14_02_2021_10_29/"
checkpoint = torch.load(os.path.join(checkpoint_dir, "model_2.pth"), map_location=device)
hyper_params = checkpoint["hyper_params"]
model = model.double().to(device)
model.load_state_dict(checkpoint["model_state_dict"])

def val_test(dataloader, best_of_n=20, print_every=50):
    model.eval()
    l2error_avg_dest_list = []
    l2error_dest_list = []
    best_fde, mean_fde = [], []
    predicted_goals = []
    ref_pos_list = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            hist, nbrs, mask, _, _, fut, _, ref_pos = data
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
            all_guesses_reshaped = np.swapaxes(all_guesses, 0, 1)
            predicted_goals.append(all_guesses_reshaped)
            ref_pos_list.append(ref_pos.cpu().numpy())

            if i % print_every == 0:
                print('Batch: {}, Test time error in destination best: {:0.3f} and mean: {:0.3f}'.format(i, l2error_dest, l2error_avg_dest))
        best_fde = np.concatenate(best_fde)
        mean_fde = np.concatenate(mean_fde)
        predicted_goals = np.concatenate(predicted_goals)
        ref_pos_list = np.concatenate(ref_pos_list)
        print("Best destination error (mean): {}, Mean desitation error : {}".format(np.mean(best_fde), np.mean(mean_fde)))
        return np.mean(best_fde), np.mean(mean_fde), predicted_goals, ref_pos_list

# best_fde, mean_fde, predicted_goals, ref_pos_list = val_test(test_loader, best_of_n=20, print_every=50)
best_fde, mean_fde, predicted_goals, ref_pos_list = val_test(test_loader, best_of_n=100, print_every=50)
print("Test time error in destination best: {:0.3f} and mean: {:0.3f}".format(best_fde, mean_fde))
print("predicted_goals.shape: {}, ref_pos_list.shape: {}".format(predicted_goals.shape, ref_pos_list.shape))

np.save(os.path.join(checkpoint_dir, "predicted_goals_100_pred.npy"), predicted_goals)
np.save(os.path.join(checkpoint_dir, "ref_pos_100_pred.npy"), ref_pos_list)