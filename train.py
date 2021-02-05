import yaml
import numpy as np

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from models.PEC_CSP import PEC_CSP_Net
from models.CSPEncoder import CSPEncoder
from utils.ngsim_dataloader import ngsimDataset


# CUDA / CPU Device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Initialize training, val and test dataset
batch_size = 128
train_dataset = ngsimDataset(mat_file="./datasets/v2/TrainSet.mat", t_h=32, t_f=48, d_s=4)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=train_dataset.collate_fn)

val_dataset = ngsimDataset(mat_file="./datasets/v2/ValSet.mat", t_h=32, t_f=48, d_s=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, collate_fn=val_dataset.collate_fn)

test_dataset = ngsimDataset(mat_file="./datasets/v2/TestSet.mat", t_h=32, t_f=48, d_s=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, collate_fn=test_dataset.collate_fn)

# Network parameters
CSP_args = {}
CSP_args['encoder_size'] = 64
CSP_args['grid_size'] = (13, 3)
CSP_args['soc_conv_depth'] = 64
CSP_args['conv_3x1_depth'] = 16
CSP_args['dyn_embedding_size'] = 32
CSP_args['input_embedding_size'] = 32

# Loss function
def calculate_loss(x, reconstructed_x, mean, log_var):
    # Weights
    w1 = torch.tensor([10, 1]).to(device)
    w2 = torch.tensor([10, 1] * 11).to(device)
    
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

hyper_params = load_hyper_parameters()

# Initialize network
model = PEC_CSP_Net(
    hyper_params["enc_dest_size"],
    hyper_params["enc_latent_size"],
    hyper_params["dec_size"],
    hyper_params["fdim"],
    hyper_params["zdim"],
    hyper_params["sigma"],
    CSP_args,
    device,
    verbose=True
)

model = model.double().to(device)
optimizer = optim.Adam(model.parameters())

def train():
    model.train()
    train_loss = 0
    total_rcl, total_kld = 0, 0
    for i, data in enumerate(train_loader):
        hist, nbrs, mask, _, _, fut, _ = data

        hist = hist.double().to(device)
        fut = fut.double().to(device)
        nbrs = nbrs.double().to(device)
        dest = fut[-1, :, :]

        dest_recon, mu, var = model.forward(hist, nbrs, mask, dest, device)

        optimizer.zero_grad()
        rcl, kld = calculate_loss(dest, dest_recon, mu, var)
        loss = rcl + kld * hyper_params["kld_reg"]
        loss.backward()

        train_loss += loss.item()
        total_rcl += rcl.item()
        total_kld += kld.item()
        print("batch_num: {}, train_loss: {}, total_rcl: {}, total_kld: {}".format(i, loss.item(), rcl.item(), kld.item()))
        optimizer.step()
        if i >= 30:
            break
    return train_loss, total_rcl, total_kld

def val_test(dataloader, best_of_n=20):
    model.eval()
    l2error_avg_dest_list = []
    l2error_dest_list = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            hist, nbrs, mask, _, _, fut, _ = data

            hist = hist.double().to(device)
            fut = fut.double().to(device)
            nbrs = nbrs.double().to(device)
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
            
            print('Batch: {}, Test time error in destination best: {:0.3f} and mean: {:0.3f}'.format(i, l2error_dest, l2error_avg_dest))
        mean_l2error_best = np.mean(l2error_dest_list)
        mean_l2error_avg = np.mean(l2error_avg_dest_list)
        print("Best destination error (mean): {}, Mean desitation error : {}".format(mean_l2error_best, mean_l2error_avg))
        return mean_l2error_best, mean_l2error_avg


for epoch in range(100):
    train_loss, total_rcl, total_kld = train()
    val_l2error_best, val_l2error_mean = val_test(val_loader, best_of_n=20)
    test_l2error_best, test_l2error_mean = val_test(test_loader, best_of_n=20)
    print("-"*50)
    print("Epoch Num: {}".format(epoch))
    print("train_loss: {}, total_rcl: {}, total_kld: {}".format(train_loss, total_rcl, total_kld))
    print("val_l2error_best: {}, val_l2error_mean: {}".format(val_l2error_best, val_l2error_mean))
    print("test_l2error_best: {}, test_l2error_mean: {}".format(test_l2error_best, test_l2error_mean))
    print("-"*100)
