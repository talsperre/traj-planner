import os
import csv
import math
import time
import random
import pickle

import yaml
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils import data
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.distributions.normal import Normal

from sklearn.cluster import KMeans
from matplotlib.patches import Rectangle

from torch.utils.data import DataLoader
from utils.pecnet_dataloader import ngsimDataset

# Make Directory Structure to Save the Models and plots:
base_dir = os.path.dirname(os.path.realpath(__file__))
exp_dir = os.path.join(base_dir, 'cache', 'pecnet', time.strftime("%d_%m_%Y_%H_%M"))
loss_dir = os.path.join(exp_dir, 'loss')
os.makedirs(exp_dir, exist_ok=True)
os.makedirs(loss_dir, exist_ok=True)

# CUDA / CPU Device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# Initialize training, val and test dataset
batch_size = 128
train_dataset = ngsimDataset(mat_file="./datasets/v3/TrainSet.mat", t_h=32, t_f=48, d_s=4)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=train_dataset.collate_fn)

test_dataset = ngsimDataset(mat_file="./datasets/v3/TestSet.mat", t_h=32, t_f=48, d_s=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, collate_fn=test_dataset.collate_fn)

# Model initialization
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x

class PECNet(nn.Module):
    def __init__(self, 
                 enc_past_size, 
                 enc_dest_size, 
                 enc_latent_size, 
                 dec_size, 
                 predictor_size, 
                 fdim, 
                 zdim, 
                 sigma,
                 past_length, 
                 future_length, 
                 verbose):
        '''
        Args:
            size parameters: Dimension sizes
            sigma: Standard deviation used for sampling N(0, sigma)
            past_length: Length of past history (number of timesteps)
            future_length: Length of future trajectory to be predicted
        '''
        super(PECNet, self).__init__()

        self.zdim = zdim
        self.sigma = sigma

        # takes in the past
        self.encoder_past = MLP(input_dim = past_length*2, output_dim = fdim, hidden_size=enc_past_size)

        self.encoder_dest = MLP(input_dim = 2, output_dim = fdim, hidden_size=enc_dest_size)

        self.encoder_latent = MLP(input_dim = 2*fdim, output_dim = 2*zdim, hidden_size=enc_latent_size)

        self.decoder = MLP(input_dim = fdim + zdim, output_dim = 2, hidden_size=dec_size)

        self.predictor = MLP(input_dim = 2*fdim, output_dim = 2*(future_length-1), hidden_size=predictor_size)

        architecture = lambda net: [l.in_features for l in net.layers] + [net.layers[-1].out_features]

        if verbose:
            print("Past Encoder architecture : {}".format(architecture(self.encoder_past)))
            print("Dest Encoder architecture : {}".format(architecture(self.encoder_dest)))
            print("Latent Encoder architecture : {}".format(architecture(self.encoder_latent)))
            print("Decoder architecture : {}".format(architecture(self.decoder)))
            print("Predictor architecture : {}".format(architecture(self.predictor)))

    def forward(self, x, dest = None, device=torch.device('cpu')):
        # provide destination iff training
        # assert model.training
        assert self.training ^ (dest is None)

        # encode
        ftraj = self.encoder_past(x)

        if not self.training:
            z = torch.Tensor(x.size(0), self.zdim)
            z.normal_(0, self.sigma)

        else:
            # during training, use the destination to produce generated_dest and use it again to predict final future points

            # CVAE code
            dest_features = self.encoder_dest(dest)
            features = torch.cat((ftraj, dest_features), dim = 1)
            latent =  self.encoder_latent(features)

            mu = latent[:, 0:self.zdim] # 2-d array
            logvar = latent[:, self.zdim:] # 2-d array

            var = logvar.mul(0.5).exp_()
            eps = torch.DoubleTensor(var.size()).normal_()
            eps = eps.to(device)
            z = eps.mul(var).add_(mu)

        z = z.double().to(device)
        decoder_input = torch.cat((ftraj, z), dim = 1)
        generated_dest = self.decoder(decoder_input)

        if self.training:
            generated_dest_features = self.encoder_dest(generated_dest)

            prediction_features = torch.cat((ftraj, generated_dest_features), dim = 1)

            pred_future = self.predictor(prediction_features)
            return generated_dest, mu, logvar, pred_future

        return generated_dest

    # separated for forward to let choose the best destination
    # def predict(self, past, generated_dest, mask, initial_pos):
    def predict(self, past, generated_dest):
        ftraj = self.encoder_past(past)
        generated_dest_features = self.encoder_dest(generated_dest)

        prediction_features = torch.cat((ftraj, generated_dest_features), dim = 1)

        interpolated_future = self.predictor(prediction_features)
        return interpolated_future

# Loss Function
def calculate_loss(x, reconstructed_x, mean, log_var, criterion, future, interpolated_future):
    # Weights
    w1 = torch.tensor([10, 1]).to(device)
    w2 = torch.tensor([10, 1] * 11).to(device)
    
    # reconstruction loss
    RCL_dest = torch.mean(w1 * (x - reconstructed_x) ** 2)
    
    ADL_traj = torch.mean(w2 * (future - interpolated_future) ** 2)
    
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return RCL_dest, KLD, ADL_traj

# Training
def train(train_loader, model, optimizer):
    model.train()
    train_loss = 0
    total_rcl, total_kld, total_adl = 0, 0, 0
    criterion = nn.MSELoss()
    num_datapoints = 0

    for i, data in enumerate(train_loader):
        hist, nbrs, mask, _, _, fut, _, ref_pos = data
        num_datapoints += ref_pos.size()[0]
        ref_pos = ref_pos.unsqueeze(1)
        hist = hist.permute(1, 0, 2) + ref_pos
        fut = fut.permute(1, 0, 2) + ref_pos

        fut = fut - hist[:, :1, :]
        hist = hist - hist[:, :1, :]

        # Converting data into format suitable for PECNET
        x = hist
        y = fut

        # reshape the data
        x = x.contiguous().view(-1, x.shape[1]*x.shape[2]).double()
        x = x.to(device)
        y = y.contiguous().double()

        dest = y[:, -1, :].to(device)
        future = y[:, :-1, :].view(y.size(0),-1).to(device)
        
        dest_recon, mu, var, interpolated_future = model.forward(x, dest=dest, device=device)
        
        optimizer.zero_grad()
        rcl, kld, adl = calculate_loss(dest, dest_recon, mu, var, criterion, future, interpolated_future)
        loss = rcl + kld*hyper_params["kld_reg"] + adl*hyper_params["adl_reg"]
        loss.backward()

        train_loss += loss.item()
        total_rcl += rcl.item()
        total_kld += kld.item()
        total_adl += adl.item()
        optimizer.step()
        
        if i % 25 == 0:
            print('Batch: {}, Train Loss: {:0.3f}, RCL: {:0.3f}, KLD: {:0.3f}, ADL: {:0.3f}'.format(i, loss.item(), rcl.item(), kld.item(), adl.item()))
    return train_loss, total_rcl, total_kld, total_adl, num_datapoints

# Testing
def test(test_loader, model, best_of_n = 1):
    model.eval()
    assert best_of_n >= 1 and type(best_of_n) == int
    with torch.no_grad():
        best_fde, mean_fde = [], []    
        for i, data in enumerate(test_loader):
            hist, nbrs, mask, _, _, fut, _, ref_pos = data
            ref_pos = ref_pos.unsqueeze(1)
            hist = hist.permute(1, 0, 2) + ref_pos
            fut = fut.permute(1, 0, 2) + ref_pos

            fut = fut - hist[:, :1, :]
            hist = hist - hist[:, :1, :]

            # Converting data into format suitable for PECNET
            x = hist
            y = fut.cpu().numpy()

            # reshape the data
            x = x.contiguous().view(-1, x.shape[1]*x.shape[2]).double()
            x = x.to(device)

            dest = y[:, -1, :]
            all_l2_errors_dest = []
            all_guesses = []
            for _ in range(best_of_n):
                dest_recon = model.forward(x, device=device)
                dest_recon = dest_recon.cpu().numpy()
                all_guesses.append(dest_recon)

                l2error_sample = np.linalg.norm(dest_recon - dest, axis = 1)
                all_l2_errors_dest.append(l2error_sample)

            all_l2_errors_dest = np.array(all_l2_errors_dest)
            all_guesses = np.array(all_guesses)

            # average error
            l2error_avg_dest = np.mean(all_l2_errors_dest)
            # taking the minimum error out of all guess
            l2error_dest = np.mean(np.min(all_l2_errors_dest, axis = 0))

            best_fde.append(np.min(all_l2_errors_dest, axis = 0))
            mean_fde.append(np.mean(all_l2_errors_dest, axis = 0))
            if i % 25 == 0:
                print('Batch: {}, Test time error in destination best: {:0.3f} and mean: {:0.3f}'.format(i, l2error_dest, l2error_avg_dest))
    print("-"*50)
    best_fde = np.concatenate(best_fde)
    mean_fde = np.concatenate(mean_fde)
    print("Best destination error (mean): {}, Mean desitation error : {}".format(np.mean(best_fde), np.mean(mean_fde)))
    return np.mean(best_fde), np.mean(mean_fde)

# Loading hyperparameters
def load_hyper_parameters(file_name='./pecnet/optimal.yaml'):
    with open(file_name, 'r') as file:
        hyper_params = yaml.load(file)
    
    return hyper_params

hyper_params = load_hyper_parameters()
hyper_params["data_scale"] = 1

# Load pretrained model and checkpoint
model = PECNet(
    hyper_params["enc_past_size"],
    hyper_params["enc_dest_size"],
    hyper_params["enc_latent_size"],
    hyper_params["dec_size"],
    hyper_params["predictor_hidden_size"],
    hyper_params["fdim"], 
    hyper_params["zdim"], 
    hyper_params["sigma"], 
    hyper_params["past_length"], 
    hyper_params["future_length"], 
    verbose=True
)
model = model.double().to(device)
optimizer = optim.Adam(model.parameters(), lr= hyper_params["learning_rate"])

best_test_loss = 50 # start saving after this threshold
best_endpoint_loss = 50
N = hyper_params["n_values"]

epoch_losses = [["Epoch", "train_loss", "rcl", "kld", "adl", "test_fde", "mean_fde"]]

for e in range(hyper_params['num_epochs']):  
    train_loss, rcl, kld, adl, num_datapoints = train(train_loader, model, optimizer)
    best_fde, mean_fde = test(test_loader, model, best_of_n = N)

    epoch_losses.append([
        e,
        train_loss / num_datapoints,
        rcl / num_datapoints,
        kld / num_datapoints,
        adl / num_datapoints,
        best_fde,
        mean_fde
    ])

    if best_test_loss > best_fde:
        print("Epoch: {}".format(e))
        print('####### BEST PERFORMANCE {:0.2f} ########'.format(best_fde))
        best_test_loss = best_fde
        if best_test_loss < 5.0:
            save_path = os.path.join(exp_dir, 'model_{}.pth'.format(e))
            torch.save({
                'hyper_params': hyper_params,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, save_path)
            print("Saved model to:\n{}".format(save_path)) 

with open(os.path.join(loss_dir, 'loss.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerows(epoch_losses)
