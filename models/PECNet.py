import os
import math
import random

import matplotlib
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils import data
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.distributions.normal import Normal

from MLP import MLP

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
        return generated_dest

    # separated for forward to let choose the best destination
    def predict(self, past, generated_dest):
        ftraj = self.encoder_past(past)
        generated_dest_features = self.encoder_dest(generated_dest)

        prediction_features = torch.cat((ftraj, generated_dest_features), dim = 1)

        interpolated_future = self.predictor(prediction_features)
        return interpolated_future