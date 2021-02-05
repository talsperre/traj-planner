import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils import data
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.distributions.normal import Normal

from .MLP import MLP
from .CSPEncoder import CSPEncoder

class PEC_CSP_Net(nn.Module):
    def __init__(self,  
                 enc_dest_size, 
                 enc_latent_size, 
                 dec_size,
                 fdim, 
                 zdim, 
                 sigma,
                 CSP_args,
                 device,
                 verbose):
        '''
        Args:
            size parameters: Dimension sizes
            sigma: Standard deviation used for sampling N(0, sigma)
            past_length: Length of past history (number of timesteps)
            future_length: Length of future trajectory to be predicted
        '''
        super(PEC_CSP_Net, self).__init__()

        self.zdim = zdim
        self.sigma = sigma

        # takes in the past
        self.encoder_past = CSPEncoder(CSP_args)
        self.encoder_past = self.encoder_past.double().to(device)
        self.csp_encoder_size = self.encoder_past.soc_embedding_size + self.encoder_past.dyn_embedding_size
        self.encoder_dest = MLP(input_dim=2, output_dim=fdim, hidden_size=enc_dest_size)
        self.encoder_latent = MLP(input_dim=self.csp_encoder_size + fdim, output_dim=2*zdim, hidden_size=enc_latent_size)
        self.decoder = MLP(input_dim=self.csp_encoder_size + zdim, output_dim = 2, hidden_size=dec_size)

        architecture = lambda net: [l.in_features for l in net.layers] + [net.layers[-1].out_features]

        if verbose:
            print("Dest Encoder architecture : {}".format(architecture(self.encoder_dest)))
            print("Latent Encoder architecture : {}".format(architecture(self.encoder_latent)))
            print("Decoder architecture : {}".format(architecture(self.decoder)))

    def forward(self, hist, nbrs, masks, dest=None, device=torch.device('cpu')):
        # provide destination iff training
        # assert model.training
        assert self.training ^ (dest is None)

        # encode
        ftraj = self.encoder_past(hist, nbrs, masks, device)

        if not self.training:
            z = torch.Tensor(hist.size(1), self.zdim)
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
        
        # print("z.size(): {}".format(z.size()))
        z = z.double().to(device)
        decoder_input = torch.cat((ftraj, z), dim = 1)
        generated_dest = self.decoder(decoder_input)
        if self.training:
            return generated_dest, mu, logvar
        return generated_dest
