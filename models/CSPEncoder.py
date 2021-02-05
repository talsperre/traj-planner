from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn

class CSPEncoder(nn.Module):
    ## Initialization
    def __init__(self,args):
        super(CSPEncoder, self).__init__()

        ## Unpack arguments
        self.args = args

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.grid_size = args['grid_size']
        self.soc_conv_depth = args['soc_conv_depth']
        self.conv_3x1_depth = args['conv_3x1_depth']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        self.soc_embedding_size = (((args['grid_size'][0] - 4) + 1) // 2) * self.conv_3x1_depth

        ## Define network weights
        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)

        # Encoder LSTM
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)
        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size, self.dyn_embedding_size)
        # Convolutional social pooling layer and social embedding layer
        self.soc_conv = torch.nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3)
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3, 1))
        self.soc_maxpool = torch.nn.MaxPool2d((2, 1), padding=(1, 0))

        # FC social pooling layer (for comparison):
        # self.soc_fc = torch.nn.Linear(self.soc_conv_depth * self.grid_size[0] * self.grid_size[1], (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()

    ## Forward Pass
    def forward(self, hist, nbrs, masks, device):
        ## Forward pass hist:
        _, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1], hist_enc.shape[2])))

        ## Forward pass nbrs
        _, (nbrs_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        ## Masked scatter
        # soc_enc = torch.zeros_like(masks).float().to(device)
        soc_enc = torch.zeros_like(masks).double().to(device)
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)
        soc_enc = soc_enc.permute(0, 3, 2, 1)

        ## Apply convolutional social pooling:
        soc_enc = self.soc_maxpool(self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc)))))
        soc_enc = soc_enc.view(-1, self.soc_embedding_size)

        ## Apply fc soc pooling
        # soc_enc = soc_enc.contiguous()
        # soc_enc = soc_enc.view(-1, self.soc_conv_depth * self.grid_size[0] * self.grid_size[1])
        # soc_enc = self.leaky_relu(self.soc_fc(soc_enc))

        ## Concatenate encodings:
        enc = torch.cat((soc_enc, hist_enc), 1)
        return enc
