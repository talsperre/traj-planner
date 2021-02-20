import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from convLSTM import ConvLSTM


class INFER(nn.Module):
    def __init__(self, activation, init_type, num_channels, image_height, image_width, batchnorm=False):
        super(INFER, self).__init__()
        
        self.batchnorm = batchnorm
        self.bias = not self.batchnorm
        self.init_type = init_type
        self.activation = None
        self.num_channels = num_channels

        # Encoder
        self.conv1 = nn.Conv2d(self.num_channels, 16, 3, 1, 1, bias=self.bias)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=self.bias)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=self.bias)

        # Decoder
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)
        self.deconv1 = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)

        # # Conv LSTM
        self.conv_lstm = ConvLSTM(
            input_dim=64,
            hidden_dim=64,
            kernel_size=(3,3),
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        
        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Conv2d(8, 1, 1)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.SELU(inplace=True)

        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(32)
            self.bn5 = nn.BatchNorm2d(16)
            self.bn6 = nn.BatchNorm2d(8)
    
    def forward(self, ego_grid, veh_grid):
        x = torch.cat([ego_grid, veh_grid], dim=2)
        batch_size, traj_length, num_channels, h, w = x.size()
        
        # First apply 2D convolution layers on all the trajectories in the batch
        x = x.view(batch_size * traj_length, num_channels, h, w)
        # Encoder
        if self.batchnorm:
            conv1_ = self.pool(self.bn1(self.activation(self.conv1(x))))
            conv2_ = self.pool(self.bn2(self.activation(self.conv2(conv1_))))
            conv3_ = self.pool(self.bn3(self.activation(self.conv3(conv2_))))
        else:
            conv1_ = self.pool(self.activation(self.conv1(x)))
            conv2_ = self.pool(self.activation(self.conv2(conv1_)))
            conv3_ = self.pool(self.activation(self.conv3(conv2_)))
        
        # print("conv3_.size(): {}".format(conv3_.size()))
        conv3_ = conv3_.view(batch_size, traj_length, 64, int(h / 8), int(w / 8))
        # print("conv3_.size(): {}".format(conv3_.size()))
        _, lstm_out_ = self.conv_lstm(conv3_)
        # print("lstm_out_: {}".format(type(lstm_out_)))
        hidden = lstm_out_[0][0]
        # print("hidden.size(): {}".format(hidden.size()))
        
        # Decoder
        if self.batchnorm:    
            deconv3_ = self.bn4(self.activation(self.deconv3(hidden)))
            # print("deconv3_.size(): {}".format(deconv3_.size()))
            deconv2_ = self.bn5(self.activation(self.deconv2(deconv3_)))
            # print("deconv2_.size(): {}".format(deconv2_.size()))
            deconv1_ = self.bn6(self.activation(self.deconv1(deconv2_)))
            # print("deconv1_.size(): {}".format(deconv1_.size()))
        else:
            deconv3_ = self.activation(self.deconv3(hidden))
            # print("deconv3_.size(): {}".format(deconv3_.size()))
            deconv2_ = self.activation(self.deconv2(deconv3_))
            # print("deconv2_.size(): {}".format(deconv2_.size()))
            deconv1_ = self.activation(self.deconv1(deconv2_))
            # print("deconv1_.size(): {}".format(deconv1_.size()))
        
        score = self.classifier(deconv1_)
        return score

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init_type == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                if self.init_type == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()