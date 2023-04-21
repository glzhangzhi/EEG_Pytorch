"""
The following is an import of PyTorch libraries.
"""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image

"""
A Convolutional Variational Autoencoder
"""
class VAE(nn.Module):
    
    def __init__(self, imgChannels=1, featureDim=32*170, zDim=256):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv1d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv1d(16, 32, 5)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose1d(32, 16, 5)
        self.decConv2 = nn.ConvTranspose1d(16, imgChannels, 5)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        
        # 1 x 178
        x = F.relu(self.encConv1(x))
        # 16 x 174
        x = F.relu(self.encConv2(x))
        # 32 x 170
        
        x = x.view(-1, 32*170)
        # 1 x (32x170)
        mu = self.encFC1(x)
        # 1 x 256
        logVar = self.encFC2(x)
        # 1 x 256
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps
        # 1 x 256

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        
        # 1 x 256
        x = F.relu(self.decFC1(z))
        # 1 x (32x170)
        x = x.view(-1, 32, 170)
        # 32 x 170
        x = F.relu(self.decConv1(x))
        # 16 x 174
        x = torch.tanh(self.decConv2(x))
        # 1 x 178
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

