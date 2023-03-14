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
    
    def __init__(self, in_features, out_features, featureDim, zDim):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.fn1 = nn.Linear(in_features, out_features)
        self.fn2 = nn.Linear(out_features, featureDim)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.fn3 = nn.Linear(featureDim, out_features)
        self.fn4 = nn.Linear(out_features, in_features)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        
        # 1 x 28 x 28
        x = F.relu(self.fn1(x))
        # 16 x 24 x 24
        x = F.relu(self.fn2(x))
        # 32 x 20 x 20
        
        # 1 x (32x20x20)
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
        # 1 x (32x20x20)
        # 32 x 20 x 20
        x = F.relu(self.fn3(x))
        # 16 x 24 x 24
        x = torch.sigmoid(self.fn4(x))
        # 1 x 28 x 28
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

