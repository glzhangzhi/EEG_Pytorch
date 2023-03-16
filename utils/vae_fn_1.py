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
    
    def __init__(self):
        super(VAE, self).__init__()

        self.efn1 = nn.Linear(196, 400)
        self.efn2 = nn.Linear(400, 400)
        self.efn3 = nn.Linear(400, 400)
        self.efn4 = nn.Linear(400, 256)
        
        self.muFC = nn.Linear(256, 256)
        self.varFC = nn.Linear(256, 256)
        
        self.zFC = nn.Linear(256, 256)
        
        self.dfn4 = nn.Linear(256, 400)
        self.dfn3 = nn.Linear(400, 400)
        self.dfn2 = nn.Linear(400, 400)
        self.dfn1 = nn.Linear(400, 196)

    def encoder(self, x):

        x = F.relu(self.efn1(x))
        x = F.relu(self.efn2(x))
        x = F.relu(self.efn3(x))
        x = F.relu(self.efn4(x))
        
        mu = self.muFC(x)
        logVar = self.varFC(x)
        
        return mu, logVar

    def reparameterize(self, mu, logVar):

        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        
        x = F.relu(self.zFC(z))
        
        x = F.relu(self.dfn4(x))
        x = F.relu(self.dfn3(x))
        x = F.relu(self.dfn2(x))
        x = F.relu(self.dfn1(x))
        
        return x

    def forward(self, x):
        
        x = x.view(-1, 196)
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        out = out.view(-1, 1, 14, 14)
        return out, mu, logVar

