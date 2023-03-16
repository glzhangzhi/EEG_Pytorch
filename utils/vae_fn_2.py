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

        self.efns = nn.Sequential(
            nn.Linear(196, 400),
            nn.ReLU(),
            nn.Linear(400, 800),
            nn.ReLU(),
            nn.Linear(800, 1200),
            nn.ReLU(),
            nn.Linear(1200, 2400),
            nn.ReLU(),
            nn.Linear(2400, 5000),
            nn.ReLU(),
            nn.Linear(5000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 256),
            nn.ReLU()
        )
        
        self.muFC = nn.Linear(256, 256)
        self.varFC = nn.Linear(256, 256)
        
        self.zFC = nn.Linear(256, 256)
        
        self.dfns = nn.Sequential(
            nn.Linear(256, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 2400),
            nn.ReLU(),
            nn.Linear(2400, 1200),
            nn.ReLU(),
            nn.Linear(1200, 800),
            nn.ReLU(),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Linear(400, 196),
            nn.ReLU()
        )
        
    def encoder(self, x):
        
        x = self.efns(x)

        mu = self.muFC(x)
        logVar = self.varFC(x)
        
        return mu, logVar

    def reparameterize(self, mu, logVar):

        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        
        x = F.relu(self.zFC(z))
        
        x = self.dfns(x)

        return x

    def forward(self, x):
        
        x = x.view(-1, 196)
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        out = out.view(-1, 1, 14, 14)
        return out, mu, logVar

