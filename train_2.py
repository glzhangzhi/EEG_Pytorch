import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from utils.datasets import ESR
from utils.vae_cnn import VAE

batch_size = 128
learning_rate = 1e-3
num_epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = ESR('D:')
# dataset = datasets.MNIST('D:', train=True, download=True,
#                     transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1)


"""
Initialize the network and the Adam optimizer
"""
net = VAE().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


"""
Training the network for a given number of epochs
The loss after every epoch is printed
"""
for epoch in range(num_epochs):
    for idx, data in enumerate(train_loader, 0):
        imgs, _ = data
        imgs = imgs.to(device)
        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, logVar = net(imgs)

        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        loss = torch.nn.MSELoss(reduction='sum')(out, imgs) + kl_divergence

        # Backpropagation based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch {}: Loss {}'.format(epoch, loss))