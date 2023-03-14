import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from utils.datasets import ESR
from utils.vae_cnn import VAE

batch_size = 20
learning_rate = 1e-3
num_epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = ESR('D:')
# dataset = datasets.MNIST('D:', train=True, download=True,
#                     transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1)


net = VAE().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for x, y in train_loader:
        
        x = x.to(device)

        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, logVar = net(x)

        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        loss = F.binary_cross_entropy(out, x, reduction='mean') + kl_divergence
        print(loss.item())
        # Backpropagation based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch {}: Loss {}'.format(epoch, loss))