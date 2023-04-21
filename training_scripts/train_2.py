import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import trange

from utils.datasets import ESR
from utils.vae_cnn import VAE

logging.basicConfig(filename='training.log',
                    format='%(asctime)s %(message)s',
                    encoding='utf-8',
                    level=logging.DEBUG,
                    filemode='a')

batch_size = 128
learning_rate = 1e-3
num_epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = ESR('D:')

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1)

model = VAE(256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in trange(num_epochs):
    
    for x, y in train_loader:
        
        x = x.to(device)
        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, logVar = model(x)

        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        loss_r = torch.nn.MSELoss(reduction='sum')(out, x)
        loss = loss_r + kl_divergence

        # Backpropagation based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logging.debug(f'Epoch {epoch}: Loss_R {loss_r:.3f} Loss_K {kl_divergence:.3f} Loss {loss:.3f}')
    
    if epoch % 5 == 0:
        # TODO 计算一次H值
        torch.save(model.state_dict(), f'ckp_cnn_vae_1/{epoch}.ckpt')