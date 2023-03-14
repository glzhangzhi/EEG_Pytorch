import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm, trange

from utils.models.vae import Complex_VAE, EEG_Dataset, Res_VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32

# dataset = EEG_Dataset(root='EEG')
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

dataset = torchvision.datasets.MNIST(root='D:',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)
dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

model = Res_VAE(batch_size).to(device)
model.load_state_dict(torch.load('models/20.ckpt'))
# summary(model, (1, 40*64))

zs = []

for data in tqdm(dataloader):
    
    x = data[0].view(batch_size, 1, -1)
    x = x.float().to(device)

    x = model.encoder(x)
    mu, var = model.parameterize(x)
    z = model.reparameterize(mu, var)

    zs.append([(z[0] * 100).tolist(), str(data[1].tolist()[0])[:3]])

with open('output_data/zs20.pkl', 'wb') as f:
    pickle.dump(zs, f)
