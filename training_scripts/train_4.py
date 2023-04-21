import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from utils.datasets import ESR
from utils.hopkins import get_H
from utils.vae_trans import VAE
from utils.vae_complex import Complex_VAE

batch_size = 512
learning_rate = 1e-3
num_epochs = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = ESR('D:')

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1)

model = Complex_VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    
    for x, y in train_loader:
        
        x = x.to(device)
        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        recon_x, mu, logVar = model(x)

        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        loss_r = torch.nn.MSELoss(reduction='sum')(recon_x, x)
        loss = loss_r + kl_divergence

        # Backpropagation based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch}: Loss_R {loss_r:.3f} Loss_K {kl_divergence:.3f} Loss {loss:.3f}')
    
    torch.save(model.state_dict(), f'ckp_vae_cnn_1d/{epoch}.ckpt')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 512

dataset = ESR('D:')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():

    for epoch in trange(50):

        model = VAE().to(device)

        model.load_state_dict(torch.load(f'ckp_vae_cnn_1d/{epoch}.ckpt'))

        zs = None

        for x, y in dataloader:
            
            x = x.to(device)
            mu, var = model.encode(x)
            z = model.reparameterize(mu, var)

            z = z.cpu()
            
            if zs is None:
                zs = z
            else:
                zs = np.vstack([zs, z])

        with open(f'ckp_vae_cnn_1d/{epoch}.pkl', 'wb') as f:
            pickle.dump(zs, f)
