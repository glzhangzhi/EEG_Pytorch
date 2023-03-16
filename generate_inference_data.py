import pickle

import numpy as np
import torch
from tqdm import trange

from utils.datasets import ESR
from utils.vae_fn_2 import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 512

dataset = ESR('D:')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():

    for epoch in trange(50):

        model = VAE().to(device)

        model.load_state_dict(torch.load(f'ckp_vae_fn_2/{epoch}.ckpt'))

        zs = None

        for x, y in dataloader:
            
            x = x.to(device)
            x = x.view(-1, 196)
            mu, var = model.encoder(x)
            z = model.reparameterize(mu, var)

            z = z.cpu()
            
            if zs is None:
                zs = z
            else:
                zs = np.vstack([zs, z])

        with open(f'ckp_vae_fn_2/{epoch}.pkl', 'wb') as f:
            pickle.dump(zs, f)
