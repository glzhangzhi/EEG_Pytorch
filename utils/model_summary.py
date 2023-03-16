import torch
import torch.nn as nn

# from torchsummary import summary
from torchinfo import summary

from utils.vae_fn_1 import VAE
from utils.datasets import ESR

dataset = ESR('D:')

model = VAE(178, 512, 512, 256)

summary(model, dataset.get_x_shape())