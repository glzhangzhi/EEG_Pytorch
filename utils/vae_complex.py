import torch
import torch.nn as nn
import torch.nn.functional as F

latent_dim = 10

class Complex_VAE(nn.Module):
    def __init__(self):
        super(Complex_VAE, self).__init__()
        self.ec11 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.ec12 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.em1 = nn.MaxPool1d(kernel_size=2)
        
        self.ec21 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.ec22 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.em2 = nn.MaxPool1d(kernel_size=2)
        
        self.ec31 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.ec32 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.em3 = nn.MaxPool1d(kernel_size=2)
        
        self.ec41 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.ec42 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.em4 = nn.MaxPool1d(kernel_size=2)
        
        # self.el = nn.Linear(in_features=128*160, out_features=latent_dim * 2)
        self.el = nn.Linear(in_features=6272, out_features=latent_dim * 2)
        self.elmu = nn.Linear(in_features=latent_dim * 2, out_features=latent_dim)
        self.elvar = nn.Linear(in_features=latent_dim * 2, out_features=latent_dim)
        
        self.dl1 = nn.Linear(in_features=latent_dim, out_features=latent_dim*2)
        # self.dl2 = nn.Linear(in_features=latent_dim*2, out_features=128*160)
        self.dl2 = nn.Linear(in_features=latent_dim*2, out_features=6272)
        
        self.noc = nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.dc1 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.dc2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.dc3 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.dc4 = nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=2, stride=2, padding=0)


    def reparameterize(self, mu, var):
        std = torch.exp(var/2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encoder(self, x):
        x = nn.ReLU()(self.ec11(x))
        x = nn.ReLU()(self.ec12(x))
        x = self.em1(x)
        
        x = nn.ReLU()(self.ec21(x))
        x = nn.ReLU()(self.ec22(x))
        x = self.em2(x)
        
        x = nn.ReLU()(self.ec31(x))
        x = nn.ReLU()(self.ec32(x))
        x = self.em3(x)
        
        x = nn.ReLU()(self.ec41(x))
        x = nn.ReLU()(self.ec42(x))
        x = self.em4(x)

        x = nn.Flatten()(x)
        
        x = self.el(x)
        mu = self.elmu(x)
        var = self.elvar(x)

        return mu, var

    def decoder(self, z):
        x = nn.ReLU()(self.dl1(z))
        x = nn.ReLU()(self.dl2(x))

        # x = nn.Unflatten(1, (128, 160))(x)
        x = nn.Unflatten(1, (128, 49))(x)

        x = nn.ReLU()(self.noc(x))

        x = nn.ReLU()(self.dc1(x))
        x = nn.ReLU()(self.dc2(x))
        x = nn.ReLU()(self.dc3(x))
        x = nn.ReLU()(self.dc4(x))
        
        return x

    def forward(self, x):
        mu, var = self.encoder(x)
        z = self.reparameterize(mu, var)
        x_r = self.decoder(z)

        return x_r, mu, var