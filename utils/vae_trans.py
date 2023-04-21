import torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.stride = stride

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class VAE(nn.Module):
    def __init__(self, latent_size=128):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            # b x 1 x 178
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # b x 32 x 178
            nn.MaxPool1d(kernel_size=2, stride=2),
            # b x 32 x 89
            ResidualBlock(32, 64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # b x 64 x 44
            ResidualBlock(64, 128),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # b x 128 x 22
            ResidualBlock(128, 256),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # b x 256 x 11
            nn.Flatten(),
            # b x 2816
            nn.Linear(256*11, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # b x 512
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
            # b x 256
        )

        
        self.fc_mu = nn.Linear(256, latent_size)
        self.fc_var = nn.Linear(256, latent_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # b x 256
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # b x 512
            nn.Linear(512, 256*11),
            nn.BatchNorm1d(256*11),
            nn.ReLU(inplace=True),
            # b x 2816
            View((256, 11)),
            # b x 256 x 11
            ResidualBlock(256, 128),
            # b x 128 x 11
            nn.ConvTranspose1d(128, 128, kernel_size=2, stride=2, padding=0),
            # b x 128 x 22
            ResidualBlock(128, 64),
            nn.ReLU(),
            # b x 64 x 22
            nn.ConvTranspose1d(64, 64, kernel_size=2, stride=2, padding=0),
            # b x 64 x 44
            ResidualBlock(64, 32),
            nn.ReLU(),
            # b x 32 x 44
            nn.ConvTranspose1d(32, 32, kernel_size=2, stride=2, padding=0),
            # b x 32 x 88
            nn.ConvTranspose1d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            # b x 32 x 176
            nn.ConvTranspose1d(32, 1, kernel_size=5, stride=1, padding=1),
            # b x 1 x 178
            nn.Sigmoid()
        )
        
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar
    
    def decode(self, z):
        z = self.decoder(z)
        return z
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
