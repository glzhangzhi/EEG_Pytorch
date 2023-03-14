import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)
        
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = y + x
        return F.relu(y)

class ResidualTranspose(nn.Module):
    
    def __init__(self, input_channels, num_channels, specific_output_shape):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(input_channels, input_channels, kernel_size=3, padding=1, stride=1)
        if input_channels == num_channels:
            self.conv2 = nn.ConvTranspose1d(input_channels, num_channels, kernel_size=3, padding=1, stride=1)  # 不变
        else:
            if specific_output_shape:
                self.conv2 = nn.ConvTranspose1d(input_channels, num_channels, kernel_size=3, padding=1, stride=2)  # 翻倍指定
            else:
                self.conv2 = nn.ConvTranspose1d(input_channels, num_channels, kernel_size=4, padding=1, stride=2)  # 翻倍
        self.bn1 = nn.BatchNorm1d(input_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)
    
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y)

class Reshape(nn.Module):
    
    def __init__(self, shape, batch_size):
        super(Reshape, self).__init__()
        self.shape = shape
        self.batch_size = batch_size
    
    def forward(self, x):
        return x.view((self.batch_size, -1, 1))

class Repeat(nn.Module):
    
    def __init__(self, n):
        super(Repeat, self).__init__()
        self.n = n
    
    def forward(self, x):
        return x.repeat(1, 1, self.n) 

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

def resnet_transpose_block(input_channels, num_channels, num_residuals, last_block=False, specific_output_shape=False):
    blk = []
    for i in range(num_residuals):
        if i != num_residuals - 1 and not last_block:
            blk.append(ResidualTranspose(input_channels, input_channels, specific_output_shape))
        else:
            blk.append(ResidualTranspose(input_channels, num_channels, specific_output_shape))
    return blk

class Res_VAE(nn.Module):
    def __init__(self, batch_size):
        super(Res_VAE, self).__init__()
        b1 = nn.Sequential(nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm1d(64),
                   nn.ReLU(),
                   nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))

        self.encoder = nn.Sequential(b1, b2, b3, b4, b5,
                            nn.AdaptiveAvgPool1d(1),
                            nn.Flatten(), nn.Linear(512, 20))

        a1 = nn.Sequential(*resnet_transpose_block(512, 256, 2, specific_output_shape=True))
        a2 = nn.Sequential(*resnet_transpose_block(256, 128, 2))
        a3 = nn.Sequential(*resnet_transpose_block(128, 64, 2))
        a4 = nn.Sequential(*resnet_transpose_block(64, 64, 2, last_block=True))
        a5 = nn.Sequential(nn.ConvTranspose1d(64, 64, kernel_size=4, padding=1, stride=2),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.ConvTranspose1d(64, 1, kernel_size=4, padding=1, stride=2),
                        nn.BatchNorm1d(1),
                        nn.ReLU())

        self.decoder = nn.Sequential(nn.Linear(10, 20),
                                nn.Linear(20, 512),
                                Reshape((1, -1, 1), batch_size),
                                Repeat(25),
                                a1, a2, a3, a4, a5
                                )
        
        self.l1 = nn.Linear(20, 10)
        self.l2 = nn.Linear(20, 10)
    
    def parameterize(self, x):
        mu = self.l1(x)
        var = self.l2(x)
        return mu, var
        
    
    def reparameterize(self, mu, var):
        std = torch.exp(var/2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = self.encoder(x)
        mu, var = self.parameterize(x)
        z = self.reparameterize(mu, var)
        x_r = self.decoder(z)
        return x_r, mu, var