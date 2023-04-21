import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from tqdm import tqdm

from utils import config
from utils.models.vae import Complex_VAE, EEG_Dataset, Res_VAE

path_log = Path('./output_data')
path_log.mkdir(exist_ok=True)
path_mnist_dataset = Path('D:')
path_models = Path('./models')
path_models.mkdir(exist_ok=True)

# 可以配置日志输出文件，编码形式，打印级别，日志文件记录方式
logging.basicConfig(filename=path_log / 'training.log',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    encoding='utf-8',
                    level=logging.DEBUG,
                    filemode=config['log_file_mode'])
# 也可以通过在命令行传入参数--log=INFO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = config['learning_rate']
start_epoch = config['start_epoch']
num_epochs = config['num_epochs']
batch_size = config['batch_size']

if config['dataset'] == 'eeg':
    dataloader = DataLoader(EEG_Dataset(root='EEG'), batch_size=batch_size, shuffle=True)
elif config['dataset'] == 'mnist':
    dataset = torchvision.datasets.MNIST(root=path_mnist_dataset,
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)

if config['model'] == 'vae':
    model = Complex_VAE().to(device)
elif config['model'] == 'resnet':
    model = Res_VAE(batch_size).to(device)

if start_epoch != 1:
    model.load_state_dict(torch.load(f'./models/{start_epoch - 1}.ckpt'))

# summary(model, (batch_size, 1, 40*64))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if config['use_lr_scheduler']:
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=config['patience'])

for epoch in range(start_epoch, start_epoch + num_epochs):
    i = 1
    for data in tqdm(dataloader):
        
        if config['dataset'] == 'eeg':
            patien_id, x = data
        elif config['dataset'] == 'mnist':
            x = data[0].view(batch_size, 1, -1)
        
        x = x.float().to(device)
        x_r, mu, var = model(x)
        
        # loss_r = F.binary_cross_entropy(x_r, x, reduction='sum')
        loss_r = torch.nn.MSELoss(reduction='sum')(x_r, x)
        
        kl_div = -5e-4 * torch.mean(1 + var - mu.pow(2) - var.exp())
        
        loss = torch.mean(kl_div + loss_r)
        
        optimizer.zero_grad()
        if config['loss'] == 'r+d':
            loss.backward()
        elif config['loss'] == 'r':
            loss_r.backward()
        optimizer.step()
        if config['use_lr_scheduler']:
            scheduler.step(loss)
        if i % 10 == 0:
            logging.info(f"Epoch[{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], Reconst Loss: {loss_r.item():.4f}, KL Div: {kl_div.item():.4f}")
        i += 1
    
    torch.save(model.state_dict(), path_models / f'{epoch}.ckpt')