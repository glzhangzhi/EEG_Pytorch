
import torch

from utils.compare_x import compare_x
from utils.datasets import ESR, Single
from utils.vae_cnn_1d import VAE

batch_size = 50
learning_rate = 1e-3
num_epochs = 500

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Single()
# dataset = ESR()

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size, shuffle=False)

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    
    for x in train_loader:
        
        x = x.to(device)
        # x = x[0].to(device)
        
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
        # with torch.no_grad():
        #     compare_x(x.cpu().reshape(178,), recon_x.cpu().reshape(178,), epoch)
        # break
    
    with torch.no_grad():
        
        recon_x, mu, logVar = model(x[-1])
        compare_x(x[-1].cpu().reshape(178,), recon_x.cpu().reshape(178,), epoch)
    
    # compare_x(x, recon_x.reshape(-1,), epoch)

    print(f'Epoch {epoch}: Loss_R {loss_r:.3f} Loss_K {kl_divergence:.3f} Loss {loss:.3f}')
    
    # torch.save(model.state_dict(), f'ckp_vae_cnn_1d/{epoch}.ckpt')

