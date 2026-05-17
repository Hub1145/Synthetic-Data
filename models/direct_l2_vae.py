import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectL2VAE(nn.Module):
    def __init__(self, timesteps=96, num_features=4, latent_dim=64):
        super(DirectL2VAE, self).__init__()
        
        self.timesteps = timesteps
        self.num_features = num_features
        self.latent_dim = latent_dim
        
        # Encoder: Convolutional layers to capture temporal orderbook patterns
        self.encoder = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, stride=2, padding=1), # [B, 32, 48]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),           # [B, 64, 24]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),          # [B, 128, 12)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (timesteps // 8), 256),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: Transposed convolutions to reconstruct the orderbook sequence
        self.decoder_input = nn.Linear(latent_dim, 256)
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 128 * (timesteps // 8)),
            nn.ReLU(),
            nn.Unflatten(1, (128, timesteps // 8)),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1), # [B, 64, 24]
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 48]
            nn.ReLU(),
            nn.ConvTranspose1d(32, num_features, kernel_size=4, stride=2, padding=1), # [B, 4, 96]
        )

    def encode(self, x):
        # x shape: [Batch, Timesteps, Features] -> [Batch, Features, Timesteps]
        x = x.transpose(1, 2)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        recon = self.decoder(h)
        # recon shape: [Batch, Features, Timesteps] -> [Batch, Timesteps, Features]
        return recon.transpose(1, 2)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD
