import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from models.utils import create_sliding_windows, reconstruct_from_windows

class BaseVAE(nn.Module):
    def __init__(
        self, 
        window_size=24,
        num_features=None,
        latent_dim=8,
        epochs=20,
        batch_size=256,
        lr=1e-3,
        device=None,
        scaler=None,
        verbose=True
    ):
        super(BaseVAE, self).__init__()
        self.window_size = window_size
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.scaler = scaler

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.encoder_conv1 = None
        self.encoder_conv2 = None
        self.encoder_pool = None
        self.fc_mu = None
        self.fc_logvar = None
        self.decoder_fc = None
        self.decoder_conv1 = None
        self.decoder_conv2 = None
        self.decoder_upsample = None
        self.is_trained = False

        if self.num_features is not None:
            self._initialize_layers()

    def _initialize_layers(self):
        self.encoder_conv1 = nn.Conv1d(self.num_features, 32, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.encoder_pool = nn.AdaptiveAvgPool1d(8)
        self.fc_mu = nn.Linear(64 * 8, self.latent_dim)
        self.fc_logvar = nn.Linear(64 * 8, self.latent_dim)
        self.decoder_fc = nn.Linear(self.latent_dim, 64 * 8)
        self.decoder_conv1 = nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.ConvTranspose1d(32, self.num_features, kernel_size=3, padding=1)
        self.decoder_upsample = nn.Upsample(size=self.window_size, mode='linear', align_corners=False)
        self.to(self.device)

    def encode(self, x):
        h = F.relu(self.encoder_conv1(x))
        h = F.relu(self.encoder_conv2(h))
        h = self.encoder_pool(h)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 64, 8)
        h = F.relu(self.decoder_conv1(h))
        h = F.relu(self.decoder_conv2(h))
        h = self.decoder_upsample(h)
        return h

    def forward(self, x):
        if x.dim() == 3: # Already (B, D, W)
            pass
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def fit(self, windows):
        num_windows, window_size, num_features = windows.shape
        self.num_features = num_features
        self._initialize_layers()

        windows_transposed = torch.tensor(windows.transpose(0, 2, 1), dtype=torch.float32)
        dataset = TensorDataset(windows_transposed)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch_data,) in dataloader:
                batch_data = batch_data.to(self.device)
                optimizer.zero_grad()
                recon, mu, logvar = self.forward(batch_data)
                recon_loss = F.mse_loss(recon, batch_data, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.1 * kl_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if self.verbose and epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss / len(dataloader.dataset):.6f}")

        self.is_trained = True

    def generate(self, num_samples):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            samples = self.decode(z)
            return samples.cpu().numpy().transpose(0, 2, 1)
