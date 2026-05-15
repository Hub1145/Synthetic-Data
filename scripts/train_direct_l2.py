import os
import sys
# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.direct_l2_vae import DirectL2VAE, vae_loss_function

# Configuration
DATA_PATH = "data/l2_training_samples.npy"
MODEL_SAVE_PATH = "models/direct_l2_vae_v1.pth"
EPOCHS = 100
BATCH_SIZE = 32
LATENT_DIM = 64

def train():
    if not os.path.exists(DATA_PATH):
        print(f"Data not found: {DATA_PATH}")
        return

    # Load data
    data = np.load(DATA_PATH)
    data_tensor = torch.FloatTensor(data)
    
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    model = DirectL2VAE(timesteps=96, num_features=4, latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            
            recon_x, mu, logvar = model(x)
            loss = vae_loss_function(recon_x, x, mu, logvar)
            
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader.dataset):.4f}")
            
    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
