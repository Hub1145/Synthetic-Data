import os
import pandas as pd
import numpy as np
import glob
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import joblib
from models.base_vae import BaseVAE

# Configuration
WINDOW_SIZE = 24
LATENT_DIM = 32 # Increased for global diversity
EPOCHS = 100
BATCH_SIZE = 64
MODEL_SAVE_PATH = "models/global_vae_v1.pth"
SCALER_SAVE_PATH = "models/global_scaler_v1.pkl"

def load_all_data():
    # Scan for all kline files in the real/ hierarchy
    # Matches: real/[EXCHANGE]/[REGIME]/[SYMBOL]/processed/*.csv
    files = glob.glob("real/**/**/*.csv", recursive=True)
    print(f"Found {len(files)} data files for global training.")
    
    all_windows = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            # Ensure we have o,h,l,c,v and they are numeric
            cols = ['o', 'h', 'l', 'c', 'v']
            if not all(col in df.columns for col in cols):
                continue
                
            # Filter for only numeric data (skips headers if accidentally duplicated)
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            df = df.dropna(subset=cols)
            
            data = df[cols].values
            
            # Slide window
            if len(data) >= WINDOW_SIZE:
                for i in range(len(data) - WINDOW_SIZE):
                    window = data[i : i + WINDOW_SIZE]
                    all_windows.append(window)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
            
    return np.array(all_windows)

def train():
    os.makedirs("models", exist_ok=True)
    
    # 1. Load and Scale
    raw_data = load_all_data()
    print(f"Total training windows: {len(raw_data)}")
    
    # Flatten for scaling
    n_samples, n_steps, n_features = raw_data.shape
    raw_data_flat = raw_data.reshape(-1, n_features)
    
    scaler = StandardScaler()
    scaled_data_flat = scaler.fit_transform(raw_data_flat)
    scaled_data = scaled_data_flat.reshape(n_samples, n_steps, n_features)
    
    # Save scaler
    joblib.dump(scaler, SCALER_SAVE_PATH)
    
    # 2. Prepare Torch
    dataset = TensorDataset(torch.FloatTensor(scaled_data))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Model
    model = BaseVAE(window_size=WINDOW_SIZE, num_features=n_features, latent_dim=LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            x = batch[0].transpose(1, 2).to(model.device) # Shape (B, 5, 24)
            optimizer.zero_grad()
            
            recon_x, mu, logvar = model(x)
            
            # Loss Calculation (Matching fit() logic)
            recon_loss = F.mse_loss(recon_x, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.1 * kl_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader.dataset):.4f}")
            
    # 5. Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Global Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
