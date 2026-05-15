import os
import sys
# Add root directory to sys.path to allow importing from 'models'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import torch
import joblib
import glob
from models.base_vae import BaseVAE

# Configuration
WINDOW_SIZE = 24
LATENT_DIM = 32
MODEL_PATH = "models/global_vae_v1.pth"
SCALER_PATH = "models/global_scaler_v1.pkl"
OUTPUT_BASE = "synthetic"
NUM_SAMPLES_PER_EXCHANGE = 100 # 50 Pump, 50 Control

def generate_global_data():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}. Waiting for training to finish...")
        return

    # Load model and scaler
    scaler = joblib.load(SCALER_PATH)
    model = BaseVAE(window_size=WINDOW_SIZE, num_features=5, latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    exchanges = ['binance', 'bybit', 'kucoin', 'okx', 'huobi', 'mexc', 'gateio', 'bitget']
    
    for ex in exchanges:
        print(f"\n--- Generating Synthetic Data for {ex.upper()} ---")
        
        # Generate baseline noise from Global VAE
        total_samples = NUM_SAMPLES_PER_EXCHANGE
        with torch.no_grad():
            z = torch.randn(total_samples, LATENT_DIM)
            # Sample from decoder: output shape (B, 5, 24)
            samples_tensor = model.decode(z)
            # Transpose back to (B, 24, 5) and move to CPU
            samples = samples_tensor.cpu().numpy().transpose(0, 2, 1)
            
        for i in range(total_samples):
            # 1. Inverse transform
            sample_real = scaler.inverse_transform(samples[i])
            
            is_pump = i < (NUM_SAMPLES_PER_EXCHANGE // 2)
            regime = "pumps" if is_pump else "control"
            import string
            import random
            random_chars = ''.join(random.choices(string.ascii_uppercase, k=3))
            symbol = f"S_{random_chars}USDT"
            
            if is_pump:
                # Inject Pump Signature
                pump_window = np.arange(WINDOW_SIZE)
                peak_idx = 12
                # Use slightly randomized intensities for variety
                intensity = np.random.uniform(0.05, 0.15)
                price_mult = 1.0 + intensity * np.exp(-((pump_window - peak_idx)**2) / 6.0)
                vol_mult = 1.0 + 15.0 * np.exp(-((pump_window - peak_idx)**2) / 3.0)
                
                sample_real[:, 0:4] *= price_mult[:, np.newaxis]
                sample_real[:, 4] *= vol_mult
            
            # 2. Save using strict schema
            target_dir = os.path.join(OUTPUT_BASE, ex, regime, symbol)
            os.makedirs(target_dir, exist_ok=True)
            
            df = pd.DataFrame(sample_real, columns=['o', 'h', 'l', 'c', 'v'])
            df['is_pump'] = 1 if is_pump else 0
            
            output_file = os.path.join(target_dir, f"{symbol}_synthetic.csv")
            df.to_csv(output_file, index=False)
            print(f"Generated {regime.upper()} for {ex}: {target_dir}")

if __name__ == "__main__":
    generate_global_data()
