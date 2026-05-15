import os
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
import joblib

# Configuration
INPUT_BASE = "reconstructed"
OUTPUT_DATA = "data/l2_training_samples.npy"
SCALER_PATH = "models/l2_scaler.pkl"
TIMESTEPS = 96
FEATURES = ['bid_price', 'ask_price', 'bid_size', 'ask_size']

def prepare_data():
    print(f"Scanning {INPUT_BASE} for reconstructed L2 files...")
    files = glob.glob(os.path.join(INPUT_BASE, "**/*_L2.csv"), recursive=True)
    print(f"Found {len(files)} files.")
    
    all_samples = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) < TIMESTEPS:
                continue
            
            # Take the first TIMESTEPS
            data = df[FEATURES].values[:TIMESTEPS]
            
            # Normalize prices relative to the first mid-price in the window
            mid0 = (data[0, 0] + data[0, 1]) / 2
            data[:, 0:2] /= mid0
            
            all_samples.append(data)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if not all_samples:
        print("No valid samples found.")
        return
        
    samples_array = np.array(all_samples) # [Samples, 96, 4]
    print(f"Created {len(samples_array)} training samples. Shape: {samples_array.shape}")
    
    # Flatten for scaling
    B, T, F = samples_array.shape
    samples_flat = samples_array.reshape(-1, F)
    
    scaler = StandardScaler()
    samples_scaled = scaler.fit_transform(samples_flat)
    samples_reshaped = samples_scaled.reshape(B, T, F)
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_DATA), exist_ok=True)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    np.save(OUTPUT_DATA, samples_reshaped)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved training data to {OUTPUT_DATA} and scaler to {SCALER_PATH}")

if __name__ == "__main__":
    prepare_data()
