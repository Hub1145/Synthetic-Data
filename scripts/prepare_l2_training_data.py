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
    files = [p for p in glob.glob(os.path.join(INPUT_BASE, "**/*_L2.csv"), recursive=True)
             if "_market_ctx" not in p]
    print(f"Found {len(files)} files.")

    all_samples = []
    skipped_nan = 0
    skipped_mid = 0

    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) < TIMESTEPS:
                continue

            # Take the first TIMESTEPS rows
            data = df[FEATURES].values[:TIMESTEPS].astype(np.float64)

            # Drop samples that already contain NaN/Inf before normalisation
            if not np.isfinite(data).all():
                skipped_nan += 1
                continue

            # Normalise prices relative to first mid-price
            mid0 = (data[0, 0] + data[0, 1]) / 2.0
            if mid0 <= 0:
                skipped_mid += 1
                continue
            data[:, 0:2] /= mid0

            # Drop samples that became NaN/Inf during normalisation
            if not np.isfinite(data).all():
                skipped_nan += 1
                continue

            # Drop samples with non-positive sizes (guard against corrupt data)
            mean_sz = data[:, 2:4].mean()
            if mean_sz <= 0:
                skipped_nan += 1
                continue
            data[:, 2:4] /= mean_sz

            all_samples.append(data.astype(np.float32))
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print(f"  Skipped (mid0 ≤ 0)  : {skipped_mid}")
    print(f"  Skipped (NaN/Inf)   : {skipped_nan}")
            
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
