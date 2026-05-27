import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.direct_l2_vae import DirectL2VAE, vae_loss_function

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH       = "data/l2_training_samples.npy"
MODEL_SAVE_PATH = "models/direct_l2_vae_v1.pth"
EPOCHS          = 100
BATCH_SIZE      = 32
LATENT_DIM      = 64
GRAD_CLIP       = 1.0     # max gradient norm — prevents explosion → NaN weights
MIN_GOOD_EPOCHS = 5       # abort if loss is still NaN after this many epochs


def train():
    if not os.path.exists(DATA_PATH):
        print(f"Data not found: {DATA_PATH}")
        return

    # ── Load and validate data ─────────────────────────────────────────────────
    data = np.load(DATA_PATH)

    n_before = len(data)
    valid = np.isfinite(data).all(axis=(1, 2))
    data  = data[valid]
    n_after = len(data)

    if n_before != n_after:
        print(f"  [WARN] Dropped {n_before - n_after} NaN/Inf samples from training array "
              f"({n_after} remain).")

    if n_after == 0:
        print("ERROR: No finite samples in training data. "
              "Re-run scripts/prepare_l2_training_data.py first.")
        return

    print(f"  Training samples : {n_after}  shape: {data.shape}")

    data_tensor = torch.FloatTensor(data)
    dataset     = TensorDataset(data_tensor)
    dataloader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device           : {device}")

    model     = DirectL2VAE(timesteps=96, num_features=4,
                            latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=8, factor=0.5, min_lr=1e-5)

    best_loss  = float("inf")
    nan_epochs = 0

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        n_batches  = 0

        for (x,) in dataloader:
            x = x.to(device)

            # Skip any batch that somehow still has NaN
            if not torch.isfinite(x).all():
                continue

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss_function(recon_x, x, mu, logvar)

            if not torch.isfinite(loss):
                continue   # skip bad batch, don't backprop NaN

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        if n_batches == 0:
            nan_epochs += 1
            if nan_epochs >= MIN_GOOD_EPOCHS:
                print(f"  [ERROR] Loss was NaN/Inf for {nan_epochs} consecutive epochs. "
                      "Check your training data for corrupt values.")
                return
            continue

        avg_loss = total_loss / n_batches
        nan_epochs = 0

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        scheduler.step(avg_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>3}/{EPOCHS}  loss: {avg_loss:.6f}  "
                  f"(best: {best_loss:.6f}  lr: {optimizer.param_groups[0]['lr']:.2e})")

    if best_loss < float("inf"):
        print(f"\n  Best loss        : {best_loss:.6f}")
        print(f"  Model saved to   : {MODEL_SAVE_PATH}")
    else:
        print("\n  [ERROR] Training produced no valid checkpoint. "
              "Inspect data/l2_training_samples.npy for NaN values.")


if __name__ == "__main__":
    train()
