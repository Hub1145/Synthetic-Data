"""
DirectL2VAE synthetic generator — Type-B.

Generates L2 orderbook + tradebook + market context directly from the trained
DirectL2VAE, across all 8 exchanges. Each sample is tagged with:
  - coin_regime  : pump | control
  - market_regime: normal | uncertain | pumped   (what BTC was doing simultaneously)
  - peak_bucket  : micro | small | medium | large | major | extreme  (pump samples only)

This produces training data for the 3×3 coin×market regime matrix:

  Coin: Normal    Coin: Uncertain   Coin: Pumped
  baseline        weak signal       MOST SUSPICIOUS   ← Market: Normal
  noise           noise             ambiguous         ← Market: Uncertain
  coin lagging    coin catching up  least suspicious  ← Market: Pumped

Files saved per sample (in synthetic/[exchange]/[regime]/[symbol]/):
  [SYMBOL]_direct_L2.csv    — coin orderbook (96 × 10-level depth)
  [SYMBOL]_trades.csv       — coin tradebook (timestamp_idx, side, price, size)
  [SYMBOL]_market_ctx.csv   — market (BTC) context (96 × 6 features)
  [SYMBOL]_meta.json        — coin_regime, market_regime, peak_bucket, peak_pct, peak_idx

Usage (from project root):
    python scripts/generate_direct_synthetic.py
"""

import os
import sys
import json
import string
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import numpy as np
import pandas as pd
import joblib
from models.direct_l2_vae import DirectL2VAE

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_PATH  = "models/direct_l2_vae_v1.pth"
SCALER_PATH = "models/l2_scaler.pkl"
OUTPUT_BASE = "synthetic"
TIMESTEPS   = 96
LATENT_DIM  = 64
N_DEPTH     = 10

EXCHANGES = ['binance', 'bybit', 'kucoin', 'okx', 'huobi', 'mexc', 'gateio', 'bitget']

# Pump magnitude buckets matching Andy's 5–10% increment scheme
PEAK_BUCKETS = [
    ("micro",   0.05, 0.10),
    ("small",   0.10, 0.20),
    ("medium",  0.20, 0.30),
    ("large",   0.30, 0.40),
    ("major",   0.40, 0.50),
    ("extreme", 0.50, 1.00),
]

MARKET_REGIMES = ["normal", "uncertain", "pumped"]

NUM_PUMP_SAMPLES    = 54   # 6 buckets × 3 market regimes = 18 cells × 3 = 54
NUM_CONTROL_SAMPLES = 51   # 3 market regimes × 17 each = 51


# ── Symbol helper ──────────────────────────────────────────────────────────────

def get_random_symbol():
    return "S_" + "".join(random.choices(string.ascii_uppercase, k=3)) + "USDT"


# ── Orderbook depth helper ─────────────────────────────────────────────────────

def generate_depth_levels(bid_price, ask_price, bid_size, ask_size, n_levels=N_DEPTH):
    T   = len(bid_price)
    mid = (bid_price + ask_price) / 2.0
    imbalance    = np.clip(bid_size / np.maximum(ask_size, 1e-10), 0.1, 10.0)
    base_step    = np.random.uniform(0.0005, 0.0015, size=T)
    out = {}
    bp, ap = bid_price.copy(), ask_price.copy()
    bs, as_ = bid_size.copy(), ask_size.copy()
    for k in range(2, n_levels + 1):
        step = mid * base_step * (1.0 + 0.15 * (k - 1))
        bp   = bp - step * np.random.uniform(0.85, 1.15, size=T)
        ap   = ap + step * np.random.uniform(0.85, 1.15, size=T)
        bp   = np.maximum(bp, mid * 0.5)
        bd   = np.clip(np.random.uniform(0.60, 0.80, size=T) + 0.05 * np.log(imbalance), 0.40, 0.90)
        ad   = np.clip(np.random.uniform(0.60, 0.80, size=T) - 0.05 * np.log(imbalance), 0.40, 0.90)
        bs   = np.maximum(bs * bd * np.random.uniform(0.80, 1.20, size=T), 1e-8)
        as_  = np.maximum(as_ * ad * np.random.uniform(0.80, 1.20, size=T), 1e-8)
        out[f"bid_price_{k}"] = bp.copy()
        out[f"bid_size_{k}"]  = bs.copy()
        out[f"ask_price_{k}"] = ap.copy()
        out[f"ask_size_{k}"]  = as_.copy()
    return out


# ── Market context generator ───────────────────────────────────────────────────

def generate_market_ctx(regime: str) -> pd.DataFrame:
    """
    Generate a synthetic BTC-like market context sequence (96 timesteps, 6 features).
    Saved as _market_ctx.csv — NOT a coin L2 file, so add_orderbook_depth.py skips it.

    Features: bid_price, ask_price, bid_size, ask_size, buy_ratio, aggressor_imbalance
    """
    T = TIMESTEPS

    if regime == "normal":
        # BTC flat/slight drift — market doing nothing special
        noise  = np.random.randn(T) * 0.003
        price  = np.cumprod(1 + noise)
        b_size = np.random.uniform(0.8, 1.2, T)
        a_size = np.random.uniform(0.8, 1.2, T)
        br     = np.random.uniform(0.44, 0.56, T)

    elif regime == "uncertain":
        # BTC showing moderate moves, no clear direction
        noise  = np.random.randn(T) * 0.008
        trend  = np.random.choice([-1, 1]) * np.linspace(0, 0.05, T)
        price  = np.cumprod(1 + noise) + trend
        price  = np.maximum(price, 0.5)
        b_size = np.random.uniform(0.4, 2.0, T)
        a_size = np.random.uniform(0.4, 2.0, T)
        br     = pd.Series(np.random.uniform(0.28, 0.72, T)).rolling(5, min_periods=1).mean().values

    else:  # pumped — BTC itself has a pump-like pattern
        mkt_peak      = np.random.randint(15, T - 10)
        mkt_intensity = np.random.uniform(0.05, 0.15)
        sharpness     = np.random.uniform(60, 120)
        window        = np.arange(T)
        left  = window <= mkt_peak
        right = ~left
        gauss = np.zeros(T)
        gauss[left]  = np.exp(-((window[left]  - mkt_peak) ** 2) / (sharpness * 0.6))
        gauss[right] = np.exp(-((window[right] - mkt_peak) ** 2) / (sharpness * 1.4))
        price  = 1.0 + mkt_intensity * gauss
        b_size = np.ones(T)
        a_size = np.ones(T)
        b_size[left]  += mkt_intensity * 2.0 * gauss[left]
        a_size[right] += mkt_intensity * 3.0 * gauss[right]
        br = np.full(T, 0.50)
        br[left]  = np.clip(0.70 + 0.10 * gauss[left],  0, 1)
        br[right] = np.clip(0.30 - 0.10 * gauss[right], 0, 1)

    spread    = np.random.uniform(0.0001, 0.0005, T)
    bid_price = price * (1 - spread)
    ask_price = price * (1 + spread)
    ask_price = np.maximum(ask_price, bid_price + 1e-8)
    br        = np.clip(br, 0.0, 1.0)
    agg_imb   = np.clip(br * 2 - 1, -1.0, 1.0)

    return pd.DataFrame({
        "bid_price":            bid_price,
        "ask_price":            ask_price,
        "bid_size":             b_size,
        "ask_size":             a_size,
        "buy_ratio":            br,
        "aggressor_imbalance":  agg_imb,
    })


# ── Pump injection ─────────────────────────────────────────────────────────────

def inject_pump(sample: np.ndarray,
                lo: float, hi: float) -> tuple[np.ndarray, float, int]:
    """
    Apply an asymmetric pump-and-dump signature to a base VAE sample.
    Returns (modified_sample, actual_intensity, peak_idx).
    """
    intensity = np.random.uniform(lo, hi)
    peak_idx  = np.random.randint(20, TIMESTEPS - 15)
    sharpness = np.random.uniform(40.0, 150.0)
    window    = np.arange(TIMESTEPS)
    left      = window <= peak_idx
    right     = ~left
    gauss     = np.zeros(TIMESTEPS)
    gauss[left]  = np.exp(-((window[left]  - peak_idx) ** 2) / (sharpness * 0.6))
    gauss[right] = np.exp(-((window[right] - peak_idx) ** 2) / (sharpness * 1.4))

    price_mult       = 1.0 + intensity * gauss
    bid_mult         = np.ones(TIMESTEPS)
    ask_mult         = np.ones(TIMESTEPS)
    bid_mult[left]   = 1.0 + (intensity * 4.0) * gauss[left]
    ask_mult[right]  = 1.0 + (intensity * 5.0) * gauss[right]
    ask_mult[left]   = 1.0 - (intensity * 0.3) * gauss[left]
    bid_mult[right]  = 1.0 - (intensity * 0.2) * gauss[right]
    bid_mult         = np.maximum(bid_mult, 0.1)
    ask_mult         = np.maximum(ask_mult, 0.1)

    out = sample.copy()
    out[:, 0:2] *= price_mult[:, np.newaxis]
    out[:, 2]   *= bid_mult
    out[:, 3]   *= ask_mult
    return out, intensity, peak_idx


# ── Tradebook generator ────────────────────────────────────────────────────────

def generate_trades(df: pd.DataFrame, is_pump: bool,
                    peak_idx: int | None) -> pd.DataFrame:
    rows = []
    for t in range(TIMESTEPS):
        if is_pump:
            buy_prob = 0.80 if t <= peak_idx else 0.25
        else:
            buy_prob = 0.50
        for _ in range(np.random.randint(1, 8)):
            side  = "buy" if np.random.random() < buy_prob else "sell"
            price = float(df.iloc[t]["ask_price"] if side == "buy"
                          else df.iloc[t]["bid_price"])
            price *= (1 + np.random.uniform(-0.0001, 0.0001))
            size  = float(np.random.uniform(0.01, 2.0 if is_pump else 1.0))
            rows.append({"timestamp_idx": t, "side": side,
                         "price": price, "size": size})
    return pd.DataFrame(rows)


# ── Main generation loop ───────────────────────────────────────────────────────

def generate_direct_data():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    scaler = joblib.load(SCALER_PATH)
    device = torch.device("cpu")
    model  = DirectL2VAE(timesteps=TIMESTEPS, num_features=4,
                         latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Pre-build ordered cell lists so samples are evenly distributed
    # Pump cells: 18 combinations (6 buckets × 3 market regimes)
    pump_cells = [
        (bname, lo, hi, mkt)
        for bname, lo, hi in PEAK_BUCKETS
        for mkt in MARKET_REGIMES
    ]  # 18 entries
    # Control cells: 3 market regimes
    ctrl_cells = MARKET_REGIMES  # 3 entries

    total_generated = 0

    for ex in EXCHANGES:
        print(f"\n{'='*55}")
        print(f"  Exchange: {ex.upper()}")
        print(f"{'='*55}")

        # ── PUMP samples ──
        for i in range(NUM_PUMP_SAMPLES):
            cell              = pump_cells[i % len(pump_cells)]
            bucket_name, lo, hi, market_regime = cell
            symbol = get_random_symbol()

            with torch.no_grad():
                z      = torch.randn(1, LATENT_DIM)
                sample = model.decode(z).squeeze(0).numpy()

            sample = scaler.inverse_transform(sample)
            sample, intensity, peak_idx = inject_pump(sample, lo, hi)
            sample[:, 1] = np.maximum(sample[:, 1], sample[:, 0] + 1e-8)

            df = pd.DataFrame(sample,
                              columns=["bid_price", "ask_price", "bid_size", "ask_size"])
            df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
            for col, arr in generate_depth_levels(
                    df["bid_price"].values, df["ask_price"].values,
                    df["bid_size"].values,  df["ask_size"].values).items():
                df[col] = arr

            trades_df  = generate_trades(df, is_pump=True, peak_idx=peak_idx)
            market_ctx = generate_market_ctx(market_regime)
            meta = {
                "coin_regime":   "pump",
                "market_regime": market_regime,
                "peak_bucket":   bucket_name,
                "peak_pct":      round(intensity * 100, 2),
                "peak_idx":      int(peak_idx),
            }

            out_dir = os.path.join(OUTPUT_BASE, ex, "pumps", symbol)
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(        os.path.join(out_dir, f"{symbol}_direct_L2.csv"),   index=False)
            trades_df.to_csv( os.path.join(out_dir, f"{symbol}_trades.csv"),      index=False)
            market_ctx.to_csv(os.path.join(out_dir, f"{symbol}_market_ctx.csv"),  index=False)
            with open(        os.path.join(out_dir, f"{symbol}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            total_generated += 1
            if i % 18 == 0:
                print(f"  [pump {i+1:>3}/{NUM_PUMP_SAMPLES}] "
                      f"market={market_regime:<10} bucket={bucket_name}")

        # ── CONTROL samples ──
        for i in range(NUM_CONTROL_SAMPLES):
            market_regime = ctrl_cells[i % len(ctrl_cells)]
            symbol = get_random_symbol()

            with torch.no_grad():
                z      = torch.randn(1, LATENT_DIM)
                sample = model.decode(z).squeeze(0).numpy()

            sample = scaler.inverse_transform(sample)
            sample[:, 1] = np.maximum(sample[:, 1], sample[:, 0] + 1e-8)

            df = pd.DataFrame(sample,
                              columns=["bid_price", "ask_price", "bid_size", "ask_size"])
            df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
            for col, arr in generate_depth_levels(
                    df["bid_price"].values, df["ask_price"].values,
                    df["bid_size"].values,  df["ask_size"].values).items():
                df[col] = arr

            trades_df  = generate_trades(df, is_pump=False, peak_idx=None)
            market_ctx = generate_market_ctx(market_regime)
            meta = {
                "coin_regime":   "control",
                "market_regime": market_regime,
                "peak_bucket":   None,
                "peak_pct":      None,
                "peak_idx":      None,
            }

            out_dir = os.path.join(OUTPUT_BASE, ex, "control", symbol)
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(        os.path.join(out_dir, f"{symbol}_direct_L2.csv"),   index=False)
            trades_df.to_csv( os.path.join(out_dir, f"{symbol}_trades.csv"),      index=False)
            market_ctx.to_csv(os.path.join(out_dir, f"{symbol}_market_ctx.csv"),  index=False)
            with open(        os.path.join(out_dir, f"{symbol}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            total_generated += 1

        print(f"  {ex}: {NUM_PUMP_SAMPLES} pump + {NUM_CONTROL_SAMPLES} control = "
              f"{NUM_PUMP_SAMPLES + NUM_CONTROL_SAMPLES} samples")

    print(f"\n{'='*55}")
    print(f"  Total generated: {total_generated} samples across {len(EXCHANGES)} exchanges")
    print(f"  Pump cells covered: 6 peak buckets × 3 market regimes = 18")
    print(f"  Control cells: 3 market regimes")
    print(f"{'='*55}")


if __name__ == "__main__":
    generate_direct_data()
