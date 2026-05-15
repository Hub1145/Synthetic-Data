"""
Expand all 1-level L2 files in reconstructed/ to 10-level orderbook depth.
Adds columns: bid_price_2..10, bid_size_2..10, ask_price_2..10, ask_size_2..10
(Level 1 = existing bid_price / ask_price / bid_size / ask_size columns, unchanged)

Depth model:
  - Price gaps widen slightly at each level (log-normal with drift)
  - Sizes decay with distance from best price, modulated by bid/ask imbalance
  - Pump-signature preserved: heavy bid side → thicker bid depth, heavy ask → thicker ask depth
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import glob
import numpy as np
import pandas as pd

RECONSTRUCTED_BASE = "reconstructed"
N_LEVELS = 10
ALREADY_DONE_COL = "bid_price_2"   # presence means depth already added


def generate_depth(bid_price: np.ndarray, ask_price: np.ndarray,
                   bid_size: np.ndarray, ask_size: np.ndarray,
                   n_levels: int = 10, seed: int = 0) -> dict:
    """
    Vectorised depth generation across all timesteps in a file.

    Returns dict of column arrays ready to be added to the DataFrame.
    """
    rng = np.random.default_rng(seed)
    T = len(bid_price)

    mid = (bid_price + ask_price) / 2.0
    spread = np.maximum(ask_price - bid_price, mid * 1e-6)

    # Imbalance ratio: bid_size / ask_size — drives which side is thicker
    imbalance = np.clip(bid_size / np.maximum(ask_size, 1e-10), 0.1, 10.0)

    # Base price step per level: ~0.05–0.15% of mid, wider as levels go deeper
    base_step_pct = rng.uniform(0.0005, 0.0015, size=T)

    out = {}

    bid_prev_price = bid_price.copy()
    ask_prev_price = ask_price.copy()
    bid_prev_size  = bid_size.copy()
    ask_prev_size  = ask_size.copy()

    for k in range(2, n_levels + 1):
        # Price step grows with depth
        depth_factor = 1.0 + 0.15 * (k - 1)
        price_step   = mid * base_step_pct * depth_factor

        # Bid levels go DOWN in price
        noise_b = rng.uniform(0.85, 1.15, size=T)
        bid_p   = bid_prev_price - price_step * noise_b
        bid_p   = np.maximum(bid_p, mid * 0.5)  # sanity floor

        # Ask levels go UP in price
        noise_a = rng.uniform(0.85, 1.15, size=T)
        ask_p   = ask_prev_price + price_step * noise_a

        # Size decay: base decay + imbalance effect
        # High imbalance (buy pressure) → bid levels stay thick, ask thins faster
        bid_decay = rng.uniform(0.60, 0.80, size=T)
        ask_decay = rng.uniform(0.60, 0.80, size=T)

        # Imbalance > 1 = buy pressure: bids decay slower, asks faster
        bid_decay = bid_decay + 0.05 * np.log(imbalance)   # positive when buy pressure
        ask_decay = ask_decay - 0.05 * np.log(imbalance)   # negative when buy pressure
        bid_decay = np.clip(bid_decay, 0.40, 0.90)
        ask_decay = np.clip(ask_decay, 0.40, 0.90)

        bid_s = bid_prev_size * bid_decay * rng.uniform(0.80, 1.20, size=T)
        ask_s = ask_prev_size * ask_decay * rng.uniform(0.80, 1.20, size=T)
        bid_s = np.maximum(bid_s, 1e-8)
        ask_s = np.maximum(ask_s, 1e-8)

        out[f"bid_price_{k}"] = bid_p
        out[f"bid_size_{k}"]  = bid_s
        out[f"ask_price_{k}"] = ask_p
        out[f"ask_size_{k}"]  = ask_s

        bid_prev_price = bid_p
        ask_prev_price = ask_p
        bid_prev_size  = bid_s
        ask_prev_size  = ask_s

    return out


def add_depth_to_file(filepath: str) -> bool:
    try:
        df = pd.read_csv(filepath)

        required = {"bid_price", "ask_price", "bid_size", "ask_size"}
        if not required.issubset(df.columns):
            return False

        if ALREADY_DONE_COL in df.columns:
            return False   # already has depth

        df = df.dropna(subset=list(required))
        if len(df) == 0:
            return False

        bp = df["bid_price"].values.astype(np.float64)
        ap = df["ask_price"].values.astype(np.float64)
        bs = df["bid_size"].values.astype(np.float64)
        as_ = df["ask_size"].values.astype(np.float64)

        # Use file path hash as seed for reproducibility
        seed = hash(filepath) % (2**31)
        depth_cols = generate_depth(bp, ap, bs, as_, n_levels=N_LEVELS, seed=seed)

        for col, arr in depth_cols.items():
            df[col] = arr

        df.to_csv(filepath, index=False)
        return True

    except Exception as e:
        print(f"  [WARN] {filepath}: {e}")
        return False


def main():
    print("=" * 62)
    print(f"  Orderbook Depth Expansion  ({N_LEVELS} levels)")
    print(f"  Target: {RECONSTRUCTED_BASE}/")
    print("=" * 62)

    all_l2 = glob.glob(
        os.path.join(RECONSTRUCTED_BASE, "**", "*_L2.csv"), recursive=True
    )
    print(f"\n  Found {len(all_l2)} L2 files\n")

    updated = skipped = failed = 0
    prev_exchange = None

    for i, fpath in enumerate(all_l2):
        norm  = fpath.replace("\\", "/")
        parts = norm.split("/")
        try:
            rec_idx  = parts.index("reconstructed")
            exchange = parts[rec_idx + 1]
        except (ValueError, IndexError):
            exchange = "?"

        if exchange != prev_exchange:
            print(f"\n  [{exchange.upper()}]")
            prev_exchange = exchange

        result = add_depth_to_file(fpath)
        if result:
            updated += 1
        else:
            # Check if it was already done or skipped
            try:
                df = pd.read_csv(fpath, nrows=1)
                if ALREADY_DONE_COL in df.columns:
                    skipped += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

        if (i + 1) % 500 == 0:
            print(f"    Progress: {i+1}/{len(all_l2)}  "
                  f"(updated={updated}, skipped={skipped}, failed={failed})")

    print(f"\n{'='*62}")
    print(f"  Total L2 files : {len(all_l2)}")
    print(f"  Depth added    : {updated}")
    print(f"  Already had depth : {skipped}")
    print(f"  Failed/skipped : {failed}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
