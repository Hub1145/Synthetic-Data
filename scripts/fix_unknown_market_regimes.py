"""
Fix unknown market_regime in all meta.json files and generate missing _market_ctx.csv.

For every *_meta.json with market_regime == "unknown":
  1. Assigns a regime from [normal, uncertain, pumped] cycling evenly
  2. Generates a synthetic market context CSV using the same logic as generate_direct_synthetic.py
  3. Saves the *_market_ctx.csv alongside the L2 file
  4. Updates market_regime in the meta.json

Usage:
    python scripts/fix_unknown_market_regimes.py
    python scripts/fix_unknown_market_regimes.py --dirs reconstructed synthetic
"""

import os
import re
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TIMESTEPS    = 96
REGIMES      = ["normal", "uncertain", "pumped"]
BG_VOLATILITIES = ["calm", "normal", "volatile"]

# All 9 combinations cycled evenly
REGIME_COMBOS = [
    (r, v) for r in REGIMES for v in BG_VOLATILITIES
]


def generate_market_ctx(regime: str) -> pd.DataFrame:
    T = TIMESTEPS
    if regime == "normal":
        noise  = np.random.randn(T) * 0.003
        price  = np.cumprod(1 + noise)
        b_size = np.random.uniform(0.8, 1.2, T)
        a_size = np.random.uniform(0.8, 1.2, T)
        br     = np.random.uniform(0.44, 0.56, T)
    elif regime == "uncertain":
        noise  = np.random.randn(T) * 0.008
        trend  = np.random.choice([-1, 1]) * np.linspace(0, 0.05, T)
        price  = np.cumprod(1 + noise) + trend
        price  = np.maximum(price, 0.5)
        b_size = np.random.uniform(0.4, 2.0, T)
        a_size = np.random.uniform(0.4, 2.0, T)
        br     = pd.Series(np.random.uniform(0.28, 0.72, T)).rolling(5, min_periods=1).mean().values
    else:  # pumped
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
        "bid_price":           bid_price,
        "ask_price":           ask_price,
        "bid_size":            b_size,
        "ask_size":            a_size,
        "buy_ratio":           br,
        "aggressor_imbalance": agg_imb,
    })


def ctx_path_for(meta_path: str) -> str:
    return re.sub(r'_meta\.json$', '_market_ctx.csv', meta_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+",
                        default=["reconstructed", "synthetic"],
                        help="Top-level directories to scan")
    args = parser.parse_args()

    fixed = skipped = failed = 0
    counter = 0  # cycles through all 9 (regime, volatility) combos

    for base in args.dirs:
        meta_files = glob.glob(os.path.join(base, "**", "*_meta.json"), recursive=True)
        print(f"\nScanning {base}/ — {len(meta_files)} meta files found")

        for mpath in meta_files:
            try:
                with open(mpath) as f:
                    meta = json.load(f)
            except Exception:
                failed += 1
                continue

            needs_regime = meta.get("market_regime", "unknown") == "unknown"
            needs_vol    = "background_volatility" not in meta

            if not needs_regime and not needs_vol:
                skipped += 1
                continue

            regime, bg_vol = REGIME_COMBOS[counter % len(REGIME_COMBOS)]
            counter += 1

            ctx_path = ctx_path_for(mpath)
            try:
                if needs_regime:
                    ctx_df = generate_market_ctx(regime)
                    ctx_df.to_csv(ctx_path, index=False)
                    meta["market_regime"] = regime

                meta["background_volatility"] = bg_vol
                with open(mpath, "w") as f:
                    json.dump(meta, f, indent=2)

                fixed += 1
            except Exception as e:
                print(f"  [ERR] {mpath}: {e}")
                failed += 1

    print(f"\n{'='*60}")
    print(f"  Fixed   : {fixed}")
    print(f"  Skipped : {skipped}  (already complete)")
    print(f"  Failed  : {failed}")
    print(f"  Combos  : {len(REGIME_COMBOS)} (3 regimes x 3 volatilities, cycled evenly)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
