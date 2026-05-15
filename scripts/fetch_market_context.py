"""
Fetch BTC market context for all events in real/ and reconstructed/.

For each unique event date, downloads BTCUSDT 1-minute OHLCV from
Binance Data Vision (full history, free). From that kline, it:
  1. Derives a 6-feature market context L2 sequence (bid/ask price+size, buy_ratio, agg_imb)
  2. Classifies the BTC behaviour during that window as:
       normal    — BTC flat, < 3% peak move
       uncertain — BTC had a 3–8% move without confirmed retracement
       pumped    — BTC itself had a >= 5% spike with >= 20% retracement
  3. Saves the market context alongside each matched L2 file as *_market_ctx.csv
  4. Updates the *_meta.json market_regime field for every matched file

BTC data is cached in data/market_context/ so each date is only downloaded once.

Usage (from project root):
    python scripts/fetch_market_context.py
    python scripts/fetch_market_context.py --exchanges binance kucoin
    python scripts/fetch_market_context.py --overwrite   # rebuild existing ctx files
"""

import os
import sys
import re
import io
import json
import glob
import time
import zipfile
import argparse
import urllib.request
import urllib.error

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Config ─────────────────────────────────────────────────────────────────────

REAL_BASE          = "real"
RECONSTRUCTED_BASE = "reconstructed"
BTC_CACHE_DIR      = "data/market_context"

BINANCE_KLINE_URL = (
    "https://data.binance.vision/data/spot/daily/klines/"
    "BTCUSDT/1m/BTCUSDT-1m-{date}.zip"
)
RATE_SLEEP = 0.4

# BTC regime thresholds (looser than coin pump — market moves matter less)
MKT_UNCERTAIN_THRESH  = 0.03   # 3%+ move without clear dump = uncertain
MKT_PUMP_SPIKE_THRESH = 0.05   # 5%+ spike
MKT_PUMP_RETRACE      = 0.20   # 20%+ retracement = BTC was pumped

TIMESTEPS = 96   # match the coin window length

COLS_12 = ['open_ts_ms', 'o', 'h', 'l', 'c', 'v',
           'close_t', 'quote_v', 'n_trades',
           'taker_buy_base', 'taker_buy_quote', 'ignore']


# ── BTC kline fetch + cache ────────────────────────────────────────────────────

def fetch_btc_kline(date: str) -> pd.DataFrame | None:
    """Download and cache BTC 1m kline for a given date. Returns DataFrame or None."""
    os.makedirs(BTC_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(BTC_CACHE_DIR, f"BTCUSDT-1m-{date}.csv")

    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path)
        except Exception:
            pass

    url = BINANCE_KLINE_URL.format(date=date)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"    [HTTP {e.code}] BTC {date}")
        return None
    except Exception as e:
        print(f"    [ERR] BTC {date}: {e}")
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(f, header=None, names=COLS_12)
    except Exception as e:
        print(f"    [PARSE ERR] BTC {date}: {e}")
        return None

    df.to_csv(cache_path, index=False)
    return df


# ── BTC regime classification ──────────────────────────────────────────────────

def classify_btc_regime(df: pd.DataFrame) -> str:
    """
    Classify BTC behaviour in a kline window as normal / uncertain / pumped.
    Uses the same spike+retracement logic as coin pump labelling, with looser thresholds.
    """
    if df is None or len(df) < 5:
        return "normal"

    prices = df["c"].astype(float).values
    highs  = df["h"].astype(float).values
    p0     = float(prices[0])
    if p0 <= 0:
        return "normal"

    peak_idx  = int(np.argmax(highs))
    peak_high = float(highs[peak_idx])
    increase  = (peak_high - p0) / p0

    if increase < MKT_UNCERTAIN_THRESH:
        return "normal"

    if increase < MKT_PUMP_SPIKE_THRESH:
        return "uncertain"

    # Check for retracement after peak
    after = prices[peak_idx:]
    if len(after) < 2:
        return "uncertain"
    min_after   = float(after.min())
    price_range = peak_high - p0
    retracement = (peak_high - min_after) / price_range if price_range > 0 else 0.0

    return "pumped" if retracement >= MKT_PUMP_RETRACE else "uncertain"


# ── Market context L2 builder ──────────────────────────────────────────────────

def build_market_ctx(btc_df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """
    Convert BTC 1m kline data into a 6-feature market context sequence
    aligned to TIMESTEPS candles.

    Features: bid_price, ask_price, bid_size, ask_size, buy_ratio, aggressor_imbalance
    All prices normalised to the first mid-price (same as coin normalisation).
    """
    df = btc_df.copy()
    for col in ("o", "h", "l", "c", "v", "taker_buy_base"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["o", "h", "l", "c"]).reset_index(drop=True)

    # Trim or pad to TIMESTEPS
    if len(df) > TIMESTEPS:
        # Centre the window around the peak candle
        peak_idx = int(df["h"].argmax())
        start    = max(0, peak_idx - TIMESTEPS // 2)
        start    = min(start, len(df) - TIMESTEPS)
        df       = df.iloc[start : start + TIMESTEPS].reset_index(drop=True)
    if len(df) < TIMESTEPS:
        # Pad by repeating the last row
        pad = pd.concat([df.iloc[[-1]]] * (TIMESTEPS - len(df)), ignore_index=True)
        df  = pd.concat([df, pad], ignore_index=True)

    close   = df["c"].values.astype(float)
    high    = df["h"].values.astype(float)
    low     = df["l"].values.astype(float)
    volume  = df["v"].values.astype(float)
    taker_b = df["taker_buy_base"].values.astype(float) if "taker_buy_base" in df.columns else volume * 0.5

    # Buy ratio from taker data
    total_vol = np.maximum(volume, 1e-10)
    buy_ratio = np.clip(taker_b / total_vol, 0.0, 1.0)
    agg_imb   = np.clip(buy_ratio * 2 - 1, -1.0, 1.0)

    # Construct synthetic bid/ask from OHLCV
    mid_price = (high + low) / 2
    spread    = (high - low) / np.maximum(mid_price, 1e-10) * 0.1   # 10% of HL range
    bid_price = mid_price * (1 - spread * 0.5)
    ask_price = mid_price * (1 + spread * 0.5)
    ask_price = np.maximum(ask_price, bid_price + 1e-8)

    # Size proxy: use volume normalised to window mean
    mean_vol  = np.maximum(total_vol.mean(), 1e-10)
    bid_size  = total_vol * buy_ratio / mean_vol
    ask_size  = total_vol * (1 - buy_ratio) / mean_vol

    return pd.DataFrame({
        "bid_price":            bid_price,
        "ask_price":            ask_price,
        "bid_size":             bid_size,
        "ask_size":             ask_size,
        "buy_ratio":            buy_ratio,
        "aggressor_imbalance":  agg_imb,
    })


# ── File helpers ───────────────────────────────────────────────────────────────

def meta_path_for(l2_path: str) -> str:
    return re.sub(r'(_direct_L2|_synthetic_L2|_L2)\.csv$', '_meta.json', l2_path)


def ctx_path_for(l2_path: str) -> str:
    return re.sub(r'(_direct_L2|_synthetic_L2|_L2)\.csv$', '_market_ctx.csv', l2_path)


def update_meta_regime(meta_path: str, regime: str):
    try:
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}
        meta["market_regime"] = regime
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass


# ── Per-exchange processing ────────────────────────────────────────────────────

def process_exchange(exchange: str, overwrite: bool):
    saved = skipped = failed = 0

    for base in [REAL_BASE, RECONSTRUCTED_BASE]:
        exch_dir = os.path.join(base, exchange)
        if not os.path.isdir(exch_dir):
            continue

        l2_files = [
            p for p in glob.glob(
                os.path.join(exch_dir, "**", "*_L2.csv"), recursive=True
            )
            if "_market_ctx" not in p
        ]

        for l2_path in l2_files:
            cp = ctx_path_for(l2_path)
            if os.path.exists(cp) and not overwrite:
                skipped += 1
                continue

            # Extract date from path or filename
            m = re.search(r'(\d{4}-\d{2}-\d{2})', l2_path)
            if not m:
                failed += 1
                continue
            date = m.group(1)

            btc_df = fetch_btc_kline(date)
            time.sleep(RATE_SLEEP)

            if btc_df is None:
                failed += 1
                continue

            regime = classify_btc_regime(btc_df)
            ctx_df = build_market_ctx(btc_df, regime)

            try:
                ctx_df.to_csv(cp, index=False)
                update_meta_regime(meta_path_for(l2_path), regime)
                saved += 1
            except Exception as e:
                print(f"    [SAVE ERR] {cp}: {e}")
                failed += 1

    return saved, skipped, failed


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchanges", nargs="+",
                        default=["binance", "bybit", "kucoin", "okx",
                                 "huobi", "mexc", "gateio", "bitget"],
                        help="Which exchanges to process")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild _market_ctx.csv files that already exist")
    args = parser.parse_args()

    os.makedirs(BTC_CACHE_DIR, exist_ok=True)

    print("=" * 65)
    print("  Market Context Builder")
    print("  Source: data.binance.vision (BTCUSDT 1m klines, cached)")
    print("  Output: *_market_ctx.csv + market_regime in *_meta.json")
    print("=" * 65)

    total_saved = total_skipped = total_failed = 0

    for ex in args.exchanges:
        print(f"\n--- {ex.upper()} ---")
        s, sk, f = process_exchange(ex, args.overwrite)
        total_saved   += s
        total_skipped += sk
        total_failed  += f
        print(f"  saved={s}  skipped={sk}  failed={f}")

    print(f"\n{'='*65}")
    print(f"  Total saved   : {total_saved}")
    print(f"  Total skipped : {total_skipped}")
    print(f"  Total failed  : {total_failed}")
    print(f"  BTC klines cached in: {BTC_CACHE_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
