"""
Generate trades files for every L2 file that lacks one.

Priority order per L2 file:
  1. Already has *_trades.csv → skip
  2. Source is *_direct_L2.csv  → skip (generator always creates trades)
  3. Source is *_synthetic_L2.csv → find OHLCV in synthetic/
  4. Source is real *_L2.csv     → find OHLCV in real/
  5. Fallback: derive trades from the L2 mid_price column directly

Trade generation from OHLCV:
  buy_ratio = (close - low) / (high - low + ε)   [0..1]
  Generates 2–8 individual trades per minute.
  Prices are jittered ±0.01% around the appropriate bid/ask.
  Sizes drawn proportionally from OHLCV volume.

This gives every file a non-neutral trades signal that reflects the
real (or synthetic OHLCV) price direction — better than the 0.5 fill
used during training when no trades file exists.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import glob
import re
import numpy as np
import pandas as pd

REAL_BASE          = "real"
SYNTHETIC_BASE     = "synthetic"
RECONSTRUCTED_BASE = "reconstructed"

RNG = np.random.default_rng(42)


# ──────────────────────────────────────────────────────────────────────────────
# Trade generation helpers
# ──────────────────────────────────────────────────────────────────────────────

def trades_from_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Convert OHLCV rows to individual trade ticks.

    Handles both formats:
      - 12-col Binance (headerless): uses taker_buy_base if available
      - 6-col CCXT: open_ts_ms, o, h, l, c, v  (with header)
    """
    rows = []

    # Detect format
    has_header = isinstance(ohlcv.columns[0], str) and ohlcv.columns[0].startswith("open")
    if has_header:
        # CCXT 6-col
        o_col, h_col, l_col, c_col, v_col = ("o", "h", "l", "c", "v")
        taker_col = None
    else:
        # Binance 12-col headerless (cols 1–11 = o,h,l,c,v,...)
        # col indices: 0=ts, 1=o, 2=h, 3=l, 4=c, 5=v, 9=taker_buy_base
        ohlcv.columns = range(len(ohlcv.columns))
        o_col, h_col, l_col, c_col, v_col = 1, 2, 3, 4, 5
        taker_col = 9 if len(ohlcv.columns) > 9 else None

    for t, row in enumerate(ohlcv.itertuples(index=False)):
        try:
            o = float(getattr(row, str(o_col)) if has_header else row[o_col])
            h = float(getattr(row, str(h_col)) if has_header else row[h_col])
            l = float(getattr(row, str(l_col)) if has_header else row[l_col])
            c = float(getattr(row, str(c_col)) if has_header else row[c_col])
            v = float(getattr(row, str(v_col)) if has_header else row[v_col])
        except (AttributeError, IndexError, TypeError, ValueError):
            continue

        if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(l)
                and np.isfinite(c) and np.isfinite(v) and v > 0):
            continue

        # Buy ratio from price position in the candle
        hl_range = h - l
        if taker_col is not None:
            try:
                buy_vol = float(row[taker_col])
                buy_ratio = np.clip(buy_vol / max(v, 1e-12), 0.0, 1.0)
            except Exception:
                buy_ratio = np.clip((c - l) / max(hl_range, 1e-12), 0.05, 0.95)
        else:
            buy_ratio = np.clip((c - l) / max(hl_range, 1e-12), 0.05, 0.95)

        num_trades = int(RNG.integers(2, 9))
        sides = RNG.random(num_trades) < buy_ratio
        # Volume per trade ~ Dirichlet so they sum to v
        sizes = RNG.dirichlet(np.ones(num_trades)) * v

        mid = (h + l) / 2.0
        spread_half = max(hl_range * 0.0005, mid * 0.00005)

        for k in range(num_trades):
            side = "buy" if sides[k] else "sell"
            # Buys execute near ask (mid + spread), sells near bid (mid - spread)
            price = mid + (spread_half if side == "buy" else -spread_half)
            price *= (1.0 + RNG.uniform(-0.0001, 0.0001))
            rows.append({
                "timestamp_idx": t,
                "side":  side,
                "price": price,
                "size":  float(sizes[k]),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["timestamp_idx", "side", "price", "size"]
    )


def trades_from_l2(l2: pd.DataFrame) -> pd.DataFrame:
    """Fallback: derive trades from mid_price movement in the L2 file."""
    rows = []
    mid = l2["mid_price"].values if "mid_price" in l2.columns else (
        (l2["bid_price"] + l2["ask_price"]).values / 2.0
    )
    bid = l2["bid_price"].values
    ask = l2["ask_price"].values
    bid_sz = l2["bid_size"].values
    ask_sz = l2["ask_size"].values

    for t in range(len(mid)):
        if not np.isfinite(mid[t]):
            continue
        delta = mid[t] - mid[t - 1] if t > 0 else 0.0
        total = bid_sz[t] + ask_sz[t]
        buy_ratio = np.clip(0.5 + 0.5 * delta / max(abs(delta), mid[t] * 1e-6),
                            0.05, 0.95) if delta != 0 else 0.5

        num_trades = int(RNG.integers(2, 7))
        sides = RNG.random(num_trades) < buy_ratio
        sizes = RNG.dirichlet(np.ones(num_trades)) * max(total * 0.1, 1e-8)

        for k in range(num_trades):
            side = "buy" if sides[k] else "sell"
            price = ask[t] if side == "buy" else bid[t]
            price *= (1.0 + RNG.uniform(-0.0001, 0.0001))
            rows.append({"timestamp_idx": t, "side": side,
                         "price": float(price), "size": float(sizes[k])})

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["timestamp_idx", "side", "price", "size"]
    )


# ──────────────────────────────────────────────────────────────────────────────
# Source OHLCV lookup
# ──────────────────────────────────────────────────────────────────────────────

def find_ohlcv(exchange: str, regime: str, symbol: str,
               l2_basename: str) -> str | None:
    """Locate the source OHLCV file for a given L2 file."""
    # Extract date
    m = re.search(r"(\d{4}-\d{2}-\d{2})", l2_basename)
    date = m.group(1) if m else ""

    # Try real/
    real_sym = os.path.join(REAL_BASE, exchange, regime, symbol)
    if os.path.isdir(real_sym):
        candidates = (
            glob.glob(os.path.join(real_sym, f"*{date}*.csv")) +
            glob.glob(os.path.join(real_sym, "*.csv"))
        )
        # Exclude trades files
        candidates = [c for c in candidates if "trades" not in c.lower()]
        if candidates:
            return candidates[0]

    # Try synthetic/
    synth_sym = os.path.join(SYNTHETIC_BASE, exchange, regime, symbol)
    if os.path.isdir(synth_sym):
        candidates = glob.glob(os.path.join(synth_sym, "*.csv"))
        if candidates:
            return candidates[0]

    return None


def read_ohlcv(path: str) -> pd.DataFrame | None:
    try:
        # Try with header first
        df = pd.read_csv(path, nrows=2)
        if df.shape[1] >= 5:
            return pd.read_csv(path)
    except Exception:
        pass
    try:
        # Headerless Binance
        return pd.read_csv(path, header=None)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Missing Trades Generator")
    print("  Fills every L2 file that lacks a *_trades.csv")
    print("=" * 65)

    all_l2 = glob.glob(
        os.path.join(RECONSTRUCTED_BASE, "**", "*_L2.csv"), recursive=True
    )
    print(f"\n  Total L2 files : {len(all_l2)}")

    missing = [f for f in all_l2
               if not os.path.exists(f.replace("_L2.csv", "_trades.csv"))]
    print(f"  Missing trades : {len(missing)}")
    print()

    generated = skipped_direct = ohlcv_source = l2_fallback = failed = 0
    prev_exchange = None

    for i, l2_path in enumerate(missing):
        norm  = l2_path.replace("\\", "/")
        parts = norm.split("/")
        try:
            rec_idx  = parts.index("reconstructed")
            exchange = parts[rec_idx + 1]
            regime   = parts[rec_idx + 2]
            symbol   = parts[rec_idx + 3]
        except (ValueError, IndexError):
            failed += 1
            continue

        if exchange != prev_exchange:
            print(f"  [{exchange.upper()}]")
            prev_exchange = exchange

        basename = parts[-1]

        # Skip direct synthetic — generator always creates trades alongside
        if "_direct_L2" in basename:
            skipped_direct += 1
            continue

        out_path = l2_path.replace("_L2.csv", "_trades.csv")

        # Try OHLCV source first
        ohlcv_path = find_ohlcv(exchange, regime, symbol, basename)
        trades_df = None

        if ohlcv_path:
            ohlcv = read_ohlcv(ohlcv_path)
            if ohlcv is not None and len(ohlcv) > 0:
                trades_df = trades_from_ohlcv(ohlcv)
                if not trades_df.empty:
                    ohlcv_source += 1

        # Fallback: derive from L2 mid_price
        if trades_df is None or trades_df.empty:
            try:
                l2_df = pd.read_csv(l2_path, usecols=lambda c: c in
                                    {"bid_price", "ask_price", "bid_size",
                                     "ask_size", "mid_price"})
                trades_df = trades_from_l2(l2_df)
                if not trades_df.empty:
                    l2_fallback += 1
            except Exception:
                pass

        if trades_df is None or trades_df.empty:
            failed += 1
            continue

        trades_df.to_csv(out_path, index=False)
        generated += 1

        if (i + 1) % 500 == 0:
            print(f"    Progress: {i+1}/{len(missing)}  "
                  f"(gen={generated}, ohlcv={ohlcv_source}, "
                  f"l2fallback={l2_fallback}, fail={failed})")

    print()
    print("=" * 65)
    print(f"  Generated      : {generated}")
    print(f"    from OHLCV   : {ohlcv_source}")
    print(f"    from L2 mid  : {l2_fallback}")
    print(f"  Skipped direct : {skipped_direct}  (already have trades)")
    print(f"  Failed         : {failed}")
    print("=" * 65)


if __name__ == "__main__":
    main()
