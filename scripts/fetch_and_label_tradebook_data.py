"""
Verify pump labels and enrich confirmed events with real trade ticks.

Two operations in one pass over real/:

  1. LABEL — apply the pump confirmation criteria to every event in real/pumps/:
       - Price spike >= 5%  (MIN_INCREASE)
       - Retracement >= 30% (RETRACEMENT_THRESHOLD)
     Events that fail are moved to real/[exchange]/control/[symbol]/.

  2. ENRICH — for confirmed pumps on exchanges that have a free historical tick
     archive, immediately fetch real trade ticks:
       Binance → data.binance.vision/data/spot/daily/trades/
       Bybit   → public.bybit.com/trading/  (rolling ~90 days only)
     Ticks are saved to real/[exchange]/[regime]/[symbol]/[SYM]-trades-[DATE].csv
     and also upgrade the matching reconstructed/ trades file.

Usage (from project root):
    python scripts/label_real_data.py
    python scripts/label_real_data.py --skip-tradebook
    python scripts/label_real_data.py --exchanges binance
    python scripts/label_real_data.py --within-days 90
"""

import os
import sys
import re
import io
import gzip
import glob
import shutil
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

MIN_INCREASE          = 0.05   # 5%  minimum price spike
RETRACEMENT_THRESHOLD = 0.30   # 30% dump required from peak

# Peak magnitude buckets — used by the 9-cell market×coin regime matrix
PEAK_BUCKETS = [
    ("micro",   0.05, 0.10),
    ("small",   0.10, 0.20),
    ("medium",  0.20, 0.30),
    ("large",   0.30, 0.40),
    ("major",   0.40, 0.50),
    ("extreme", 0.50, float("inf")),
]

CANDLE_US  = 60_000_000        # 1 minute in µs
CANDLE_MS  = 60_000
RATE_SLEEP = 0.5               # seconds between HTTP requests

BINANCE_TRADES_URL = "https://data.binance.vision/data/spot/daily/trades/{sym}/{sym}-trades-{date}.zip"
BYBIT_TRADES_URL   = "https://public.bybit.com/trading/{sym}/{sym}{date}.csv.gz"

COLS_12 = ['open_ts_ms', 'o', 'h', 'l', 'c', 'v',
           'close_t', 'quote_v', 'n_trades',
           'taker_buy_base', 'taker_buy_quote', 'ignore']
COLS_6  = ['open_ts_ms', 'o', 'h', 'l', 'c', 'v']


# ── OHLCV helpers ──────────────────────────────────────────────────────────────

def load_ohlcv(filepath):
    peek = pd.read_csv(filepath, nrows=1, header=None)
    first_cell = str(peek.iloc[0, 0])
    try:
        float(first_cell)
        is_headerless = True
    except ValueError:
        is_headerless = False

    if is_headerless:
        df = pd.read_csv(filepath, header=None)
        if df.shape[1] == 12:
            df.columns = COLS_12
        elif df.shape[1] == 6:
            df.columns = COLS_6
        else:
            raise ValueError(f"Unexpected column count ({df.shape[1]}) in headerless file")
    else:
        df = pd.read_csv(filepath)
    return df


def calculate_pump_score(filepath):
    """
    Returns (is_pump, retracement, increase).
    A genuine P&D requires: >= MIN_INCREASE spike AND >= RETRACEMENT_THRESHOLD dump.
    """
    df = load_ohlcv(filepath)
    required = {'o', 'h', 'l', 'c'}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing OHLCV columns: {df.columns.tolist()}")

    df = df.dropna(subset=['o', 'h', 'l', 'c']).reset_index(drop=True)
    if len(df) < 5:
        return False, 0.0, 0.0

    peak_loc      = int(df['h'].argmax())
    peak_high     = float(df['h'].iloc[peak_loc])
    initial_price = float(df['o'].iloc[0])

    if initial_price <= 0:
        return False, 0.0, 0.0

    increase = (peak_high - initial_price) / initial_price
    if increase < MIN_INCREASE:
        return False, 0.0, increase

    after_peak = df.iloc[peak_loc:]
    if len(after_peak) < 2:
        return False, 0.0, increase

    min_close_after = float(after_peak['c'].min())
    price_range     = peak_high - initial_price
    if price_range <= 0:
        return False, 0.0, increase

    retracement = (peak_high - min_close_after) / price_range
    return retracement >= RETRACEMENT_THRESHOLD, retracement, increase


def get_peak_bucket(increase: float) -> str:
    for name, lo, hi in PEAK_BUCKETS:
        if lo <= increase < hi:
            return name
    return "extreme"


def save_event_meta(exchange: str, regime: str, symbol: str, date: str,
                    increase: float):
    """Write _meta.json into the matching reconstructed/ symbol directory."""
    import json
    rec_dir = os.path.join(RECONSTRUCTED_BASE, exchange, regime, symbol)
    if not os.path.isdir(rec_dir):
        return
    meta = {
        "coin_regime":   regime.rstrip("s"),   # "pumps" → "pump", "control" → "control"
        "market_regime": "unknown",             # filled later by fetch_market_context.py
        "peak_pct":      round(increase * 100, 2),
        "peak_bucket":   get_peak_bucket(increase) if regime == "pumps" else None,
        "event_date":    date,
    }
    # Write alongside every L2 file in this dir that matches the date
    for fname in os.listdir(rec_dir):
        if fname.endswith("_L2.csv") and (date in fname if date else True):
            stem = re.sub(r'(_direct_L2|_synthetic_L2|_L2)\.csv$', '', fname)
            meta_path = os.path.join(rec_dir, stem + "_meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)


def extract_date(filename):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    return m.group(1) if m else None


def move_event_files(src_dir, dest_dir, date_str):
    os.makedirs(dest_dir, exist_ok=True)
    moved = []
    for fname in os.listdir(src_dir):
        if date_str in fname:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dest_dir, fname)
            if not os.path.exists(dst):
                shutil.move(src, dst)
                moved.append(fname)
    return moved


# ── Tradebook helpers ──────────────────────────────────────────────────────────

def get_kline_start(kline_path):
    """Return (start_us, n_candles) from a kline file."""
    try:
        df     = pd.read_csv(kline_path, nrows=1)
        ts_raw = int(df.iloc[0, 0])
        if ts_raw > 1e13:
            start_us = ts_raw
        elif ts_raw > 1e10:
            start_us = ts_raw * 1_000
        else:
            start_us = ts_raw * 1_000_000

        with open(kline_path, encoding="utf-8", errors="replace") as f:
            n_lines = sum(1 for _ in f)
        return start_us, max(n_lines - 1, 1)
    except Exception:
        return None, None


def to_standard(df, start_us, n_candles):
    df["timestamp_idx"] = ((df["time_us"] - start_us) // CANDLE_US).astype(int)
    out = df[["timestamp_idx", "side", "price", "size"]].copy()
    out = out[(out["timestamp_idx"] >= 0) & (out["timestamp_idx"] < n_candles)]
    return out.reset_index(drop=True) if not out.empty else None


def save_trades(df, out_path):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        return True
    except Exception as e:
        print(f"    [ERR] Could not save {out_path}: {e}")
        return False


def upgrade_reconstructed(exchange, regime, symbol, real_trades_df):
    """Overwrite OHLCV-derived trades in reconstructed/ with real tick data."""
    rec_dir = os.path.join(RECONSTRUCTED_BASE, exchange, regime, symbol)
    if not os.path.isdir(rec_dir):
        return
    for l2_path in glob.glob(os.path.join(rec_dir, "*_L2.csv")):
        trades_path = l2_path.replace("_L2.csv", "_trades.csv")
        try:
            real_trades_df.to_csv(trades_path, index=False)
        except Exception:
            pass


# ── Exchange tick fetchers ─────────────────────────────────────────────────────

def fetch_binance_ticks(symbol, date):
    url = BINANCE_TRADES_URL.format(sym=symbol, date=date)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"    [HTTP {e.code}]")
        return None
    except Exception as e:
        print(f"    [ERR] {e}")
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                raw = pd.read_csv(
                    f, header=None,
                    names=["trade_id", "price", "qty", "quote_qty",
                           "time", "is_buyer_maker", "is_best_match"],
                )
    except Exception as e:
        print(f"    [PARSE ERR] {e}")
        return None

    if raw.empty:
        return None

    times = raw["time"].astype(np.int64)
    sample = int(times.iloc[0])
    raw["time_us"] = times if sample > 1e13 else times * 1_000 if sample > 1e10 else times * 1_000_000
    raw["side"]    = raw["is_buyer_maker"].apply(
        lambda x: "sell" if str(x).strip().lower() == "true" else "buy"
    )
    raw["price"]   = raw["price"].astype(float)
    raw["size"]    = raw["qty"].astype(float)
    return raw[["time_us", "side", "price", "size"]]


def fetch_bybit_ticks(symbol, date):
    url = BYBIT_TRADES_URL.format(sym=symbol, date=date)
    try:
        with urllib.request.urlopen(url, timeout=180) as resp:
            compressed = resp.read()
        mb = len(compressed) // 1024 // 1024
        print(f"    {mb} MB", end="  ", flush=True)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"    [HTTP {e.code}]", end="  ", flush=True)
        return None
    except Exception as e:
        print(f"    [ERR: {e}]", end="  ", flush=True)
        return None

    try:
        with gzip.open(io.BytesIO(compressed), "rt", encoding="utf-8") as gz:
            raw = pd.read_csv(gz)
    except Exception as e:
        print(f"    [PARSE ERR: {e}]", end="  ", flush=True)
        return None

    if raw.empty:
        return None

    ts_col    = next((c for c in raw.columns if "time" in c.lower()), None)
    side_col  = next((c for c in raw.columns if "side" in c.lower()), None)
    size_col  = next((c for c in raw.columns if c.lower() in ("size", "qty", "quantity")), None)
    price_col = next((c for c in raw.columns if "price" in c.lower()), None)

    if not all([ts_col, side_col, size_col, price_col]):
        return None

    ts_vals = raw[ts_col].astype(float)
    sample  = float(ts_vals.iloc[0])
    time_us = (ts_vals if sample > 1e13
               else ts_vals * 1_000 if sample > 1e10
               else ts_vals * 1_000_000).astype(np.int64)

    return pd.DataFrame({
        "time_us": time_us,
        "side":    raw[side_col].str.strip().str.lower().map(
                       {"buy": "buy", "sell": "sell"}).fillna("buy"),
        "price":   raw[price_col].astype(float),
        "size":    raw[size_col].astype(float),
    })


TICK_FETCHERS = {
    "binance": fetch_binance_ticks,
    "bybit":   fetch_bybit_ticks,
}


def enrich_event(exchange, regime, symbol, sym_dir, kline_path, date,
                 tick_exchanges, within_days):
    """Fetch trade ticks for a single confirmed event."""
    if exchange not in tick_exchanges:
        return

    if within_days is not None:
        from datetime import datetime, timedelta
        if datetime.strptime(date, "%Y-%m-%d") < datetime.now() - timedelta(days=within_days):
            print(f"    [SKIP] {symbol} {date}: older than {within_days} days")
            return

    out_path = os.path.join(sym_dir, f"{symbol}-trades-{date}.csv")
    if os.path.exists(out_path):
        try:
            if len(pd.read_csv(out_path)) > 200:
                return   # already have real ticks
        except Exception:
            pass

    start_us, n_candles = get_kline_start(kline_path)
    if start_us is None:
        return

    print(f"    [ticks] {symbol} {date} ...", end=" ", flush=True)
    raw = TICK_FETCHERS[exchange](symbol, date)
    time.sleep(RATE_SLEEP)

    if raw is None:
        print("not available")
        return

    standard = to_standard(raw, start_us, n_candles)
    if standard is None:
        print("no ticks in window")
        return

    if save_trades(standard, out_path):
        print(f"{len(standard):,} ticks saved")
        upgrade_reconstructed(exchange, regime, symbol, standard)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tradebook", action="store_true",
                        help="Label only — skip trade tick fetching")
    parser.add_argument("--exchanges", nargs="+",
                        default=list(TICK_FETCHERS.keys()),
                        help="Exchanges to fetch ticks for (default: binance bybit)")
    parser.add_argument("--within-days", type=int, default=None,
                        help="Only fetch ticks for events within this many days (Bybit: use 90)")
    args = parser.parse_args()

    tick_exchanges = set() if args.skip_tradebook else set(args.exchanges)

    print("=" * 65)
    print("  Label & Enrich — Real Data Verification Pass")
    print(f"  Criteria: >{MIN_INCREASE*100:.0f}% spike AND >{RETRACEMENT_THRESHOLD*100:.0f}% retracement")
    if tick_exchanges:
        print(f"  Tick fetch: {sorted(tick_exchanges)}")
        if args.within_days:
            print(f"  Date filter: within {args.within_days} days")
    else:
        print("  Tick fetch: disabled (--skip-tradebook)")
    print("=" * 65)

    report_rows    = []
    total_checked  = 0
    confirmed      = 0
    relabeled      = 0

    if not os.path.exists(REAL_BASE):
        print(f"ERROR: '{REAL_BASE}' directory not found. Run from project root.")
        return

    for exchange in sorted(os.listdir(REAL_BASE)):
        pump_dir = os.path.join(REAL_BASE, exchange, "pumps")
        if not os.path.isdir(pump_dir):
            continue

        print(f"\n--- {exchange.upper()} ---")

        for symbol in sorted(os.listdir(pump_dir)):
            sym_dir = os.path.join(pump_dir, symbol)
            if not os.path.isdir(sym_dir):
                continue

            kline_files = sorted([
                f for f in os.listdir(sym_dir)
                if ('-1m-' in f or '_klines' in f) and f.endswith('.csv')
                and '_processed' not in f
            ])
            if not kline_files:
                kline_files = sorted([
                    f for f in os.listdir(sym_dir)
                    if f.endswith('_processed.csv') and '_processed_processed' not in f
                ])

            for fname in kline_files:
                filepath = os.path.join(sym_dir, fname)
                date_str = extract_date(fname)
                total_checked += 1

                try:
                    is_pump, retracement, increase = calculate_pump_score(filepath)
                    label = "PUMP" if is_pump else "VOLATILE_CONTROL"

                    bucket = get_peak_bucket(increase) if is_pump else None
                    report_rows.append({
                        'exchange':        exchange,
                        'symbol':          symbol,
                        'date':            date_str,
                        'file':            fname,
                        'increase_pct':    round(increase * 100, 2),
                        'retracement_pct': round(retracement * 100, 2),
                        'peak_bucket':     bucket,
                        'label':           label,
                    })

                    tag = "[PUMP]" if is_pump else "[ORGANIC]"
                    bucket_tag = f"  [{bucket}]" if bucket else ""
                    print(f"  {tag:<10} {symbol:<12} {date_str}  "
                          f"+{increase*100:.1f}% spike  {retracement*100:.1f}% retracement"
                          + bucket_tag + ("" if is_pump else "  -> control"))

                    if is_pump:
                        confirmed += 1
                        save_event_meta(exchange, "pumps", symbol, date_str, increase)
                        if tick_exchanges and date_str:
                            enrich_event(exchange, "pumps", symbol, sym_dir,
                                         filepath, date_str,
                                         tick_exchanges, args.within_days)
                    else:
                        relabeled += 1
                        if date_str:
                            dest_dir = os.path.join(REAL_BASE, exchange, "control", symbol)
                            moved = move_event_files(sym_dir, dest_dir, date_str)
                            if moved:
                                print(f"     Moved {len(moved)} file(s) to control/")
                        else:
                            print(f"     WARNING: no date in '{fname}' — skipping move")

                except Exception as e:
                    print(f"  [ERROR]    {symbol} / {fname}: {e}")
                    report_rows.append({
                        'exchange': exchange, 'symbol': symbol, 'date': date_str,
                        'file': fname, 'increase_pct': None,
                        'retracement_pct': None, 'label': 'ERROR',
                    })

    os.makedirs("data", exist_ok=True)
    report_path = "data/labeling_report.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False)

    print("\n" + "=" * 65)
    print(f"  Total events checked:          {total_checked}")
    print(f"  Confirmed pumps:               {confirmed}")
    print(f"  Relabeled -> control:          {relabeled}")
    print(f"  Report saved to:               {report_path}")
    print("=" * 65)

    if relabeled > 0:
        print("\n  NOTE: Re-run reconstruct_orderbook.py to rebuild L2 files")
        print("  for the new control/ directories.")

    if tick_exchanges:
        unavailable = ["kucoin", "okx", "gateio", "mexc", "huobi", "bitget"]
        print(f"\n  Exchanges with no free historical tick data: {', '.join(unavailable)}")
        print("  Use Tardis.dev or Kaiko for paid historical tick coverage.")


if __name__ == "__main__":
    main()
