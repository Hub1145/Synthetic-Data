"""
Bulk-fetch script for ArdiaD/PumpDump dataset.

Reads data/list_pd_events.csv (1,160 Binance P&D events) and downloads the
corresponding 1-minute OHLCV kline files from the Binance Data Vision public
archive for every successful event that hasn't been fetched yet.

Output schema: real/binance/pumps/[SYMBOL]/[SYMBOL]-1m-[DATE].csv

Usage (from project root):
    python scripts/fetch_ardia_pumps.py
"""

import os
import io
import time
import zipfile
import requests
import pandas as pd
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────
EVENTS_CSV    = "data/list_pd_events.csv"
PUMP_BASE     = "real/binance/pumps"
VISION_BASE   = "https://data.binance.vision/data/spot/daily/klines"
RATE_LIMIT_S  = 0.3   # seconds between requests
MAX_RETRIES   = 2

# ── Helpers ───────────────────────────────────────────────────────────────────
def already_fetched() -> set:
    """Return a set of (symbol_upper, YYYY-MM-DD) pairs already on disk."""
    done = set()
    if not os.path.exists(PUMP_BASE):
        return done
    for sym in os.listdir(PUMP_BASE):
        sym_dir = os.path.join(PUMP_BASE, sym)
        if not os.path.isdir(sym_dir):
            continue
        for fname in os.listdir(sym_dir):
            if "-1m-" in fname and fname.endswith(".csv") and "_processed" not in fname:
                date_part = fname.replace(sym + "-1m-", "").replace(".csv", "")
                done.add((sym.upper(), date_part))
    return done


def download_klines(symbol: str, date_str: str) -> pd.DataFrame | None:
    """
    Download 1-minute klines from Binance Data Vision for a given symbol/date.
    Returns a DataFrame or None on failure.
    Tries BTC pair first, falls back to USDT.
    """
    # The archive uses the exact symbol that was listed on Binance
    for suffix in ["BTC", "USDT"]:
        full_symbol = symbol + suffix
        url = f"{VISION_BASE}/{full_symbol}/1m/{full_symbol}-1m-{date_str}.zip"
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                        name = z.namelist()[0]
                        with z.open(name) as f:
                            df = pd.read_csv(f, header=None)
                    df.columns = [
                        'open_ts_ms', 'o', 'h', 'l', 'c', 'v',
                        'close_t', 'quote_v', 'n_trades',
                        'taker_buy_base', 'taker_buy_quote', 'ignore'
                    ]
                    return df, full_symbol
                elif resp.status_code == 404:
                    break   # this suffix doesn't exist — try next
                else:
                    if attempt < MAX_RETRIES:
                        time.sleep(1)
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(1)
                else:
                    print(f"    [ERR] {full_symbol} {date_str}: {e}")
        time.sleep(RATE_LIMIT_S)
    return None, None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  ArdiaD/PumpDump Bulk Fetch")
    print("  Source: data/list_pd_events.csv  (1,160 Binance events)")
    print("=" * 65)

    df = pd.read_csv(EVENTS_CSV)

    # Keep only successful pumps that have a known pump timestamp
    df = df[(df["success"] == 1) & df["pump_date"].notna()].copy()
    df["pump_date"] = pd.to_datetime(df["pump_date"], utc=True)
    df["date_str"] = df["pump_date"].dt.strftime("%Y-%m-%d")
    df["Currency"] = df["Currency"].str.strip().str.upper()

    print(f"  Eligible events : {len(df)}")

    done = already_fetched()
    print(f"  Already on disk : {len(done)}")

    # Build list of events to fetch
    todo = []
    for _, row in df.iterrows():
        sym   = row["Currency"]
        date  = row["date_str"]
        # Check both BTC and USDT variants
        if (sym + "BTC", date) not in done and (sym + "USDT", date) not in done:
            todo.append((sym, date))

    # Deduplicate (same symbol+date can appear in multiple pump announcements)
    todo = list(dict.fromkeys(todo))
    print(f"  To fetch        : {len(todo)}")
    print()

    success_count = 0
    fail_count    = 0

    for i, (sym, date) in enumerate(todo, 1):
        print(f"  [{i:>3}/{len(todo)}]  {sym:<10}  {date}  ", end="", flush=True)

        df_klines, full_symbol = download_klines(sym, date)

        if df_klines is None:
            print("NOT FOUND")
            fail_count += 1
            time.sleep(RATE_LIMIT_S)
            continue

        # Save
        out_dir = os.path.join(PUMP_BASE, full_symbol)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{full_symbol}-1m-{date}.csv")
        df_klines.to_csv(out_path, index=False)
        print(f"OK  -> {full_symbol}  ({len(df_klines)} rows)")
        success_count += 1
        time.sleep(RATE_LIMIT_S)

    print()
    print("=" * 65)
    print(f"  Downloaded : {success_count}")
    print(f"  Not found  : {fail_count}")
    print(f"  Total done : {len(done) + success_count}")
    print("=" * 65)

    if success_count > 0:
        print()
        print("  Next steps:")
        print("    1. python scripts/label_real_data.py   (re-verify labels)")
        print("    2. python scripts/reconstruct_orderbook.py  (build L2 files)")
        print("    3. python scripts/train_pump_detector.py    (re-train model)")


if __name__ == "__main__":
    main()
