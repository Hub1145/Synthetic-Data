"""
Comprehensive multi-exchange pump event fetcher (Phase 6).

Data sources
------------
Binance  : ArdiaD/PumpDump list (data/list_pd_events.csv, 322 confirmed events)
           → fetched from Binance Data Vision public archive (no auth needed)
KuCoin   : data/global_deep_scan_pumps.csv  (1,118 events)
Bybit    : data/global_deep_scan_pumps.csv  + data/bybit_specific_pumps.csv  (695 events)
OKX      : data/global_deep_scan_pumps.csv  (387 events)

Output schema:
    real/[exchange]/pumps/[SYMBOL]/[SYMBOL]_[DATE]_klines.csv

Usage (run from project root):
    python scripts/fetch_all_pump_data.py
    python scripts/fetch_all_pump_data.py --exchange binance
    python scripts/fetch_all_pump_data.py --exchange bybit
"""

import os
import io
import sys
import time
import zipfile
import argparse
import requests
import pandas as pd
import ccxt
from datetime import datetime, timezone

# Force UTF-8 output so exchange error messages with non-ASCII chars don't crash on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────────
REAL_BASE          = "real"
ARDIA_CSV          = "data/list_pd_events.csv"
GLOBAL_CSV         = "data/global_deep_scan_pumps.csv"
BYBIT_CSV          = "data/bybit_specific_pumps.csv"
BINANCE_VISION_BASE = "https://data.binance.vision/data/spot/daily/klines"

# Minutes of OHLCV context to fetch around the event (via CCXT)
WINDOW_MINUTES = 200
CCXT_RATE_S    = 0.5   # seconds between CCXT requests
VISION_RATE_S  = 0.25  # seconds between Data Vision downloads
MAX_RETRIES    = 2


# ── Already-fetched detection ─────────────────────────────────────────────────
def already_fetched(exchange: str) -> set:
    """Return a set of (SYMBOL_UPPER, YYYY-MM-DD) pairs already on disk."""
    done = set()
    pump_dir = os.path.join(REAL_BASE, exchange, "pumps")
    if not os.path.exists(pump_dir):
        return done
    for sym in os.listdir(pump_dir):
        sym_dir = os.path.join(pump_dir, sym)
        if not os.path.isdir(sym_dir):
            continue
        for fname in os.listdir(sym_dir):
            if fname.endswith(".csv") and "_processed" not in fname:
                # Accept both   SYMBOL-1m-DATE.csv  and  SYMBOL_DATE_klines.csv
                base = fname.replace(".csv", "")
                for marker in ["-1m-", "_klines"]:
                    if marker in base:
                        # extract YYYY-MM-DD
                        idx = base.find(marker) + len(marker)
                        candidate = base[idx:idx+10]
                        if len(candidate) == 10 and candidate[4] == "-":
                            done.add((sym.upper(), candidate))
    return done


# ── Binance: Data Vision archive ──────────────────────────────────────────────
def binance_vision_fetch(currency: str, date_str: str) -> tuple:
    """Try CURRENCYBTC then CURRENCYUSDT from Binance Data Vision."""
    for suffix in ("BTC", "USDT"):
        full = currency + suffix
        url  = f"{BINANCE_VISION_BASE}/{full}/1m/{full}-1m-{date_str}.zip"
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                        with z.open(z.namelist()[0]) as f:
                            df = pd.read_csv(f, header=None)
                    df.columns = [
                        "open_ts_ms", "o", "h", "l", "c", "v",
                        "close_t", "quote_v", "n_trades",
                        "taker_buy_base", "taker_buy_quote", "ignore"
                    ]
                    return df, full
                elif resp.status_code == 404:
                    break
                else:
                    if attempt < MAX_RETRIES:
                        time.sleep(1)
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(1)
                else:
                    print(f"    [ERR] {full} {date_str}: {e}")
        time.sleep(VISION_RATE_S)
    return None, None


def fetch_binance(target_exchanges=None):
    if target_exchanges and "binance" not in target_exchanges:
        return
    print("\n" + "=" * 65)
    print("  BINANCE  —  ArdiaD/PumpDump dataset (322 confirmed events)")
    print("=" * 65)

    df = pd.read_csv(ARDIA_CSV)
    df = df[(df["success"] == 1) & df["pump_date"].notna()].copy()
    df["pump_date"] = pd.to_datetime(df["pump_date"], utc=True)
    df["date_str"]  = df["pump_date"].dt.strftime("%Y-%m-%d")
    df["Currency"]  = df["Currency"].str.strip().str.upper()

    done = already_fetched("binance")
    print(f"  Eligible: {len(df)}   Already on disk: {len(done)}")

    todo = []
    for _, row in df.iterrows():
        sym  = row["Currency"]
        date = row["date_str"]
        if (sym + "BTC", date) not in done and (sym + "USDT", date) not in done:
            todo.append((sym, date))
    todo = list(dict.fromkeys(todo))
    print(f"  To fetch: {len(todo)}\n")

    ok = fail = 0
    for i, (sym, date) in enumerate(todo, 1):
        print(f"  [{i:>3}/{len(todo)}]  {sym:<12} {date}  ", end="", flush=True)
        df_k, full_sym = binance_vision_fetch(sym, date)
        if df_k is None:
            print("NOT FOUND")
            fail += 1
            continue
        out_dir = os.path.join(REAL_BASE, "binance", "pumps", full_sym)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{full_sym}-1m-{date}.csv")
        df_k.to_csv(out_path, index=False)
        print(f"OK -> {full_sym}  ({len(df_k)} rows)")
        ok += 1
        time.sleep(VISION_RATE_S)

    print(f"\n  Binance: {ok} downloaded, {fail} not found in archive")


# ── CCXT exchanges (all non-Binance) ─────────────────────────────────────────
# Maps exchange names in our CSVs to their CCXT identifiers
CCXT_EXCHANGE_MAP = {
    "kucoin":  "kucoin",
    "bybit":   "bybit",
    "okx":     "okx",
    "gateio":  "gateio",
    "mexc":    "mexc",
    "huobi":   "huobi",
    "bitget":  "bitget",
    "kraken":  "kraken",
    "poloniex":"poloniex",
}

def symbol_to_ccxt(raw: str) -> str:
    """Convert LINKUSDT -> LINK/USDT,  BTCBTC -> BTC/BTC, etc."""
    for quote in ("USDT", "BTC", "ETH", "BUSD", "USD"):
        if raw.endswith(quote):
            base = raw[: -len(quote)]
            return f"{base}/{quote}"
    # fallback: assume last 3 chars are quote
    return f"{raw[:-3]}/{raw[-3:]}"


# Timeframe cascade — tried in order until one succeeds.
# Coarser timeframes have longer API lookback windows (exchanges limit by bar count, not time).
TIMEFRAME_CASCADE = ["1m", "5m", "15m", "1h"]

def ccxt_fetch(exchange_obj, ccxt_sym: str, date_str: str) -> pd.DataFrame | None:
    """
    Fetch WINDOW_MINUTES worth of OHLCV starting at midnight of date_str.
    Tries 1m → 5m → 15m → 1h until one succeeds (handles exchanges like Gate.io
    that restrict 1m lookback to the last ~7 days).
    """
    dt_ms = int(datetime.strptime(date_str, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc).timestamp() * 1000)

    last_err = None
    for tf in TIMEFRAME_CASCADE:
        try:
            bars = exchange_obj.fetch_ohlcv(
                ccxt_sym, timeframe=tf, since=dt_ms, limit=WINDOW_MINUTES
            )
            if not bars:
                return None
            df = pd.DataFrame(bars, columns=["open_ts_ms", "o", "h", "l", "c", "v"])
            return df
        except Exception as e:
            msg = str(e)
            # Only fall through to a coarser timeframe on lookback-limit errors
            if any(k in msg for k in ("too long ago", "too early", "INVALID_PARAM", "invalid start")):
                last_err = msg
                time.sleep(CCXT_RATE_S)
                continue
            # Any other error — raise immediately
            raise RuntimeError(msg)

    raise RuntimeError(f"All timeframes exhausted. Last error: {last_err}")


def fetch_ccxt_exchange(exchange_id: str, events: pd.DataFrame):
    print(f"\n--- {exchange_id.upper()}  ({len(events)} events) ---")

    try:
        ex_cls  = getattr(ccxt, CCXT_EXCHANGE_MAP[exchange_id])
        ex_obj  = ex_cls({"enableRateLimit": True})
    except Exception as e:
        print(f"  [ERR] Could not init {exchange_id}: {e}")
        return

    done = already_fetched(exchange_id)
    print(f"  Already on disk: {len(done)}")

    todo = []
    for _, row in events.iterrows():
        sym  = str(row["symbol"]).strip().upper()
        date = str(row["date"]).strip()[:10]
        if (sym, date) not in done:
            todo.append((sym, date))
    todo = list(dict.fromkeys(todo))
    print(f"  To fetch: {len(todo)}\n")

    ok = fail = 0
    for i, (sym, date) in enumerate(todo, 1):
        print(f"  [{i:>4}/{len(todo)}]  {sym:<14} {date}  ", end="", flush=True)
        ccxt_sym = symbol_to_ccxt(sym)
        try:
            df_k = ccxt_fetch(ex_obj, ccxt_sym, date)
            if df_k is None or len(df_k) == 0:
                print("NO DATA")
                fail += 1
                time.sleep(CCXT_RATE_S)
                continue
            out_dir = os.path.join(REAL_BASE, exchange_id, "pumps", sym)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{sym}_{date}_klines.csv")
            df_k.to_csv(out_path, index=False)
            print(f"OK  ({len(df_k)} rows)")
            ok += 1
        except Exception as e:
            print(f"ERR: {e}")
            fail += 1
        time.sleep(CCXT_RATE_S)

    print(f"\n  {exchange_id}: {ok} downloaded, {fail} failed/not found")


def fetch_non_binance(target_exchanges=None):
    global_df = pd.read_csv(GLOBAL_CSV)

    # Merge Bybit-specific CSV in case it has events not in global scan
    if os.path.exists(BYBIT_CSV):
        bybit_extra = pd.read_csv(BYBIT_CSV)
        global_df = pd.concat([global_df, bybit_extra], ignore_index=True).drop_duplicates(
            subset=["exchange", "symbol", "date"]
        )

    # Dynamically handle every exchange present in the scan CSV
    all_exchanges = sorted(global_df["exchange"].dropna().unique())
    for exch_id in all_exchanges:
        if exch_id == "binance":
            continue   # handled separately via Data Vision
        if target_exchanges and exch_id not in target_exchanges:
            continue
        if exch_id not in CCXT_EXCHANGE_MAP:
            print(f"\n  [SKIP] {exch_id} — no CCXT mapping defined")
            continue
        subset = global_df[global_df["exchange"] == exch_id].copy()
        if subset.empty:
            continue
        fetch_ccxt_exchange(exch_id, subset)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetch pump event OHLCV data for all exchanges")
    parser.add_argument(
        "--exchange", nargs="+",
        choices=list(CCXT_EXCHANGE_MAP.keys()) + ["binance"],
        help="Limit to specific exchange(s). Omit to fetch all."
    )
    args = parser.parse_args()
    target = set(args.exchange) if args.exchange else None

    print("=" * 65)
    print("  Multi-Exchange Pump Data Fetcher  (Phase 6)")
    print("=" * 65)
    if target:
        print(f"  Exchanges: {sorted(target)}")
    else:
        print("  Exchanges: binance, kucoin, bybit, okx")
    print()

    fetch_binance(target)
    fetch_non_binance(target)

    print("\n" + "=" * 65)
    print("  Fetch complete.")
    print("  Next steps:")
    print("    1. python scripts/label_real_data.py     (re-verify pump labels)")
    print("    2. python scripts/reconstruct_orderbook.py  (rebuild L2 files)")
    print("    3. python scripts/train_pump_detector.py    (re-train model)")
    print("=" * 65)


if __name__ == "__main__":
    main()
