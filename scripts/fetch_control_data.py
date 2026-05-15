"""
Control data fetcher — Phase 6.

Reads volatile_control events from data/scanned_pumps/[exchange]_pumps.csv
(events with large spikes that failed the 30% retracement threshold — organic
volatility, NOT pump-and-dumps) and fetches their 1-minute OHLCV from CCXT.

These form the hard-negative control set: the model must learn that a spike
alone is insufficient — the dump confirmation is required.

Output schema: real/[exchange]/control/[SYMBOL]/[SYMBOL]_[DATE]_klines.csv

Usage (run from project root):
    python scripts/fetch_control_data.py
    python scripts/fetch_control_data.py --exchange huobi mexc gateio
"""

import os
import sys
import time
import argparse
import pandas as pd
import ccxt
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────────
REAL_BASE       = "real"
SCAN_BASE       = "data/scanned_pumps"
WINDOW_MINUTES  = 200
CCXT_RATE_S     = 0.5
TIMEFRAME_CASCADE = ["1m", "5m", "15m", "1h"]

CCXT_EXCHANGE_MAP = {
    "binance": "binance",
    "bybit":   "bybit",
    "kucoin":  "kucoin",
    "okx":     "okx",
    "gateio":  "gateio",
    "mexc":    "mexc",
    "huobi":   "huobi",
    "bitget":  "bitget",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def already_fetched(exchange: str) -> set:
    done = set()
    ctrl_dir = os.path.join(REAL_BASE, exchange, "control")
    if not os.path.exists(ctrl_dir):
        return done
    for sym in os.listdir(ctrl_dir):
        sym_dir = os.path.join(ctrl_dir, sym)
        if not os.path.isdir(sym_dir):
            continue
        for fname in os.listdir(sym_dir):
            if fname.endswith(".csv") and "_processed" not in fname:
                base = fname.replace(".csv", "")
                for marker in ["-1m-", "_klines", "_control"]:
                    if marker in base:
                        idx = base.find(marker) + len(marker)
                        candidate = base[idx:idx+10]
                        if len(candidate) == 10 and candidate[4] == "-":
                            done.add((sym.upper(), candidate))
    return done


def symbol_to_ccxt(raw: str) -> str:
    for quote in ("USDT", "BTC", "ETH", "BUSD", "USD"):
        if raw.endswith(quote):
            return f"{raw[:-len(quote)]}/{quote}"
    return f"{raw[:-3]}/{raw[-3:]}"


def ccxt_fetch(ex_obj, ccxt_sym: str, date_str: str) -> pd.DataFrame | None:
    dt_ms = int(datetime.strptime(date_str, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc).timestamp() * 1000)
    last_err = None
    for tf in TIMEFRAME_CASCADE:
        try:
            bars = ex_obj.fetch_ohlcv(ccxt_sym, timeframe=tf,
                                       since=dt_ms, limit=WINDOW_MINUTES)
            if not bars:
                return None
            return pd.DataFrame(bars, columns=["open_ts_ms", "o", "h", "l", "c", "v"])
        except Exception as e:
            msg = str(e)
            if any(k in msg for k in ("too long ago", "too early", "INVALID_PARAM", "invalid start")):
                last_err = msg
                time.sleep(CCXT_RATE_S)
                continue
            raise RuntimeError(msg)
    raise RuntimeError(f"All timeframes exhausted. Last: {last_err}")


# ── Per-exchange fetch ────────────────────────────────────────────────────────
def fetch_exchange_control(exchange_id: str):
    scan_path = os.path.join(SCAN_BASE, f"{exchange_id}_pumps.csv")
    if not os.path.exists(scan_path):
        print(f"  [SKIP] {exchange_id} — no scan file at {scan_path}")
        return

    df = pd.read_csv(scan_path)
    if "label" not in df.columns:
        print(f"  [SKIP] {exchange_id} — scan file has no label column")
        return

    ctrl_events = df[df["label"] == "volatile_control"].copy()
    if ctrl_events.empty:
        print(f"  [SKIP] {exchange_id} — no volatile_control events in scan")
        return

    print(f"\n--- {exchange_id.upper()}  ({len(ctrl_events)} control events) ---")

    try:
        ex_cls = getattr(ccxt, CCXT_EXCHANGE_MAP[exchange_id])
        ex_obj = ex_cls({"enableRateLimit": True})
    except Exception as e:
        print(f"  [ERR] Could not init {exchange_id}: {e}")
        return

    done = already_fetched(exchange_id)
    print(f"  Already on disk: {len(done)}")

    todo = []
    for _, row in ctrl_events.iterrows():
        sym  = str(row["symbol"]).strip().upper()
        date = str(row["date"]).strip()[:10]
        if (sym, date) not in done:
            todo.append((sym, date))
    todo = list(dict.fromkeys(todo))
    print(f"  To fetch: {len(todo)}\n")

    ok = fail = 0
    for i, (sym, date) in enumerate(todo, 1):
        print(f"  [{i:>4}/{len(todo)}]  {sym:<16} {date}  ", end="", flush=True)
        ccxt_sym = symbol_to_ccxt(sym)
        try:
            df_k = ccxt_fetch(ex_obj, ccxt_sym, date)
            if df_k is None or len(df_k) == 0:
                print("NO DATA")
                fail += 1
                time.sleep(CCXT_RATE_S)
                continue
            out_dir = os.path.join(REAL_BASE, exchange_id, "control", sym)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{sym}_{date}_klines.csv")
            df_k.to_csv(out_path, index=False)
            print(f"OK  ({len(df_k)} rows)")
            ok += 1
        except Exception as e:
            print(f"ERR: {str(e)[:80]}")
            fail += 1
        time.sleep(CCXT_RATE_S)

    print(f"\n  {exchange_id}: {ok} control events downloaded, {fail} failed")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetch volatile-control OHLCV for all exchanges")
    parser.add_argument("--exchange", nargs="+",
                        choices=list(CCXT_EXCHANGE_MAP.keys()),
                        help="Limit to specific exchange(s). Omit for all.")
    args = parser.parse_args()
    target = set(args.exchange) if args.exchange else set(CCXT_EXCHANGE_MAP.keys())

    print("=" * 65)
    print("  Control Data Fetcher  (volatile_control events from scanner)")
    print(f"  Exchanges: {sorted(target)}")
    print("=" * 65)

    for ex_id in sorted(target):
        fetch_exchange_control(ex_id)

    print("\n" + "=" * 65)
    print("  Control fetch complete.")
    print("  Next: python scripts/run_pipeline.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
