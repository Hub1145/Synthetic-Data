"""
Control data fetcher — Phase 6.

Reads volatile_control events from data/scanned_pumps/[exchange]_pumps.csv
(events with large spikes that failed the 30% retracement threshold — organic
volatility, NOT pump-and-dumps) and fetches their 1-minute OHLCV using each
exchange's native REST API directly (no CCXT).

These form the hard-negative control set: the model must learn that a spike
alone is insufficient — the dump confirmation is required.

Data sources (same as fetch_all_pump_data.py):
    Binance : data.binance.vision public archive       (full history, no auth)
    Bybit   : api.bybit.com/v5/market/kline            (explicit start/end ms)
    KuCoin  : api.kucoin.com/api/v1/market/candles     (startAt/endAt, Unix s)
    OKX     : okx.com/api/v5/market/history-candles    (paginated backwards)
    Gate.io : api.gateio.ws/api/v4/spot/candlesticks   (from/to, Unix s, cascade)
    MEXC    : api.mexc.com/api/v3/klines               (startTime/endTime ms, cascade)
    Bitget  : api.bitget.com/api/v2/spot/market/history-candles

Output schema: real/[exchange]/control/[SYMBOL]/[SYMBOL]_[DATE]_klines.csv

Usage (run from project root):
    python scripts/fetch_control_data.py
    python scripts/fetch_control_data.py --exchange mexc gateio
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────────
REAL_BASE      = "real"
SCAN_BASE      = "data/scanned_pumps"
WINDOW_MINUTES = 200
RATE_LIMIT_S   = 0.35
MAX_RETRIES    = 2

EXCHANGES = ["binance", "bybit", "kucoin", "okx", "gateio", "mexc", "bitget"]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "pump-detector-research/1.0", "Accept": "application/json"})


# ── Timestamp helpers ─────────────────────────────────────────────────────────
def window_ms(date_str: str) -> tuple:
    start = int(
        datetime.strptime(date_str, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    )
    return start, start + WINDOW_MINUTES * 60 * 1000


def window_s(date_str: str) -> tuple:
    ms0, ms1 = window_ms(date_str)
    return ms0 // 1000, ms1 // 1000


# ── Symbol conversion helpers ─────────────────────────────────────────────────
_QUOTES = ("USDT", "BUSD", "USDC", "BTC", "ETH", "USD")


def split_symbol(raw: str) -> tuple:
    raw = raw.upper()
    for q in _QUOTES:
        if raw.endswith(q):
            return raw[: -len(q)], q
    return raw[:-3], raw[-3:]


def to_dash(raw: str) -> str:
    b, q = split_symbol(raw)
    return f"{b}-{q}"


def to_underscore(raw: str) -> str:
    b, q = split_symbol(raw)
    return f"{b}_{q}"


# ── Already-fetched detection ─────────────────────────────────────────────────
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
                        candidate = base[idx : idx + 10]
                        if len(candidate) == 10 and candidate[4] == "-":
                            done.add((sym.upper(), candidate))
    return done


def save_klines(df: pd.DataFrame, exchange: str, symbol: str, date_str: str) -> None:
    out_dir = os.path.join(REAL_BASE, exchange, "control", symbol)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}_{date_str}_klines.csv")
    df.to_csv(out_path, index=False)


# ── Per-exchange 1-minute OHLCV fetchers ─────────────────────────────────────
# Same endpoints and logic as fetch_all_pump_data.py.

def fetch_binance(symbol: str, date_str: str) -> "pd.DataFrame | None":
    """Binance Data Vision public archive — full history, no auth.
    symbol is the full raw symbol (e.g. 'ETHUSDT'). If only a base is passed,
    both the BTC and USDT variants are tried in order.
    """
    import io, zipfile
    # Build candidate list: if symbol is already fully-qualified use it once,
    # otherwise try base+BTC then base+USDT (same logic as fetch_all_pump_data.py).
    if symbol.endswith(("BTC", "USDT", "BUSD", "USDC", "ETH")):
        candidates = [symbol]
    else:
        candidates = [symbol + "BTC", symbol + "USDT"]
    for full in candidates:
        url = f"https://data.binance.vision/data/spot/daily/klines/{full}/1m/{full}-1m-{date_str}.zip"
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = SESSION.get(url, timeout=30)
                if resp.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                        with z.open(z.namelist()[0]) as f:
                            df = pd.read_csv(f, header=None)
                    df.columns = [
                        "open_ts_ms", "o", "h", "l", "c", "v",
                        "close_t", "quote_v", "n_trades",
                        "taker_buy_base", "taker_buy_quote", "ignore",
                    ]
                    return df[["open_ts_ms", "o", "h", "l", "c", "v"]]
                elif resp.status_code == 404:
                    break
                elif attempt < MAX_RETRIES:
                    time.sleep(1)
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(1)
        time.sleep(0.25)
    return None


def fetch_bybit(symbol: str, date_str: str) -> "pd.DataFrame | None":
    start_ms, end_ms = window_ms(date_str)
    params = {
        "category": "spot",
        "symbol":   symbol,
        "interval": "1",
        "start":    str(start_ms),
        "end":      str(end_ms),
        "limit":    WINDOW_MINUTES,
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = SESSION.get("https://api.bybit.com/v5/market/kline",
                               params=params, timeout=15).json()
            if resp.get("retCode") != 0:
                return None
            raw = resp.get("result", {}).get("list", [])
            if not raw:
                return None
            raw = list(reversed(raw))
            df = pd.DataFrame(raw, columns=["open_ts_ms", "o", "h", "l", "c", "v", "turnover"])
            df["open_ts_ms"] = df["open_ts_ms"].astype("int64")
            df = df[(df["open_ts_ms"] >= start_ms) & (df["open_ts_ms"] < end_ms)]
            return df[["open_ts_ms", "o", "h", "l", "c", "v"]].reset_index(drop=True)
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(1)
            else:
                print(f"      [ERR] bybit {symbol}: {e}")
    return None


def fetch_kucoin(symbol: str, date_str: str) -> "pd.DataFrame | None":
    start_s, end_s = window_s(date_str)
    params = {"symbol": to_dash(symbol), "type": "1min", "startAt": start_s, "endAt": end_s}
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = SESSION.get("https://api.kucoin.com/api/v1/market/candles",
                               params=params, timeout=15).json()
            if resp.get("code") != "200000":
                return None
            raw = resp.get("data", [])
            if not raw:
                return None
            df = pd.DataFrame(raw, columns=["ts", "o", "c", "h", "l", "v", "turnover"])
            df["open_ts_ms"] = df["ts"].astype("int64") * 1000
            df = df.sort_values("open_ts_ms")
            return df[["open_ts_ms", "o", "h", "l", "c", "v"]].reset_index(drop=True)
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(1)
            else:
                print(f"      [ERR] kucoin {symbol}: {e}")
    return None


def fetch_okx(symbol: str, date_str: str) -> "pd.DataFrame | None":
    start_ms, end_ms = window_ms(date_str)
    inst_id  = to_dash(symbol)
    all_bars = []
    cursor   = end_ms + 1

    for _ in range(5):
        params = {"instId": inst_id, "bar": "1m", "after": str(cursor), "limit": 100}
        try:
            resp = SESSION.get("https://www.okx.com/api/v5/market/history-candles",
                               params=params, timeout=15).json()
        except Exception as e:
            print(f"      [ERR] okx {symbol}: {e}")
            break
        if resp.get("code") != "0":
            break
        data = resp.get("data", [])
        if not data:
            break
        all_bars.extend(data)
        earliest = int(data[-1][0])
        cursor   = earliest
        if earliest <= start_ms:
            break
        time.sleep(RATE_LIMIT_S)

    if not all_bars:
        return None
    df = pd.DataFrame(all_bars,
                      columns=["open_ts_ms", "o", "h", "l", "c", "v",
                               "volCcy", "volCcyQuote", "confirm"])
    df["open_ts_ms"] = df["open_ts_ms"].astype("int64")
    df = df[(df["open_ts_ms"] >= start_ms) & (df["open_ts_ms"] < end_ms)]
    df = df.sort_values("open_ts_ms")
    return df[["open_ts_ms", "o", "h", "l", "c", "v"]].reset_index(drop=True)


_GATEIO_CASCADE = ["1m", "5m", "15m", "1h", "4h"]

def _gateio_parse_raw(raw: list) -> pd.DataFrame:
    rows = []
    for candle in raw:
        rows.append({
            "open_ts_ms": int(candle[0]) * 1000,
            "o": candle[5], "h": candle[3], "l": candle[4],
            "c": candle[2], "v": candle[6],
        })
    return pd.DataFrame(rows).sort_values("open_ts_ms").reset_index(drop=True)


def fetch_gateio(symbol: str, date_str: str) -> "pd.DataFrame | None":
    start_s, end_s = window_s(date_str)
    pair = to_underscore(symbol)

    for interval in _GATEIO_CASCADE:
        params = {
            "currency_pair": pair, "interval": interval,
            "from": start_s, "to": end_s,
            "limit": min(WINDOW_MINUTES, 1000),
        }
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = SESSION.get("https://api.gateio.ws/api/v4/spot/candlesticks",
                                   params=params, timeout=15)
                if resp.status_code == 400:
                    break
                if resp.status_code != 200:
                    if attempt < MAX_RETRIES:
                        time.sleep(1)
                        continue
                    break
                raw = resp.json()
                if not raw:
                    break
                return _gateio_parse_raw(raw)
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(1)
                else:
                    print(f"      [ERR] gateio {symbol} ({interval}): {e}")
                    break
    return None


_MEXC_CASCADE = [
    ("1m",  WINDOW_MINUTES),
    ("5m",  WINDOW_MINUTES // 5  + 1),
    ("15m", WINDOW_MINUTES // 15 + 1),
    ("30m", WINDOW_MINUTES // 30 + 1),
    ("60m", WINDOW_MINUTES // 60 + 1),
    ("4h",  max(1, WINDOW_MINUTES // 240 + 1)),
    ("1d",  1),
]

def fetch_mexc(symbol: str, date_str: str) -> "pd.DataFrame | None":
    start_ms, end_ms = window_ms(date_str)

    for interval, limit in _MEXC_CASCADE:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": str(start_ms), "endTime": str(end_ms),
            "limit": limit,
        }
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = SESSION.get("https://api.mexc.com/api/v3/klines",
                                   params=params, timeout=15)
                if resp.status_code != 200:
                    break
                raw = resp.json()
                if not raw or isinstance(raw, dict):
                    break
                df = pd.DataFrame(raw).iloc[:, :8]
                df.columns = ["open_ts_ms", "o", "h", "l", "c", "v", "close_t", "quote_v"]
                df["open_ts_ms"] = df["open_ts_ms"].astype("int64")
                return df[["open_ts_ms", "o", "h", "l", "c", "v"]].sort_values(
                    "open_ts_ms").reset_index(drop=True)
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(1)
                else:
                    print(f"      [ERR] mexc {symbol} ({interval}): {e}")
                    break
        time.sleep(RATE_LIMIT_S)
    return None


def fetch_bitget(symbol: str, date_str: str) -> "pd.DataFrame | None":
    start_ms, end_ms = window_ms(date_str)
    params = {
        "symbol": symbol, "granularity": "1min",
        "startTime": str(start_ms), "endTime": str(end_ms),
        "limit": WINDOW_MINUTES,
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = SESSION.get(
                "https://api.bitget.com/api/v2/spot/market/history-candles",
                params=params, timeout=15
            ).json()
            if resp.get("code") != "00000":
                return None
            raw = resp.get("data", [])
            if not raw:
                return None
            df = pd.DataFrame(raw,
                              columns=["open_ts_ms", "o", "h", "l", "c",
                                       "v", "quote_v", "usdt_v"])
            df["open_ts_ms"] = df["open_ts_ms"].astype("int64")
            df = df.sort_values("open_ts_ms")
            return df[["open_ts_ms", "o", "h", "l", "c", "v"]].reset_index(drop=True)
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(1)
            else:
                print(f"      [ERR] bitget {symbol}: {e}")
    return None


EXCHANGE_FETCHERS = {
    "binance": fetch_binance,
    "bybit":   fetch_bybit,
    "kucoin":  fetch_kucoin,
    "okx":     fetch_okx,
    "gateio":  fetch_gateio,
    "mexc":    fetch_mexc,
    "bitget":  fetch_bitget,
}


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

    fetcher = EXCHANGE_FETCHERS.get(exchange_id)
    if fetcher is None:
        print(f"  [SKIP] {exchange_id} — no REST fetcher defined")
        return

    print(f"\n--- {exchange_id.upper()}  ({len(ctrl_events)} control events) ---")
    print(f"  Method : direct REST API (no CCXT)")

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
        try:
            df_k = fetcher(sym, date)
            if df_k is None or len(df_k) == 0:
                print("NO DATA")
                fail += 1
            else:
                save_klines(df_k, exchange_id, sym, date)
                print(f"OK  ({len(df_k)} rows)")
                ok += 1
        except Exception as e:
            print(f"ERR: {str(e)[:80]}")
            fail += 1
        time.sleep(RATE_LIMIT_S)

    print(f"\n  {exchange_id}: {ok} control events downloaded, {fail} failed")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fetch volatile-control OHLCV via direct REST APIs (no CCXT)"
    )
    parser.add_argument(
        "--exchange", nargs="+",
        choices=EXCHANGES,
        help="Limit to specific exchange(s). Omit for all."
    )
    args = parser.parse_args()
    target = set(args.exchange) if args.exchange else set(EXCHANGES)

    print("=" * 65)
    print("  Control Data Fetcher  (volatile_control events from scanner)")
    print("  Method: direct REST API — no CCXT")
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
