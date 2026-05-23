"""
Comprehensive multi-exchange pump event fetcher — Direct REST API v2.

Replaces CCXT for all non-Binance exchanges with each exchange's native
bulk-history endpoint, bypassing rolling lookback limits that caused gaps.

Data sources
------------
Binance  : data.binance.vision public archive       (full history, no auth)
Bybit    : api.bybit.com/v5/market/kline            (explicit start/end ms)
KuCoin   : api.kucoin.com/api/v1/market/candles     (startAt/endAt, Unix s)
OKX      : okx.com/api/v5/market/history-candles    (paginated backwards)
Gate.io  : api.gateio.ws/api/v4/spot/candlesticks   (from/to, Unix s, cascade)
MEXC     : api.mexc.com/api/v3/klines               (startTime/endTime ms, cascade)
Bitget   : api.bitget.com/api/v2/spot/market/history-candles (startTime/endTime ms)

Huobi/HTX excluded: their documented from/to params are silently ignored by the
server — the API only serves the most-recent ~200 bars regardless of date
requested.  No public historical REST endpoint exists.  Existing huobi/ files
on disk are preserved; new events cannot be fetched.

Output schema:
    real/[exchange]/pumps/[SYMBOL]/[SYMBOL]_[DATE]_klines.csv
    6-column: open_ts_ms, o, h, l, c, v
    MEXC saves 12-column (Binance-compat) when available for real taker-buy data.

Usage (run from project root):
    python scripts/fetch_all_pump_data.py
    python scripts/fetch_all_pump_data.py --exchange bybit
    python scripts/fetch_all_pump_data.py --exchange gateio mexc
"""

import os
import io
import sys
import time
import zipfile
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────────
REAL_BASE           = "real"
ARDIA_CSV           = "data/list_pd_events.csv"
GLOBAL_CSV          = "data/global_deep_scan_pumps.csv"
BYBIT_CSV           = "data/bybit_specific_pumps.csv"
BINANCE_VISION_BASE = "https://data.binance.vision/data/spot/daily/klines"

WINDOW_MINUTES = 200    # bars to fetch (1 bar = 1 minute)
RATE_LIMIT_S   = 0.35   # pause between requests per exchange
MAX_RETRIES    = 2

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "pump-detector-research/1.0", "Accept": "application/json"})


# ── Already-fetched detection ─────────────────────────────────────────────────
def already_fetched(exchange: str) -> set:
    """Return set of (SYMBOL_UPPER, YYYY-MM-DD) pairs already on disk."""
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
                base = fname.replace(".csv", "")
                for marker in ["-1m-", "_klines"]:
                    if marker in base:
                        idx = base.find(marker) + len(marker)
                        candidate = base[idx : idx + 10]
                        if len(candidate) == 10 and candidate[4] == "-":
                            done.add((sym.upper(), candidate))
    return done


def save_klines(df: pd.DataFrame, exchange: str, symbol: str, date_str: str) -> str:
    """Save a klines DataFrame to real/[exchange]/pumps/[symbol]/[symbol]_[date]_klines.csv."""
    out_dir = os.path.join(REAL_BASE, exchange, "pumps", symbol)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}_{date_str}_klines.csv")
    df.to_csv(out_path, index=False)
    return out_path


# ── Window timestamp helpers ──────────────────────────────────────────────────
def window_ms(date_str: str) -> tuple:
    """(start_ms, end_ms) = WINDOW_MINUTES starting at midnight UTC on date_str."""
    start = int(
        datetime.strptime(date_str, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )
    return start, start + WINDOW_MINUTES * 60 * 1000


def window_s(date_str: str) -> tuple:
    """(start_unix_s, end_unix_s) for WINDOW_MINUTES starting at midnight UTC."""
    ms0, ms1 = window_ms(date_str)
    return ms0 // 1000, ms1 // 1000


# ── Symbol conversion helpers ─────────────────────────────────────────────────
_QUOTES = ("USDT", "BUSD", "USDC", "BTC", "ETH", "USD")


def split_symbol(raw: str) -> tuple:
    """'LINKUSDT' -> ('LINK', 'USDT').  Falls back to last-3 as quote."""
    raw = raw.upper()
    for q in _QUOTES:
        if raw.endswith(q):
            return raw[: -len(q)], q
    return raw[:-3], raw[-3:]


def to_dash(raw: str) -> str:
    """LINKUSDT -> LINK-USDT  (KuCoin, OKX instId format)"""
    b, q = split_symbol(raw)
    return f"{b}-{q}"


def to_underscore(raw: str) -> str:
    """LINKUSDT -> LINK_USDT  (Gate.io currency_pair format)"""
    b, q = split_symbol(raw)
    return f"{b}_{q}"


# ── Binance: Data Vision public archive ───────────────────────────────────────
def _binance_vision_fetch(currency: str, date_str: str) -> tuple:
    """Try CURRENCYBTC then CURRENCYUSDT from Binance Data Vision."""
    for suffix in ("BTC", "USDT"):
        full = currency + suffix
        url  = f"{BINANCE_VISION_BASE}/{full}/1m/{full}-1m-{date_str}.zip"
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
        time.sleep(0.25)
    return None, None


# ── Bybit: REST API v5  (category=spot, explicit start/end ms) ────────────────
def fetch_bybit(symbol: str, date_str: str) -> "pd.DataFrame | None":
    """
    Bybit v5 kline endpoint with explicit start/end timestamps (ms).
    Returns up to 1000 bars per call; 200 bars fits in one request.
    Response list order: newest-first — reversed before returning.
    """
    start_ms, end_ms = window_ms(date_str)
    params = {
        "category": "spot",
        "symbol":   symbol,
        "interval": "1",
        "start":    str(start_ms),   # str avoids C-long overflow on Windows
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
            raw = list(reversed(raw))   # newest-first → chronological
            # Each row: [startTime_ms, open, high, low, close, volume, turnover]
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


# ── KuCoin: REST API v1  (startAt/endAt Unix seconds) ────────────────────────
def fetch_kucoin(symbol: str, date_str: str) -> "pd.DataFrame | None":
    """
    KuCoin candle endpoint with startAt/endAt (Unix seconds).
    Allows arbitrary historical access beyond the CCXT default lookback.
    Response: newest-first; each row = [ts_s, open, close, high, low, vol, turnover].
    Note unusual column order: close before high/low, open at index 1.
    """
    start_s, end_s = window_s(date_str)
    params = {
        "symbol":  to_dash(symbol),
        "type":    "1min",
        "startAt": start_s,
        "endAt":   end_s,
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = SESSION.get("https://api.kucoin.com/api/v1/market/candles",
                               params=params, timeout=15).json()
            if resp.get("code") != "200000":
                return None
            raw = resp.get("data", [])
            if not raw:
                return None
            # [ts_s, open, close, high, low, volume, turnover]
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


# ── OKX: history-candles endpoint  (paginated backwards, 100 bars/call) ───────
def fetch_okx(symbol: str, date_str: str) -> "pd.DataFrame | None":
    """
    OKX /api/v5/market/history-candles goes arbitrarily far back.
    Max 100 bars per page; we walk backwards from end_ms until start_ms covered.
    'after' param = return records earlier than this timestamp (ms).
    Response columns: [ts_ms, o, h, l, c, vol, volCcy, volCcyQuote, confirm].
    Using int64 explicitly to avoid C-long overflow on large ms timestamps.
    """
    start_ms, end_ms = window_ms(date_str)
    inst_id  = to_dash(symbol)
    all_bars = []
    cursor   = end_ms + 1     # start paging from just after the window end

    for _ in range(5):        # max 5 × 100 = 500 bars, covers 200-min window
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
        earliest = int(data[-1][0])   # data is newest-first; last entry is oldest
        cursor   = earliest
        if earliest <= start_ms:
            break
        time.sleep(RATE_LIMIT_S)

    if not all_bars:
        return None

    # [ts_ms, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    df = pd.DataFrame(all_bars,
                      columns=["open_ts_ms", "o", "h", "l", "c", "v",
                               "volCcy", "volCcyQuote", "confirm"])
    df["open_ts_ms"] = df["open_ts_ms"].astype("int64")
    df = df[(df["open_ts_ms"] >= start_ms) & (df["open_ts_ms"] < end_ms)]
    df = df.sort_values("open_ts_ms")
    return df[["open_ts_ms", "o", "h", "l", "c", "v"]].reset_index(drop=True)


# ── Gate.io: REST API v4  (from/to Unix seconds, timeframe cascade) ───────────
# Gate.io enforces a hard 10 000-point rolling limit per timeframe regardless of
# the requested interval.  For 1m that is ~7 days; for 4h it is ~4.5 years.
# We cascade through coarser timeframes until one succeeds.
_GATEIO_CASCADE = ["1m", "5m", "15m", "1h", "4h"]

def _gateio_parse_raw(raw: list) -> pd.DataFrame:
    """
    Gate.io candle format (all intervals):
    [ts_s, quote_vol, close, high, low, open, base_vol, is_closed]
    Index:   0          1       2      3    4    5        6          7
    """
    rows = []
    for candle in raw:
        rows.append({
            "open_ts_ms": int(candle[0]) * 1000,
            "o":          candle[5],
            "h":          candle[3],
            "l":          candle[4],
            "c":          candle[2],
            "v":          candle[6],   # base volume (not quote)
        })
    return pd.DataFrame(rows).sort_values("open_ts_ms").reset_index(drop=True)


def fetch_gateio(symbol: str, date_str: str) -> "pd.DataFrame | None":
    """
    Tries 1m → 5m → 15m → 1h → 4h until Gate.io accepts the request.
    The 10 000-point rolling limit means 1m only covers ~7 days back, but
    4h covers ~4.5 years, ensuring most historical events are reachable.
    Coarser intervals are accepted by reconstruct_orderbook.py unchanged.
    """
    start_s, end_s = window_s(date_str)
    pair = to_underscore(symbol)

    for interval in _GATEIO_CASCADE:
        params = {
            "currency_pair": pair,
            "interval":      interval,
            "from":          start_s,
            "to":            end_s,
            "limit":         min(WINDOW_MINUTES, 1000),
        }
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = SESSION.get("https://api.gateio.ws/api/v4/spot/candlesticks",
                                   params=params, timeout=15)
                if resp.status_code == 400:
                    # "Candlestick too long ago" — try coarser timeframe
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
        # 400 or empty at this interval → try the next coarser one

    return None


# ── MEXC: Binance-compatible API  (startTime/endTime ms, timeframe cascade) ───
# MEXC rolling limits per interval (confirmed by testing):
#   1m  → ~35 days     5m/15m/30m → ~180 days     60m → 865+ days     4h/1d → years
# Note: MEXC uses "60m" for the 1-hour interval; "1h" returns an invalid-interval error.
_MEXC_CASCADE = [
    ("1m",   WINDOW_MINUTES),
    ("5m",   WINDOW_MINUTES // 5  + 1),
    ("15m",  WINDOW_MINUTES // 15 + 1),
    ("30m",  WINDOW_MINUTES // 30 + 1),
    ("60m",  WINDOW_MINUTES // 60 + 1),
    ("4h",   max(1, WINDOW_MINUTES // 240 + 1)),
    ("1d",   1),
]

def fetch_mexc(symbol: str, date_str: str) -> "pd.DataFrame | None":
    """
    Cascades through 1m → 5m → 15m → 30m → 60m → 4h → 1d until MEXC
    returns data for the target date. '60m' reaches 865+ days back;
    '4h'/'1d' cover the full dataset history.
    Format: [ts_ms, open, high, low, close, vol, close_ts, quote_vol] (8 cols).
    Note: '1h' is an invalid interval on MEXC; the cascade uses '60m' instead.
    """
    start_ms, end_ms = window_ms(date_str)

    for interval, limit in _MEXC_CASCADE:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": str(start_ms),
            "endTime":   str(end_ms),
            "limit":     limit,
        }
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = SESSION.get("https://api.mexc.com/api/v3/klines",
                                   params=params, timeout=15)
                if resp.status_code != 200:
                    break
                raw = resp.json()
                if not raw or isinstance(raw, dict):
                    break           # empty → try coarser interval
                df = pd.DataFrame(raw)
                # 8 columns: [ts_ms, o, h, l, c, vol, close_ts, quote_vol]
                df = df.iloc[:, :8]
                df.columns = ["open_ts_ms", "o", "h", "l", "c", "v", "close_t", "quote_v"]
                df["open_ts_ms"] = df["open_ts_ms"].astype("int64")
                return df[["open_ts_ms", "o", "h", "l", "c", "v"]].sort_values(
                    "open_ts_ms"
                ).reset_index(drop=True)
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(1)
                else:
                    print(f"      [ERR] mexc {symbol} ({interval}): {e}")
                    break
        time.sleep(RATE_LIMIT_S)

    return None


# ── Bitget: history-candles v2  (startTime/endTime ms, 200 bars/call) ─────────
def fetch_bitget(symbol: str, date_str: str) -> "pd.DataFrame | None":
    """
    Bitget v2 spot history-candles with explicit startTime/endTime (ms).
    Max 200 bars per call — matches WINDOW_MINUTES exactly (one request).
    Response: oldest-first; 8 columns:
    [ts_ms, open, high, low, close, baseVol, quoteVol, usdtVol]
    """
    start_ms, end_ms = window_ms(date_str)
    params = {
        "symbol":      symbol,
        "granularity": "1min",
        "startTime":   str(start_ms),
        "endTime":     str(end_ms),
        "limit":       WINDOW_MINUTES,
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
            # 8 columns: [ts_ms, open, high, low, close, baseVol, quoteVol, usdtVol]
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


# ── Exchange dispatch table ───────────────────────────────────────────────────
EXCHANGE_FETCHERS = {
    "bybit":  fetch_bybit,
    "kucoin": fetch_kucoin,
    "okx":    fetch_okx,
    "gateio": fetch_gateio,
    "mexc":   fetch_mexc,
    "bitget": fetch_bitget,
}


# ── Binance orchestration ─────────────────────────────────────────────────────
def fetch_binance(target_exchanges=None):
    if target_exchanges and "binance" not in target_exchanges:
        return
    print("\n" + "=" * 65)
    print("  BINANCE  —  ArdiaD/PumpDump dataset (322 confirmed events)")
    print("  Source : data.binance.vision public archive")
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
        df_k, full_sym = _binance_vision_fetch(sym, date)
        if df_k is None:
            print("NOT FOUND")
            fail += 1
            continue
        out_dir = os.path.join(REAL_BASE, "binance", "pumps", full_sym)
        os.makedirs(out_dir, exist_ok=True)
        df_k.to_csv(os.path.join(out_dir, f"{full_sym}-1m-{date}.csv"), index=False)
        print(f"OK -> {full_sym}  ({len(df_k)} rows)")
        ok += 1
        time.sleep(0.25)

    print(f"\n  Binance: {ok} downloaded, {fail} not found in archive")


# ── Non-Binance orchestration ─────────────────────────────────────────────────
def fetch_exchange(exchange_id: str, events: pd.DataFrame):
    fetcher = EXCHANGE_FETCHERS.get(exchange_id)
    if fetcher is None:
        print(f"\n  [SKIP] {exchange_id} — no direct fetcher defined")
        return

    print(f"\n{'=' * 65}")
    print(f"  {exchange_id.upper()}  ({len(events)} events)")
    print(f"  Method : direct REST API (no CCXT, no rolling-window limit)")
    print(f"{'=' * 65}")

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
        print(f"  [{i:>4}/{len(todo)}]  {sym:<16} {date}  ", end="", flush=True)
        try:
            df_k = fetcher(sym, date)
            if df_k is None or len(df_k) == 0:
                print("NO DATA")
                fail += 1
            else:
                save_klines(df_k, exchange_id, sym, date)
                print(f"OK  ({len(df_k)} bars)")
                ok += 1
        except Exception as e:
            print(f"ERR: {e}")
            fail += 1
        time.sleep(RATE_LIMIT_S)

    print(f"\n  {exchange_id}: {ok} downloaded, {fail} failed/not found")


def fetch_non_binance(target_exchanges=None):
    global_df = pd.read_csv(GLOBAL_CSV)

    if os.path.exists(BYBIT_CSV):
        bybit_extra = pd.read_csv(BYBIT_CSV)
        global_df   = pd.concat([global_df, bybit_extra], ignore_index=True).drop_duplicates(
            subset=["exchange", "symbol", "date"]
        )

    for exch_id in sorted(global_df["exchange"].dropna().unique()):
        if exch_id == "binance":
            continue
        if target_exchanges and exch_id not in target_exchanges:
            continue
        subset = global_df[global_df["exchange"] == exch_id].copy()
        if subset.empty:
            continue
        fetch_exchange(exch_id, subset)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fetch pump OHLCV via direct exchange REST APIs (no CCXT)"
    )
    parser.add_argument(
        "--exchange", nargs="+",
        choices=list(EXCHANGE_FETCHERS.keys()) + ["binance"],
        help="Limit to specific exchange(s). Omit to fetch all.",
    )
    args   = parser.parse_args()
    target = set(args.exchange) if args.exchange else None

    print("=" * 65)
    print("  Multi-Exchange Pump Data Fetcher  (Direct REST API v2)")
    print("  No CCXT — each exchange uses its native bulk history endpoint")
    print("=" * 65)
    if target:
        print(f"  Exchanges : {sorted(target)}")
    else:
        print(f"  Exchanges : binance + {sorted(EXCHANGE_FETCHERS.keys())}")
    print()

    fetch_binance(target)
    fetch_non_binance(target)

    print("\n" + "=" * 65)
    print("  Fetch complete.")
    print("  Next steps:")
    print("    1. python scripts/label_real_data.py")
    print("    2. python scripts/reconstruct_orderbook.py")
    print("    3. python scripts/train_pump_detector.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
