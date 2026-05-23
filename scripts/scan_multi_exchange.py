"""
Multi-exchange pump-and-dump scanner — Direct REST API.

Scans OHLCV daily candles across 7 active exchanges for pump-and-dump events
using the 30% retracement rule (Andy's threshold):

    increase    = (high - initial_open) / initial_open  >= 5%
    retracement = (peak_high - min_close_after) / (peak_high - initial_open) >= 30%

Events passing both thresholds → pumps/
Events with large spike but no sufficient dump → volatile_control/ (hard negatives)

Data sources (no CCXT):
    Binance : api.binance.com/api/v3           (exchangeInfo + klines)
    Bybit   : api.bybit.com/v5/market          (instruments-info + kline)
    KuCoin  : api.kucoin.com/api/v1            (symbols + market/candles)
    OKX     : okx.com/api/v5/public            (instruments + history-candles)
    Gate.io : api.gateio.ws/api/v4/spot        (currency_pairs + candlesticks)
    MEXC    : api.mexc.com/api/v3              (exchangeInfo + klines)
    Bitget  : api.bitget.com/api/v2/spot       (public/symbols + history-candles)

Output:
    data/global_deep_scan_pumps.csv   — all confirmed pump events (all exchanges)
    data/scanned_pumps/[exchange].csv — per-exchange results (pumps + volatile_control)

Usage (run from project root):
    python scripts/scan_multi_exchange.py
    python scripts/scan_multi_exchange.py --exchange bybit okx
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────────
EXCHANGES = ["binance", "bybit", "kucoin", "okx", "gateio", "mexc", "bitget"]

QUOTE_ASSETS          = {"USDT", "BTC"}
MIN_INCREASE          = 0.05    # 5% spike minimum
RETRACEMENT_THRESHOLD = 0.30    # 30% dump from peak required to confirm P&D
DISCOVERY_MIN_MOVE    = 0.50    # 50% spike threshold for quick discovery pass
CANDLE_LIMIT          = 200     # daily candles (~6–7 months)
RATE_LIMIT_S          = 0.20    # pause between per-symbol requests
OUTPUT_BASE           = "data/scanned_pumps"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "pump-detector-research/1.0", "Accept": "application/json"})


# ── Symbol helpers ────────────────────────────────────────────────────────────
_QUOTES = ("USDT", "BUSD", "USDC", "BTC", "ETH", "USD")


def split_symbol(raw: str) -> tuple:
    """'BTCUSDT' → ('BTC', 'USDT').  Falls back to last-3 as quote."""
    raw = raw.upper()
    for q in _QUOTES:
        if raw.endswith(q):
            return raw[: -len(q)], q
    return raw[:-3], raw[-3:]


def to_raw(symbol: str) -> str:
    """'BTC-USDT' or 'BTC_USDT' → 'BTCUSDT'"""
    return symbol.replace("-", "").replace("_", "").upper()


def to_dash(raw: str) -> str:
    b, q = split_symbol(raw)
    return f"{b}-{q}"


def to_underscore(raw: str) -> str:
    b, q = split_symbol(raw)
    return f"{b}_{q}"


# ── Symbol discovery — one function per exchange ───────────────────────────────

def get_symbols_binance() -> list:
    resp = SESSION.get("https://api.binance.com/api/v3/exchangeInfo", timeout=20).json()
    return [
        s["symbol"]
        for s in resp.get("symbols", [])
        if s.get("status") == "TRADING" and s.get("quoteAsset") in QUOTE_ASSETS
    ]


def get_symbols_bybit() -> list:
    resp = SESSION.get(
        "https://api.bybit.com/v5/market/instruments-info",
        params={"category": "spot"}, timeout=20
    ).json()
    return [
        s["symbol"]
        for s in resp.get("result", {}).get("list", [])
        if s.get("quoteCoin") in QUOTE_ASSETS and s.get("status") == "Trading"
    ]


def get_symbols_kucoin() -> list:
    resp = SESSION.get("https://api.kucoin.com/api/v1/symbols", timeout=20).json()
    symbols = []
    for s in resp.get("data", []):
        if s.get("quoteCurrency") in QUOTE_ASSETS and s.get("enableTrading"):
            symbols.append(to_raw(s["symbol"]))   # BTC-USDT → BTCUSDT
    return symbols


def get_symbols_okx() -> list:
    resp = SESSION.get(
        "https://www.okx.com/api/v5/public/instruments",
        params={"instType": "SPOT"}, timeout=20
    ).json()
    symbols = []
    for s in resp.get("data", []):
        if s.get("quoteCcy") in QUOTE_ASSETS and s.get("state") == "live":
            symbols.append(to_raw(s["instId"]))   # BTC-USDT → BTCUSDT
    return symbols


def get_symbols_gateio() -> list:
    resp = SESSION.get("https://api.gateio.ws/api/v4/spot/currency_pairs", timeout=20).json()
    symbols = []
    for s in resp:
        if s.get("quote") in QUOTE_ASSETS and s.get("trade_status") == "tradable":
            symbols.append(to_raw(s["id"]))   # BTC_USDT → BTCUSDT
    return symbols


def get_symbols_mexc() -> list:
    resp = SESSION.get("https://api.mexc.com/api/v3/exchangeInfo", timeout=20).json()
    return [
        s["symbol"]
        for s in resp.get("symbols", [])
        if s.get("quoteAsset") in QUOTE_ASSETS and s.get("status") == "ENABLED"
    ]


def get_symbols_bitget() -> list:
    resp = SESSION.get(
        "https://api.bitget.com/api/v2/spot/public/symbols", timeout=20
    ).json()
    return [
        s["symbol"]
        for s in resp.get("data", [])
        if s.get("quoteCoin") in QUOTE_ASSETS and s.get("status") == "online"
    ]


SYMBOL_FETCHERS = {
    "binance": get_symbols_binance,
    "bybit":   get_symbols_bybit,
    "kucoin":  get_symbols_kucoin,
    "okx":     get_symbols_okx,
    "gateio":  get_symbols_gateio,
    "mexc":    get_symbols_mexc,
    "bitget":  get_symbols_bitget,
}


# ── Daily OHLCV — one function per exchange ────────────────────────────────────
# Each returns a list of [ts_ms, open, high, low, close, volume] rows,
# sorted chronologically.  Returns [] on any error.

def fetch_daily_binance(symbol: str) -> list:
    try:
        resp = SESSION.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": "1d", "limit": CANDLE_LIMIT},
            timeout=15
        )
        if resp.status_code != 200:
            return []
        return [[int(r[0]), r[1], r[2], r[3], r[4], r[5]] for r in resp.json()]
    except Exception:
        return []


def fetch_daily_bybit(symbol: str) -> list:
    """Max 200 candles per call; response is newest-first — reverse for chronological order."""
    try:
        resp = SESSION.get(
            "https://api.bybit.com/v5/market/kline",
            params={"category": "spot", "symbol": symbol,
                    "interval": "D", "limit": min(CANDLE_LIMIT, 200)},
            timeout=15
        ).json()
        if resp.get("retCode") != 0:
            return []
        raw = resp.get("result", {}).get("list", [])
        bars = [[int(r[0]), r[1], r[2], r[3], r[4], r[5]] for r in raw]
        return sorted(bars, key=lambda x: x[0])
    except Exception:
        return []


def fetch_daily_kucoin(symbol: str) -> list:
    """
    KuCoin daily candles.  Symbol converted to dash format internally.
    Response: newest-first [ts_s, open, close, high, low, vol, turnover]
    Note unusual column order: close before high/low.
    """
    try:
        resp = SESSION.get(
            "https://api.kucoin.com/api/v1/market/candles",
            params={"symbol": to_dash(symbol), "type": "1day"},
            timeout=15
        ).json()
        if resp.get("code") != "200000":
            return []
        raw = resp.get("data", [])
        bars = []
        for r in raw:
            # [ts_s, open, close, high, low, vol, turnover]
            bars.append([int(r[0]) * 1000, r[1], r[3], r[4], r[2], r[5]])
        return sorted(bars, key=lambda x: x[0])[-CANDLE_LIMIT:]
    except Exception:
        return []


def fetch_daily_okx(symbol: str) -> list:
    """OKX max 100 candles per page; paginate backwards until CANDLE_LIMIT reached."""
    inst_id  = to_dash(symbol)
    all_bars = []
    cursor   = None

    while len(all_bars) < CANDLE_LIMIT:
        params = {"instId": inst_id, "bar": "1D", "limit": 100}
        if cursor:
            params["after"] = cursor
        try:
            resp = SESSION.get(
                "https://www.okx.com/api/v5/market/history-candles",
                params=params, timeout=15
            ).json()
        except Exception:
            break
        if resp.get("code") != "0":
            break
        data = resp.get("data", [])
        if not data:
            break
        for r in data:
            all_bars.append([int(r[0]), r[1], r[2], r[3], r[4], r[5]])
        cursor = str(int(data[-1][0]))   # oldest ts → page further back
        if len(data) < 100:
            break
        time.sleep(RATE_LIMIT_S)

    return sorted(all_bars, key=lambda x: x[0])[-CANDLE_LIMIT:]


def fetch_daily_gateio(symbol: str) -> list:
    """
    Gate.io daily candles.
    Response row format: [ts_s, quote_vol, close, high, low, open, base_vol, is_closed]
    """
    try:
        resp = SESSION.get(
            "https://api.gateio.ws/api/v4/spot/candlesticks",
            params={"currency_pair": to_underscore(symbol),
                    "interval": "1d", "limit": CANDLE_LIMIT},
            timeout=15
        )
        if resp.status_code != 200:
            return []
        bars = []
        for r in resp.json():
            # index: 0=ts_s, 2=close, 3=high, 4=low, 5=open, 6=base_vol
            bars.append([int(r[0]) * 1000, r[5], r[3], r[4], r[2], r[6]])
        return sorted(bars, key=lambda x: x[0])
    except Exception:
        return []


def fetch_daily_mexc(symbol: str) -> list:
    try:
        resp = SESSION.get(
            "https://api.mexc.com/api/v3/klines",
            params={"symbol": symbol, "interval": "1d", "limit": CANDLE_LIMIT},
            timeout=15
        )
        if resp.status_code != 200:
            return []
        raw = resp.json()
        if not raw or isinstance(raw, dict):
            return []
        return [[int(r[0]), r[1], r[2], r[3], r[4], r[5]] for r in raw]
    except Exception:
        return []


def fetch_daily_bitget(symbol: str) -> list:
    """Bitget max 200 candles per call; sorted oldest-first in response."""
    try:
        resp = SESSION.get(
            "https://api.bitget.com/api/v2/spot/market/history-candles",
            params={"symbol": symbol, "granularity": "1day",
                    "limit": min(CANDLE_LIMIT, 200)},
            timeout=15
        ).json()
        if resp.get("code") != "00000":
            return []
        raw = resp.get("data", [])
        # [ts_ms, open, high, low, close, baseVol, quoteVol, usdtVol]
        bars = [[int(r[0]), r[1], r[2], r[3], r[4], r[5]] for r in raw]
        return sorted(bars, key=lambda x: x[0])
    except Exception:
        return []


DAILY_FETCHERS = {
    "binance": fetch_daily_binance,
    "bybit":   fetch_daily_bybit,
    "kucoin":  fetch_daily_kucoin,
    "okx":     fetch_daily_okx,
    "gateio":  fetch_daily_gateio,
    "mexc":    fetch_daily_mexc,
    "bitget":  fetch_daily_bitget,
}


# ── Pump scan logic ───────────────────────────────────────────────────────────

def scan_bars(bars: list) -> "dict | None":
    """
    Scan a list of [ts_ms, o, h, l, c, v] daily bars for a pump-and-dump event.
    Uses a sliding 20-candle window across the history.
    Returns a match dict for the first qualifying event, or None.
    """
    if len(bars) < 5:
        return None

    df = pd.DataFrame(bars, columns=["ts", "o", "h", "l", "c", "v"])
    df = df.apply(pd.to_numeric, errors="coerce")

    for i in range(len(df) - 4):
        window = df.iloc[i : i + 20] if i + 20 <= len(df) else df.iloc[i:]
        initial_price = float(window["o"].iloc[0])
        if initial_price <= 0:
            continue

        peak_idx_local = int(window["h"].argmax())
        peak_high      = float(window["h"].iloc[peak_idx_local])
        increase       = (peak_high - initial_price) / initial_price

        if increase < DISCOVERY_MIN_MOVE:
            continue

        after_peak      = window.iloc[peak_idx_local:]
        if len(after_peak) < 2:
            continue
        min_close_after = float(after_peak["c"].min())
        price_range     = peak_high - initial_price
        if price_range <= 0:
            continue
        retracement = (peak_high - min_close_after) / price_range

        is_pump = (increase >= MIN_INCREASE) and (retracement >= RETRACEMENT_THRESHOLD)

        return {
            "date":        datetime.fromtimestamp(int(window["ts"].iloc[0]) / 1000).strftime("%Y-%m-%d"),
            "increase":    round(increase, 4),
            "retracement": round(retracement, 4),
            "label":       "pump" if is_pump else "volatile_control",
        }

    return None


# ── Per-exchange scan ─────────────────────────────────────────────────────────

def scan_exchange(exchange_id: str) -> list:
    print(f"\n--- Scanning {exchange_id.upper()} ---")

    try:
        symbols = SYMBOL_FETCHERS[exchange_id]()
    except Exception as e:
        print(f"  [ERR] Could not load symbols for {exchange_id}: {e}")
        return []

    print(f"  Scanning {len(symbols)} symbols (USDT + BTC pairs)...")

    daily_fetcher = DAILY_FETCHERS[exchange_id]
    matches = []

    for sym in symbols:
        try:
            bars = daily_fetcher(sym)
            time.sleep(RATE_LIMIT_S)
            if not bars or len(bars) < 5:
                continue

            match = scan_bars(bars)
            if match is None:
                continue

            raw_sym = to_raw(sym)
            record  = {"exchange": exchange_id, "symbol": raw_sym, **match}
            matches.append(record)

            tag = "[PUMP]" if match["label"] == "pump" else "[ORGANIC]"
            print(
                f"  {tag:<10} {raw_sym:<16} {match['date']}  "
                f"+{match['increase']*100:.1f}%  ret={match['retracement']*100:.1f}%"
            )
        except Exception:
            continue

    pump_count = sum(1 for m in matches if m["label"] == "pump")
    print(f"  {exchange_id}: {pump_count} pump events found  ({len(matches)} total candidates)")
    return matches


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Multi-exchange P&D scanner — direct REST API (no CCXT)"
    )
    parser.add_argument(
        "--exchange", nargs="+",
        choices=EXCHANGES,
        help="Limit to specific exchange(s). Omit to scan all."
    )
    args = parser.parse_args()
    target = set(args.exchange) if args.exchange else set(EXCHANGES)

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print("=" * 65)
    print("  Multi-Exchange P&D Scanner  —  Direct REST API (no CCXT)")
    print(f"  Exchanges : {sorted(target)}")
    print(f"  Threshold : >{MIN_INCREASE*100:.0f}% spike "
          f"AND >{RETRACEMENT_THRESHOLD*100:.0f}% retracement")
    print(f"  Window    : {CANDLE_LIMIT} daily candles per symbol")
    print("=" * 65)

    all_matches = []

    for ex_id in sorted(target):
        events = scan_exchange(ex_id)
        all_matches.extend(events)

        if events:
            df_ex = pd.DataFrame(events)
            df_ex.to_csv(os.path.join(OUTPUT_BASE, f"{ex_id}_pumps.csv"), index=False)

    if not all_matches:
        print("\nNo pump events found across any exchange.")
        return

    df_all   = pd.DataFrame(all_matches)
    df_pumps = df_all[df_all["label"] == "pump"].drop(columns=["label"])
    df_pumps.to_csv("data/global_deep_scan_pumps.csv", index=False)

    print("\n" + "=" * 65)
    print(f"  Total candidates     : {len(df_all)}")
    print(f"  Confirmed pumps      : {len(df_pumps)}")
    print(f"  Volatile control     : {len(df_all) - len(df_pumps)}")
    print(f"  Global CSV           : data/global_deep_scan_pumps.csv")
    per_ex = df_pumps.groupby("exchange").size()
    for ex, cnt in per_ex.items():
        print(f"    {ex:<12} {cnt} pumps")
    print("=" * 65)
    print()
    print("  Next steps:")
    print("    1. python scripts/fetch_all_pump_data.py")
    print("    2. python scripts/fetch_control_data.py")


if __name__ == "__main__":
    main()
