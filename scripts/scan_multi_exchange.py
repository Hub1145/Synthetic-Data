"""
Multi-exchange pump-and-dump scanner.

Scans OHLCV daily candles across 8 major exchanges for pump-and-dump events
using the 30% retracement rule (Andy's threshold):

    increase    = (high - initial_open) / initial_open  >= 5%
    retracement = (peak_high - min_close_after) / (peak_high - initial_open) >= 30%

Events passing both thresholds → pumps/
Events with large spike but no sufficient dump → control/

Output:
    data/global_deep_scan_pumps.csv   — all confirmed pump events (all exchanges)
    data/scanned_pumps/[exchange].csv — per-exchange pump lists

Usage (run from project root):
    python scripts/scan_multi_exchange.py
"""

import os
import ccxt
import pandas as pd
from datetime import datetime
import time

# ── Config ────────────────────────────────────────────────────────────────────
EXCHANGES = [
    "binance",
    "bybit",
    "kucoin",
    "okx",
    "gateio",    # Gate.io — heavy micro-cap listing activity
    "mexc",      # MEXC — common P&D venue for new token launches
    "huobi",     # HTX/Huobi — historical Asian exchange P&D data
    "bitget",    # Bitget — growing derivatives + spot exchange
]

MIN_INCREASE          = 0.05   # 5% spike minimum to be a candidate
RETRACEMENT_THRESHOLD = 0.30   # 30% dump from peak to confirm P&D (Andy's rule)
DISCOVERY_MIN_MOVE    = 0.50   # 50% spike for quick discovery pass before retracement check
CANDLE_LIMIT          = 365    # 1 year of daily candles per symbol
RATE_LIMIT_S          = 0.15   # between symbol requests within an exchange
OUTPUT_BASE           = "data/scanned_pumps"


# ── Scanner ───────────────────────────────────────────────────────────────────
def scan_exchange(exchange_id: str) -> list:
    print(f"\n--- Scanning {exchange_id.upper()} ---")
    try:
        ex_cls = getattr(ccxt, exchange_id)
        ex     = ex_cls({"enableRateLimit": True})
        ex.load_markets()
    except Exception as e:
        print(f"  [ERR] Could not connect to {exchange_id}: {e}")
        return []

    # Collect USDT and BTC quote pairs (both are common P&D venues)
    symbols = [
        s for s in ex.symbols
        if s.endswith("/USDT") or s.endswith("/BTC")
    ]
    print(f"  Scanning {len(symbols)} symbols (USDT + BTC pairs)...")

    matches = []
    for sym in symbols:
        try:
            bars = ex.fetch_ohlcv(sym, timeframe="1d", limit=CANDLE_LIMIT)
            time.sleep(RATE_LIMIT_S)
            if not bars or len(bars) < 5:
                continue

            df = pd.DataFrame(bars, columns=["ts", "o", "h", "l", "c", "v"])

            # Slide a 20-candle window to find pump episodes
            for i in range(len(df) - 4):
                window = df.iloc[i:i + 20] if i + 20 <= len(df) else df.iloc[i:]
                initial_price = float(window["o"].iloc[0])
                if initial_price <= 0:
                    continue

                peak_idx_local = int(window["h"].argmax())
                peak_high      = float(window["h"].iloc[peak_idx_local])
                increase       = (peak_high - initial_price) / initial_price

                # Must exceed discovery threshold
                if increase < DISCOVERY_MIN_MOVE:
                    continue

                # Measure retracement from peak to end of window
                after_peak     = window.iloc[peak_idx_local:]
                if len(after_peak) < 2:
                    continue
                min_close_after = float(after_peak["c"].min())
                price_range     = peak_high - initial_price
                if price_range <= 0:
                    continue
                retracement = (peak_high - min_close_after) / price_range

                is_pump = (increase >= MIN_INCREASE) and (retracement >= RETRACEMENT_THRESHOLD)
                label   = "pump" if is_pump else "volatile_control"

                date_str  = datetime.fromtimestamp(
                    int(window["ts"].iloc[0]) / 1000
                ).strftime("%Y-%m-%d")
                raw_symbol = sym.replace("/", "")

                matches.append({
                    "exchange":       exchange_id,
                    "symbol":         raw_symbol,
                    "date":           date_str,
                    "increase":       round(increase, 4),
                    "retracement":    round(retracement, 4),
                    "label":          label,
                })

                tag = "[PUMP]" if is_pump else "[ORGANIC]"
                print(
                    f"  {tag:<10} {raw_symbol:<16} {date_str}  "
                    f"+{increase*100:.1f}%  ret={retracement*100:.1f}%"
                )
                break   # one event per window start to avoid flood

        except Exception:
            continue

    print(f"  {exchange_id}: {len(matches)} pump events found")
    return matches


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print("=" * 65)
    print("  Multi-Exchange P&D Scanner")
    print(f"  Exchanges : {EXCHANGES}")
    print(f"  Threshold : >{MIN_INCREASE*100:.0f}% spike "
          f"AND >{RETRACEMENT_THRESHOLD*100:.0f}% retracement")
    print("=" * 65)

    all_matches = []

    for ex_id in EXCHANGES:
        events = scan_exchange(ex_id)
        all_matches.extend(events)

        if events:
            df_ex = pd.DataFrame(events)
            df_ex.to_csv(os.path.join(OUTPUT_BASE, f"{ex_id}_pumps.csv"), index=False)

    if not all_matches:
        print("\nNo pump events found across any exchange.")
        return

    df_all = pd.DataFrame(all_matches)

    # Only keep confirmed pumps for the global output (control events recorded per-exchange)
    df_pumps = df_all[df_all["label"] == "pump"].drop(columns=["label"])
    df_pumps.to_csv("data/global_deep_scan_pumps.csv", index=False)

    print("\n" + "=" * 65)
    print(f"  Total events scanned : {len(df_all)}")
    print(f"  Confirmed pumps      : {len(df_pumps)}")
    print(f"  Global CSV saved to  : data/global_deep_scan_pumps.csv")
    per_ex = df_pumps.groupby("exchange").size()
    for ex, cnt in per_ex.items():
        print(f"    {ex:<12} {cnt} pumps")
    print("=" * 65)
    print()
    print("  Next step: python scripts/fetch_all_pump_data.py")


if __name__ == "__main__":
    main()
