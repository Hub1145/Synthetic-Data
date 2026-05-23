"""
Live Pump Detector — terminal entry point.

Examples
--------
# Watch specific symbols on Binance:
  python main.py --exchange binance --symbols ETH/USDT SOL/USDT DOGE/USDT

# Auto-select top 50 USDT pairs by 24h volume on KuCoin:
  python main.py --exchange kucoin --top 50

# Scan every 30 seconds with verbose debug output:
  python main.py --exchange bybit --top 30 --interval 30 --verbose

Notes
-----
- The buffer warms up after 96 candles (~96 minutes). No alerts until then.
- Requires pump_detector_v3.pth in models/ (run the training pipeline first).
- Public endpoints only — no API keys needed.
"""

import argparse
import asyncio
import logging
import sys

import ccxt

from config import SCAN_INTERVAL_SECONDS, EXCHANGES, PUMPABLE_THRESHOLD, CNN_THRESHOLD
from cascade import LiveCascade


def get_top_usdt_pairs(exchange_id: str, n: int) -> list:
    """Fetch the top-N USDT spot pairs ranked by 24h quote volume."""
    exc_class = getattr(ccxt, exchange_id, None)
    if exc_class is None:
        print(f"Exchange '{exchange_id}' not found in ccxt.", file=sys.stderr)
        sys.exit(1)

    exc = exc_class({"enableRateLimit": True})
    try:
        markets = exc.load_markets()
    except Exception as e:
        print(f"Failed to load markets for {exchange_id}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        tickers = exc.fetch_tickers()
    except Exception as e:
        print(f"Failed to fetch tickers for {exchange_id}: {e}", file=sys.stderr)
        sys.exit(1)

    usdt_pairs = [
        (sym, float(t.get("quoteVolume") or 0))
        for sym, t in tickers.items()
        if sym.endswith("/USDT") and markets.get(sym, {}).get("spot", False)
    ]
    usdt_pairs.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in usdt_pairs[:n]]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Live pump-and-dump detector — terminal mode (no web server)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exchange",
        default="binance",
        choices=EXCHANGES,
        help="Exchange to monitor (default: binance)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYMBOL",
        help="Symbols to watch, e.g. ETH/USDT SOL/USDT DOGE/USDT",
    )
    parser.add_argument(
        "--top",
        type=int,
        metavar="N",
        help="Auto-select top N USDT pairs by 24h volume",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=SCAN_INTERVAL_SECONDS,
        help=f"Scan interval in seconds (default: {SCAN_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=10,
        help="Minutes between repeat alerts for the same coin (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.symbols and not args.top:
        parser.error("Provide --symbols SYMBOL [SYMBOL ...] or --top N")

    if args.top:
        print(f"Fetching top {args.top} USDT pairs from {args.exchange} ...")
        symbols = get_top_usdt_pairs(args.exchange, args.top)
        preview = symbols[:5]
        more = f" ... +{len(symbols)-5} more" if len(symbols) > 5 else ""
        print(f"Monitoring {len(symbols)} symbols: {preview}{more}\n")
    else:
        symbols = args.symbols

    cascade = LiveCascade(
        exchange_id=args.exchange,
        symbols=symbols,
        cooldown_minutes=args.cooldown,
    )

    print("=" * 62)
    print(f"  Pump Detector — {args.exchange.upper()}")
    print(f"  Symbols    : {len(symbols)}")
    print(f"  Thresholds : pumpable > {PUMPABLE_THRESHOLD}  |  CNN prob > {CNN_THRESHOLD}")
    print(f"  Interval   : {args.interval}s")
    print(f"  Cooldown   : {args.cooldown} min")
    print(f"  Warmup     : ~96 min (buffer fills before first alert)")
    print("  Press Ctrl+C to stop.")
    print("=" * 62 + "\n")

    try:
        asyncio.run(cascade.run(interval=args.interval))
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
