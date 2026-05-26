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
from pathlib import Path

import ccxt

from config import SCAN_INTERVAL_SECONDS, EXCHANGES, PUMPABLE_THRESHOLD, CNN_THRESHOLD
from cascade import LiveCascade


def get_usdt_pairs(exchange_id: str, top_n: int = 0) -> list:
    """Fetch USDT spot pairs ranked by 24h quote volume.

    Args:
        top_n: Return only the top N pairs. 0 means return all.

    Note: fetch_tickers() on some exchanges (e.g. Bybit) returns futures keys
    like 'BTC/USDT:USDT' rather than spot keys 'BTC/USDT'. We load spot symbols
    from markets first and pass them explicitly to fetch_tickers() to avoid the
    mismatch.
    """
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

    # All active USDT spot symbols according to the markets endpoint
    spot_symbols = [
        sym for sym, m in markets.items()
        if sym.endswith("/USDT")
        and m.get("spot", False)
        and m.get("active", True)
    ]

    if not top_n:
        # --all: no volume sort needed, just return everything
        return sorted(spot_symbols)

    # --top N: fetch tickers only for the spot symbols so keys always match
    try:
        tickers = exc.fetch_tickers(spot_symbols)
    except Exception:
        # Fallback: some exchanges don't support filtered fetch_tickers
        try:
            tickers = exc.fetch_tickers()
        except Exception as e:
            print(f"Failed to fetch tickers for {exchange_id}: {e}", file=sys.stderr)
            sys.exit(1)

    usdt_pairs = [
        (sym, float(tickers.get(sym, {}).get("quoteVolume") or 0))
        for sym in spot_symbols
    ]
    usdt_pairs.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in usdt_pairs[:top_n]]


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
        "--all",
        action="store_true",
        help="Monitor every USDT spot pair on the exchange (can be 500–1500+ symbols)",
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
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        help=(
            "Write output to a log file in addition to the terminal. "
            "Creates two files: PATH (full session log) and PATH with "
            "'.log' replaced by '_alerts.jsonl' (one JSON line per alert). "
            "Example: --log-file logs/session.log"
        ),
    )

    args = parser.parse_args()

    # ── Logging setup ──────────────────────────────────────────────────────────
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_fmt   = "%(asctime)s  %(levelname)-7s  %(message)s"
    log_date  = "%H:%M:%S"

    handlers = [logging.StreamHandler()]   # always log to terminal

    alert_log: Path | None = None
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
        # alerts JSONL: swap extension to _alerts.jsonl
        alert_log = log_path.with_name(log_path.stem + "_alerts.jsonl")
        print(f"Session log : {log_path}")
        print(f"Alerts log  : {alert_log}\n")

    logging.basicConfig(level=log_level, format=log_fmt, datefmt=log_date, handlers=handlers)

    if not args.symbols and not args.top and not args.all:
        parser.error("Provide --symbols SYMBOL [SYMBOL ...], --top N, or --all")

    if args.all:
        print(f"Fetching all USDT spot pairs from {args.exchange} ...")
        symbols = get_usdt_pairs(args.exchange)
        preview = symbols[:5]
        more = f" ... +{len(symbols)-5} more" if len(symbols) > 5 else ""
        print(f"Monitoring {len(symbols)} symbols: {preview}{more}\n")
    elif args.top:
        print(f"Fetching top {args.top} USDT pairs from {args.exchange} ...")
        symbols = get_usdt_pairs(args.exchange, args.top)
        preview = symbols[:5]
        more = f" ... +{len(symbols)-5} more" if len(symbols) > 5 else ""
        print(f"Monitoring {len(symbols)} symbols: {preview}{more}\n")
    else:
        symbols = args.symbols

    cascade = LiveCascade(
        exchange_id=args.exchange,
        symbols=symbols,
        cooldown_minutes=args.cooldown,
        alert_log=alert_log,
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
