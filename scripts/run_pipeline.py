"""
Full post-fetch pipeline runner.

Chains together all steps that must run after new real data is fetched:

  DATA COLLECTION
  1. fetch_all_pump_data.py           — scan & download pump events from all 8 exchanges

  LABELLING & ENRICHMENT
  2. fetch_and_label_tradebook_data.py — verify pump labels (30% retracement) + fetch real
                                         trade ticks + save peak_bucket in _meta.json

  RECONSTRUCTION
  3. reconstruct_orderbook.py         — build L2 + trades files for all new OHLCV
  4. add_orderbook_depth.py           — expand all L2 files to 10-level orderbook depth

  METADATA & MARKET CONTEXT
  5. tag_peak_buckets.py              — compute peak_pct/peak_bucket for reconstructed/ + synthetic/
  6. fetch_market_context.py          — fetch BTC OHLCV, build _market_ctx.csv + fill market_regime

  SYNTHETIC GENERATION
  7. generate_direct_synthetic.py     — DirectL2VAE: 3 market regimes × 6 peak buckets
                                        + _market_ctx.csv + _meta.json per sample
  8. generate_missing_trades.py       — fill trades files for any L2 missing a tradebook

  TRAINING
  9. train_pump_detector.py           — retrain CNN PumpDetector v3 (dual-stream, 9-cell weights)

Usage (run from project root):
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-fetch       (data already downloaded)
    python scripts/run_pipeline.py --skip-label
    python scripts/run_pipeline.py --skip-reconstruct
    python scripts/run_pipeline.py --skip-synthetic
    python scripts/run_pipeline.py --train-only       (training step only)
"""

import sys
import argparse
import subprocess
import time

STEPS = [
    ("fetch",       "scripts/fetch_all_pump_data.py",              "Scan & download pump events from all 8 exchanges"),
    ("label",       "scripts/fetch_and_label_tradebook_data.py",   "Verify pump labels + fetch trade ticks + save peak_bucket metadata"),
    ("reconstruct", "scripts/reconstruct_orderbook.py",            "L2 + trades reconstruction from OHLCV"),
    ("depth",       "scripts/add_orderbook_depth.py",              "Expand all L2 files to 10-level orderbook depth"),
    ("tag",         "scripts/tag_peak_buckets.py",                 "Tag peak_pct/peak_bucket for all reconstructed/ and synthetic/ L2 files"),
    ("mktctx",      "scripts/fetch_market_context.py",             "Fetch BTC OHLCV + build _market_ctx.csv for all events"),
    ("synthetic",   "scripts/generate_direct_synthetic.py",        "DirectL2VAE — 3 market regimes × 6 peak buckets + _market_ctx.csv + _meta.json"),
    ("fill-trades", "scripts/generate_missing_trades.py",          "Fill trades files for any L2 missing a tradebook"),
    ("train",       "scripts/train_pump_detector.py",              "CNN PumpDetector v3 training (dual-stream, 9-cell weights, zero-shot Bybit+OKX)"),
]


def run_step(name: str, script: str, description: str) -> bool:
    print()
    print("=" * 65)
    print(f"  STEP: {description}")
    print(f"  Script: {script}")
    print("=" * 65)
    start = time.time()
    result = subprocess.run([sys.executable, script], check=False)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n  [{name}] Done in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  [{name}] FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False


def main():
    parser = argparse.ArgumentParser(description="Full post-fetch pipeline runner")
    parser.add_argument("--skip-fetch",        action="store_true", help="Skip fetch_all_pump_data.py")
    parser.add_argument("--skip-label",        action="store_true", help="Skip fetch_and_label_tradebook_data.py")
    parser.add_argument("--skip-reconstruct",  action="store_true", help="Skip reconstruct_orderbook.py and add_orderbook_depth.py")
    parser.add_argument("--skip-meta",         action="store_true", help="Skip tag_peak_buckets.py and fetch_market_context.py")
    parser.add_argument("--skip-synthetic",    action="store_true", help="Skip synthetic generation steps")
    parser.add_argument("--train-only",        action="store_true", help="Run train_pump_detector.py only")
    args = parser.parse_args()

    skip = set()
    if args.skip_fetch or args.train_only:
        skip.add("fetch")
    if args.skip_label or args.train_only:
        skip.add("label")
    if args.skip_reconstruct or args.train_only:
        skip.update({"reconstruct", "depth"})
    if args.skip_meta or args.train_only:
        skip.update({"tag", "mktctx"})
    if args.skip_synthetic or args.train_only:
        skip.update({"synthetic", "fill-trades"})

    print("=" * 65)
    print("  Pump-and-Dump Pipeline Runner")
    print("=" * 65)

    for name, script, description in STEPS:
        if name in skip:
            print(f"\n  [SKIP] {description}")
            continue
        ok = run_step(name, script, description)
        if not ok:
            print(f"\n  Pipeline aborted at step [{name}].")
            sys.exit(1)

    print()
    print("=" * 65)
    print("  Pipeline complete.")
    print("  Trained model: models/pump_detector_v3.pth")
    print("  Run inference: see detector.md for scoring examples.")
    print("=" * 65)


if __name__ == "__main__":
    main()
