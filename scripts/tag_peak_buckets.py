"""
Tag all reconstructed/ and synthetic/ L2 files with peak magnitude metadata.

For every *_L2.csv file that does not already have a paired *_meta.json,
this script:
  1. Computes the peak price increase from the mid-price curve
  2. Measures retracement after the peak
  3. Assigns a peak_bucket (micro/small/medium/large/major/extreme)
  4. Writes a *_meta.json alongside the L2 file

Regime is inferred from the directory path ([regime] = pumps | control).
market_regime is set to "unknown" — run fetch_market_context.py afterwards
to fill it in for real/reconstructed events.

Applies to:
  reconstructed/[exchange]/[pumps|control]/[symbol]/*_L2.csv
  synthetic/[exchange]/[pumps|control]/[symbol]/*_L2.csv
    (Type-A synthetic: *_synthetic_L2.csv)
    (Type-B _direct_L2.csv files already get meta.json from generate_direct_synthetic.py)

Usage (from project root):
    python scripts/tag_peak_buckets.py
    python scripts/tag_peak_buckets.py --overwrite   # re-tag files that already have meta
"""

import os
import sys
import re
import json
import glob
import argparse

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Config ─────────────────────────────────────────────────────────────────────

SCAN_BASES = ["reconstructed", "synthetic"]

MIN_INCREASE          = 0.05
RETRACEMENT_THRESHOLD = 0.30

PEAK_BUCKETS = [
    ("micro",   0.05, 0.10),
    ("small",   0.10, 0.20),
    ("medium",  0.20, 0.30),
    ("large",   0.30, 0.40),
    ("major",   0.40, 0.50),
    ("extreme", 0.50, float("inf")),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_peak_bucket(increase: float) -> str:
    for name, lo, hi in PEAK_BUCKETS:
        if lo <= increase < hi:
            return name
    return "extreme"


def meta_path_for(l2_path: str) -> str:
    return re.sub(r'(_direct_L2|_synthetic_L2|_L2)\.csv$', '_meta.json', l2_path)


def regime_from_path(l2_path: str) -> str:
    """Extract 'pump' or 'control' from the directory structure."""
    parts = l2_path.replace("\\", "/").split("/")
    for p in parts:
        if p == "pumps":
            return "pump"
        if p in ("control", "normal", "uncertain"):
            return "control"
    return "unknown"


def compute_peak(l2_path: str) -> tuple[float, float, int]:
    """
    Compute (increase, retracement, peak_idx) from an L2 CSV mid-price series.
    Returns (0.0, 0.0, 0) on failure.
    """
    try:
        df = pd.read_csv(l2_path, usecols=["bid_price", "ask_price"])
        if len(df) < 5:
            return 0.0, 0.0, 0
        mid   = ((df["bid_price"] + df["ask_price"]) / 2).values
        p0    = float(mid[0])
        if p0 <= 0:
            return 0.0, 0.0, 0

        peak_idx  = int(np.argmax(mid))
        peak_val  = float(mid[peak_idx])
        increase  = (peak_val - p0) / p0

        if increase < MIN_INCREASE or peak_idx >= len(mid) - 2:
            return increase, 0.0, peak_idx

        after       = mid[peak_idx:]
        min_after   = float(after.min())
        price_range = peak_val - p0
        retracement = (peak_val - min_after) / price_range if price_range > 0 else 0.0
        return increase, retracement, peak_idx
    except Exception:
        return 0.0, 0.0, 0


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-tag files that already have a _meta.json")
    args = parser.parse_args()

    tagged   = 0
    skipped  = 0
    failed   = 0

    for base in SCAN_BASES:
        if not os.path.isdir(base):
            print(f"  [SKIP] {base}/ not found")
            continue

        # Collect all L2 files; exclude _market_ctx.csv (not a coin L2)
        l2_files = [
            p for p in glob.glob(
                os.path.join(base, "**", "*_L2.csv"), recursive=True
            )
            if "_market_ctx" not in p
        ]

        print(f"\n{base}/ — {len(l2_files)} L2 files")

        for l2_path in l2_files:
            mp = meta_path_for(l2_path)

            if os.path.exists(mp) and not args.overwrite:
                skipped += 1
                continue

            # Skip Type-B direct files — meta generated at creation time
            if "_direct_L2.csv" in l2_path and os.path.exists(mp):
                skipped += 1
                continue

            coin_regime = regime_from_path(l2_path)
            increase, retracement, peak_idx = compute_peak(l2_path)

            is_pump = (coin_regime == "pump" and
                       increase >= MIN_INCREASE and
                       retracement >= RETRACEMENT_THRESHOLD)

            meta = {
                "coin_regime":   coin_regime,
                "market_regime": "unknown",
                "peak_pct":      round(increase * 100, 2) if coin_regime == "pump" else None,
                "peak_bucket":   get_peak_bucket(increase) if is_pump else None,
                "peak_idx":      int(peak_idx) if coin_regime == "pump" else None,
            }

            try:
                with open(mp, "w") as f:
                    json.dump(meta, f, indent=2)
                tagged += 1
            except Exception as e:
                print(f"  [ERR] {l2_path}: {e}")
                failed += 1

    print(f"\n{'='*55}")
    print(f"  Tagged  : {tagged}")
    print(f"  Skipped (already have meta): {skipped}")
    print(f"  Failed  : {failed}")
    print(f"  Run fetch_market_context.py next to fill in market_regime.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
