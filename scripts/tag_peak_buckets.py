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

# Fix 1: rolling mean window for smoothing before argmax
SMOOTH_WINDOW = 5   # bars; adaptive — capped at len(mid)//5

# Fix 2: flag events where fewer than this many bars exist after the peak
MIN_POST_PEAK_BARS = 5

# Background volatility thresholds (std dev of 1-min BTC mid-price returns)
VOL_CALM_THRESHOLD     = 0.0010   # < 0.10% per bar  → calm
VOL_VOLATILE_THRESHOLD = 0.0030   # > 0.30% per bar  → volatile
                                   # in between       → normal

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


def ctx_path_for(l2_path: str) -> str:
    return re.sub(r'(_direct_L2|_synthetic_L2|_L2)\.csv$', '_market_ctx.csv', l2_path)


def compute_background_volatility(l2_path: str) -> str:
    """
    Classify background market volatility from the paired _market_ctx.csv.
    Returns calm / normal / volatile / unknown.
    """
    ctx_path = ctx_path_for(l2_path)
    if not os.path.exists(ctx_path):
        return "unknown"
    try:
        df = pd.read_csv(ctx_path, usecols=["bid_price", "ask_price"])
        if len(df) < 5:
            return "unknown"
        mid     = ((df["bid_price"] + df["ask_price"]) / 2).values
        returns = np.diff(mid) / mid[:-1]
        vol     = float(np.std(returns))
        if vol < VOL_CALM_THRESHOLD:
            return "calm"
        if vol > VOL_VOLATILE_THRESHOLD:
            return "volatile"
        return "normal"
    except Exception:
        return "unknown"


def _smooth(mid: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling mean with edge-padding so the output length matches the input.
    Uses 'edge' padding (repeats first/last value) to avoid boundary artifacts.
    """
    if window < 2 or len(mid) < window:
        return mid.copy()
    half = window // 2
    padded = np.pad(mid, half, mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(mid)]


def _null_peak() -> dict:
    return {"increase": 0.0, "retracement": 0.0, "peak_idx": 0, "peak_truncated": False}


def compute_peak(l2_path: str) -> dict:
    """
    Returns a dict with keys:
      increase        float   fractional price rise from bar 0 to peak
      retracement     float   fraction of gain given back after peak
      peak_idx        int     bar index of the detected peak
      peak_truncated  bool    True if fewer than MIN_POST_PEAK_BARS exist after peak
    """
    try:
        df = pd.read_csv(l2_path, usecols=["bid_price", "ask_price"])
        if len(df) < 5:
            return _null_peak()

        mid = ((df["bid_price"] + df["ask_price"]) / 2).values
        p0  = float(mid[0])
        if p0 <= 0:
            return _null_peak()

        # ── Fix 1: smooth before argmax ──────────────────────────────────────
        win      = max(3, min(SMOOTH_WINDOW, len(mid) // 5))
        smoothed = _smooth(mid, win)
        peak_idx = int(np.argmax(smoothed))

        # Use the raw mid value at the smoothed-identified peak index
        peak_val = float(mid[peak_idx])
        increase = (peak_val - p0) / p0

        if increase < MIN_INCREASE:
            return {"increase": increase, "retracement": 0.0,
                    "peak_idx": peak_idx, "peak_truncated": False}

        # ── Fix 2: always compute retracement with available post-peak data ──
        after          = mid[peak_idx:]
        peak_truncated = len(after) < MIN_POST_PEAK_BARS

        if len(after) < 2:
            # Peak is literally the last bar — no dump data at all
            return {"increase": increase, "retracement": 0.0,
                    "peak_idx": peak_idx, "peak_truncated": True}

        min_after   = float(after.min())
        price_range = peak_val - p0
        retracement = (peak_val - min_after) / price_range if price_range > 0 else 0.0

        return {
            "increase":       increase,
            "retracement":    retracement,
            "peak_idx":       peak_idx,
            "peak_truncated": peak_truncated,
        }

    except Exception:
        return _null_peak()


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
            res = compute_peak(l2_path)
            increase = res["increase"]
            retracement = res["retracement"]
            peak_idx = res["peak_idx"]
            peak_truncated = res["peak_truncated"]

            is_pump = (coin_regime == "pump" and
                       increase >= MIN_INCREASE and
                       retracement >= RETRACEMENT_THRESHOLD)

            # Preserve existing fields (e.g. market_regime already set)
            existing = {}
            if os.path.exists(mp):
                try:
                    with open(mp) as f:
                        existing = json.load(f)
                except Exception:
                    pass

            existing.update({
                "coin_regime":          coin_regime,
                "market_regime":        existing.get("market_regime", "unknown"),
                "background_volatility": compute_background_volatility(l2_path),
                "peak_pct":             round(increase * 100, 2) if coin_regime == "pump" else None,
                "peak_bucket":          get_peak_bucket(increase) if is_pump else None,
                "peak_idx":             int(peak_idx) if coin_regime == "pump" else None,
                "peak_truncated":       peak_truncated,
            })

            try:
                with open(mp, "w") as f:
                    json.dump(existing, f, indent=2)
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
