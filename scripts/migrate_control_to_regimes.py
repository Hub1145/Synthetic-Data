"""
Migrate control/ directories into three market-regime subdirectories.

Old structure:
    [tier]/[exchange]/control/[symbol]/[files]

New structure:
    [tier]/[exchange]/normal/[symbol]/[files]     ← control + market normal
    [tier]/[exchange]/uncertain/[symbol]/[files]  ← control + market uncertain
    [tier]/[exchange]/pumped/[symbol]/[files]     ← control + market pumped

Paired files moved together: *_L2.csv, *_trades.csv, *_market_ctx.csv, *_meta.json

Usage:
    python scripts/migrate_control_to_regimes.py
    python scripts/migrate_control_to_regimes.py --dirs reconstructed synthetic
    python scripts/migrate_control_to_regimes.py --dry-run
"""

import os
import sys
import re
import json
import glob
import shutil
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REGIME_DIRS = {"normal", "uncertain", "pumped"}
DEFAULT_REGIME = "normal"

# BTC market regime → destination directory
# "pumped" BTC during a control event → still "uncertain" (coin just followed BTC)
REGIME_MAP = {
    "normal":    "normal",
    "uncertain": "uncertain",
    "pumped":    "uncertain",   # BTC pumped but coin is control → uncertain bucket
    "unknown":   "normal",
}


def find_paired_files(l2_path: str):
    """Return all files in the same directory that share the same stem."""
    stem = re.sub(r'(_direct_L2|_synthetic_L2|_L2)\.csv$', '', l2_path)
    candidates = []
    base_dir = os.path.dirname(l2_path)
    for f in os.listdir(base_dir):
        fpath = os.path.join(base_dir, f)
        if fpath.startswith(stem) or fpath == l2_path:
            candidates.append(fpath)
    # include all files in the symbol directory (safest approach)
    return [os.path.join(base_dir, f) for f in os.listdir(base_dir)]


def get_dest_dir_name(meta_path: str) -> str:
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        regime = meta.get("market_regime", "unknown")
        return REGIME_MAP.get(regime, DEFAULT_REGIME)
    except Exception:
        return DEFAULT_REGIME


def migrate_exchange(base: str, exchange: str, dry_run: bool):
    ctrl_dir = os.path.join(base, exchange, "control")
    if not os.path.isdir(ctrl_dir):
        return 0, 0

    moved = failed = 0

    for symbol in os.listdir(ctrl_dir):
        sym_dir = os.path.join(ctrl_dir, symbol)
        if not os.path.isdir(sym_dir):
            continue

        # Find the meta.json for this symbol directory
        meta_files = glob.glob(os.path.join(sym_dir, "*_meta.json"))
        if meta_files:
            regime = get_dest_dir_name(meta_files[0])
        else:
            regime = DEFAULT_REGIME

        dest_dir = os.path.join(base, exchange, regime, symbol)

        try:
            if not dry_run:
                os.makedirs(dest_dir, exist_ok=True)
                # Move all files in this symbol directory
                for fname in os.listdir(sym_dir):
                    src = os.path.join(sym_dir, fname)
                    dst = os.path.join(dest_dir, fname)
                    if os.path.isfile(src):
                        shutil.move(src, dst)
                # Remove now-empty symbol dir
                try:
                    os.rmdir(sym_dir)
                except OSError:
                    pass
            else:
                print(f"  [DRY] {sym_dir}  →  {dest_dir}")
            moved += 1
        except Exception as e:
            print(f"  [ERR] {sym_dir}: {e}")
            failed += 1

    # Remove empty control/ dir if empty
    if not dry_run:
        try:
            os.rmdir(ctrl_dir)
        except OSError:
            pass  # not empty — some files remain

    return moved, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+",
                        default=["real", "reconstructed", "synthetic"],
                        help="Top-level directories to migrate")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview moves without touching files")
    args = parser.parse_args()

    exchanges = ["binance", "bybit", "kucoin", "okx", "huobi", "mexc", "gateio", "bitget"]

    total_moved = total_failed = 0

    for base in args.dirs:
        print(f"\n=== {base.upper()} ===")
        for ex in exchanges:
            moved, failed = migrate_exchange(base, ex, args.dry_run)
            if moved or failed:
                print(f"  {ex:<10} moved={moved}  failed={failed}")
            total_moved  += moved
            total_failed += failed

    print(f"\n{'='*60}")
    if args.dry_run:
        print("  DRY RUN — no files were moved")
    print(f"  Symbol dirs moved : {total_moved}")
    print(f"  Failed            : {total_failed}")
    print(f"\n  New structure:")
    print(f"  [tier]/[exchange]/pumps/[symbol]/     ← confirmed pumps (label=1)")
    print(f"  [tier]/[exchange]/normal/[symbol]/    ← control, BTC normal (label=0)")
    print(f"  [tier]/[exchange]/uncertain/[symbol]/ ← control, BTC uncertain (label=0)")
    print(f"  [tier]/[exchange]/pumped/[symbol]/    ← control, BTC pumped (label=0)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
