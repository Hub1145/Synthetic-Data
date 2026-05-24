"""
main.py -- Full pipeline orchestrator for the crypto pump-and-dump detection project.

Runs every stage from raw data collection through model training and threshold
calibration, and produces a structured run report with dataset statistics and
model performance metrics.

Pipeline stages
---------------
  Step  0  Scan exchanges for new pump events   (opt-in: --with-scan)
  Step  1  Fetch pump OHLCV klines              (7 exchanges — direct REST)
  Step  2  Fetch Ardia Binance pump dataset     (ArdiaD/PumpDump, 322 confirmed events)
  Step  3  Fetch volatile-control OHLCV         (7 exchanges — direct REST)
  Step  4  Verify pump labels + fetch real trade ticks
  Step  5  Reconstruct L2 orderbook from OHLCV
  Step  6  Expand to 10-level orderbook depth
  Step  7  Tag peak magnitude buckets  (meta.json)
  Step  8  Fetch market regime context  (classify normal / uncertain / pumped)
  Step  9  Fix remaining unknown market regimes + generate missing context
  Step 10  Migrate control/ -> normal/ + uncertain/ by market regime
  Step 11  Prepare L2 training data   (vectorise reconstructed/ into .npy)
  Step 12  Train DirectL2VAE          (Type-B generator, latent dim 64)
  Step 13  Generate Type-B synthetic  (DirectL2VAE, 840 files)
  Step 14  Fill missing trades files  (OHLCV buy-ratio method)
  Step 15  Train PumpDetectorV3       (dual-stream CNN, 18-cell weights)
  Step 16  Calibrate detection threshold  (PR curve on Bybit + OKX zero-shot set)

Usage
-----
  python scripts/main.py                       # full pipeline (scan skipped by default)
  python scripts/main.py --with-scan           # also run exchange scanner (Step 0)
  python scripts/main.py --skip-fetch          # skip data download (Steps 1-3)
  python scripts/main.py --skip-label          # skip pump label verification (Step 4)
  python scripts/main.py --skip-reconstruct    # skip L2 reconstruction + depth (Steps 5-6)
  python scripts/main.py --skip-meta           # skip peak tagging + market context (Steps 7-10)
  python scripts/main.py --skip-vae-train      # skip DirectL2VAE training (Step 12, use existing)
  python scripts/main.py --skip-synthetic      # skip VAE training + generation entirely (Steps 12-13)
  python scripts/main.py --skip-trades         # skip missing-trades fill (Step 14)
  python scripts/main.py --skip-calibrate      # skip threshold calibration (Step 16)
  python scripts/main.py --train-only          # Steps 15-16 only (skip all data prep)
  python scripts/main.py --report-only         # generate report without running any steps
  python scripts/main.py --dry-run             # print plan without executing any scripts
  python scripts/main.py --continue-on-error   # keep going even if a step fails
  python scripts/main.py --resume              # auto-detect completed steps and skip them
"""

import argparse
import glob
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOTAL_STEPS  = 16
EXCHANGES    = ["binance", "bybit", "kucoin", "okx", "mexc", "gateio", "bitget"]
REGIMES      = ["pumps", "normal", "uncertain"]
TIERS        = ["real", "reconstructed", "synthetic"]

# main.py lives in scripts/; project root is one level up
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUN_TS   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_PATH = os.path.join(OUTPUT_DIR, f"run_{RUN_TS}.log")
RPT_PATH = os.path.join(OUTPUT_DIR, f"run_report_{RUN_TS}.txt")
JSN_PATH = os.path.join(OUTPUT_DIR, f"run_report_{RUN_TS}.json")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class _Formatter(logging.Formatter):
    def format(self, record):
        ts = datetime.now().strftime("%H:%M:%S")
        return f"[{ts}] {record.levelname[:4]}  {record.getMessage()}"


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_Formatter())
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8", errors="replace")
    fh.setFormatter(_Formatter())
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


log = _setup_logging()

# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def _run_step(step_num: int, name: str, script: str,
              extra_args: list = None, dry_run: bool = False) -> tuple:
    """
    Execute one pipeline step as a subprocess.
    Returns (success: bool, captured_lines: list[str]).
    """
    extra_args = extra_args or []
    sep = "-" * 70
    log.info(sep)
    log.info(f"Step {step_num:02d}/{TOTAL_STEPS}  --  {name}")
    log.info(sep)

    if dry_run:
        cmd_str = f"python scripts/{script} {' '.join(extra_args)}".strip()
        log.info(f"  [DRY RUN] would run: {cmd_str}")
        return True, []

    cmd = [sys.executable, os.path.join("scripts", script)] + extra_args
    t0  = time.time()
    lines: list = []

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=PROJECT_ROOT,
            env=env,
        )
        for raw in proc.stdout:
            line = raw.rstrip()
            if line:
                log.info(f"    {line}")
                lines.append(line)
        proc.wait()
        elapsed = time.time() - t0

        if proc.returncode != 0:
            log.error(f"  Step {step_num:02d} FAILED (exit {proc.returncode}) "
                      f"after {elapsed:.1f}s")
            return False, lines

        log.info(f"  Done in {elapsed:.1f}s")
        return True, lines

    except FileNotFoundError:
        log.error(f"  Script not found: scripts/{script}")
        return False, lines
    except Exception as exc:
        log.error(f"  Unexpected error in step {step_num:02d}: {exc}")
        return False, lines

# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def _count(pattern: str) -> int:
    return len(glob.glob(pattern, recursive=True))


def collect_dataset_stats() -> dict:
    stats: dict = {}

    for tier in TIERS:
        stats[tier] = {}
        for ex in EXCHANGES:
            ex_dir = os.path.join(PROJECT_ROOT, tier, ex)
            if not os.path.isdir(ex_dir):
                continue
            stats[tier][ex] = {}
            for regime in REGIMES:
                rdir = os.path.join(ex_dir, regime)
                if not os.path.isdir(rdir):
                    stats[tier][ex][regime] = 0
                    continue
                n = len([d for d in os.listdir(rdir)
                         if os.path.isdir(os.path.join(rdir, d))])
                stats[tier][ex][regime] = n

        tier_root = os.path.join(PROJECT_ROOT, tier)
        stats[f"{tier}_totals"] = {
            "l2_files":     _count(os.path.join(tier_root, "**", "*_L2.csv")),
            "trades_files": _count(os.path.join(tier_root, "**", "*_trades.csv")),
            "market_ctx":   _count(os.path.join(tier_root, "**", "*_market_ctx.csv")),
            "meta_json":    _count(os.path.join(tier_root, "**", "*_meta.json")),
            "kline_files":  (_count(os.path.join(tier_root, "**", "*_klines.csv")) +
                             _count(os.path.join(tier_root, "**", "*-1m-*.csv"))),
        }

    return stats

# ---------------------------------------------------------------------------
# Scan results summary
# ---------------------------------------------------------------------------

def collect_scan_stats() -> dict:
    """Read data/global_deep_scan_pumps.csv and return per-exchange counts."""
    scan_path = os.path.join(PROJECT_ROOT, "data", "global_deep_scan_pumps.csv")
    if not os.path.exists(scan_path):
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(scan_path)
        if "exchange" in df.columns:
            return df["exchange"].value_counts().to_dict()
    except Exception:
        pass
    return {}

# ---------------------------------------------------------------------------
# Parse training output for model metrics
# ---------------------------------------------------------------------------

def parse_training_output(lines: list) -> dict:
    m = {
        "best_val_auc":        None,
        "mean_val_auc":        None,
        "std_val_auc":         None,
        "zero_shot_auc":       None,
        "zero_shot_accuracy":  None,
        "peak_mae_candles":    None,
        "total_train_windows": None,
        "pump_windows":        None,
        "control_windows":     None,
        "epochs_completed":    None,
        "classification_report": [],
    }
    in_report = False

    for line in lines:
        # Best validation AUC
        hit = re.search(r"best\s+val(?:idation)?\s+auc[:\s=]+([0-9]\.[0-9]+)", line, re.I)
        if hit:
            m["best_val_auc"] = float(hit.group(1))

        # Mean validation AUC and std
        hit = re.search(r"mean\s+val(?:idation)?\s+auc[:\s=]+([0-9]\.[0-9]+)\s*\(.*?([0-9]\.[0-9]+)", line, re.I)
        if hit:
            m["mean_val_auc"] = float(hit.group(1))
            m["std_val_auc"]  = float(hit.group(2))

        # Zero-shot AUC
        hit = re.search(r"zero.?shot\s+auc[:\s=]+([0-9]\.[0-9]+)", line, re.I)
        if hit:
            m["zero_shot_auc"] = float(hit.group(1))

        # Peak MAE in candles
        hit = re.search(r"peak\s+reg\s+mae[:\s=]+([0-9]+\.?[0-9]*)\s*candles", line, re.I)
        if hit:
            m["peak_mae_candles"] = float(hit.group(1))

        # Accuracy
        hit = re.search(r"accuracy[:\s=]+([0-9]+\.?[0-9]*)%?", line, re.I)
        if hit and m["zero_shot_accuracy"] is None:
            m["zero_shot_accuracy"] = float(hit.group(1))

        # Window counts
        hit = re.search(r"train\s+total[:\s=]+([0-9,]+)\s+windows", line, re.I)
        if hit:
            m["total_train_windows"] = int(hit.group(1).replace(",", ""))
        hit = re.search(r"\(([0-9,]+)\s+pump", line, re.I)
        if hit and m["pump_windows"] is None:
            m["pump_windows"] = int(hit.group(1).replace(",", ""))
        hit = re.search(r"([0-9,]+)\s+control\)", line, re.I)
        if hit and m["control_windows"] is None:
            m["control_windows"] = int(hit.group(1).replace(",", ""))

        # Epoch
        hit = re.search(r"^\s*([0-9]+)\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+", line)
        if hit:
            m["epochs_completed"] = int(hit.group(1))

        # Classification report block
        if re.search(r"precision\s+recall\s+f1", line, re.I):
            in_report = True
        if in_report:
            m["classification_report"].append(line)
        if in_report and "weighted avg" in line.lower():
            in_report = False

    return m

# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _fmt_elapsed(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def _auc_bar(val: float, width: int = 20) -> str:
    if val is None:
        return "[" + " " * width + "] N/A"
    filled = int(round(val * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {val:.4f}"


def write_report(stats: dict, metrics: dict, timings: dict,
                 step_results: dict, args, run_start: float):
    total_elapsed = time.time() - run_start
    elapsed_str   = _fmt_elapsed(total_elapsed)
    ts_str        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep70  = "=" * 70
    sep70d = "-" * 70

    out = []

    out.append(sep70)
    out.append("  CRYPTO PUMP-AND-DUMP DETECTION PIPELINE  --  RUN REPORT")
    out.append(sep70)
    out.append(f"  Generated  : {ts_str}")
    out.append(f"  Duration   : {elapsed_str}")
    out.append(f"  Platform   : {platform.system()} {platform.machine()}, "
               f"Python {platform.python_version()}")
    out.append(f"  Log file   : {LOG_PATH}")
    out.append("")

    # Pipeline step summary
    out.append(sep70d)
    out.append("  PIPELINE STEP SUMMARY")
    out.append(sep70d)
    step_names = {
        0:  "Scan exchanges for new pump events",
        1:  "Fetch pump OHLCV (7 exchanges)",
        2:  "Fetch Ardia Binance pump dataset",
        3:  "Fetch volatile-control OHLCV (7 exchanges)",
        4:  "Verify labels + fetch trades",
        5:  "Reconstruct L2 orderbook",
        6:  "Expand to 10-level depth",
        7:  "Tag peak magnitude buckets",
        8:  "Fetch market regime context",
        9:  "Fix unknown market regimes",
        10: "Migrate control -> normal/uncertain",
        11: "Prepare L2 training data",
        12: "Train DirectL2VAE",
        13: "Generate Type-B synthetic",
        14: "Fill missing trades files",
        15: "Train PumpDetectorV3",
        16: "Calibrate detection threshold",
    }
    for num, name in step_names.items():
        result = step_results.get(num)
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "OK"
        else:
            status = "FAILED"
        t = timings.get(f"Step {num:02d} -- {name}", 0)
        t_str = _fmt_elapsed(t) if t > 0 else "---"
        out.append(f"  Step {num:02d}  [{status:<7}]  {name:<44} {t_str}")
    out.append("")

    # ---- Scanner results (if scan ran) ------------------------------------
    scan_stats = collect_scan_stats()
    if scan_stats:
        out.append(sep70d)
        out.append("  SCANNER RESULTS  (data/global_deep_scan_pumps.csv)")
        out.append(sep70d)
        total_scan = sum(scan_stats.values())
        for ex, cnt in sorted(scan_stats.items(), key=lambda x: -x[1]):
            out.append(f"  {ex:<12} {cnt:>6,} pump events")
        out.append(f"  {'TOTAL':<12} {total_scan:>6,}")
        out.append("")

    # ---- Dataset statistics ------------------------------------------------
    out.append(sep70d)
    out.append("  DATASET STATISTICS")
    out.append(sep70d)

    grand_l2 = 0
    for tier in TIERS:
        tier_data = stats.get(tier, {})
        tot       = stats.get(f"{tier}_totals", {})
        out.append(f"\n  {tier.upper()}/")
        out.append(f"    {'Exchange':<12} {'pumps':>7} {'normal':>8} {'uncertain':>10}")
        out.append(f"    {'-'*12} {'-'*7} {'-'*8} {'-'*10}")
        gp = gn = gu = 0
        for ex in EXCHANGES:
            d = tier_data.get(ex, {})
            p = d.get("pumps", 0)
            n = d.get("normal", 0)
            u = d.get("uncertain", 0)
            if p + n + u > 0:
                out.append(f"    {ex:<12} {p:>7,} {n:>8,} {u:>10,}")
            gp += p; gn += n; gu += u
        out.append(f"    {'TOTAL':<12} {gp:>7,} {gn:>8,} {gu:>10,}")
        l2 = tot.get("l2_files", 0)
        grand_l2 += l2
        out.append(f"    >> L2 files: {l2:,}  |  trades: {tot.get('trades_files',0):,}  |  "
                   f"market_ctx: {tot.get('market_ctx',0):,}  |  "
                   f"klines: {tot.get('kline_files',0):,}")

    out.append(f"\n  Grand total L2 files across all tiers : {grand_l2:,}")
    out.append("")

    # ---- Detection matrix --------------------------------------------------
    out.append(sep70d)
    out.append("  18-CELL DETECTION MATRIX  (coin_regime × market_regime × volatility)")
    out.append(sep70d)
    out.append("")
    out.append("  Label         | Mkt Normal | Mkt Uncertain | Mkt Pumped")
    out.append("  --------------|------------|---------------|------------")
    out.append("  Pump  + calm  |   4.0      |      2.5      |    1.5")
    out.append("  Pump  + norm  |   3.0      |      2.0      |    1.0")
    out.append("  Pump  + vol   |   2.0      |      1.5      |    1.0")
    out.append("  Ctrl  + calm  |   3.5      |      2.0      |    1.5")
    out.append("  Ctrl  + norm  |   2.5      |      1.5      |    1.0")
    out.append("  Ctrl  + vol   |   1.5      |      1.0      |    1.0")
    out.append("")

    # ---- Model performance -------------------------------------------------
    out.append(sep70d)
    out.append("  MODEL PERFORMANCE  --  PumpDetectorV3")
    out.append(sep70d)
    out.append("")
    out.append("  Architecture")
    out.append("    Coin stream  : 3-block Conv1D  (6 features × 96 timesteps)")
    out.append("    Market stream: 3-block Conv1D  (6 features × 96 timesteps)")
    out.append("    Trunk        : Concat(256) → Linear(128) → Dropout(0.3)")
    out.append("    cls_head     : Linear(1) → Sigmoid  [pump probability]")
    out.append("    reg_head     : Linear(1) → Sigmoid  [peak_pos = peak_idx/96]")
    out.append("    Loss         : BCE + 0.3 × MSE (pump samples only)")
    out.append("    Weights file : models/pump_detector_v3.pth")
    out.append("")
    out.append("  Training configuration")
    out.append("    Train exchanges : Binance, KuCoin, MEXC, Gate.io, Bitget")
    out.append("    Zero-shot test  : Bybit, OKX  (never seen during training)")
    out.append("    Pump window cap : 5,000 per exchange  (prevents Gate.io dominance)")
    out.append("    Window          : 96 timesteps  |  50% overlap (step=48)")
    out.append("    Epochs          : 60  |  Batch: 64  |  Optimizer: Adam (lr=1e-3)")
    out.append("    Class handling  : WeightedRandomSampler + 18-cell sample weights")
    out.append("")

    if metrics.get("total_train_windows"):
        tw = metrics["total_train_windows"]
        pw = metrics.get("pump_windows") or 0
        cw = metrics.get("control_windows") or 0
        out.append(f"  Training windows  : {tw:,}  (pump={pw:,} / control={cw:,})")
    if metrics.get("epochs_completed"):
        out.append(f"  Epochs completed  : {metrics['epochs_completed']}")
    out.append("")

    bval  = metrics.get("best_val_auc")
    mval  = metrics.get("mean_val_auc")
    sval  = metrics.get("std_val_auc")
    btest = metrics.get("zero_shot_auc")
    bacc  = metrics.get("zero_shot_accuracy")
    pmae  = metrics.get("peak_mae_candles")

    if bval is not None or btest is not None:
        out.append("  Results:")
        out.append(f"    Best val AUC    (train set)        : {_auc_bar(bval)}")
        if mval is not None:
            std_str = f"  ±{sval:.4f}" if sval is not None else ""
            out.append(f"    Mean val AUC    (all epochs)       : {_auc_bar(mval)}{std_str}")
        out.append(f"    Zero-shot AUC   (Bybit + OKX)     : {_auc_bar(btest)}")
        if bacc is not None:
            out.append(f"    Zero-shot accuracy               : {bacc:.1f}%")
        if pmae is not None:
            out.append(f"    Peak timing MAE                  : {pmae:.1f} candles")
        out.append("")

        if btest is not None:
            if btest >= 0.80:
                quality = "EXCELLENT -- strong cross-exchange generalisation"
            elif btest >= 0.72:
                quality = "GOOD -- solid zero-shot transfer"
            elif btest >= 0.60:
                quality = "MODERATE -- usable but may need more data or retraining"
            else:
                quality = "WEAK -- review data quality and exchange balance"
            out.append(f"    Assessment : {quality}")
            if mval is not None and bval is not None and (bval - mval) > 0.05:
                out.append(f"    Note       : Best AUC is {bval-mval:.3f} above mean — "
                           f"AUC is unstable across epochs. Check exchange balance.")
            out.append("")
    else:
        out.append("  Results : training was skipped or metrics not captured")
        out.append("")

    if metrics.get("classification_report"):
        out.append("  Classification report (zero-shot, Bybit + OKX):")
        for cl in metrics["classification_report"]:
            out.append(f"    {cl}")
        out.append("")

    # ---- Threshold guidance ------------------------------------------------
    out.append(sep70d)
    out.append("  THRESHOLD GUIDANCE")
    out.append(sep70d)
    out.append("")
    out.append("  Threshold  |  Effect")
    out.append("  -----------|-------------------------------------------------------")
    out.append("  0.30       |  High recall — catches most pumps, more false alarms")
    out.append("  0.50       |  Balanced (default)")
    out.append("  0.65-0.75  |  Recommended for live alerting (precision / recall balance)")
    out.append("  0.85+      |  Very conservative — flags only the most obvious events")
    out.append("")
    out.append("  Production cascade:")
    out.append("    PUMPABLE COINS score > 70")
    out.append("      -> PumpDetectorV3(x_coin, x_market) > CNN_THRESHOLD  ->  ALERT")
    out.append("")
    if step_results.get(16):
        out.append("  Step 16 ran — see models/pr_curve.png for the PR curve.")
        out.append("  Update CNN_THRESHOLD in live/config.py with the printed value.")
    else:
        out.append("  Run scripts/calibrate_threshold.py to find the optimal threshold.")
    out.append("")

    # ---- Pipeline timing ---------------------------------------------------
    out.append(sep70d)
    out.append("  PIPELINE TIMING")
    out.append(sep70d)
    for step_key, elapsed in timings.items():
        t_str = _fmt_elapsed(elapsed) if elapsed > 0 else "skipped"
        out.append(f"  {step_key:<54} {t_str}")
    out.append(f"\n  {'Total pipeline':54} {elapsed_str}")
    out.append("")

    # ---- Saved artifacts ---------------------------------------------------
    out.append(sep70d)
    out.append("  SAVED ARTIFACTS")
    out.append(sep70d)
    artifacts = [
        ("models/pump_detector_v3.pth",  "PumpDetectorV3 trained weights"),
        ("models/direct_l2_vae_v1.pth",  "DirectL2VAE generator weights"),
        ("models/l2_scaler.pkl",          "L2 StandardScaler (DirectL2VAE)"),
        ("models/pr_curve.png",           "Precision-recall curve (threshold calibration)"),
        ("data/global_deep_scan_pumps.csv", "Multi-exchange pump scan results"),
    ]
    for path, desc in artifacts:
        tag = "[found]  " if os.path.exists(os.path.join(PROJECT_ROOT, path)) else "[missing]"
        out.append(f"  {tag}  {path:<42}  {desc}")
    out.append("")
    out.append(f"  [output]   {LOG_PATH}")
    out.append(f"  [output]   {RPT_PATH}")
    out.append(f"  [output]   {JSN_PATH}")
    out.append("")

    # ---- Known limitations -------------------------------------------------
    out.append(sep70d)
    out.append("  KNOWN LIMITATIONS & NEXT STEPS")
    out.append(sep70d)
    out.append("")
    out.append("  Real L2 unavailability:")
    out.append("    KuCoin, OKX, Gate.io, MEXC, Bitget have no free orderbook")
    out.append("    archives. real/ for these exchanges holds reconstructed L2 as a proxy.")
    out.append("")
    out.append("  Trade tick coverage:")
    out.append("    Only Binance (full history) provides real taker-flow ticks.")
    out.append("    Other exchanges use OHLCV-derived approximations.")
    out.append("    Paid providers: Tardis.dev, Kaiko.")
    out.append("")
    out.append("  Real L2 (Phase 6+):")
    out.append("    hftbacktest + reach.stratosphere.capital for true L2 at sub-second")
    out.append("    granularity. Will significantly improve DirectL2VAE fidelity.")
    out.append("")
    out.append("  GPU training:")
    out.append("    Run on a CUDA-capable machine for full 60-epoch training.")
    out.append("    CPU estimate: ~5-8 min/epoch on a modern laptop.")
    out.append("")

    out.append(sep70)
    out.append("  END OF REPORT")
    out.append(sep70)

    report_text = "\n".join(out)

    with open(RPT_PATH, "w", encoding="utf-8", errors="replace") as f:
        f.write(report_text)

    json_payload = {
        "run_timestamp":    RUN_TS,
        "duration_seconds": total_elapsed,
        "dataset_stats":    stats,
        "model_metrics":    metrics,
        "pipeline_timings": timings,
        "step_results":     {str(k): v for k, v in step_results.items()},
        "args":             vars(args),
    }
    with open(JSN_PATH, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, default=str)

    log.info(report_text)
    log.info(f"Report   -> {RPT_PATH}")
    log.info(f"JSON     -> {JSN_PATH}")

# ---------------------------------------------------------------------------
# Auto-skip (--resume)
# ---------------------------------------------------------------------------

def _auto_skip(args) -> None:
    root = PROJECT_ROOT

    def has_files(pattern: str) -> bool:
        return len(glob.glob(os.path.join(root, pattern), recursive=True)) > 0

    def exists(rel_path: str) -> bool:
        return os.path.exists(os.path.join(root, rel_path))

    checks = {}

    # Step 0 scan — skip if scan CSV already exists
    checks["scan"]        = exists("data/global_deep_scan_pumps.csv")

    # Steps 1-3 fetch — real/ has kline files
    checks["fetch"]       = (has_files("real/**/*_klines.csv") or
                             has_files("real/**/*-1m-*.csv"))

    # Step 4 label
    checks["label"]       = checks["fetch"]

    # Steps 5-6 reconstruct — reconstructed/ has L2 files
    checks["reconstruct"] = has_files("reconstructed/**/*_L2.csv")

    # Steps 7-10 meta — all L2 files have market_ctx
    l2_count  = len(glob.glob(os.path.join(root, "reconstructed/**/*_L2.csv"), recursive=True))
    ctx_count = len(glob.glob(os.path.join(root, "reconstructed/**/*_market_ctx.csv"), recursive=True))
    checks["meta"]        = l2_count > 0 and ctx_count >= l2_count * 0.9

    # Steps 11-12 VAE training
    checks["vae_train"]   = (exists("models/direct_l2_vae_v1.pth") and
                             exists("models/l2_scaler.pkl"))

    # Step 13 synthetic generation
    checks["synthetic"]   = has_files("synthetic/**/*_L2.csv")

    # Step 14 trades fill
    trades_count = len(glob.glob(os.path.join(root, "reconstructed/**/*_trades.csv"), recursive=True))
    checks["trades"]      = l2_count > 0 and trades_count >= l2_count * 0.9

    # Step 16 calibration
    checks["calibrate"]   = exists("models/pr_curve.png")

    if checks["scan"] and not args.with_scan:
        args.with_scan = False   # already have scan output, stay off by default
    if checks["fetch"]:       args.skip_fetch       = True
    if checks["label"]:       args.skip_label       = True
    if checks["reconstruct"]: args.skip_reconstruct = True
    if checks["meta"]:        args.skip_meta        = True
    if checks["vae_train"]:   args.skip_vae_train   = True
    if checks["synthetic"]:   args.skip_synthetic   = True
    if checks["trades"]:      args.skip_trades      = True
    if checks["calibrate"]:   args.skip_calibrate   = True

    log.info("  --resume: auto-detected completed steps:")
    log.info(f"    Step  0   (scan)                  : {'SKIP' if not args.with_scan     else 'RUN'}")
    log.info(f"    Steps 1-3 (fetch pump+ardia+ctrl) : {'SKIP' if checks['fetch']        else 'RUN'}")
    log.info(f"    Step  4   (verify labels)          : {'SKIP' if checks['label']        else 'RUN'}")
    log.info(f"    Steps 5-6 (reconstruct + depth)   : {'SKIP' if checks['reconstruct']  else 'RUN'}")
    log.info(f"    Steps 7-10 (meta + context)       : {'SKIP' if checks['meta']         else 'RUN'}")
    log.info(f"    Steps 11-12 (train DirectL2VAE)   : {'SKIP' if checks['vae_train']    else 'RUN'}")
    log.info(f"    Step  13  (generate synthetic)    : {'SKIP' if checks['synthetic']    else 'RUN'}")
    log.info(f"    Step  14  (fill trades)           : {'SKIP' if checks['trades']       else 'RUN'}")
    log.info(f"    Step  15  (train PumpDetectorV3)  : RUN")
    log.info(f"    Step  16  (calibrate threshold)   : {'SKIP' if checks['calibrate']    else 'RUN'}")
    log.info("")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Full crypto pump-and-dump detection pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--with-scan",          action="store_true",
                        help="Run exchange scanner to discover new pump events (Step 0)")
    parser.add_argument("--skip-fetch",         action="store_true",
                        help="Skip data download (Steps 1-3)")
    parser.add_argument("--skip-label",         action="store_true",
                        help="Skip pump label verification (Step 4)")
    parser.add_argument("--skip-reconstruct",   action="store_true",
                        help="Skip L2 reconstruction + depth expansion (Steps 5-6)")
    parser.add_argument("--skip-meta",          action="store_true",
                        help="Skip peak tagging + market context (Steps 7-10)")
    parser.add_argument("--skip-vae-train",     action="store_true",
                        help="Skip DirectL2VAE training (Step 12, use existing weights)")
    parser.add_argument("--skip-synthetic",     action="store_true",
                        help="Skip VAE training + generation entirely (Steps 12-13)")
    parser.add_argument("--skip-trades",        action="store_true",
                        help="Skip missing-trades fill (Step 14)")
    parser.add_argument("--skip-calibrate",     action="store_true",
                        help="Skip threshold calibration (Step 16)")
    parser.add_argument("--train-only",         action="store_true",
                        help="Run training + calibration only (skip Steps 0-14)")
    parser.add_argument("--report-only",        action="store_true",
                        help="Collect stats and write report only — no scripts run")
    parser.add_argument("--dry-run",            action="store_true",
                        help="Print the plan without executing any scripts")
    parser.add_argument("--continue-on-error",  action="store_true",
                        help="Keep running even if a step fails")
    parser.add_argument("--resume",             action="store_true",
                        help="Auto-detect completed steps and skip them")
    args = parser.parse_args()

    # Derived flag logic
    if args.skip_synthetic:
        args.skip_vae_train = True

    if args.train_only:
        args.with_scan      = False
        args.skip_fetch     = True
        args.skip_label     = True
        args.skip_reconstruct = True
        args.skip_meta      = True
        args.skip_vae_train = True
        args.skip_synthetic = True
        args.skip_trades    = True

    if args.resume:
        _auto_skip(args)

    run_start     = time.time()
    timings:      dict = {}
    step_results: dict = {}
    all_ok        = True

    # Banner
    sep = "=" * 70
    log.info(sep)
    log.info("  CRYPTO PUMP-AND-DUMP DETECTION PIPELINE")
    log.info(f"  Run: {RUN_TS}")
    if args.dry_run:
        log.info("  MODE: DRY RUN -- no scripts will be executed")
    log.info(f"  Log: {LOG_PATH}")
    log.info(sep)
    log.info("")

    # Step wrapper
    def step(num: int, name: str, script: str,
             extra: list = None, skip: bool = False):
        nonlocal all_ok
        key = f"Step {num:02d} -- {name}"

        if skip or args.report_only:
            reason = "report-only mode" if args.report_only else "flag"
            log.info(f"  Step {num:02d}/{TOTAL_STEPS}  [SKIPPED -- {reason}]  {name}")
            timings[key]      = 0.0
            step_results[num] = None
            return []

        if not all_ok and not args.continue_on_error:
            log.warning(f"  Step {num:02d} ({name}) skipped -- earlier step failed")
            timings[key]      = 0.0
            step_results[num] = None
            return []

        t0      = time.time()
        ok, out = _run_step(num, name, script, extra, dry_run=args.dry_run)
        elapsed = time.time() - t0

        timings[key]      = elapsed
        step_results[num] = ok

        if not ok:
            all_ok = False
            if not args.continue_on_error:
                log.error("  Halting pipeline. Use --continue-on-error to keep going.")

        return out

    # ------------------------------------------------------------------
    # Step 0 — Scan exchanges for new pump events  (opt-in)
    # ------------------------------------------------------------------
    step(0,  "Scan exchanges for new pump events",
         "scan_multi_exchange.py",
         skip=not args.with_scan)

    # ------------------------------------------------------------------
    # Step 1 — Fetch pump OHLCV (7 exchanges via direct REST)
    # ------------------------------------------------------------------
    step(1,  "Fetch pump OHLCV klines (7 exchanges)",
         "fetch_all_pump_data.py",
         skip=args.skip_fetch)

    # ------------------------------------------------------------------
    # Step 2 — Fetch Ardia Binance pump dataset (322 confirmed events)
    # ------------------------------------------------------------------
    step(2,  "Fetch Ardia Binance pump dataset",
         "fetch_ardia_pumps.py",
         skip=args.skip_fetch)

    # ------------------------------------------------------------------
    # Step 3 — Fetch volatile-control OHLCV (7 exchanges)
    # ------------------------------------------------------------------
    step(3,  "Fetch volatile-control OHLCV (7 exchanges)",
         "fetch_control_data.py",
         skip=args.skip_fetch)

    # ------------------------------------------------------------------
    # Step 4 — Verify pump labels + fetch real trade ticks
    # ------------------------------------------------------------------
    step(4,  "Verify pump labels (30% retracement) + fetch real trades",
         "fetch_and_label_tradebook_data.py",
         skip=args.skip_label)

    # ------------------------------------------------------------------
    # Step 5 — Reconstruct L2 orderbook from OHLCV
    # ------------------------------------------------------------------
    step(5,  "Reconstruct L2 orderbook from OHLCV",
         "reconstruct_orderbook.py",
         skip=args.skip_reconstruct)

    # ------------------------------------------------------------------
    # Step 6 — Expand to 10-level orderbook depth
    # ------------------------------------------------------------------
    step(6,  "Expand to 10-level orderbook depth",
         "add_orderbook_depth.py",
         skip=args.skip_reconstruct)

    # ------------------------------------------------------------------
    # Step 7 — Tag peak magnitude buckets (meta.json)
    # ------------------------------------------------------------------
    step(7,  "Tag peak magnitude buckets (meta.json)",
         "tag_peak_buckets.py",
         skip=args.skip_meta)

    # ------------------------------------------------------------------
    # Step 8 — Fetch market regime context
    # ------------------------------------------------------------------
    step(8,  "Fetch market regime context (normal / uncertain / pumped)",
         "fetch_market_context.py",
         skip=args.skip_meta)

    # ------------------------------------------------------------------
    # Step 9 — Fix unknown market regimes
    # ------------------------------------------------------------------
    step(9,  "Fix remaining unknown market regimes + generate missing context",
         "fix_unknown_market_regimes.py",
         skip=args.skip_meta)

    # ------------------------------------------------------------------
    # Step 10 — Migrate control/ -> normal/ + uncertain/
    # ------------------------------------------------------------------
    step(10, "Migrate control/ -> normal/ + uncertain/ by market regime",
         "migrate_control_to_regimes.py",
         skip=args.skip_meta)

    # ------------------------------------------------------------------
    # Step 11 — Prepare L2 training data for DirectL2VAE
    # ------------------------------------------------------------------
    step(11, "Prepare L2 training data (vectorise reconstructed/ -> .npy)",
         "prepare_l2_training_data.py",
         skip=args.skip_vae_train)

    # ------------------------------------------------------------------
    # Step 12 — Train DirectL2VAE
    # ------------------------------------------------------------------
    step(12, "Train DirectL2VAE (Type-B generator, latent dim 64)",
         "train_direct_l2.py",
         skip=args.skip_vae_train)

    # ------------------------------------------------------------------
    # Step 13 — Generate Type-B synthetic data
    # ------------------------------------------------------------------
    step(13, "Generate Type-B synthetic (DirectL2VAE, 840 files)",
         "generate_direct_synthetic.py",
         skip=args.skip_synthetic)

    # ------------------------------------------------------------------
    # Step 14 — Fill missing trades files
    # ------------------------------------------------------------------
    step(14, "Fill missing trades files (OHLCV buy-ratio method)",
         "generate_missing_trades.py",
         skip=args.skip_trades)

    # ------------------------------------------------------------------
    # Step 15 — Train PumpDetectorV3
    # ------------------------------------------------------------------
    training_out = step(15, "Train PumpDetectorV3 (dual-stream CNN, 18-cell weights)",
                        "train_pump_detector.py")

    # ------------------------------------------------------------------
    # Step 16 — Calibrate detection threshold
    # ------------------------------------------------------------------
    step(16, "Calibrate detection threshold (PR curve, Bybit + OKX)",
         "calibrate_threshold.py",
         extra=["--no-plot"],           # headless — plot saved to models/pr_curve.png
         skip=args.skip_calibrate)

    # ------------------------------------------------------------------
    # Collect stats and write report
    # ------------------------------------------------------------------
    log.info("-" * 70)
    log.info("Collecting dataset statistics and writing report ...")
    log.info("-" * 70)

    metrics = parse_training_output(training_out or [])

    log.info("  Scanning file system for dataset statistics ...")
    stats = collect_dataset_stats()

    log.info("  Writing report ...")
    write_report(stats, metrics, timings, step_results, args, run_start)

    # Final status
    total_elapsed = time.time() - run_start
    log.info("")
    if all_ok or args.continue_on_error:
        log.info(f"Pipeline finished in {_fmt_elapsed(total_elapsed)}")
        log.info(f"Report -> {RPT_PATH}")
        log.info(f"JSON   -> {JSN_PATH}")
    if not all_ok:
        log.error("One or more steps FAILED -- review the log above")
        sys.exit(1)


if __name__ == "__main__":
    main()
