"""
Threshold calibration for PumpDetectorV3.

Loads the trained model, runs it on the zero-shot test set (Bybit + OKX),
plots the precision-recall curve, and prints the threshold that maximises F1.

The recommended threshold replaces CNN_THRESHOLD in live/config.py.

Usage (from project root):
    python scripts/calibrate_threshold.py
    python scripts/calibrate_threshold.py --exchanges bybit okx
    python scripts/calibrate_threshold.py --no-plot
"""

import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score,
    f1_score, precision_score, recall_score,
)

from models.pump_detector import PumpDetectorV3
from scripts.train_pump_detector import (
    load_dataset, make_loader, PumpDataset,
    NUM_FEATURES, BATCH_SIZE, TIMESTEPS,
)

MODEL_PATH      = "models/pump_detector_v3.pth"
DEFAULT_EXCHANGES = ["bybit", "okx"]
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(exchanges: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob) arrays for the given exchanges."""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run scripts/train_pump_detector.py first.")
        sys.exit(1)

    model = PumpDetectorV3(
        num_coin_features=NUM_FEATURES,
        num_market_features=NUM_FEATURES,
    ).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    print(f"\nLoading data for: {exchanges}")
    X_coin, X_mkt, y, peak, w = load_dataset(exchanges)
    if len(X_coin) == 0:
        print("ERROR: No data found for the specified exchanges.")
        sys.exit(1)

    ds     = PumpDataset(X_coin, X_mkt, y, peak, w)
    loader = make_loader(ds, BATCH_SIZE, shuffle=False)

    preds, labels = [], []
    with torch.no_grad():
        for x_coin, x_mkt, yb, _ in loader:
            pump_prob, _ = model(x_coin.to(DEVICE), x_mkt.to(DEVICE))
            preds.extend(pump_prob.cpu().numpy())
            labels.extend(yb.numpy())

    return np.array(labels), np.array(preds)


# ── Threshold analysis ────────────────────────────────────────────────────────

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the threshold that maximises F1 on the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # sklearn appends a final precision=1, recall=0 with no threshold; drop it
    f1 = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
        0.0,
    )
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx])


def print_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
        0.0,
    )

    auc = roc_auc_score(y_true, y_prob)
    bin_pred = (y_prob >= threshold).astype(int)
    p  = precision_score(y_true, bin_pred, zero_division=0)
    r  = recall_score(y_true, bin_pred, zero_division=0)
    f  = f1_score(y_true, bin_pred, zero_division=0)
    tp = int(((bin_pred == 1) & (y_true == 1)).sum())
    fp = int(((bin_pred == 1) & (y_true == 0)).sum())
    fn = int(((bin_pred == 0) & (y_true == 1)).sum())
    tn = int(((bin_pred == 0) & (y_true == 0)).sum())

    print("\n" + "=" * 60)
    print("  Precision-Recall Threshold Calibration")
    print("=" * 60)
    print(f"  ROC-AUC              : {auc:.4f}")
    print(f"  Best F1 threshold    : {threshold:.4f}")
    print(f"  At this threshold:")
    print(f"    Precision          : {p:.4f}")
    print(f"    Recall             : {r:.4f}")
    print(f"    F1                 : {f:.4f}")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print()

    # Print a table of precision/recall/F1 at common thresholds
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("  " + "-" * 44)
    sample_thresholds = np.arange(0.3, 0.96, 0.05)
    for t in sample_thresholds:
        bp = (y_prob >= t).astype(int)
        pp = precision_score(y_true, bp, zero_division=0)
        rr = recall_score(y_true, bp, zero_division=0)
        ff = f1_score(y_true, bp, zero_division=0)
        marker = " ◄ best F1" if abs(t - threshold) < 0.026 else ""
        print(f"  {t:>10.2f}  {pp:>10.4f}  {rr:>8.4f}  {ff:>8.4f}{marker}")

    print()
    print(f"  Recommended:  set CNN_THRESHOLD = {threshold:.2f}  in live/config.py")
    print("=" * 60)


def plot_curve(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping plot)")
        return

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, lw=2, label="PR curve")

    # Mark the best-F1 point
    bin_pred = (y_prob >= threshold).astype(int)
    p = precision_score(y_true, bin_pred, zero_division=0)
    r = recall_score(y_true, bin_pred, zero_division=0)
    plt.scatter([r], [p], s=120, zorder=5,
                label=f"Best F1 @ t={threshold:.2f}")

    # Mark current default
    default = 0.65
    bin_default = (y_prob >= default).astype(int)
    pd_ = precision_score(y_true, bin_default, zero_division=0)
    rd_ = recall_score(y_true, bin_default, zero_division=0)
    plt.scatter([rd_], [pd_], s=80, marker="x", zorder=5, color="red",
                label=f"Current default @ t={default:.2f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PumpDetectorV3 — Precision-Recall (zero-shot test set)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = "models/pr_curve.png"
    plt.savefig(out, dpi=150)
    print(f"  Plot saved to {out}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Threshold calibration for PumpDetectorV3")
    parser.add_argument("--exchanges", nargs="+", default=DEFAULT_EXCHANGES,
                        help="Exchanges to use as calibration set")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the matplotlib PR-curve plot")
    args = parser.parse_args()

    y_true, y_prob = run_inference(args.exchanges)
    best_t = find_best_threshold(y_true, y_prob)
    print_report(y_true, y_prob, best_t)

    if not args.no_plot:
        plot_curve(y_true, y_prob, best_t)


if __name__ == "__main__":
    main()
