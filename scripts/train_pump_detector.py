"""
CNN PumpDetector v3 training script — dual-stream (coin + market context).

Architecture: PumpDetectorV3
  - Separate 3-block 1D-CNN towers for coin L2 and market context (BTC)
  - Towers → shared trunk (256→128) → two heads:
      cls_head  sigmoid pump probability          (BCE loss)
      reg_head  sigmoid peak_pos = peak_idx/96   (MSE loss, pump samples only)
  - Captures the full 3×3 coin×market regime matrix

Input per sample:
  x_coin   [96, 6]  — bid_price, ask_price, bid_size, ask_size, buy_ratio, aggressor_imbalance
  x_market [96, 6]  — same 6 features for BTC/market context
                       (neutral fill if no _market_ctx.csv found)

18-cell sample weights (coin_regime × market_regime × background_volatility):
  Coin:pump   + Market:normal    + calm     → 4.0   most suspicious
  Coin:pump   + Market:normal    + normal   → 3.0
  Coin:pump   + Market:normal    + volatile → 2.0
  Coin:pump   + Market:uncertain + calm     → 2.5
  Coin:pump   + Market:uncertain + normal   → 2.0
  Coin:pump   + Market:uncertain + volatile → 1.5
  Coin:pump   + Market:pumped    + calm     → 1.5
  Coin:pump   + Market:pumped    + normal   → 1.0
  Coin:pump   + Market:pumped    + volatile → 1.0
  Coin:control + Market:normal    + calm    → 3.5   hardest negative
  Coin:control + Market:normal    + normal  → 2.5
  ... (see CELL_WEIGHTS dict for all 18)

Usage (from project root):
    python scripts/train_pump_detector.py
"""

import os
import sys
import re
import json
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, classification_report

from models.pump_detector import PumpDetectorV3

# ── Configuration ─────────────────────────────────────────────────────────────

RECONSTRUCTED_BASE  = "reconstructed"
SYNTHETIC_BASE      = "synthetic"
MODEL_SAVE_PATH     = "models/pump_detector_v3.pth"
TIMESTEPS           = 96

L2_FEATURES    = ['bid_price', 'ask_price', 'bid_size', 'ask_size']
TRADE_FEATURES = ['buy_ratio', 'aggressor_imbalance']
ALL_FEATURES   = L2_FEATURES + TRADE_FEATURES
NUM_FEATURES   = len(ALL_FEATURES)   # 6

TRAIN_EXCHANGES = ['binance', 'kucoin', 'mexc', 'gateio', 'bitget']
TEST_EXCHANGES  = ['bybit', 'okx']

EPOCHS           = 60
BATCH_SIZE       = 64
LR               = 1e-3
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXCHANGE_PUMP_CAP = 5_000   # max pump windows per exchange (prevents Binance dominance)
PEAK_REG_WEIGHT  = 0.3      # λ — weight of regression loss relative to BCE

NEUTRAL_BUY_RATIO = 0.5
NEUTRAL_AGGRESSOR = 0.0

# 18-cell weight matrix: coin_regime × market_regime × background_volatility
# calm background raises weight (no market noise to hide in — cleaner signal)
# volatile background lowers weight (move may be noise amplification)
CELL_WEIGHTS = {
    # coin      market       volatility     weight
    ("pump",    "normal",    "calm"):       4.0,
    ("pump",    "normal",    "normal"):     3.0,
    ("pump",    "normal",    "volatile"):   2.0,
    ("pump",    "uncertain", "calm"):       2.5,
    ("pump",    "uncertain", "normal"):     2.0,
    ("pump",    "uncertain", "volatile"):   1.5,
    ("pump",    "pumped",    "calm"):       1.5,
    ("pump",    "pumped",    "normal"):     1.0,
    ("pump",    "pumped",    "volatile"):   1.0,
    ("control", "normal",    "calm"):       3.5,
    ("control", "normal",    "normal"):     2.5,
    ("control", "normal",    "volatile"):   1.5,
    ("control", "uncertain", "calm"):       2.0,
    ("control", "uncertain", "normal"):     1.5,
    ("control", "uncertain", "volatile"):   1.0,
    ("control", "pumped",    "calm"):       1.5,
    ("control", "pumped",    "normal"):     1.0,
    ("control", "pumped",    "volatile"):   1.0,
}


# ── Trade-flow loader ─────────────────────────────────────────────────────────

def load_trade_features(trades_path: str, n_timesteps: int) -> np.ndarray:
    out = np.full((n_timesteps, 2),
                  [NEUTRAL_BUY_RATIO, NEUTRAL_AGGRESSOR], dtype=np.float32)
    try:
        df = pd.read_csv(trades_path)
        if not {"timestamp_idx", "side", "size"}.issubset(df.columns):
            return out
        df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0)
        for idx, grp in df.groupby("timestamp_idx"):
            if not (0 <= idx < n_timesteps):
                continue
            total   = grp["size"].sum()
            if total <= 0:
                continue
            buy_vol = grp.loc[grp["side"] == "buy", "size"].sum()
            sell_vol = total - buy_vol
            out[int(idx), 0] = buy_vol / total
            out[int(idx), 1] = (buy_vol - sell_vol) / total
    except Exception:
        pass
    return out


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_window(window: np.ndarray) -> np.ndarray:
    """Prices ÷ first mid-price, sizes ÷ mean size, trade features unchanged."""
    w = window.copy().astype(np.float32)
    mid0 = (w[0, 0] + w[0, 1]) / 2.0
    if mid0 > 0:
        w[:, 0:2] /= mid0
    mean_sz = w[:, 2:4].mean()
    if mean_sz > 0:
        w[:, 2:4] /= mean_sz
    return w


# ── Market context loader ─────────────────────────────────────────────────────

NEUTRAL_MARKET = None   # built lazily on first use

def neutral_market() -> np.ndarray:
    global NEUTRAL_MARKET
    if NEUTRAL_MARKET is None:
        NEUTRAL_MARKET = np.column_stack([
            np.ones(TIMESTEPS),             # bid_price (flat)
            np.ones(TIMESTEPS),             # ask_price (flat)
            np.ones(TIMESTEPS),             # bid_size
            np.ones(TIMESTEPS),             # ask_size
            np.full(TIMESTEPS, 0.5),        # buy_ratio (neutral)
            np.zeros(TIMESTEPS),            # aggressor_imbalance (neutral)
        ]).astype(np.float32)
    return NEUTRAL_MARKET.copy()


def load_market_ctx(l2_path: str) -> np.ndarray:
    ctx_path = re.sub(r'(_direct_L2|_synthetic_L2|_L2)\.csv$',
                      '_market_ctx.csv', l2_path)
    if not os.path.exists(ctx_path):
        return neutral_market()
    try:
        df = pd.read_csv(ctx_path)
        needed = ['bid_price', 'ask_price', 'bid_size', 'ask_size']
        if not set(needed).issubset(df.columns):
            return neutral_market()
        data = df[needed].values[:TIMESTEPS].astype(np.float32)
        if len(data) < TIMESTEPS:
            return neutral_market()
        if 'buy_ratio' in df.columns and 'aggressor_imbalance' in df.columns:
            tf = df[['buy_ratio', 'aggressor_imbalance']].values[:TIMESTEPS].astype(np.float32)
        else:
            tf = np.column_stack([np.full(TIMESTEPS, 0.5),
                                  np.zeros(TIMESTEPS)]).astype(np.float32)
        return np.concatenate([data, tf], axis=1)  # [96, 6]
    except Exception:
        return neutral_market()


# ── Meta loader ───────────────────────────────────────────────────────────────

def load_meta(l2_path: str, label: int) -> tuple[float, float]:
    """
    Returns (cell_weight, peak_pos) for a given L2 file.

    peak_pos = peak_idx / TIMESTEPS — the normalised peak location in [0, 1].
    Sourced from meta.json fields (in priority order):
      1. peak_pct   — written by tag_peak_buckets.py (float 0-1)
      2. peak_idx   — written by generate_direct_synthetic.py (int)
      3. peak_bucket — written by tag_peak_buckets.py (int 0-5, mapped to bucket centre)
    Falls back to 0.5 (midpoint) if none present.
    """
    meta_path = re.sub(r'(_direct_L2|_synthetic_L2|_L2)\.csv$',
                       '_meta.json', l2_path)
    coin_regime   = "pump" if label == 1 else "control"
    market_regime = "normal"
    bg_volatility = "normal"
    peak_pos      = 0.5   # neutral default — masked out for control samples anyway

    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            coin_regime   = meta.get("coin_regime",           coin_regime)
            market_regime = meta.get("market_regime",         "normal")
            bg_volatility = meta.get("background_volatility", "normal")
            if market_regime == "unknown":
                market_regime = "normal"
            if bg_volatility == "unknown":
                bg_volatility = "normal"

            if "peak_pct" in meta:
                peak_pos = float(meta["peak_pct"])
            elif "peak_idx" in meta:
                peak_pos = float(meta["peak_idx"]) / TIMESTEPS
            elif "peak_bucket" in meta:
                # Bucket centres: 0→0.08, 1→0.25, 2→0.42, 3→0.58, 4→0.75, 5→0.92
                bucket = int(meta["peak_bucket"])
                peak_pos = (bucket * 2 + 1) / 12.0
        except Exception:
            pass

    cell_weight = CELL_WEIGHTS.get((coin_regime, market_regime, bg_volatility), 1.5)
    return cell_weight, peak_pos


# ── Window extraction ─────────────────────────────────────────────────────────

def extract_windows(l2_df: pd.DataFrame, l2_path: str,
                    label: int, step_ratio: float = 0.5):
    """
    Returns lists of (coin_window, market_window, cell_weight, peak_pos).
    Each window is [TIMESTEPS, 6] float32, already normalised.
    peak_pos is the same for all windows of a given file (event-level label).
    """
    l2_data = l2_df[L2_FEATURES].values.astype(np.float32)
    n_rows  = len(l2_data)

    tp = l2_path.replace("_L2.csv", "_trades.csv")
    tp = tp.replace("_direct_L2.csv", "_trades.csv")
    tp = tp.replace("_synthetic_L2.csv", "_trades.csv")
    trade_data = (load_trade_features(tp, n_rows)
                  if os.path.exists(tp)
                  else np.full((n_rows, 2),
                               [NEUTRAL_BUY_RATIO, NEUTRAL_AGGRESSOR],
                               dtype=np.float32))

    coin_full   = np.concatenate([l2_data, trade_data], axis=1)  # [N, 6]
    market_full = load_market_ctx(l2_path)                        # [96, 6]
    cell_weight, peak_pos = load_meta(l2_path, label)

    step = max(1, int(TIMESTEPS * step_ratio))
    coin_windows, market_windows, weights, peak_positions = [], [], [], []

    for start in range(0, n_rows - TIMESTEPS + 1, step):
        w_coin = coin_full[start:start + TIMESTEPS]
        if len(w_coin) != TIMESTEPS or not np.isfinite(w_coin).all():
            continue
        if len(market_full) >= TIMESTEPS:
            w_mkt = market_full[:TIMESTEPS]
        else:
            pad = np.zeros((TIMESTEPS - len(market_full), market_full.shape[1]), dtype=np.float32)
            pad[:, 0] = 1.0
            pad[:, 1] = 1.0
            pad[:, 4] = NEUTRAL_BUY_RATIO
            w_mkt = np.concatenate([market_full, pad], axis=0)

        coin_windows.append(normalize_window(w_coin))
        market_windows.append(normalize_window(w_mkt))
        weights.append(cell_weight)
        peak_positions.append(peak_pos)

    return coin_windows, market_windows, weights, peak_positions


# ── Dataset class ─────────────────────────────────────────────────────────────

class PumpDataset(Dataset):
    def __init__(self, X_coin, X_market, y, peak_pos, sample_weights):
        self.X_coin    = torch.FloatTensor(X_coin)
        self.X_market  = torch.FloatTensor(X_market)
        self.y         = torch.FloatTensor(y)
        self.peak_pos  = torch.FloatTensor(peak_pos)
        self.weights   = sample_weights   # numpy array

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_coin[idx], self.X_market[idx], self.y[idx], self.peak_pos[idx]


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset(exchanges: list):
    X_coin_all, X_mkt_all, y_all, peak_all, w_all = [], [], [], [], []

    for exchange in exchanges:
        search_bases = [
            os.path.join(RECONSTRUCTED_BASE, exchange),
            os.path.join(SYNTHETIC_BASE,     exchange),
        ]

        all_regime_dirs = set()
        for b in search_bases:
            if os.path.exists(b):
                all_regime_dirs.update(os.listdir(b))

        if not all_regime_dirs:
            print(f"  [SKIP] {exchange} — not found in reconstructed/ or synthetic/")
            continue

        # Collect pump and control windows separately for per-exchange capping
        exc_pump_coin, exc_pump_mkt, exc_pump_peak, exc_pump_w = [], [], [], []
        exc_ctrl_coin, exc_ctrl_mkt, exc_ctrl_peak, exc_ctrl_w = [], [], [], []

        for regime_dir in sorted(all_regime_dirs):
            label = 1 if regime_dir == "pumps" else 0
            csv_files = []
            for b in search_bases:
                rp = os.path.join(b, regime_dir)
                if os.path.isdir(rp):
                    csv_files.extend(glob.glob(
                        os.path.join(rp, "**", "*_L2.csv"), recursive=True
                    ))
            csv_files = [p for p in csv_files if "_market_ctx" not in p]

            n_windows = 0
            for fpath in csv_files:
                try:
                    df = pd.read_csv(fpath)
                    missing = [c for c in L2_FEATURES if c not in df.columns]
                    if missing or len(df) < TIMESTEPS:
                        continue
                    df = df.dropna(subset=L2_FEATURES)
                    coin_w, mkt_w, wts, peaks = extract_windows(df, fpath, label)
                    if label == 1:
                        exc_pump_coin.extend(coin_w)
                        exc_pump_mkt.extend(mkt_w)
                        exc_pump_w.extend(wts)
                        exc_pump_peak.extend(peaks)
                    else:
                        exc_ctrl_coin.extend(coin_w)
                        exc_ctrl_mkt.extend(mkt_w)
                        exc_ctrl_w.extend(wts)
                        exc_ctrl_peak.extend(peaks)
                    n_windows += len(coin_w)
                except Exception as e:
                    print(f"  [WARN] {fpath}: {e}")

            if csv_files:
                print(f"  {exchange}/{regime_dir}: {len(csv_files)} files → "
                      f"{n_windows} windows  (label={label})")

        # Apply per-exchange pump cap
        n_pump = len(exc_pump_coin)
        if n_pump > EXCHANGE_PUMP_CAP:
            idx = np.random.choice(n_pump, EXCHANGE_PUMP_CAP, replace=False)
            exc_pump_coin = [exc_pump_coin[i] for i in idx]
            exc_pump_mkt  = [exc_pump_mkt[i]  for i in idx]
            exc_pump_w    = [exc_pump_w[i]    for i in idx]
            exc_pump_peak = [exc_pump_peak[i] for i in idx]
            print(f"  {exchange}: pump windows capped {n_pump} → {EXCHANGE_PUMP_CAP}")

        X_coin_all.extend(exc_pump_coin);  X_coin_all.extend(exc_ctrl_coin)
        X_mkt_all.extend(exc_pump_mkt);   X_mkt_all.extend(exc_ctrl_mkt)
        y_all.extend([1] * len(exc_pump_coin)); y_all.extend([0] * len(exc_ctrl_coin))
        peak_all.extend(exc_pump_peak);    peak_all.extend(exc_ctrl_peak)
        w_all.extend(exc_pump_w);          w_all.extend(exc_ctrl_w)

    if not X_coin_all:
        empty = np.empty((0, TIMESTEPS, NUM_FEATURES), dtype=np.float32)
        return empty, empty, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0)

    return (np.array(X_coin_all,  dtype=np.float32),
            np.array(X_mkt_all,   dtype=np.float32),
            np.array(y_all,       dtype=np.float32),
            np.array(peak_all,    dtype=np.float32),
            np.array(w_all,       dtype=np.float32))


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_loader(dataset: PumpDataset, batch_size: int,
                shuffle: bool = True) -> DataLoader:
    if shuffle:
        y        = dataset.y.numpy().astype(int)
        counts   = np.bincount(y, minlength=2).clip(min=1)
        cls_w    = (1.0 / counts)[y]
        combined = cls_w * dataset.weights
        sampler  = WeightedRandomSampler(combined, num_samples=len(combined))
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    print("=" * 65)
    print("  CNN PumpDetector v3  (dual-stream: coin + market context)")
    print(f"  Features per stream : {ALL_FEATURES}")
    print(f"  Train exchanges     : {TRAIN_EXCHANGES}")
    print(f"  Test  exchanges     : {TEST_EXCHANGES}  (zero-shot)")
    print(f"  Device              : {DEVICE}")
    print(f"  Pump window cap     : {EXCHANGE_PUMP_CAP} per exchange")
    print(f"  Peak reg. weight λ  : {PEAK_REG_WEIGHT}")
    print("=" * 65)

    print("\nLoading training data...")
    X_coin_tr, X_mkt_tr, y_tr, peak_tr, w_tr = load_dataset(TRAIN_EXCHANGES)
    if len(X_coin_tr) == 0:
        print("\nERROR: No training data found.")
        print("Run scripts/reconstruct_orderbook.py first.")
        return

    pumps_tr = int(y_tr.sum())
    ctrl_tr  = int((y_tr == 0).sum())
    print(f"\n  Train total  : {len(X_coin_tr)} windows  ({pumps_tr} pump, {ctrl_tr} control)")
    print(f"  Market ctx   : {'_market_ctx.csv found for some files' if X_mkt_tr.mean() != 1.0 else 'neutral fill (run fetch_market_context.py)'}")

    print("\nLoading zero-shot test data...")
    X_coin_te, X_mkt_te, y_te, peak_te, w_te = load_dataset(TEST_EXCHANGES)
    if len(X_coin_te):
        print(f"  Test  total  : {len(X_coin_te)} windows  "
              f"({int(y_te.sum())} pump, {int((y_te==0).sum())} control)")

    # 80/20 split
    idx   = np.random.permutation(len(X_coin_tr))
    split = int(0.8 * len(idx))
    tr_idx, val_idx = idx[:split], idx[split:]

    train_ds = PumpDataset(X_coin_tr[tr_idx],  X_mkt_tr[tr_idx],
                           y_tr[tr_idx],        peak_tr[tr_idx],  w_tr[tr_idx])
    val_ds   = PumpDataset(X_coin_tr[val_idx], X_mkt_tr[val_idx],
                           y_tr[val_idx],       peak_tr[val_idx], w_tr[val_idx])

    train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(val_ds,   BATCH_SIZE, shuffle=False)

    model     = PumpDetectorV3(num_coin_features=NUM_FEATURES,
                               num_market_features=NUM_FEATURES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=6, factor=0.5)
    bce_loss_fn = nn.BCELoss()
    mse_loss_fn = nn.MSELoss()

    best_val_auc = 0.0
    auc_history: list[float] = []
    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'BCE':>8}  {'MSE':>8}  {'Val AUC':>9}")
    print("-" * 52)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = total_bce = total_mse = 0.0

        for x_coin, x_mkt, yb, peak_pos in train_loader:
            x_coin   = x_coin.to(DEVICE)
            x_mkt    = x_mkt.to(DEVICE)
            yb       = yb.to(DEVICE)
            peak_pos = peak_pos.to(DEVICE)

            optimizer.zero_grad()
            pump_prob, peak_pred = model(x_coin, x_mkt)

            bce = bce_loss_fn(pump_prob, yb)

            # Regression loss only on pump samples where peak_idx is meaningful
            pump_mask = yb > 0.5
            if pump_mask.any():
                mse = mse_loss_fn(peak_pred[pump_mask], peak_pos[pump_mask])
            else:
                mse = torch.tensor(0.0, device=DEVICE)

            loss = bce + PEAK_REG_WEIGHT * mse
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce  += bce.item()
            total_mse  += mse.item()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x_coin, x_mkt, yb, _ in val_loader:
                pump_prob, _ = model(x_coin.to(DEVICE), x_mkt.to(DEVICE))
                preds.extend(pump_prob.cpu().numpy())
                labels.extend(yb.numpy())

        val_auc = roc_auc_score(labels, preds)
        auc_history.append(val_auc)
        scheduler.step(val_auc)

        n = len(train_loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {total_loss/n:>12.4f}  "
                  f"{total_bce/n:>8.4f}  {total_mse/n:>8.4f}  {val_auc:>9.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    mean_val_auc = float(np.mean(auc_history))
    std_val_auc  = float(np.std(auc_history))
    print("-" * 52)
    print(f"  Best validation AUC : {best_val_auc:.4f}")
    print(f"  Mean validation AUC : {mean_val_auc:.4f}  (±{std_val_auc:.4f} across {EPOCHS} epochs)")
    print(f"  Model saved to      : {MODEL_SAVE_PATH}")

    # ── Zero-shot transfer test ───────────────────────────────────────────────
    if len(X_coin_te) == 0:
        print("\n  No test exchange data — skipping zero-shot evaluation.")
        return

    print("\n" + "=" * 65)
    print("  Zero-Shot Transfer Test  (Bybit + OKX — never seen during training)")
    print("=" * 65)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE,
                                     weights_only=True))
    model.eval()

    test_ds     = PumpDataset(X_coin_te, X_mkt_te, y_te, peak_te, w_te)
    test_loader = make_loader(test_ds, BATCH_SIZE, shuffle=False)
    preds, labels, peak_preds, peak_true = [], [], [], []

    with torch.no_grad():
        for x_coin, x_mkt, yb, peak_pos in test_loader:
            pump_prob, peak_pred = model(x_coin.to(DEVICE), x_mkt.to(DEVICE))
            preds.extend(pump_prob.cpu().numpy())
            labels.extend(yb.numpy())
            peak_preds.extend(peak_pred.cpu().numpy())
            peak_true.extend(peak_pos.numpy())

    preds      = np.array(preds)
    labels     = np.array(labels)
    peak_preds = np.array(peak_preds)
    peak_true  = np.array(peak_true)

    test_auc = roc_auc_score(labels, preds)
    bin_pred = (preds >= 0.5).astype(int)

    # Peak regression MAE — on pump samples only
    pump_mask = labels > 0.5
    if pump_mask.any():
        peak_mae = np.abs(peak_preds[pump_mask] - peak_true[pump_mask]).mean()
        peak_mae_candles = peak_mae * TIMESTEPS
    else:
        peak_mae_candles = float("nan")

    print(f"\n  Zero-shot AUC       : {test_auc:.4f}")
    print(f"  Peak reg MAE        : {peak_mae_candles:.1f} candles (pump samples)")
    print()
    print(classification_report(labels, bin_pred,
                                target_names=["control", "pump"],
                                zero_division=0))
    print(f"  Hint: run scripts/calibrate_threshold.py to find the optimal")
    print(f"  operating threshold using precision-recall curves on this test set.")

    if test_auc >= 0.80:
        print("  RESULT: Strong cross-exchange generalisation.")
    elif test_auc >= 0.65:
        print("  RESULT: Moderate generalisation — more real data will help.")
    else:
        print("  RESULT: Weak transfer — consider domain adaptation.")


if __name__ == "__main__":
    train()
