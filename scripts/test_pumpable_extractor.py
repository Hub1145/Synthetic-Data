"""
PumpableCoin Extractor — Full Data Quality Test
Tests real, reconstructed, and synthetic L2 + tradebook data.
Reads multi-level orderbook (up to 10 levels) and paired *_trades.csv files.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PUMPABLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "..", "PUMPABLE COINS")
sys.path.insert(0, os.path.abspath(PUMPABLE_DIR))
from pumpable_coin_extractor import PumpableCoinExtractor

RECONSTRUCTED_BASE = "reconstructed"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Disable liquidity gate — synthetic data not in USD scale
EXTRACTOR_CONFIG = {
    "min_liquidity":      0,
    "max_liquidity":      float("inf"),
    "depth_levels":       10,
    "buy_threshold":      5.0,
    "sell_threshold":     0.2,
    "impact_notional_pct": 0.05,
}

N_LEVELS = 10


def classify_source(filepath: str) -> str:
    fname = os.path.basename(filepath)
    if "_direct_L2" in fname:
        return "synthetic_direct"
    if "_synthetic" in fname:
        return "synthetic_typeA"
    return "real"


def build_orderbook(row: pd.Series) -> dict:
    """
    Build multi-level orderbook dict from a DataFrame row.
    Uses bid_price/bid_size (level 1) + bid_price_2..N / bid_size_2..N if present.
    """
    bids, asks = [], []

    for lvl in range(1, N_LEVELS + 1):
        bp_col = "bid_price" if lvl == 1 else f"bid_price_{lvl}"
        bs_col = "bid_size"  if lvl == 1 else f"bid_size_{lvl}"
        ap_col = "ask_price" if lvl == 1 else f"ask_price_{lvl}"
        as_col = "ask_size"  if lvl == 1 else f"ask_size_{lvl}"

        if bp_col not in row.index:
            break

        bp = float(row[bp_col])
        bs = float(row[bs_col])
        ap = float(row[ap_col])
        as_ = float(row[as_col])

        if bp > 0 and bs > 0:
            bids.append([bp, bs])
        if ap > 0 and as_ > 0:
            asks.append([ap, as_])

    return {"bids": bids, "asks": asks}


def load_trades(trades_path: str, n_timesteps: int) -> list[list]:
    """
    Load *_trades.csv → list of per-timestep trade lists for the extractor.
    Each element is a list of {'price': x, 'amount': y, 'side': 'buy'/'sell'}.
    """
    per_ts = [[] for _ in range(n_timesteps)]
    if not trades_path or not os.path.exists(trades_path):
        return per_ts
    try:
        df = pd.read_csv(trades_path)
        required = {"timestamp_idx", "side", "price", "size"}
        if not required.issubset(df.columns):
            return per_ts
        for _, r in df.iterrows():
            idx = int(r["timestamp_idx"])
            if 0 <= idx < n_timesteps:
                per_ts[idx].append({
                    "price":  float(r["price"]),
                    "amount": float(r["size"]),
                    "side":   str(r["side"]),
                })
    except Exception:
        pass
    return per_ts


def score_file(filepath: str, extractor: PumpableCoinExtractor, symbol: str) -> dict | None:
    try:
        df = pd.read_csv(filepath)
        required = {"bid_price", "ask_price", "bid_size", "ask_size"}
        if not required.issubset(df.columns):
            return None
        df = df.dropna(subset=list(required)).reset_index(drop=True)
        if len(df) == 0:
            return None

        n_ts = len(df)
        trades_path = filepath.replace("_L2.csv", "_trades.csv")
        per_ts_trades = load_trades(trades_path, n_ts)
        has_trades = os.path.exists(trades_path)

        scores, imbalances, spreads, impacts = [], [], [], []

        for t, row in df.iterrows():
            ob = build_orderbook(row)
            if not ob["bids"] or not ob["asks"]:
                continue
            trades_t = per_ts_trades[t] if t < len(per_ts_trades) else []
            feats = extractor.extract(symbol, ob, trades=trades_t or None)
            scores.append(feats.get("pump_score", 0.0))
            imbalances.append(feats.get("pump_buy_sell_ratio", 1.0))
            spreads.append(feats.get("pump_spread_pct", 0.0))
            impacts.append(feats.get("pump_max_impact_pct", 0.0))

        if not scores:
            return None

        scores_arr = np.array(scores)
        peak_idx = int(np.argmax(scores_arr))

        return {
            "score_mean":      float(np.mean(scores_arr)),
            "score_max":       float(np.max(scores_arr)),
            "score_std":       float(np.std(scores_arr)),
            "score_peak_t":    peak_idx,
            "n_timesteps":     len(scores),
            "n_ob_levels":     len(ob["bids"]),
            "has_trades":      has_trades,
            "imbalance_mean":  float(np.mean(imbalances)),
            "spread_mean_pct": float(np.mean(spreads)),
            "impact_mean_pct": float(np.mean(impacts)),
            "_scores":         scores_arr,
        }
    except Exception as e:
        print(f"  [WARN] {filepath}: {e}")
        return None


def run_test():
    print("=" * 70)
    print("  PumpableCoin Extractor — Full Orderbook + Tradebook Test")
    print(f"  Scanning: {RECONSTRUCTED_BASE}/   (real + synthetic_direct + synthetic_typeA)")
    print("=" * 70)

    extractor = PumpableCoinExtractor(config=EXTRACTOR_CONFIG)

    all_files = glob.glob(
        os.path.join(RECONSTRUCTED_BASE, "**", "*_L2.csv"), recursive=True
    )
    print(f"\n  Found {len(all_files)} L2 files\n")

    rows = []
    trajectories: dict = {}

    for fpath in all_files:
        norm = fpath.replace("\\", "/")
        parts = norm.split("/")
        try:
            rec_idx  = parts.index("reconstructed")
            exchange = parts[rec_idx + 1]
            regime   = parts[rec_idx + 2]
            symbol   = parts[rec_idx + 3]
        except (ValueError, IndexError):
            continue

        label  = 1 if regime == "pumps" else 0
        source = classify_source(fpath)

        result = score_file(fpath, extractor, symbol)
        if result is None:
            continue

        traj = result.pop("_scores")
        key  = (source, regime)
        if key not in trajectories:
            trajectories[key] = []
        trajectories[key].append(traj)

        rows.append({
            "exchange": exchange,
            "regime":   regime,
            "label":    label,
            "source":   source,
            "symbol":   symbol,
            "file":     os.path.basename(fpath),
            **result,
        })

    if not rows:
        print("ERROR: no files scored. Check L2 column names.")
        return

    df = pd.DataFrame(rows)
    report_path = os.path.join(OUTPUT_DIR, "pumpable_extractor_report.csv")
    df.to_csv(report_path, index=False)
    print(f"  Full report saved → {report_path}\n")

    # ── Orderbook / tradebook coverage ────────────────────────────────────────
    avg_levels  = df["n_ob_levels"].mean()
    pct_trades  = df["has_trades"].mean() * 100
    print(f"  Avg orderbook depth levels : {avg_levels:.1f}")
    print(f"  Files with tradebook       : {pct_trades:.1f}%\n")

    # ── Summary by source × regime ────────────────────────────────────────────
    print("─" * 70)
    print(f"  {'SOURCE':<20} {'REGIME':<10} {'FILES':>6}  "
          f"{'MEAN SCORE':>11}  {'MAX SCORE':>10}  {'IMPACT%':>8}  {'SPREAD%':>8}")
    print("─" * 70)
    for (source, regime), grp in df.groupby(["source", "regime"]):
        print(f"  {source:<20} {regime:<10} {len(grp):>6}  "
              f"{grp['score_mean'].mean():>11.2f}  "
              f"{grp['score_max'].mean():>10.2f}  "
              f"{grp['impact_mean_pct'].mean():>8.4f}  "
              f"{grp['spread_mean_pct'].mean():>8.4f}")
    print("─" * 70)

    # ── Discrimination ─────────────────────────────────────────────────────────
    print("\n  DISCRIMINATION  (pump score separation: pump vs control)")
    print("─" * 70)
    print(f"  {'SOURCE':<22}  {'PUMP mean':>10}  {'CTRL mean':>10}  {'DELTA':>8}  {'RATIO':>7}")
    print("─" * 70)
    for source, grp in df.groupby("source"):
        pump_s = grp[grp["regime"] == "pumps"]["score_mean"]
        ctrl_s = grp[grp["regime"] == "control"]["score_mean"]
        if pump_s.empty or ctrl_s.empty:
            continue
        pm, cm = pump_s.mean(), ctrl_s.mean()
        delta  = pm - cm
        ratio  = pm / cm if cm > 0 else float("inf")
        print(f"  {source:<22}  {pm:>10.2f}  {cm:>10.2f}  {delta:>8.2f}  {ratio:>7.2f}x")
    print("─" * 70)

    # ── Trajectory ─────────────────────────────────────────────────────────────
    print("\n  TRAJECTORY  (avg pump score at key window positions)")
    print("─" * 70)
    print(f"  {'SOURCE / REGIME':<30}  {'t=0':>6}  {'t=25%':>7}  "
          f"{'t=peak':>7}  {'t=75%':>7}  {'t=end':>7}")
    print("─" * 70)

    traj_rows = []
    for (source, regime), arrays in sorted(trajectories.items()):
        padded = [a[:96] for a in arrays if len(a) >= 4]
        if not padded:
            continue
        max_len = max(len(a) for a in padded)
        stacked = np.zeros((len(padded), max_len))
        for i, a in enumerate(padded):
            stacked[i, :len(a)] = a
        avg = stacked.mean(axis=0)
        L = len(avg)
        peak_t = int(np.argmax(avg))
        print(f"  {source}/{regime:<22}  "
              f"{avg[0]:>6.2f}  {avg[int(L*0.25)]:>7.2f}  "
              f"{avg[peak_t]:>7.2f}  {avg[int(L*0.75)]:>7.2f}  {avg[-1]:>7.2f}")
        for t, v in enumerate(avg):
            traj_rows.append({"source": source, "regime": regime,
                               "timestep": t, "avg_pump_score": round(float(v), 4)})

    traj_df = pd.DataFrame(traj_rows)
    traj_path = os.path.join(OUTPUT_DIR, "pumpable_trajectories.csv")
    traj_df.to_csv(traj_path, index=False)
    print("─" * 70)
    print(f"\n  Trajectory data → {traj_path}")

    # ── Per-exchange ───────────────────────────────────────────────────────────
    print("\n  PER-EXCHANGE  (mean pump score)")
    print("─" * 70)
    print(f"  {'EXCHANGE':<12}  {'PUMP mean':>10}  {'CTRL mean':>10}  {'FILES (P/C)':>12}")
    print("─" * 70)
    for exchange, grp in df.groupby("exchange"):
        pg = grp[grp["regime"] == "pumps"]
        cg = grp[grp["regime"] == "control"]
        pm = pg["score_mean"].mean() if not pg.empty else float("nan")
        cm = cg["score_mean"].mean() if not cg.empty else float("nan")
        print(f"  {exchange:<12}  {pm:>10.2f}  {cm:>10.2f}  "
              f"{len(pg):>5} / {len(cg):<5}")
    print("─" * 70)

    print(f"\n  Total files scored : {len(df)}")
    print(f"  Reports saved in   : {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    run_test()
