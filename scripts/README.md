# Scripts Reference

All scripts are run from the **project root** (one level above `scripts/`), not from inside `scripts/`.

```
python scripts/<script_name>.py
```

Use `main.py` to run the full pipeline automatically. Individual scripts can also be run directly for targeted re-runs.

---

## Orchestration

### `main.py`

Full 16-step pipeline orchestrator. Calls every script below as a subprocess in order and writes a structured run report to `output/`.

```
python scripts/main.py                       # full pipeline (Step 0 scan skipped by default)
python scripts/main.py --with-scan           # also run exchange scanner (Step 0)
python scripts/main.py --skip-fetch          # skip data download (Steps 1-3)
python scripts/main.py --skip-label          # skip pump label verification (Step 4)
python scripts/main.py --skip-reconstruct    # skip L2 reconstruction + depth (Steps 5-6)
python scripts/main.py --skip-meta           # skip peak tagging + market context (Steps 7-10)
python scripts/main.py --skip-vae-train      # skip DirectL2VAE training (Step 12)
python scripts/main.py --skip-synthetic      # skip VAE training + generation (Steps 12-13)
python scripts/main.py --skip-trades         # skip missing-trades fill (Step 14)
python scripts/main.py --skip-calibrate      # skip threshold calibration (Step 16)
python scripts/main.py --train-only          # Steps 15-16 only (skip all data prep)
python scripts/main.py --resume              # auto-detect completed steps and skip them
python scripts/main.py --dry-run             # print plan without executing
python scripts/main.py --continue-on-error   # keep going even if a step fails
```

**Pipeline steps:**

| Step | Script | Skip flag |
|---|---|---|
| 0 (opt-in) | `scan_multi_exchange.py` | `--with-scan` to enable |
| 1 | `fetch_all_pump_data.py` | `--skip-fetch` |
| 2 | `fetch_ardia_pumps.py` | `--skip-fetch` |
| 3 | `fetch_control_data.py` | `--skip-fetch` |
| 4 | `fetch_and_label_tradebook_data.py` | `--skip-label` |
| 5 | `reconstruct_orderbook.py` | `--skip-reconstruct` |
| 6 | `add_orderbook_depth.py` | `--skip-reconstruct` |
| 7 | `tag_peak_buckets.py` | `--skip-meta` |
| 8 | `fetch_market_context.py` | `--skip-meta` |
| 9 | `fix_unknown_market_regimes.py` | `--skip-meta` |
| 10 | `migrate_control_to_regimes.py` | `--skip-meta` |
| 11 | `prepare_l2_training_data.py` | `--skip-vae-train` |
| 12 | `train_direct_l2.py` | `--skip-vae-train` |
| 13 | `generate_direct_synthetic.py` | `--skip-synthetic` |
| 14 | `generate_missing_trades.py` | `--skip-trades` |
| 15 | `train_pump_detector.py` | — |
| 16 | `calibrate_threshold.py` | `--skip-calibrate` |

The run report (`output/run_report_*.txt`) and structured JSON (`output/run_report_*.json`) capture dataset statistics, model performance (best AUC, mean AUC ± std), peak MAE, and per-step timing.

---

## Quick Map

```
Data Collection
  scan_multi_exchange.py            ← Step 0  — Scan 7 exchanges for new pump candidates (opt-in)
  fetch_all_pump_data.py            ← Step 1  — Download pump OHLCV (7 exchanges, direct REST)
  fetch_ardia_pumps.py              ← Step 2  — Binance P&D events (ArdiaD dataset, 322 confirmed)
  fetch_control_data.py             ← Step 3  — Download organic-volatility events (hard negatives)

Labelling & Enrichment
  fetch_and_label_tradebook_data.py ← Step 4  — Verify 30% retracement + fetch trade ticks + peak_bucket
  reconstruct_orderbook.py          ← Step 5  — OHLCV → Level-2 orderbook (bid/ask price + size)
  add_orderbook_depth.py            ← Step 6  — Expand all L2 files from 1-level to 10-level depth

Metadata & Market Context
  tag_peak_buckets.py               ← Step 7  — Compute peak_pct / peak_bucket for all L2 files
  fetch_market_context.py           ← Step 8  — Fetch BTC OHLCV, build _market_ctx.csv, fill market_regime
  fix_unknown_market_regimes.py     ← Step 9  — Assign regime to files with unknown market_regime
  migrate_control_to_regimes.py     ← Step 10 — One-time: split control/ → normal/ + uncertain/

Synthetic Generation
  prepare_l2_training_data.py       ← Step 11 — Vectorise reconstructed/ L2 into .npy for DirectL2VAE
  train_direct_l2.py                ← Step 12 — Train DirectL2VAE (96-step L2-space VAE, latent dim 64)
  generate_direct_synthetic.py      ← Step 13 — Type-B: DirectL2VAE → 3 regimes × 6 buckets + trades + ctx
  generate_missing_trades.py        ← Step 14 — Fill trades files for any L2 that has none

Model Training & Calibration
  train_pump_detector.py            ← Step 15 — Train PumpDetectorV3 (dual-stream CNN, 18-cell weights)
  calibrate_threshold.py            ← Step 16 — PR curve + optimal CNN threshold finder
```

---

## Data Collection

### `scan_multi_exchange.py`  *(Step 0 — opt-in)*

Scans all USDT/BTC pairs across 7 exchanges for pump-and-dump signatures using each exchange's native REST API — no CCXT. Run this when you want to discover new pump events before fetching OHLCV.

For each exchange: symbol discovery → fetch up to 200 daily candles per pair → apply 20-candle sliding window with 30% retracement rule.

Events are classified as:
- `pump` — spike ≥ 5% AND retracement ≥ 30%
- `volatile_control` — large spike but retracement < 30% (hard negative)

| Exchange | Symbol endpoint | Daily OHLCV endpoint |
|---|---|---|
| Binance | `/api/v3/exchangeInfo` | `/api/v3/klines?interval=1d` |
| Bybit | `/v5/market/instruments-info` | `/v5/market/kline?interval=D` |
| KuCoin | `/api/v1/symbols` | `/api/v1/market/candles?type=1day` |
| OKX | `/api/v5/public/instruments` | `/api/v5/market/history-candles?bar=1D` |
| Gate.io | `/api/v4/spot/currency_pairs` | `/api/v4/spot/candlesticks?interval=1d` |
| MEXC | `/api/v3/exchangeInfo` | `/api/v3/klines?interval=1d` |
| Bitget | `/api/v2/spot/public/symbols` | `/api/v2/spot/market/history-candles?granularity=1day` |

- Output: `data/scanned_pumps/[exchange]_pumps.csv` and `data/global_deep_scan_pumps.csv` (combined)
- Accepts `--exchange` flag to scan a subset (e.g. `--exchange bybit okx`)
- Output feeds into Steps 1 and 3

---

### `fetch_all_pump_data.py`  *(Step 1)*

Comprehensive multi-exchange pump OHLCV fetcher. Reads pump candidates from `data/global_deep_scan_pumps.csv` (produced by `scan_multi_exchange.py`) and downloads the corresponding 1-minute OHLCV using each exchange's native REST API directly — no CCXT.

| Exchange | Source |
|---|---|
| Binance | `data.binance.vision` public archive (full history, no auth) |
| Bybit | `api.bybit.com/v5/market/kline` with explicit start/end timestamps |
| KuCoin | `api.kucoin.com/api/v1/market/candles` with `startAt`/`endAt` (Unix seconds) |
| OKX | `okx.com/api/v5/market/history-candles` paginated backwards |
| Gate.io | `api.gateio.ws/api/v4/spot/candlesticks` with timeframe cascade (1m→5m→1h→4h) |
| MEXC | `api.mexc.com/api/v3/klines` with timeframe cascade (1m→5m→30m→60m→4h→1d) |
| Bitget | `api.bitget.com/api/v2/spot/market/history-candles` with explicit timestamps |

- Output: `real/[exchange]/pumps/[SYMBOL]/[SYMBOL]_[DATE]_klines.csv`
- Skips already-downloaded events
- `--exchange` flag to run a single exchange at a time

---

### `fetch_ardia_pumps.py`  *(Step 2)*

Downloads 1-minute OHLCV klines from **Binance Data Vision** for every event in the [ArdiaD/PumpDump](https://github.com/ArdiaD/PumpDump) dataset (`data/list_pd_events.csv`, 1,160 Binance events — 322 confirmed successful).

- Output: `real/binance/pumps/[SYMBOL]/[SYMBOL]-1m-[DATE].csv`
- Skips symbols already downloaded
- Rate-limited (0.3 s between requests, 2 retries)

---

### `fetch_control_data.py`  *(Step 3)*

Fetches OHLCV for events that produced a large price spike but **failed** the 30% retracement threshold — organic volatility events that form the hard-negative control set.

Uses the same direct REST API fetchers as `fetch_all_pump_data.py` — no CCXT. All 7 exchanges supported.

- Reads: `data/scanned_pumps/[exchange]_pumps.csv` (rows where `label == "volatile_control"`)
- Output: `real/[exchange]/control/[SYMBOL]/[SYMBOL]_[DATE]_klines.csv`
- Accepts `--exchange` flag

---

## Labelling & Enrichment

### `fetch_and_label_tradebook_data.py`  *(Step 4)*

Verifies pump labels and enriches confirmed events with real trade ticks — two operations in a single pass over `real/`.

**Label step** — applies pump confirmation criteria to every event in `real/pumps/`:
1. Price spike ≥ **5%** from the initial open
2. Retracement ≥ **30%** from peak back toward the pre-pump price

Events that pass stay in `pumps/`. Events that fail are moved to `normal/` or `uncertain/` depending on BTC market regime. A report is written to `data/labeling_report.csv`.

**Enrich step** — for each confirmed pump, fetches real trade ticks where available:
- **Binance** → `data.binance.vision/data/spot/daily/trades/` (full history)
- **Bybit** → `public.bybit.com/trading/` (rolling ~90 days)
- Other exchanges: no free source — ticks remain OHLCV-derived

```
python scripts/fetch_and_label_tradebook_data.py
python scripts/fetch_and_label_tradebook_data.py --skip-tradebook
python scripts/fetch_and_label_tradebook_data.py --exchanges binance
python scripts/fetch_and_label_tradebook_data.py --within-days 90
```

---

### `reconstruct_orderbook.py`  *(Step 5)*

Converts 1-minute OHLCV klines into synthetic Level-2 orderbook snapshots using the [ohlcv-to-orderbook](https://pypi.org/project/ohlcv-to-orderbook/) library.

- Input: `real/[exchange]/[regime]/[SYMBOL]/[SYMBOL]_*_klines.csv`
- Output: `reconstructed/[exchange]/[regime]/[SYMBOL]/[SYMBOL]_*_L2.csv` — columns: `bid_price, ask_price, bid_size, ask_size`
- Also generates a paired `*_trades.csv` using the OHLCV buy-ratio formula: `buy_ratio = (close − low) / (high − low + ε)`

---

### `add_orderbook_depth.py`  *(Step 6)*

Expands every L2 file from single-level (bid/ask level 1 only) to **10-level orderbook depth** by simulating realistic price ladders and size distributions.

- Adds columns: `bid_price_2..10`, `bid_size_2..10`, `ask_price_2..10`, `ask_size_2..10`
- Price step per level grows with depth; bid/ask size decay is asymmetric under buy pressure
- Safe to re-run — skips files that already have depth columns

---

## Metadata & Market Context

### `tag_peak_buckets.py`  *(Step 7)*

Scans every `*_L2.csv` in `reconstructed/` and `synthetic/` and writes a `*_meta.json` alongside each file (skips files that already have one). Computes the peak price increase and retracement, then assigns `peak_pct`, `peak_idx`, and a `peak_bucket`:

| Bucket | Peak increase range |
|---|---|
| micro | 5 – 10% |
| small | 10 – 20% |
| medium | 20 – 30% |
| large | 30 – 40% |
| major | 40 – 50% |
| extreme | 50%+ |

```
python scripts/tag_peak_buckets.py             # tag all untagged files
python scripts/tag_peak_buckets.py --overwrite # re-tag everything
```

---

### `fetch_market_context.py`  *(Step 8)*

For every event in `real/` and `reconstructed/`, downloads the BTCUSDT 1-minute kline for that date from Binance Data Vision (cached in `data/market_context/`), then:

1. Classifies the market's behaviour as `normal` / `uncertain` / `pumped`
2. Builds a 6-feature market context sequence derived from BTC OHLCV
3. Saves it as `*_market_ctx.csv` alongside the coin's L2 file
4. Updates `market_regime` and `background_volatility` in `*_meta.json`

Market regime thresholds: < 3% move = normal · 3–8% without retracement = uncertain · ≥ 5% spike with ≥ 20% retracement = pumped.

```
python scripts/fetch_market_context.py
python scripts/fetch_market_context.py --exchanges binance kucoin
python scripts/fetch_market_context.py --overwrite
```

---

### `fix_unknown_market_regimes.py`  *(Step 9)*

Fixes all `*_meta.json` files where `market_regime` was left as `"unknown"`. Assigns one of 9 combinations (regime × background volatility) in round-robin, and generates a synthetic `*_market_ctx.csv` for each fixed file.

- Safe to re-run — only touches files where `market_regime == "unknown"`

---

### `migrate_control_to_regimes.py`  *(Step 10)*

One-time migration that splits the legacy `control/` directory into `normal/` and `uncertain/` based on the market regime stored in each symbol's `*_meta.json`.

- `normal` market regime → `normal/`
- `uncertain` or `pumped` market regime → `uncertain/`
- Supports `--dry-run` and `--dirs` flags

---

## Synthetic Generation

### `prepare_l2_training_data.py`  *(Step 11)*

Vectorises all `reconstructed/` L2 files into a single NumPy array for training the `DirectL2VAE`.

- Reads 4 features: `bid_price, ask_price, bid_size, ask_size` — first 96 timesteps per file
- Applies StandardScaler normalisation
- Output: `data/l2_training_samples.npy` and `models/l2_scaler.pkl`

---

### `train_direct_l2.py`  *(Step 12)*

Trains the **DirectL2VAE** — the generative model that operates directly in L2 orderbook space.

- Architecture: 1D-Convolutional VAE — 3× Conv1d encoder (4→128 channels), 64-dim latent space, 3× ConvTranspose1d decoder → `[B, 96, 4]`
- Trains on `data/l2_training_samples.npy` (prepared by Step 11)
- 100 epochs, batch size 32, Adam optimizer
- Saves weights to `models/direct_l2_vae_v1.pth`

---

### `generate_direct_synthetic.py`  *(Step 13)*

**Type-B synthetic generator.** Uses the trained `DirectL2VAE` to generate 10-level orderbook sequences **directly** — no OHLCV intermediate step.

Each sample is tagged with a **coin regime** (pump/control), a **market regime** (normal/uncertain/pumped), and a **peak bucket** (micro→extreme), covering the full 3×3 coin×market matrix.

Per sample, 4 files are saved:
- `*_direct_L2.csv` — coin orderbook (96 timesteps, 10-level depth)
- `*_trades.csv` — coin tradebook with pump-phase buy/sell logic (80% buys before peak, 25% after)
- `*_market_ctx.csv` — synthetic BTC context for the assigned market regime
- `*_meta.json` — `coin_regime`, `market_regime`, `peak_bucket`, `peak_pct`, `peak_idx`

**Asymmetric bid/ask pump injection:**
- Pre-peak: `bid_size ×(1 + 4×intensity×gauss)`, `ask_size ×(1 − 0.3×intensity×gauss)` — buyers flood in, asks thin out
- Post-peak: `ask_size ×(1 + 5×intensity×gauss)`, `bid_size ×(1 − 0.2×intensity×gauss)` — sellers dump, bids retreat

- 105 samples per exchange (54 pump across 18 cells, 51 control) × 8 exchanges = **840 files**
- Requires `models/direct_l2_vae_v1.pth` and `models/l2_scaler.pkl`

---

### `generate_missing_trades.py`  *(Step 14)*

Scans all L2 files across `reconstructed/` and `synthetic/` and creates a paired `*_trades.csv` for any file that is missing one.

- Uses the OHLCV buy-ratio formula: `buy_ratio = (close − low) / (high − low + ε)`
- Ensures 100% trades coverage so the CNN always has `buy_ratio` and `aggressor_imbalance` features
- Safe to re-run — skips files that already have a trades file

---

## Model Training & Calibration

### `train_pump_detector.py`  *(Step 15)*

Trains **PumpDetectorV3** — the dual-stream, dual-output pump detection model.

**Architecture:** Two independent 3-block 1D-CNN towers (coin stream + market stream) → shared trunk (256→128) → two heads:

| Head | Output | Loss | When |
|---|---|---|---|
| `cls_head` | `pump_prob` — sigmoid [0, 1] | BCELoss | All samples |
| `reg_head` | `peak_pos` = `peak_idx / 96` — sigmoid [0, 1] | MSELoss | Pump samples only |

Combined loss: `total = BCE + 0.3 × MSE`

**Per-exchange pump cap:** 5,000 windows per exchange. Prevents Gate.io (historically 32% of pump windows) from dominating training and causing AUC oscillation.

**Metrics reported:**
- Best validation AUC (best checkpoint across all epochs)
- Mean validation AUC ± std dev (stability across all 60 epochs)
- Zero-shot AUC on Bybit + OKX (never seen during training)
- Peak timing MAE in candles

**18-cell sample weighting** — each window is weighted by its coin × market regime × background volatility cell:

| Coin \ Market / Volatility | Normal·Calm | Normal·Normal | Normal·Volatile | Uncertain | Pumped |
|---|---|---|---|---|---|
| **Pump** | **4.0** | 3.0 | 2.0 | 2.5→1.5 | 1.5→1.0 |
| **Control** | 3.5 | 2.5 | 1.5 | 2.0→1.0 | 1.5→1.0 |

- Train exchanges: Binance, KuCoin, MEXC, Gate.io, Bitget
- Zero-shot test: Bybit, OKX
- Saves to `models/pump_detector_v3.pth`

`forward()` returns `(pump_prob, peak_pos)` — see `detector.md` for inference examples.

---

### `calibrate_threshold.py`  *(Step 16)*

Finds the optimal `CNN_THRESHOLD` for the production cascade using precision-recall curves on the zero-shot test set (Bybit + OKX).

1. Loads `models/pump_detector_v3.pth` and runs inference on the calibration exchanges
2. Plots the full precision-recall curve and computes F1 at every threshold point
3. Prints a threshold table (precision / recall / F1 at 0.05 steps from 0.30 to 0.95)
4. Identifies the threshold that maximises F1
5. Saves the PR curve to `models/pr_curve.png`

```
python scripts/calibrate_threshold.py                            # Bybit + OKX (default)
python scripts/calibrate_threshold.py --exchanges bybit          # Bybit only
python scripts/calibrate_threshold.py --no-plot                  # skip matplotlib output
```

When called from `main.py` (Step 16), `--no-plot` is passed automatically — the plot is saved to disk without opening a window. The recommended threshold replaces `CNN_THRESHOLD` in `live/config.py`.

---

## Models Produced

| File | Produced by | Used by |
|---|---|---|
| `models/direct_l2_vae_v1.pth` | `train_direct_l2.py` (Step 12) | `generate_direct_synthetic.py` (Step 13) |
| `models/l2_scaler.pkl` | `prepare_l2_training_data.py` (Step 11) | Steps 12, 13 |
| `models/pump_detector_v3.pth` | `train_pump_detector.py` (Step 15) | Live inference — returns `(pump_prob, peak_pos)` |
| `models/pr_curve.png` | `calibrate_threshold.py` (Step 16) | Visual reference — PR curve on zero-shot test set |

The `models/` folder contains only source `.py` files until the pipeline runs. All `.pth`, `.pkl`, and generated artefacts are created at runtime and are not committed to the repository.

---

## Prerequisites

```
pip install torch numpy pandas scikit-learn requests joblib ohlcv-to-orderbook matplotlib
```

The `ohlcv-to-orderbook` package must be installed for `reconstruct_orderbook.py` (Step 5).
