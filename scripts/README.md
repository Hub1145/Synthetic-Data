# Scripts Reference

All scripts are run from the **project root** (one level above this folder), not from inside `scripts/`.

```
python scripts/<script_name>.py
```

Use `run_pipeline.py` to chain the common steps automatically.

---

## Quick Map

```
Data Collection
  fetch_ardia_pumps.py              ← Binance P&D events (ArdiaD dataset, 1,160 events)
  fetch_all_pump_data.py            ← Multi-exchange pump scanner + downloader (all 8)
  fetch_control_data.py             ← Download organic-volatility events as hard negatives
  scan_multi_exchange.py            ← Live scanner: find new pump candidates on any exchange

Labelling & Enrichment
  fetch_and_label_tradebook_data.py ← Verify 30% retracement + fetch trade ticks + save peak_bucket
  reconstruct_orderbook.py          ← OHLCV → Level-2 orderbook (bid/ask price + size)
  add_orderbook_depth.py            ← Expand all L2 files from 1-level to 10-level depth

Metadata & Market Context
  tag_peak_buckets.py               ← Compute peak_pct/peak_bucket for reconstructed/ + synthetic/
  fetch_market_context.py           ← Fetch BTC OHLCV, build _market_ctx.csv, fill market_regime
  fix_unknown_market_regimes.py     ← Assign normal/uncertain/pumped to files with unknown regime; generate synthetic BTC context
  migrate_control_to_regimes.py     ← One-time: split control/ → normal/ + uncertain/ based on BTC regime in meta.json

Synthetic Generation
  generate_global_synthetic.py      ← Type-A: BaseVAE OHLCV → reconstruct → L2
  generate_direct_synthetic.py      ← Type-B: DirectL2VAE → 3 regimes × 6 buckets + market ctx
  generate_missing_trades.py        ← Fill trades files for any L2 that has none

Model Training
  prepare_l2_training_data.py       ← Vectorise reconstructed/ L2 into data/l2_training_samples.npy
  train_global_vae.py               ← Train BaseVAE on real OHLCV (Type-A generator)
  train_direct_l2.py                ← Train DirectL2VAE on reconstructed L2 (Type-B generator)
  train_pump_detector.py            ← Train PumpDetectorV3 (dual-stream, 9-cell weights)

Validation
  test_pumpable_extractor.py        ← Score all L2 files using PUMPABLE COINS extractor

Orchestration
  run_pipeline.py                   ← Runs all steps in order with skip flags
```

---

## Data Collection

### `fetch_ardia_pumps.py`

Downloads 1-minute OHLCV klines from **Binance Data Vision** for every event in the [ArdiaD/PumpDump](https://github.com/ArdiaD/PumpDump) dataset (`data/list_pd_events.csv`, 1,160 Binance events).

- Output: `real/binance/pumps/[SYMBOL]/[SYMBOL]-1m-[DATE].csv`
- Skips symbols already downloaded
- Rate-limited (0.3 s between requests, 2 retries)
- Run this before `fetch_and_label_tradebook_data.py` to validate which events pass the 30% retracement filter

---

### `fetch_all_pump_data.py`

Comprehensive multi-exchange fetcher. Reads pump candidates from `data/global_deep_scan_pumps.csv` (produced by `scan_multi_exchange.py`) and downloads the corresponding 1-minute OHLCV via CCXT.

- Exchanges covered: Binance, Bybit, KuCoin, OKX, Huobi, MEXC, Gate.io, Bitget
- Output: `real/[exchange]/pumps/[SYMBOL]/[SYMBOL]_[DATE]_klines.csv`
- Skips already-downloaded events

---

### `fetch_control_data.py`

Fetches OHLCV for events that produced a large price spike but **failed** the 30% retracement threshold — these are organic volatility events, not pump-and-dumps. They form the hard-negative control set.

- Reads: `data/scanned_pumps/[exchange]_pumps.csv` (volatile_control rows)
- Output: `real/[exchange]/normal/[SYMBOL]/[SYMBOL]_[DATE]_klines.csv` (or `uncertain/` after BTC regime classification)
- Accepts `--exchange` flag to fetch one exchange at a time (e.g., `--exchange huobi mexc gateio`)
- After fetching, run `fetch_market_context.py` to assign BTC regime, then `migrate_control_to_regimes.py` to place files in the correct `normal/` or `uncertain/` subfolder

---

### `scan_multi_exchange.py`

Live scanner that queries all 8 exchanges via CCXT and identifies coins with recent anomalous price behaviour matching pump-and-dump signatures.

- Applies a spike threshold and 30% retracement filter to classify events
- Writes candidate events to `data/scanned_pumps/[exchange]_pumps.csv`
- Output feeds into `fetch_all_pump_data.py` and `fetch_control_data.py`

---

## Labelling & Enrichment

### `fetch_and_label_tradebook_data.py`

Verifies pump labels and enriches confirmed events with real trade ticks — two operations in a single pass over `real/`.

**Label step** — applies pump confirmation criteria to every event in `real/pumps/`:
1. Price spike >= **5%** from the initial open
2. Retracement >= **30%** from peak back toward the pre-pump price

Events that pass stay in `pumps/`. Events that fail (organic volatility) are moved to `normal/` or `uncertain/` depending on BTC market regime during the event. A report is written to `data/labeling_report.csv`.

**Enrich step** — for each confirmed pump, immediately fetches real trade ticks if a free archive is available:
- **Binance** → `data.binance.vision/data/spot/daily/trades/` (full history)
- **Bybit** → `public.bybit.com/trading/` (rolling ~90 days)
- Other exchanges: no free source — ticks remain OHLCV-derived

Ticks are saved to `real/[exchange]/pumps/[symbol]/[SYM]-trades-[DATE].csv` and also upgrade the matching `reconstructed/` trades file.

```
python scripts/fetch_and_label_tradebook_data.py                        # label + fetch ticks (binance + bybit)
python scripts/fetch_and_label_tradebook_data.py --skip-tradebook       # label only
python scripts/fetch_and_label_tradebook_data.py --exchanges binance    # label + fetch binance only
python scripts/fetch_and_label_tradebook_data.py --within-days 90       # skip bybit dates older than 90 days
```

---

### `reconstruct_orderbook.py`

Converts 1-minute OHLCV klines into synthetic Level-2 orderbook snapshots using the [ohlcv-to-orderbook](https://github.com/nickvdyck/ohlcv-to-orderbook) model.

- Input: `real/[exchange]/[regime]/[SYMBOL]/[SYMBOL]_*_klines.csv`
- Output: `reconstructed/[exchange]/[regime]/[SYMBOL]/[SYMBOL]_*_L2.csv` — columns: `bid_price, ask_price, bid_size, ask_size`
- Also generates a paired `*_trades.csv` using the OHLCV buy-ratio formula: `buy_ratio = (close − low) / (high − low + ε)`
- This is the primary method for producing L2 data when real orderbook snapshots are unavailable

---

### `add_orderbook_depth.py`

Expands every L2 file from single-level (bid/ask level 1 only) to **10-level orderbook depth** by simulating realistic price ladders and size distributions.

- Adds columns: `bid_price_2..10`, `bid_size_2..10`, `ask_price_2..10`, `ask_size_2..10`
- Run once after any batch of new L2 files are created
- Safe to re-run — skips files that already have depth columns
- Applied to 8,582 files in Phase 3b

---

## Metadata & Market Context

### `tag_peak_buckets.py`

Scans every `*_L2.csv` in `reconstructed/` and `synthetic/` and writes a `*_meta.json` alongside each file (skipping files that already have one). Computes the peak price increase and retracement from the mid-price curve, then assigns a `peak_bucket`:

| Bucket | Peak increase range |
|---|---|
| micro | 5 – 10% |
| small | 10 – 20% |
| medium | 20 – 30% |
| large | 30 – 40% |
| major | 40 – 50% |
| extreme | 50%+ |

Sets `market_regime = "unknown"` — run `fetch_market_context.py` next to fill it in.

```
python scripts/tag_peak_buckets.py             # tag all untagged files
python scripts/tag_peak_buckets.py --overwrite # re-tag everything
```

---

### `fetch_market_context.py`

For every event in `real/` and `reconstructed/`, downloads the BTCUSDT 1-minute kline for that date from Binance Data Vision (cached in `data/market_context/`), then:

1. Classifies BTC's behaviour during that window as `normal` / `uncertain` / `pumped`
2. Builds a 6-feature market context sequence (`bid_price, ask_price, bid_size, ask_size, buy_ratio, aggressor_imbalance`) derived from BTC OHLCV
3. Saves it as `*_market_ctx.csv` alongside the coin's L2 file
4. Updates `market_regime` in the matching `*_meta.json`

BTC regime thresholds: < 3% move = normal · 3–8% without retracement = uncertain · ≥ 5% spike with ≥ 20% retracement = pumped.

```
python scripts/fetch_market_context.py
python scripts/fetch_market_context.py --exchanges binance kucoin
python scripts/fetch_market_context.py --overwrite   # rebuild existing ctx files
```

---

### `fix_unknown_market_regimes.py`

Fixes all `*_meta.json` files across `reconstructed/` and `synthetic/` where `market_regime` was left as `"unknown"` (caused by failed BTC kline downloads for future-dated events or older Type-A synthetic files).

- Assigns `normal`, `uncertain`, or `pumped` in round-robin across all unknown files (~1,490 per regime)
- Generates a synthetic `*_market_ctx.csv` for each fixed file using statistical BTC price/volume profiles per regime
- Regime profiles: `normal` = low-noise drift, balanced bid/ask; `uncertain` = moderate drift with spikes; `pumped` = sharp spike with 40–70% retracement
- Safe to re-run — only touches files where `market_regime == "unknown"`
- Fixed 4,470 files total: 3,649 reconstructed + 821 synthetic

```
python scripts/fix_unknown_market_regimes.py
```

---

### `migrate_control_to_regimes.py`

One-time migration that splits the legacy `control/` directory into `normal/` and `uncertain/` based on the BTC market regime stored in each symbol's `*_meta.json`.

**REGIME_MAP** (BTC regime → destination folder):
- `normal` → `normal/`
- `uncertain` → `uncertain/`
- `pumped` → `uncertain/` (coin followed BTC momentum, not genuine pump — still a soft negative)
- `unknown` → `normal/` (fallback)

- Supports `--dry-run` (preview moves without touching files) and `--dirs` (specify tiers to migrate)
- Applied to 4,840 symbol dirs across `reconstructed/` and `synthetic/`; `real/` migrated separately

```
python scripts/migrate_control_to_regimes.py --dry-run          # preview
python scripts/migrate_control_to_regimes.py                    # apply to reconstructed/ + synthetic/
python scripts/migrate_control_to_regimes.py --dirs real        # apply to real/ only
```

---

## Synthetic Generation

### `generate_global_synthetic.py`

**Type-A synthetic generator.** Uses the trained `BaseVAE` (OHLCV-space, 24-step windows, latent dim 32) to sample new OHLCV sequences, then passes them through `reconstruct_orderbook.py` to produce L2 files.

- Generates 100 samples per exchange (50 pump, 50 control) × 8 exchanges = 800 files
- Output: `synthetic/[exchange]/[regime]/[symbol]/[symbol]_synthetic_L2.csv`
- Requires `models/global_vae_v1.pth` and `models/global_scaler_v1.pkl`
- Type-A synthetic has no paired trades file (neutral fill applied during training)

---

### `generate_direct_synthetic.py`

**Type-B synthetic generator.** Uses the trained `DirectL2VAE` (L2-space, 96-step windows, latent dim 64) to generate orderbook sequences **directly** — no OHLCV intermediate step.

Each sample is tagged with a **coin regime** (pump/control), a **market regime** (normal/uncertain/pumped), and a **peak bucket** (micro through extreme). This covers the full 3×3 coin×market matrix Andy defined.

Per sample, 4 files are saved:
- `*_direct_L2.csv` — coin orderbook (96 timesteps, 10-level depth)
- `*_trades.csv` — coin tradebook with pump-phase buy/sell logic
- `*_market_ctx.csv` — synthetic BTC context for the 3 market regime types
- `*_meta.json` — `coin_regime`, `market_regime`, `peak_bucket`, `peak_pct`, `peak_idx`

Pump sample distribution: 6 peak buckets × 3 market regimes = **18 cells**, cycled evenly across 54 pump samples per exchange.
Control sample distribution: 3 market regimes × 17 each = 51 control samples per exchange.

- Total: 105 samples × 8 exchanges = **840 files**
- Requires `models/direct_l2_vae_v1.pth` and `models/l2_scaler.pkl`

---

### `generate_missing_trades.py`

Scans all L2 files across `reconstructed/` and `synthetic/` and creates a paired `*_trades.csv` for any file that is missing one.

- Uses the OHLCV buy-ratio formula as the fallback: `buy_ratio = (close − low) / (high − low + ε)`
- Ensures 100% trades coverage across the full dataset so the CNN can always read `buy_ratio` and `aggressor_imbalance` features
- Safe to re-run — skips files that already have a trades file

---

## Model Training

### `prepare_l2_training_data.py`

Vectorises all `reconstructed/` L2 files into a single NumPy array for training the `DirectL2VAE`.

- Reads 4 features: `bid_price, ask_price, bid_size, ask_size` — first 96 timesteps of each file
- Applies StandardScaler normalization
- Output: `data/l2_training_samples.npy` and `models/l2_scaler.pkl`
- Run this before `train_direct_l2.py` whenever new reconstructed files have been added

---

### `train_global_vae.py`

Trains the **BaseVAE** — the Type-A generative model that operates in OHLCV space.

- Architecture: 24-step × 5-feature input, latent dim 32
- Trains on all OHLCV kline files in `real/`
- Saves weights to `models/global_vae_v1.pth` and scaler to `models/global_scaler_v1.pkl`
- Note: The BaseVAE is the older generator. The `DirectL2VAE` (Type-B) produces higher-quality L2 sequences. Both are retained.

---

### `train_direct_l2.py`

Trains the **DirectL2VAE** — the Type-B generative model that operates directly in L2 orderbook space.

- Architecture: 1D-Convolutional VAE, 96-step × 4-feature input, latent dim 64
- Trains on `data/l2_training_samples.npy` (prepared by `prepare_l2_training_data.py`)
- Saves weights to `models/direct_l2_vae_v1.pth`
- 100 epochs, batch size 32, Adam optimizer

---

### `train_pump_detector.py`

Trains **PumpDetectorV3** — the dual-stream pump detection model.

**Architecture:** Two independent 3-block 1D-CNN towers (coin stream + market context stream), each producing a 128-dim feature vector. These are concatenated → shared MLP → sigmoid output.

| Stream | Input | Source |
|---|---|---|
| Coin | `[96, 6]` — bid/ask price+size, buy_ratio, agg_imb | `*_L2.csv` + `*_trades.csv` |
| Market | `[96, 6]` — same schema for BTC context | `*_market_ctx.csv` (neutral fill if missing) |

**9-cell sample weighting** — each training window is weighted by its position in the coin×market regime matrix:

| Coin \ Market | Normal | Uncertain | Pumped |
|---|---|---|---|
| Pump | **3.0** (most suspicious) | 2.0 | 1.0 (least suspicious) |
| Control | 2.5 (hard negative) | 1.5 | 1.0 (easy negative) |

- Scans both `reconstructed/` and `synthetic/` for training windows
- Train exchanges (6): Binance, KuCoin, Huobi, MEXC, Gate.io, Bitget
- Zero-shot test exchanges (2): Bybit, OKX
- Saves to `models/pump_detector_v3.pth`

See `detector.md` (project root) for inference examples (use `PumpDetectorV3` with both coin and market windows).

---

## Validation

### `test_pumpable_extractor.py`

Scores all L2 files using the `PumpableCoinExtractor` from the PUMPABLE COINS module.

- Reads up to 10 orderbook levels and the paired `*_trades.csv`
- Scores based on: orderbook imbalance (50%), market depth/impact (30%), spread (20%)
- Writes results to `data/pumpable_scores.csv`
- Used to verify that pump samples score higher than control samples (discrimination check)
- Results: real data 1.15× pump/control ratio; synthetic data ~1.02× (pump cycle averages to near-neutral over the full 96-step window)

---

## Orchestration

### `run_pipeline.py`

Runs all pipeline steps in sequence after new data is fetched. Each step calls the corresponding script as a subprocess and aborts if any step fails.

```
python scripts/run_pipeline.py                    # run all steps
python scripts/run_pipeline.py --skip-fetch       # skip data download (already done)
python scripts/run_pipeline.py --skip-label       # skip retracement labelling
python scripts/run_pipeline.py --skip-reconstruct # skip L2 reconstruction + depth expansion
python scripts/run_pipeline.py --skip-synthetic   # skip VAE generation steps
python scripts/run_pipeline.py --train-only       # run CNN training only
```

Step order:
1. `fetch_all_pump_data.py`
2. `fetch_and_label_tradebook_data.py`
3. `reconstruct_orderbook.py`
4. `add_orderbook_depth.py`
5. `tag_peak_buckets.py`
6. `fetch_market_context.py`
7. `fix_unknown_market_regimes.py`
8. `migrate_control_to_regimes.py`
9. `generate_direct_synthetic.py`
10. `generate_missing_trades.py`
11. `train_pump_detector.py`

---

## Models Produced

| File | Produced by | Used by |
|---|---|---|
| `models/global_vae_v1.pth` | `train_global_vae.py` | `generate_global_synthetic.py` |
| `models/global_scaler_v1.pkl` | `train_global_vae.py` | `generate_global_synthetic.py` |
| `models/direct_l2_vae_v1.pth` | `train_direct_l2.py` | `generate_direct_synthetic.py` |
| `models/l2_scaler.pkl` | `prepare_l2_training_data.py` | `train_direct_l2.py`, `generate_direct_synthetic.py` |
| `models/pump_detector_v3.pth` | `train_pump_detector.py` | Inference — `PumpDetectorV3(x_coin, x_market)` (see `detector.md`) |

---

## Prerequisites

```
pip install torch numpy pandas scikit-learn ccxt requests joblib
```

The `ohlcv-to-orderbook` binary must be on PATH for `reconstruct_orderbook.py`.

The `PUMPABLE COINS` project folder must be at the same directory level as this project for `test_pumpable_extractor.py`.
