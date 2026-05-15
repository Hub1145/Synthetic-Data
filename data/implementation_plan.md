# Implementation Plan: Direct L2 Generation & Control Set Hardening

This document records the architectural pivot and implementation decisions made following Andy's final feedback. It serves as the source of truth for what was built and why.

---

## Core Architecture Decision (Andy's Rule)

Two data paths, one output directory:

```
Real OHLCV  ──►  reconstruct_orderbook.py  ──►  reconstructed/  (L2 + trades)
                 (ohlcv-to-orderbook library)

DirectL2VAE ──►  generate_direct_synthetic.py ──►  synthetic/  (L2 + trades)
                 (no OHLCV intermediate)
```

**Why**: OHLCV is an aggregated lossy representation. Synthetic data should learn from and generate the full microstructure directly. Reconstruction from OHLCV is acceptable for real data (it's the best available approximation) but synthetic data should not inherit that approximation error.

**Not all extreme events are pumps**: A genuine P&D requires a dump after the spike. The 30% retracement threshold distinguishes organized P&D from organic volatility.

**Two regime folders only**: `pumps/` (label=1) and `control/` (label=0). Organic volatility events go into `control/`, not a third folder.

---

## Phase 1: Control Set Hardening — COMPLETE

**Problem**: Events in `real/[exchange]/pumps/` included organic volatility events that spiked but did not dump. Training on these with label=1 would teach the model that "large spike = pump."

**Solution**: `scripts/label_real_data.py`

```
increase    = (peak_high - initial_open) / initial_open
retracement = (peak_high - min_close_after_peak) / (peak_high - initial_open)

PUMP:    increase >= 5% AND retracement >= 30%
CONTROL: otherwise → moved to real/[exchange]/control/[symbol]/
```

**Results**:
- 297 events checked, 272 confirmed pumps (kept), 25 relabeled and moved to control

**Files modified**: `scripts/label_real_data.py` (created), `data/labeling_report.csv` (output)

---

## Phase 2: Direct L2-VAE Architecture — COMPLETE

**Model**: `models/direct_l2_vae.py` — `DirectL2VAE`

Architecture (1D-Convolutional VAE):
- Input: `[Batch, 96, 4]` — 96 timesteps × 4 L2 features (bid_price, ask_price, bid_size, ask_size)
- Encoder: 3 × Conv1d (4→32→64→128, stride-2), flatten → Linear 1536→256, project to μ and log σ² (latent dim 64)
- Decoder: Linear 64→256→reshape, 3 × ConvTranspose1d (128→64→32→4, stride-2) → `[Batch, 96, 4]`
- Latent dim: 64 (captures richer structure than the 32-dim Type-A VAE)

**Training data**: `scripts/prepare_l2_training_data.py` vectorised `reconstructed/` into `data/l2_training_samples.npy` (470 samples, `[470, 96, 4]`)

**Weights**: `models/direct_l2_vae_v1.pth`, scaler: `models/l2_scaler.pkl`

---

## Phase 3: Direct Generation & Tradebook — COMPLETE

**Script**: `scripts/generate_direct_synthetic.py`

For each sample (100 per exchange × 8 exchanges, 50% pump / 50% control):
1. Sample `z ~ N(0, I)` in 64-dim latent space
2. Decode → `[96, 4]`, inverse-transform via scaler
3. For pump samples: inject asymmetric Gaussian price spike with randomized peak (index 20–81), 8–40% intensity, steeper rise than dump
4. **Asymmetric volume injection** (critical fix):
   - Pre-peak `bid_size` × (1 + 4×intensity×Gaussian), `ask_size` × (1 − 0.3×intensity×Gaussian)
   - Post-peak `ask_size` × (1 + 5×intensity×Gaussian), `bid_size` × (1 − 0.2×intensity×Gaussian)
5. Add 10-level depth columns (bid_price_2..10, bid_size_2..10, ask_price_2..10, ask_size_2..10)
6. Save `*_direct_L2.csv` + `*_trades.csv` (80% buys before peak, 25% after)

**Output schema**: `synthetic/[exchange]/[regime]/[symbol]/[SYMBOL]*_direct_L2.csv`

Tradebook format: `timestamp_idx, side, price, size`

---

## Phase 3b: Orderbook Depth Expansion — COMPLETE

**Script**: `scripts/add_orderbook_depth.py`

All existing L2 files (real-reconstructed and Type-A synthetic) that had only 1-level bid/ask data were expanded to 10 levels.

**Depth model**:
- Price step per level = `mid × base_step_pct × (1 + 0.15 × (k−1))`
- Bid decay = `Uniform(0.60, 0.80) + 0.05 × log(imbalance)` → clamped to [0.40, 0.90]
- Ask decay = `Uniform(0.60, 0.80) − 0.05 × log(imbalance)` → clamped to [0.40, 0.90]
- Seed = `hash(filepath) % 2^31` for reproducibility

**Result**: 8,582 / 8,582 files updated (0 skipped, 0 failed)

---

## Phase 3c: Historical Tradebook Fetch — LIMITED

**Script**: `scripts/fetch_real_trades.py`

Attempted to fetch real historical trade ticks via CCXT for all real L2 files. **Result**: All exchanges retain only 30–90 days of trade history. Since most pump events in the dataset predate the retention window, real tick data was unavailable for the vast majority of files.

**Outcome**: OHLCV-derived taker flow (Binance 12-col files only) and neutral fill (0.5 / 0.0) remain the tradebook signal for older events. Historical tick data requires paid providers (Tardis.dev, Kaiko).

---

## Phase 4: Verification — Partial

- [x] Labeling report reviewed (`data/labeling_report.csv`)
- [x] PUMPABLE COINS extractor integration test run — see Phase 7
- [ ] Visual comparison: plot a Type-A (`*_synthetic_L2.csv`) vs Type-B (`*_direct_L2.csv`) orderbook sequence
- [ ] Latent space cluster analysis: confirm pump vs control samples form distinct clusters in 64-dim space

---

## Phase 5: CNN Pump Detector — COMPLETE (v1 + v2)

### Model — `models/pump_detector.py`

3-block 1D-CNN:
- Block 1: Conv1d(num_features→32, k=5) × 2, MaxPool(2), Dropout(0.1)
- Block 2: Conv1d(32→64, k=5) × 2, MaxPool(2), Dropout(0.1)
- Block 3: Conv1d(64→128, k=3), AdaptiveAvgPool(1)
- Head: Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1) → Sigmoid

**Parameterized via `num_features`**: v1 used 4, v2 uses 6.

### v1 — 4 features (bid/ask price + size)

- Train: Binance + KuCoin (11,106 windows)
- Test: Bybit + OKX zero-shot (3,859 windows)
- Best val AUC: **0.8549**
- Zero-shot AUC: **0.7273**
- Model: `models/pump_detector_v1.pth`

### v2 — 6 features (+ buy_ratio + aggressor_imbalance from tradebook) — COMPLETE

New features derived per timestep from `*_trades.csv`:
```
buy_ratio           = buy_vol / total_vol             (0 → 1)
aggressor_imbalance = (buy_vol - sell_vol) / total_vol  (-1 → +1)
```
- Files without trades: neutral fill 0.5 / 0.0
- Binance 12-col sources: real taker flow from `taker_buy_base` column
- Synthetic: simulated buy pressure (80% before peak → 25% after peak)

**Training data (v2)**:
- Train: binance, kucoin, huobi, mexc, gateio, bitget — 96,101 windows (60,433 pump / 35,668 control)
- Test: bybit, okx — 21,788 windows (4,768 pump / 17,020 control)

**Results**:
- Best val AUC: **0.8530** (comparable to v1)
- Zero-shot AUC: **0.7899** (↑ +0.063 vs v1)
- Model: `models/pump_detector_v2.pth`

**Note on Gate.io dominance**: Gate.io contributed 19,657/60,433 pump windows (32.5%). AUC was unstable across epochs (0.54→0.84→0.57→0.85...) but recovered. If retraining, consider capping each exchange at ~5,000 pump windows for stability.

---

## Phase 6: Real Data Expansion — COMPLETE

### 6.1 ArdiaD/PumpDump Bulk Fetch — COMPLETE

**Script**: `scripts/fetch_all_pump_data.py`

- Binance: 322 confirmed events → all 291 remaining downloaded from Binance Data Vision archive (0 failures). Total Binance pump klines: 293.
- Non-Binance: CCXT fetch from `data/global_deep_scan_pumps.csv` + `data/bybit_specific_pumps.csv`

### 6.2 Multi-Exchange Scanner Expansion — COMPLETE

**Script**: `scripts/scan_multi_exchange.py`

Expanded from 4 to 8 exchanges: added Gate.io, MEXC, Huobi, Bitget.

Full symbol scan (all USDT + BTC pairs), 1-year daily candles, 30% retracement rule.

Output: `data/global_deep_scan_pumps.csv` (updated), `data/scanned_pumps/[exchange]_pumps.csv` per exchange.

### 6.3 Re-labeling + Reconstruction — COMPLETE

- `label_real_data.py` re-run: 297 checked, 272 pumps confirmed, 25 moved to control
- `reconstruct_orderbook.py` re-run: full 8-exchange reconstructed dataset

### 6.4 Type-A Synthetic for All 8 Exchanges — COMPLETE (WAS MISSING 4)

`generate_global_synthetic.py` was previously hardcoded to 4 exchanges. Fixed to generate 100 samples (50 pump/50 control) per exchange for all 8.

---

## Phase 7: PUMPABLE COINS Integration — COMPLETE

**Script**: `scripts/test_pumpable_extractor.py`

Full test of 9,363 L2 files with 10-level orderbook + paired tradebook.

**Results**:

| Source | Regime | Files | Mean Score | Max Score |
|---|---|---|---|---|
| real | control | 2,912 | 7.94 | 16.59 |
| real | pumps | 2,004 | 9.11 | 31.60 |
| synthetic_direct | control | 1,396 | 12.88 | 43.68 |
| synthetic_direct | pumps | 1,396 | 13.08 | 44.80 |
| synthetic_typeA | control | 836 | 9.11 | 14.64 |
| synthetic_typeA | pumps | 819 | 9.11 | 14.59 |

**Discrimination ratios** (pump / control mean score):
- real: **1.15x** — genuine signal ✓
- synthetic_direct: **1.02x** — weak; asymmetric injection helps price but mean score averages pre/post peak imbalance to near-neutral
- synthetic_typeA: **1.00x** — zero discrimination; inherits symmetric OHLCV reconstruction

**Key observation**: The PUMPABLE extractor scores individual timesteps and reports the mean. In a pump, bid imbalance is high pre-peak and ask imbalance is high post-peak — these average out. The real data shows 1.15x because the dump phase typically leaves a net seller excess. The synthetic direct data does not model the post-dump residual seller pressure realistically.

**Tradebook coverage**: 3.8% of files (synthetic_direct only; real files lacked accessible trade history).

**Per-exchange mean scores** (all sources combined):

| Exchange | Pump mean | Ctrl mean | Files (P/C) |
|---|---|---|---|
| binance | 9.25 | 10.47 | 682/438 |
| bitget | 11.97 | 9.31 | 211/519 |
| bybit | 10.26 | 8.97 | 615/1102 |
| gateio | 10.06 | 9.83 | 1522/759 |
| huobi | 11.69 | 9.38 | 228/618 |
| kucoin | 10.82 | 8.74 | 401/945 |
| mexc | 12.17 | 11.77 | 203/213 |
| okx | 11.30 | 9.84 | 357/550 |

---

## Verification Plan

### Automated
- Shape consistency: generated L2 CSVs have correct column order (bid_price, ask_price, bid_size, ask_size)
- Finite check: `np.isfinite(w).all()` enforced in `extract_windows()`
- Label schema: only `pumps/` and `control/` allowed as regime directories
- PUMPABLE COINS extractor: pump mean > control mean for real data ✓

### Manual
- Visual inspection: plot a Type-A vs Type-B L2 sequence side by side
- Latent space: PCA/t-SNE of 64-dim VAE embeddings — pump and control should cluster separately
- Threshold sweep: plot precision-recall curve for v2 to find optimal operating threshold

### Planned Improvements
- Improve synthetic pump discriminability: model post-dump seller residual to give net ask imbalance after the full pump+dump cycle
- Per-exchange window cap in training (~5,000 max pump windows/exchange) to reduce Gate.io dominance
- Retrain DirectL2VAE on real L2 data once hftbacktest data is available
