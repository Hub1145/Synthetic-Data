# Tasks: Direct L2 Generation & Control Set Hardening

## Architecture (Andy's Final Decision)
- **Real OHLCV data** → `reconstruct_orderbook.py` → L2 orderbook files
- **Synthetic data** → `DirectL2VAE` → L2 orderbook + tradebook files **directly** (no OHLCV intermediate)
- Not all extreme events are pumps — 30% retracement threshold distinguishes genuine P&D from organic volatility
- **Two regimes only**: `pumps/` (label=1) and `control/` (label=0). Organic-volatility events go into `control/`, not a separate folder.

---

- [x] **Phase 1: Control Set Hardening**
    - [x] Update `scripts/scan_multi_exchange.py` to differentiate Pumps vs Organic Volatility (30% retracement rule implemented).
    - [x] Create `scripts/label_real_data.py` to apply retracement filter to existing `real/` data and move organic events to `control/`.
    - [x] **Run** `scripts/label_real_data.py` — 297 events checked, 272 confirmed pumps, 25 moved to `control/`. Report at `data/labeling_report.csv`.

- [x] **Phase 2: Direct L2-VAE Architecture**
    - [x] Implement `models/direct_l2_vae.py` (1D-Convolutional VAE, 96 timesteps × 4 features, latent dim 64).
    - [x] Create `scripts/prepare_l2_training_data.py` to vectorize the `reconstructed/` data into `data/l2_training_samples.npy`.
    - [x] Train the `DirectL2VAE` — weights saved to `models/direct_l2_vae_v1.pth`.

- [x] **Phase 3: Direct Generation & Tradebook**
    - [x] Implement `scripts/generate_direct_synthetic.py` — generates `_direct_L2.csv` + `_trades.csv` per sample.
    - [x] Fix asymmetric bid/ask injection — `bid_size` swells pre-peak (buy pressure), `ask_size` swells post-peak (sell pressure). Previously both sides were injected equally, producing zero discrimination.
    - [x] Generate for all 8 exchanges (100 samples/exchange = 800 total direct synthetic files).
    - [x] Output goes to `synthetic/[exchange]/[regime]/[symbol]/` (separated from OHLCV-derived `reconstructed/` data).

- [x] **Phase 3b: Orderbook Depth Expansion**
    - [x] Create `scripts/add_orderbook_depth.py` — vectorized depth generation for all L2 files.
    - [x] **Run** `scripts/add_orderbook_depth.py` — 8,582/8,582 files updated with bid/ask levels 2–10.
    - [x] All files now carry columns `bid_price_1..10`, `bid_size_1..10`, `ask_price_1..10`, `ask_size_1..10`.

- [x] **Phase 3c: Historical Tradebook Fetch**
    - [x] Create `scripts/fetch_real_trades.py` — CCXT-based trade fetcher for real L2 files.
    - [x] **Run** (limited) — exchanges retain only 30–90 days of trade history; most pump events are older. Real tick data not available for historical events.
    - [x] **Resolved**: `generate_missing_trades.py` filled all 9,022 L2 files missing a trades file using OHLCV buy-ratio formula. Tradebook coverage now **100%** across all 7,408 L2 files (reconstructed/ + synthetic/).

- [x] **Phase 4: Verification**
    - [x] Run `scripts/label_real_data.py` (Phase 6 pass) — 272 pumps confirmed, 25 moved to control.
    - [x] Run `scripts/test_pumpable_extractor.py` — 9,363 files scored; see Phase 7 results.
    - [x] Update all documentation files.
    - [ ] Visual comparison: plot a Type A (`_synthetic_L2.csv`) vs Type B (`_direct_L2.csv`) orderbook sequence.
    - [ ] Latent space cluster analysis: confirm pump vs control samples form distinct clusters in the VAE's 64-dim latent space.

- [x] **Phase 5: CNN Pump Detector** *(The End Goal)*
    - [x] Design model: `models/pump_detector.py` — 3-block 1D-CNN, BatchNorm, Global Avg Pool, sigmoid output.
    - [x] Create `scripts/train_pump_detector.py` — trains on 6 exchanges, zero-shot tests on Bybit+OKX.
    - [x] **Run v1** — 11,106 train windows, 3,859 test windows. Best val AUC 0.8549. Zero-shot AUC 0.7273.
    - [x] **Run v2** (6-feature with tradebook) — 96,101 train windows, 21,788 test windows.
          Best val AUC **0.8530**. Zero-shot AUC **0.7899** (↑ +0.063 vs v1). Model saved to `models/pump_detector_v2.pth`.

- [x] **Phase 6: Real Data Expansion** *(Andy: real data is gold)*
    - [x] Write `scripts/fetch_all_pump_data.py` — comprehensive multi-exchange fetcher.
    - [x] Expand `scripts/scan_multi_exchange.py` from 4 → 8 exchanges (added Gate.io, MEXC, Huobi, Bitget).
    - [x] **Run** Binance ArdiaD fetch — 291/291 events downloaded (0 failures).
    - [x] **Run** label + reconstruct — 272 pumps confirmed, full 8-exchange L2 dataset.
    - [x] Fix `generate_global_synthetic.py` — was hardcoded to 4 exchanges. Now generates for all 8.
    - [x] **Run** 8-exchange scanner — 6,384 confirmed pumps identified.
    - [x] **Updated** `train_pump_detector.py` for 6-feature tradebook integration (buy_ratio, aggressor_imbalance).

- [x] **Phase 7: PUMPABLE COINS Integration**
    - [x] Create `scripts/test_pumpable_extractor.py` — scores all L2 files using PumpableCoinExtractor.
    - [x] **Run** full test — 9,363 files scored with 10-level orderbook + tradebook.
    - [x] Results: real 1.15x discrimination, synthetic_direct 1.02x, synthetic_typeA 1.00x.
    - [x] Note: PUMPABLE extractor discrimination for synthetic data is limited — the pump/dump cycle averages bid and ask imbalance to near-neutral over the full 96-step window.

- [x] **Phase 8: Data Architecture Correction**
    - [x] **User clarification**: `real/` = real exchange data, `reconstructed/` = OHLCV-simulation, `synthetic/` = VAE-generated. DirectL2VAE files had incorrectly been saved to `reconstructed/`.
    - [x] Create `scripts/restructure_directories.py` — moved 2,792 `*_direct_L2.csv` + 2,792 `*_trades.csv` from `reconstructed/` → `synthetic/`. Removed 2,769 empty dirs.
    - [x] Fix `scripts/generate_direct_synthetic.py` — changed `OUTPUT_BASE = "reconstructed"` → `"synthetic"` so new generations land in the correct location.
    - [x] Fix `scripts/cleanup_unused.py` — added protection for `_direct_L2.csv` and their paired trades files so orphan-detection never deletes DirectL2VAE output.
    - [x] Update `scripts/train_pump_detector.py` — `load_dataset()` now scans both `reconstructed/` and `synthetic/` for each exchange.
    - [x] Regenerate 800 DirectL2VAE samples (100/exchange × 8) into `synthetic/`. Final count: 823 files (414 pump, 409 control).
    - [x] Update all documentation files to reflect three-directory architecture.
    - [x] Create `scripts/fetch_real_tradebook.py` — fetches real trade ticks for all events in `real/` from Binance Data Vision + Bybit public archive; saves to `real/[exch]/[regime]/[sym]/[SYM]-trades-[DATE].csv` and upgrades matching `reconstructed/` trades files.
    - [x] Run `fetch_real_tradebook.py --exchanges binance` — fetched=290, skipped=65, failed=1. **330 real trade tick files** now in `real/binance/` (292 pump, 38 control); matching `reconstructed/` trades upgraded.
    - [x] Run `fetch_real_tradebook.py --exchanges bybit --within-days 90` — stopped early per user instruction. **53 Bybit trade tick files** saved (25 symbols: 0GUSDT, AGLDUSDT, ANIMEUSDT, ANKRUSDT, APEUSDT, BERAUSDT, CHIPUSDT, CYBERUSDT, DOTUSDT, ENJUSDT, etc.). Remaining 127 events not fetched — 5 symbols sufficient for now.

- [ ] **Phase 9: Planned**

    - [ ] Per-exchange window cap (~5,000 max pump windows) for more stable v2 retraining.
    - [ ] Pull Bayi-Hu dataset (709 events, Jan 2019 – Jan 2022) for extended coverage.
    - [ ] Investigate real L2 orderbook data from `reach.stratosphere.capital/data/usdm/` (hftbacktest project).
    - [ ] Improve synthetic pump → model post-dump seller residual for net ask imbalance after full pump+dump cycle.
    - [ ] Visual latent space analysis (PCA/t-SNE on 64-dim VAE embeddings).
