# Project Resource Hub: Quick Links

This document provides direct links to the key components of the Global Synthetic Crypto Pipeline.

### Documentation & Planning
- [Main Documentation](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/documentation.md)
- [Validation Report](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/validation_report.md)
- [Research: Data Sources & Methods](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/research.md)
- [Implementation Plan (Direct L2)](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/data/implementation_plan.md)
- [Session Scratchpad (Tasks)](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/data/scratchpad.md)
- [User Requirements Log](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/data/user_inputs.md)

### Models
- [Direct L2 VAE (Type-B Generator)](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/models/direct_l2_vae.py)
- [Base VAE (Type-A Generator)](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/models/base_vae.py)
- [CNN Pump Detector](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/models/pump_detector.py)
- [L2 Scaler](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/models/l2_scaler.pkl)

### Generation & Processing Scripts
- [Direct Synthetic Generator (Type-B)](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/scripts/generate_direct_synthetic.py) — DirectL2VAE → L2 + trades + 10-level depth
- [Global Synthetic Generator (Type-A)](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/scripts/generate_global_synthetic.py) — BaseVAE OHLCV, all 8 exchanges
- [Orderbook Reconstruction Engine](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/scripts/reconstruct_orderbook.py) — Real OHLCV → L2 + trades
- [Orderbook Depth Expander](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/scripts/add_orderbook_depth.py) — Adds bid/ask levels 2–10 to all L2 files
- [Historical Trade Fetcher](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/scripts/fetch_real_trades.py) — CCXT fetch for real trade ticks (limited by retention)
- [Multi-Exchange Scanner](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/scripts/scan_multi_exchange.py) — Pump score logic, 8 exchanges
- [Real Data Labeling / Control Set Hardening](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/scripts/label_real_data.py)

### Training & Evaluation Scripts
- [Train Pump Detector](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/scripts/train_pump_detector.py) — v1 (4-feat) + v2 (6-feat with tradebook)
- [PUMPABLE COINS Extractor Test](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/scripts/test_pumpable_extractor.py) — Full 10-level orderbook + tradebook validation

### Data Directories
- [Unified Reconstructed Data (L2)](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/reconstructed/) — 8,582 files, all 10-level depth
- [Raw Real Data (OHLCV)](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/real/)
- [Synthetic Data (OHLCV, Type-A)](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/synthetic/) — 800 files, all 8 exchanges
- [Data Reports](file:///c:/Users/abc/Desktop/WORKSPACE/Synthetic%20Data/data/) — extractor report, trajectories, labeling report

### External Projects
- [PUMPABLE COINS](file:///c:/Users/abc/Desktop/WORKSPACE/PUMPABLE%20COINS/) — PumpableCoinExtractor integration

### External Research & Data Sources
**Pump & Dump Event Datasets**
- [ArdiaD/PumpDump](https://github.com/ArdiaD/PumpDump) — 1,160 Binance P&D events (source of `data/list_pd_events.csv`). 291 events fetched.
- [SystemsLab-Sapienza/pump-and-dump-dataset](https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset) — Binance trade-level data with 5/15/25s feature windows.
- [Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency](https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency) — 709 events, Jan 2019–Jan 2022 (planned).

**Real Orderbook Data**
- [hftbacktest](https://github.com/nkaz001/hftbacktest) — Full L2/L3 reconstruction framework. Historical Binance Futures + Bybit data hosted at `reach.stratosphere.capital/data/usdm/`.

**OHLCV Reconstruction**
- [ohlcv-to-orderbook (PyPI)](https://pypi.org/project/ohlcv-to-orderbook/) — Converts OHLCV bars to simulated L2 snapshots. Used in `reconstruct_orderbook.py`.

**Benchmarks & Papers**
- [CTBench Paper (arXiv)](https://arxiv.org/html/2508.02758v1) — 452 Binance tokens, 13 evaluation metrics. Key finding: TimeVAE best for stable markets, COSCI-GAN best for volatile/pump regimes.
- [TS-Agent Paper (ACM)](https://dl.acm.org/doi/full/10.1145/3768292.3771251) — LLM agentic workflow for financial time-series, validates VAE approach for crypto.
- [TSGBench](https://github.com/YihaoAng/TSGBench) — 40+ TSG models benchmarked (general reference).
