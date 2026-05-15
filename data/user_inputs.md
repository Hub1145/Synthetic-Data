# Session Input Log: Direct L2 Generation & Control Set Hardening

## User Requests

1. **Initial Feedback Integration**: Andy's comments on labeling accuracy ("all extreme events may not be pumps") and data fidelity ("generate directly orderbook and tradebook").
2. **Strategic Pivot**: Transition from simple OHLCV reconstruction to high-dimensional "Type B" Direct Orderbook Generation.
3. **Documentation**: Maintain deep coverage and save requirements/plans to the local `data/` directory for long-term persistence.
4. **Generate synthetic data for all 8 exchanges** — `generate_global_synthetic.py` was missing huobi, mexc, gateio, bitget. Fixed.
5. **Integrate PUMPABLE COINS project** — use `PumpableCoinExtractor` to validate real, reconstructed, and synthetic data; confirm pump samples score higher than control.
6. **All data must be orderbook + tradebook** — not just OHLCV. Required: `bid_price, ask_price, bid_size, ask_size` (L2) + `timestamp_idx, side, price, size` (tradebook) for all three tiers.
7. **10-level orderbook depth** — all L2 files must carry bid/ask levels 1–10, not just level 1.
8. **Real historical trade ticks** — attempt to fetch via CCXT. Limited by exchange retention (30–90 days); accepted current coverage for historical events.
9. **Update all markdown documentation files** — `documentation.md`, `validation_report.md`, `research.md`, `data/implementation_plan.md`, `data/scratchpad.md`, `data/user_inputs.md`, `data/project_links.md`.

## Core Decisions

- **Pump Score Algorithm**: Implemented a retracement check (>30% dump) to distinguish between manipulative pumps and organic volatility.
- **CNN-VAE Pre-work**: Trained a high-dimensional Convolutional VAE (`DirectL2VAE`) to learn the structural dynamics of 96-timestamp orderbook sequences.
- **Tradebook Simulation**: Added simulated trade logs (`*_trades.csv`) to all synthetic "Type B" data to provide a complete "native" market snapshot.
- **Asymmetric Bid/Ask Injection**: Pre-peak `bid_size` swells (buyers accumulating), post-peak `ask_size` swells (sellers dumping). This was a critical fix — the original symmetric injection produced zero PUMPABLE COINS discrimination.
- **8-exchange coverage**: binance, bybit, kucoin, okx, huobi, mexc, gateio, bitget — all tiers.
- **CNN v2 training on GPU**: Deferred to high-performance PC due to Gate.io dominance issue (32% of pump windows). v2 subsequently trained and achieved zero-shot AUC **0.7899** (up from 0.7273 for v1).

## Key Results

| Metric | Value |
|---|---|
| Total L2 files with 10-level depth | 8,582 |
| PUMPABLE extractor test files | 9,363 |
| Real data discrimination (pump/ctrl) | 1.15x |
| synthetic_direct discrimination | 1.02x |
| CNN v2 validation AUC | 0.8530 |
| CNN v2 zero-shot AUC (Bybit+OKX) | 0.7899 |
