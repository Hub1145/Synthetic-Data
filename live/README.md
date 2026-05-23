# Live Inference Pipeline

Real-time pump-and-dump detector. Loads the trained model from `models/` and scans live exchange feeds every minute. Prints alerts directly to the terminal — no web server, no dashboard.

---

## Table of Contents

1. [How It Fits Into the Project](#1-how-it-fits-into-the-project)
2. [Folder Structure](#2-folder-structure)
3. [Prerequisites](#3-prerequisites)
4. [Installation](#4-installation)
5. [Quick Start](#5-quick-start)
6. [All CLI Options](#6-all-cli-options)
7. [How the Detection Cascade Works](#7-how-the-detection-cascade-works)
8. [Feature Engineering](#8-feature-engineering)
9. [The Model — PumpDetectorV3](#9-the-model--pumpdetectorv3)
10. [Peak Estimation](#10-peak-estimation)
11. [Alert Output Format](#11-alert-output-format)
12. [Buffer Warmup Period](#12-buffer-warmup-period)
13. [Thresholds and Tuning](#13-thresholds-and-tuning)
14. [Cooldown System](#14-cooldown-system)
15. [Market Context Stream](#15-market-context-stream)
16. [Known Limitations](#16-known-limitations)

---

## 1. How It Fits Into the Project

The Synthetic Data project has two separate parts that share only one file — the trained model weights:

```
Synthetic Data/
│
├── scripts/                       ← BATCH PIPELINE (data factory)
│   ├── fetch_all_pump_data.py         collects real pump events from 7 exchanges
│   ├── reconstruct_orderbook.py       converts OHLCV to L2 orderbook sequences
│   ├── generate_direct_synthetic.py   generates synthetic pump + control samples
│   ├── train_pump_detector.py         trains the CNN on real + synthetic data
│   └── ...
│
├── models/
│   ├── pump_detector_v3.pth       ← THE ONLY BRIDGE between scripts/ and live/
│   ├── pump_detector_v2.pth           (older single-stream version)
│   └── pump_detector_v1.pth           (baseline 4-feature version)
│
└── live/                          ← INFERENCE PIPELINE (this folder)
    ├── main.py                        terminal entry point
    ├── cascade.py                     3-stage detection logic
    ├── peak_detector.py               4 peak estimation methods
    ├── config.py                      paths + thresholds
    └── requirements.txt
```

**The workflow:**

```
Step 1 — Run scripts/ once (or periodically to retrain):
  fetch data → reconstruct orderbook → generate synthetic →
  train model → saves models/pump_detector_v3.pth

Step 2 — Run live/ continuously:
  loads models/pump_detector_v3.pth → scans live markets →
  prints pump alerts to terminal
```

The `scripts/` folder is the factory. The `live/` folder is the product.
They are completely independent except for the `.pth` file.

---

## 2. Folder Structure

```
live/
├── main.py            CLI entry point — argument parsing, startup banner, asyncio.run()
├── cascade.py         Core detection logic: CoinMonitor + LiveCascade classes
├── peak_detector.py   Standalone peak estimation: 4 methods + consensus combiner
├── config.py          All paths, thresholds, and settings in one place
├── requirements.txt   Python dependencies
└── README.md          This file
```

Also depends on two external locations (paths auto-configured in `config.py`):
- `../models/pump_detector_v3.pth` — trained weights (produced by `scripts/train_pump_detector.py`)
- `../../PUMPABLE COINS/pumpable_coin_extractor.py` — stage 1 scorer (sibling project)

---

## 3. Prerequisites

**Python 3.10+**

**The trained model must exist:**
```
Synthetic Data/models/pump_detector_v3.pth
```
If it doesn't exist yet, run the training pipeline in `scripts/` first.

**The PUMPABLE COINS project must be a sibling folder:**
```
WORKSPACE/
├── Synthetic Data/    ← you are here
└── PUMPABLE COINS/    ← must exist at this path
```
`config.py` auto-adds both locations to `sys.path` at startup — no manual configuration needed.

**No API keys required.** All exchange data is fetched from public endpoints.

---

## 4. Installation

From the `Synthetic Data/` root:

```bash
pip install -r live/requirements.txt
```

Dependencies:
| Package | Purpose |
|---|---|
| `torch` | Load and run PumpDetectorV3 |
| `numpy` | Feature arrays, normalization |
| `ccxt` | Live OHLCV and orderbook fetching from any exchange |
| `pandas` | Used by PumpableCoinExtractor internally |
| `scikit-learn` | Used by training scripts (not strictly needed for inference) |

---

## 5. Quick Start

Run everything from the `Synthetic Data/` root directory:

```bash
# Watch 3 specific coins on Binance
python live/main.py --exchange binance --symbols ETH/USDT SOL/USDT DOGE/USDT

# Automatically monitor the top 50 USDT pairs by 24h volume on KuCoin
python live/main.py --exchange kucoin --top 50

# Scan every 30 seconds instead of every 60
python live/main.py --exchange bybit --top 30 --interval 30

# Show debug logs (useful during the warmup period)
python live/main.py --exchange okx --top 20 --verbose

# Stop at any time with Ctrl+C
```

On startup you will see:
```
==============================================================
  Pump Detector — BINANCE
  Symbols    : 50
  Thresholds : pumpable > 70.0  |  CNN prob > 0.65
  Interval   : 60s
  Cooldown   : 10 min
  Warmup     : ~96 min (buffer fills before first alert)
  Press Ctrl+C to stop.
==============================================================
```

After ~96 minutes the buffers fill and alerts can begin firing.

---

## 6. All CLI Options

```
python live/main.py [options]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--exchange` | string | `binance` | Exchange to scan. Choices: `binance kucoin bybit okx mexc gateio bitget` |
| `--symbols` | list | — | Specific symbols to watch (e.g. `ETH/USDT SOL/USDT`). Use this OR `--top`, not both |
| `--top N` | int | — | Auto-fetch the top N USDT pairs by 24h volume from the exchange |
| `--interval` | int | `60` | Seconds between scan ticks |
| `--cooldown` | int | `10` | Minutes to suppress repeat alerts for the same coin after one fires |
| `--verbose` | flag | off | Enable DEBUG-level logging (shows per-coin scan results, fetch errors) |

**`--symbols` vs `--top`:**
- `--symbols` is best when you already know which coins to watch (faster startup, no ticker fetch needed)
- `--top N` fetches all tickers from the exchange, ranks by volume, and picks the top N — good for broad market scanning

---

## 7. How the Detection Cascade Works

Every scan tick (default: 60 seconds), for each monitored coin:

```
┌─────────────────────────────────────────────────────────────┐
│  TICK (every 60 seconds)                                    │
│                                                             │
│  1. Fetch latest closed 1-min OHLCV candle from exchange    │
│     → convert to 6 features → push to 96-step deque        │
│                                                             │
│  2. If deque not full yet (< 96 candles):                   │
│     → SKIP (still warming up)                               │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  STAGE 1 — PUMPABLE COINS SCORE                      │   │
│  │  Fetch live orderbook snapshot (20 levels)           │   │
│  │  PumpableCoinExtractor.extract(symbol, orderbook)    │   │
│  │    score = imbalance×0.5 + impact×0.3 + spread×0.2  │   │
│  │  If pump_score < 70 → stop here (no CNN call)        │   │
│  └──────────────────────────────────────────────────────┘   │
│                  ↓ score ≥ 70                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  STAGE 2 — CNN PROBABILITY                           │   │
│  │  coin window:   [96, 6] from coin's deque            │   │
│  │  market window: [96, 6] from BTC/USDT deque          │   │
│  │  Both normalized per-window before inference         │   │
│  │  PumpDetectorV3(x_coin, x_market) → prob [0,1]       │   │
│  │  If prob < 0.65 → stop here                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                  ↓ prob ≥ 0.65                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  STAGE 3 — PEAK ESTIMATION                           │   │
│  │  Method 1: price maximum (with pre/post conditions)  │   │
│  │  Method 2: order-flow reversal (buy_ratio crossover) │   │
│  │  Method 4: bid/ask imbalance flip (>1.0 → <1.0)      │   │
│  │  Consensus peak_idx = median of available estimates  │   │
│  │  Phase = accumulation / peak / dump                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                  ↓                                           │
│  ALERT PRINTED TO TERMINAL                                   │
└─────────────────────────────────────────────────────────────┘
```

**Why three stages?**

Stage 1 (PUMPABLE COINS) is a fast snapshot scorer — cheap to run on every coin every minute. It filters out 95%+ of normal coins so Stage 2 (CNN) only runs on the interesting ones. The CNN is more expensive (a forward pass through a 3-block 1D-CNN on a 96-step sequence) but far more accurate because it sees the full temporal history. Stage 3 adds timing — it tells you not just that a pump is happening, but where in the cycle you are.

---

## 8. Feature Engineering

Each 1-minute OHLCV candle is converted to 6 features before being pushed into the rolling buffer:

| Feature | Formula | Range | What it captures |
|---|---|---|---|
| `bid_price` | `close × 0.9995` | price units | Best bid approximation |
| `ask_price` | `close × 1.0005` | price units | Best ask approximation |
| `bid_size` | `volume × buy_ratio` | volume units | Bid-side depth approximation |
| `ask_size` | `volume × (1 − buy_ratio)` | volume units | Ask-side depth approximation |
| `buy_ratio` | `(close − low) / (high − low + ε)` | 0 – 1 | Buy pressure: 1 = all buyers, 0 = all sellers |
| `aggressor_imbalance` | `2 × buy_ratio − 1` | −1 – +1 | Signed imbalance: +1 = aggressive buyers, −1 = sellers |

`buy_ratio` is derived from candle body position — the same formula used in `scripts/generate_missing_trades.py` during batch training, so live features are consistent with what the model was trained on.

**Per-window normalization** (applied just before the CNN call):
```
mid0 = (window[0, bid_price] + window[0, ask_price]) / 2
window[:, bid_price] /= mid0     # → ≈ 1.0 ± small %
window[:, ask_price] /= mid0

mean_sz = mean(window[:, bid_size] + window[:, ask_size])
window[:, bid_size] /= mean_sz   # → relative depth
window[:, ask_size] /= mean_sz

buy_ratio, aggressor_imbalance: unchanged (already in fixed range)
```

This matches the normalization applied in `scripts/train_pump_detector.py` — the model never sees raw prices or volumes, only relative values.

---

## 9. The Model — PumpDetectorV3

**File:** `models/pump_detector.py` → class `PumpDetectorV3`
**Weights:** `models/pump_detector_v3.pth`

PumpDetectorV3 is a **dual-stream 1D-CNN**. It runs two separate CNN towers — one for the coin being monitored, one for the BTC/USDT market context — then fuses their outputs before the classifier head.

```
x_coin   [1, 96, 6]  →  FeatureExtractor (3-block CNN)  →  [1, 128]
                                                               ↓ concat
x_market [1, 96, 6]  →  FeatureExtractor (3-block CNN)  →  [1, 128]  →  [1, 256]
                                                               ↓
                         Linear(256→64) → ReLU → Dropout(0.3)
                         Linear(64→1)   → Sigmoid
                                                               ↓
                         pump_probability  (0.0 – 1.0)
```

Each `FeatureExtractor` block:
```
Block 1: Conv1d(6→32, k=5) + Conv1d(32→32, k=3) + MaxPool(2) + Dropout(0.1)
Block 2: Conv1d(32→64, k=5) + Conv1d(64→64, k=3) + MaxPool(2) + Dropout(0.1)
Block 3: Conv1d(64→128, k=3) + AdaptiveAvgPool1d(1)   ← collapses time axis
```

**Why dual-stream?**

The most suspicious case is when a coin spikes while the broader market (BTC) is flat. A single-stream model can't distinguish "coin pumped because market pumped" from "coin pumped independently." The second CNN tower sees the BTC context and the classifier learns to reward divergence between the two streams.

**Trained on:**
- Binance, KuCoin, MEXC, Gate.io, Bitget (+ Huobi historical) — training set
- Bybit, OKX — zero-shot test set (never seen during training)
- Best zero-shot AUC: **0.7899** (v2 baseline) — v3 extended with market stream

---

## 10. Peak Estimation

Once the CNN confirms a pump (prob ≥ 0.65), Stage 3 estimates where in the pump cycle you are. Three independent methods are combined:

**Method 1 — Price Maximum** (`peak_detector.py: estimate_peak_price_max`)

Scans the 96-candle price series for the first index where:
- Price rose ≥ 5% from the prior low
- Price fell ≥ 3% from that point to the subsequent low

Gives a peak_idx within the window. Falls back to `argmax` if no qualifying peak found.

**Method 2 — Order Flow Reversal** (`estimate_peak_order_flow`)

Looks for where `buy_ratio` drops from above 0.60 to below 0.45 (3-candle smoothed). This is the signature of insiders flipping from buying to selling — typically occurs within 1–3 candles of the price peak.

- Pre-peak `buy_ratio` average in real Binance data: **0.73**
- Post-peak `buy_ratio` average: **0.31**

**Method 4 — Imbalance Flip** (`estimate_peak_imbalance_flip`)

Looks for where `bid_size / ask_size` crosses from above 1.0 to below 1.0 (5-candle smoothed). Pre-peak: buyers inflate bids. Post-peak: sellers dominate asks.

**Consensus:** `peak_idx = median(available estimates)`

**Phase labels:**
| Phase | Meaning | Action |
|---|---|---|
| `accumulation` | Current candle is before the estimated peak | Pump is building — peak is N minutes away |
| `peak` | Current candle is at the estimated peak | Price at maximum — reversal imminent |
| `dump` | Current candle is after the estimated peak | Sell pressure dominant — distribution phase |

---

## 11. Alert Output Format

When all three stages pass, this is printed to the terminal:

```
==============================================================
  *** PUMP ALERT — SOL/USDT on BINANCE ***
  Time       : 2026-05-23T14:32:07 UTC
  Pumpable   : 82.4 / 100  (threshold 70.0)
  CNN prob   : 0.7831       (threshold 0.65)
  Phase      : ACCUMULATION
  Peak candle: 71 / 95
  Status     : ~25 min to estimated peak
  Detectors  : price_max=68  order_flow=73  imbalance_flip=71
==============================================================
```

**Field explanations:**

| Field | Description |
|---|---|
| `Pumpable` | PUMPABLE COINS score (0–100). Combines bid/ask imbalance (50%), market impact (30%), spread (20%). ≥70 to proceed. |
| `CNN prob` | PumpDetectorV3 output — probability that the 96-candle window contains a pump+dump cycle. ≥0.65 to alert. |
| `Phase` | Where in the cycle: `ACCUMULATION` (before peak), `PEAK` (at peak), `DUMP` (after peak) |
| `Peak candle` | Estimated peak position within the 96-candle window (0 = oldest candle, 95 = current candle) |
| `Status` | Human-readable action line — minutes to peak if still accumulating, or phase description |
| `Detectors` | Raw per-method peak estimates for diagnostics. `None` means that method couldn't find a peak |

---

## 12. Buffer Warmup Period

The detector needs 96 complete 1-minute candles per coin before any alerts can fire. At 60-second intervals this is **~96 minutes** of warmup.

During warmup:
- The scan loop runs normally — candles are fetched and pushed into the buffer
- All coins are skipped silently at the "buffer not ready" check
- No orderbook fetches, no CNN calls, no alerts
- With `--verbose`, you will see debug messages counting candles per coin

**Tip:** If you want faster startup for testing, lower `WINDOW_SIZE` in `config.py` (e.g. to 20) but be aware the model was trained on 96-step windows and accuracy will degrade significantly on shorter inputs.

---

## 13. Thresholds and Tuning

Both thresholds are set in `config.py` and apply globally to all coins and exchanges:

```python
PUMPABLE_THRESHOLD = 70.0   # Stage 1 gate
CNN_THRESHOLD      = 0.65   # Stage 2 gate
```

**Effect of raising/lowering:**

| Setting | Effect |
|---|---|
| Lower `PUMPABLE_THRESHOLD` (e.g. 50) | More coins pass Stage 1, more CNN calls, more false alerts |
| Raise `PUMPABLE_THRESHOLD` (e.g. 85) | Fewer coins reach Stage 2, may miss moderate pumps |
| Lower `CNN_THRESHOLD` (e.g. 0.50) | High recall — catches most pumps but more false alarms |
| Raise `CNN_THRESHOLD` (e.g. 0.75) | High precision — fewer false alarms, may miss early-stage events |
| `CNN_THRESHOLD = 0.85+` | Very conservative — only the most obvious, unambiguous pumps |

**Recommended starting point:** Keep defaults (70 / 0.65) for the first run to understand the false-positive rate in your target market. Adjust from there.

---

## 14. Cooldown System

Once an alert fires for a coin, that coin is suppressed for `--cooldown` minutes (default: 10). This prevents the same ongoing pump from generating a new alert on every scan tick.

The cooldown is per-coin, per-session. Restarting `main.py` resets all cooldowns.

If you want to completely disable cooldowns (e.g. during testing), set `--cooldown 0`.

---

## 15. Market Context Stream

PumpDetectorV3 takes two inputs — the coin stream and a market context stream. The market stream uses `BTC/USDT` 1-minute candles as a proxy for overall market conditions.

The market buffer is updated every scan tick alongside the coin buffers. If the market buffer hasn't filled yet (still in warmup), a neutral fill is used:
- `bid_price = ask_price = 1.0` (post-normalization neutral)
- `buy_ratio = 0.5` (neither buyer-dominated nor seller-dominated)
- All other features = 0

This means alerts during market warmup are possible but will be scored without true market context — slightly less accurate. In practice, both the coin and market buffers warm up at the same rate, so this only affects the very first ~96 minutes of each session.

**Why BTC/USDT?**
BTC is the benchmark for overall crypto market sentiment. A coin that pumps while BTC is flat is far more suspicious than a coin that rises alongside a broad market rally. The dual-stream architecture was trained to reward exactly this divergence.

---

## 16. Known Limitations

| Limitation | Impact | Notes |
|---|---|---|
| 96-minute warmup | No alerts for the first ~96 min per session | Expected behavior — the model needs a full window of context |
| OHLCV-derived L2 features | Less precise than a real L2 feed | The bid/ask prices and sizes are approximations. Real L2 data (e.g. from hftbacktest) would be more accurate |
| Public endpoints only | Rate limits apply | CCXT respects rate limits automatically. Monitoring >200 symbols may be slow |
| pump_detector_v3.pth may not exist | Crash on startup with clear error message | Run `scripts/train_pump_detector.py` first |
| Pump precision ~47% at 0.5 threshold | Raise threshold to 0.65–0.75 for production use | Already set to 0.65 by default |
| Gate.io AUC instability during training | v3 may have slightly weaker Gate.io performance | Retrain with per-exchange cap of ~5,000 pump windows when GPU is available |
| No persistence between runs | Cooldowns and buffers reset on restart | If you restart mid-pump, the buffer refills from scratch |
