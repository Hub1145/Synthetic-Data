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
│   ├── pump_detector.py               model architecture source
│   └── (other source .py files)       no pre-trained weights committed
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
├── main.py                      CLI entry point — argument parsing, startup banner, asyncio.run()
├── cascade.py                   Core detection logic: CoinMonitor + LiveCascade classes
├── peak_detector.py             Standalone peak estimation: 4 methods + consensus combiner
├── config.py                    All paths, thresholds, and settings in one place
├── pumpable_coin_extractor.py   Stage 1 scorer (copied here for server deployments)
├── requirements.txt             Python dependencies
└── README.md                    This file
```

Also depends on:
- `../models/pump_detector_v3.pth` — trained weights (produced by `scripts/train_pump_detector.py`)

---

## 3. Prerequisites

**Python 3.10+**

**The trained model must exist:**
```
Synthetic Data/models/pump_detector_v3.pth
```
If it doesn't exist yet, run the training pipeline in `scripts/` first.

**`pumpable_coin_extractor.py` must be accessible.** Two supported layouts:

| Environment | Location | How |
|---|---|---|
| Local dev | `WORKSPACE/PUMPABLE COINS/pumpable_coin_extractor.py` | `config.py` finds it automatically as a sibling folder |
| Server / container | `live/pumpable_coin_extractor.py` | Copy the file into `live/`; `config.py` falls back to `live/` if the sibling folder is missing |

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

Run from the `live/` directory or from the `Synthetic Data/` root:

```bash
# ── From inside live/ ────────────────────────────────────────────────
cd live/

# Watch specific coins on Binance
python main.py --exchange binance --symbols ETH/USDT SOL/USDT DOGE/USDT

# Top 50 USDT pairs by 24h volume on KuCoin
python main.py --exchange kucoin --top 50

# Every USDT spot pair on Bybit (700+ symbols)
python main.py --exchange bybit --all

# Every USDT spot pair on Binance (1500+ symbols)
python main.py --exchange binance --all

# Scan every 30 seconds with verbose debug output
python main.py --exchange bybit --top 30 --interval 30 --verbose

# Save session log + alerts JSONL
python main.py --exchange bybit --top 100 --log-file logs/bybit.log

# Disable cooldown (every alert fires every tick — for testing only)
python main.py --exchange okx --top 20 --cooldown 0

# Stop at any time with Ctrl+C


# ── From Synthetic Data/ root ────────────────────────────────────────
python live/main.py --exchange binance --top 50
python live/main.py --exchange bybit --all


# ── Scanning multiple exchanges simultaneously ────────────────────────
# One exchange per process — run each in its own terminal or screen session:
python main.py --exchange binance --top 100 --log-file logs/binance.log
python main.py --exchange bybit   --top 100 --log-file logs/bybit.log
python main.py --exchange okx     --top 100 --log-file logs/okx.log

# Or run all in background with nohup:
nohup python main.py --exchange binance --top 100 --log-file logs/binance.log > /dev/null 2>&1 &
nohup python main.py --exchange bybit   --top 100 --log-file logs/bybit.log   > /dev/null 2>&1 &
nohup python main.py --exchange okx     --top 100 --log-file logs/okx.log     > /dev/null 2>&1 &
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
| `--exchange` | string | `binance` | Exchange to scan. **One exchange per process.** Choices: `binance kucoin bybit okx mexc gateio bitget` |
| `--symbols` | list | — | Specific symbols to watch (e.g. `ETH/USDT SOL/USDT`). Mutually exclusive with `--top` / `--all` |
| `--top N` | int | — | Auto-fetch the top N USDT spot pairs by 24h volume |
| `--all` | flag | off | Monitor every USDT spot pair on the exchange (500–1500+ symbols depending on exchange) |
| `--interval` | int | `60` | Seconds between scan ticks |
| `--cooldown` | int | `10` | Minutes to suppress repeat alerts for the same coin after one fires. Use `0` to disable |
| `--log-file` | path | — | Write full session log to PATH and structured alerts to PATH with `_alerts.jsonl` suffix |
| `--verbose` | flag | off | Enable DEBUG-level logging (shows per-coin scan results, fetch errors, warmup progress) |

> **One exchange per process.** To scan multiple exchanges at the same time, launch one `main.py` process per exchange (each with its own `--log-file`). See the multi-exchange examples in Quick Start above.

**Choosing a symbol selection mode:**

| Mode | Command | Best for |
|---|---|---|
| Specific coins | `--symbols ETH/USDT SOL/USDT` | You already know which coins to watch; fastest startup |
| Top N by volume | `--top 50` | Broad coverage of the most liquid markets |
| Everything | `--all` | Full market sweep; high memory + rate-limit usage — use a longer `--interval` (120s+) |

**`--all` symbol counts by exchange (approximate):**

| Exchange | USDT spot pairs |
|---|---|
| Binance | ~1,500 |
| KuCoin | ~800 |
| Bybit | ~700 |
| OKX | ~600 |
| Gate.io | ~2,000+ |
| MEXC | ~2,000+ |
| Bitget | ~700 |

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
**Weights:** `models/pump_detector_v3.pth` (generated by `scripts/train_pump_detector.py`)

PumpDetectorV3 is a **dual-stream, dual-output 1D-CNN**. Two separate CNN towers process the coin and market streams, then feed a shared trunk that splits into two independent heads.

```
x_coin   [1, 96, 6]  →  FeatureExtractor (3-block CNN)  →  [1, 128]
                                                               ↓ concat
x_market [1, 96, 6]  →  FeatureExtractor (3-block CNN)  →  [1, 128]  →  [1, 256]
                                                               ↓
                         Shared trunk: Linear(256→128) → ReLU → Dropout(0.3)
                                                               ↓
                    ┌──────────────────────────────────────────┤
                    ↓                                          ↓
          cls_head: Linear(128→1) → Sigmoid       reg_head: Linear(128→1) → Sigmoid
          pump_prob  (0.0 – 1.0)                  peak_pos  (0.0 – 1.0)
```

**`forward()` returns a tuple:** `(pump_prob, peak_pos)`

| Output | Type | Meaning |
|---|---|---|
| `pump_prob` | float [0, 1] | Probability that the 96-candle window contains a pump event |
| `peak_pos` | float [0, 1] | Normalised peak position: `peak_idx / 96`. Multiply by 95 to get the candle index. |

The regression head (`reg_head`) is trained with MSE loss on pump samples only, using `peak_pct` / `peak_idx` stored in `*_meta.json` files. For control samples the regression loss is masked to zero (no gradient). Training combines both losses: `total = BCE + 0.3 × MSE`.

Each `FeatureExtractor` block:
```
Block 1: Conv1d(6→32, k=5) + Conv1d(32→32, k=3) + MaxPool(2) + Dropout(0.1)
Block 2: Conv1d(32→64, k=5) + Conv1d(64→64, k=3) + MaxPool(2) + Dropout(0.1)
Block 3: Conv1d(64→128, k=3) + AdaptiveAvgPool1d(1)   ← collapses time axis
```

**Why dual-stream?**
The most suspicious case is when a coin spikes while the broader market (BTC) is flat. The second CNN tower sees BTC context and the classifier learns to reward this divergence.

**Trained on:**
- Binance, KuCoin, MEXC, Gate.io, Bitget — training set (max 5,000 pump windows per exchange)
- Bybit, OKX — zero-shot test set (never seen during training)

---

## 10. Peak Estimation

Once the CNN confirms a pump (prob ≥ 0.65), Stage 3 determines where in the pump cycle you are using two layers:

### Primary — CNN Regression Head

The `reg_head` output of PumpDetectorV3 gives `peak_pos ∈ [0, 1]`, which is converted to a candle index:

```python
peak_idx = round(peak_pos × 95)   # 0 = oldest candle, 95 = current candle
minutes_since_peak = 95 − peak_idx
```

This head was trained on synthetic data where the exact `peak_idx` is known, using MSE loss masked to pump samples only. It learns the temporal pattern that distinguishes the pre-peak accumulation phase from the post-peak distribution phase.

### Diagnostic — Heuristic Cross-Check (`peak_detector.py`)

Three heuristic methods are also run and included in the alert as cross-validation:

**Method 1 — Price Maximum** (`estimate_peak_price_max`)  
Scans the 96-candle mid-price series for the first index where price rose ≥ 5% from the prior low and then fell ≥ 3% to the subsequent low. Falls back to `argmax` if no qualifying peak.

**Method 2 — Order Flow Reversal** (`estimate_peak_order_flow`)  
Looks for where `buy_ratio` drops from above 0.60 to below 0.45 (3-candle smoothed). Insiders flipping from buying to selling — typically within 1–3 candles of the price peak.

**Method 4 — Imbalance Flip** (`estimate_peak_imbalance_flip`)  
Looks for where `bid_size / ask_size` crosses from above 1.0 to below 1.0 (5-candle smoothed).

These are printed in the alert under `Detectors:` for diagnostics but do **not** override the CNN's regression estimate.

**Phase labels** (set from CNN peak_idx):

| Phase | Condition | Meaning |
|---|---|---|
| `peak` | `minutes_since_peak ≤ 5` | Peak occurred within the last 5 candles — reversal imminent or just started |
| `dump` | `minutes_since_peak > 5` | Distribution phase already underway |

Note: `accumulation` (peak in the future) is not used because the 96-step buffer contains only historical data — the peak is always in the past relative to the current candle.

---

## 11. Alert Output Format

When all three stages pass, this is printed to the terminal:

```
==============================================================
  *** PUMP ALERT — SOL/USDT on BINANCE ***
  Time       : 2026-05-23T14:32:07 UTC
  Pumpable   : 82.4 / 100  (threshold 70.0)
  CNN prob   : 0.7831       (threshold 0.65)
  Phase      : PEAK
  Peak candle: 91 / 95
  Status     : AT THE PEAK — 4 min ago
  Detectors  : price_max=89  order_flow=92  imbalance_flip=90
==============================================================
```

**Field explanations:**

| Field | Description |
|---|---|
| `Pumpable` | PUMPABLE COINS score (0–100). Combines bid/ask imbalance (50%), market impact (30%), spread (20%). ≥70 to proceed. |
| `CNN prob` | PumpDetectorV3 classification head output — probability this 96-candle window contains a pump+dump cycle. ≥0.65 to alert. |
| `Phase` | `PEAK` (peak occurred ≤5 min ago) or `DUMP` (peak was >5 min ago). Set from CNN regression head output. |
| `Peak candle` | CNN-estimated peak candle index within the 96-candle window (0 = oldest candle, 95 = current candle) |
| `Status` | Minutes since the estimated peak — `AT THE PEAK` when ≤5 min, `Post-peak dump` when >5 min |
| `Detectors` | Per-method heuristic peak estimates for cross-validation diagnostics. `None` means that method couldn't find a peak. |

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

**Finding the optimal threshold:**

After training, run the calibration script on the zero-shot test set:

```bash
python scripts/calibrate_threshold.py
```

It plots the precision-recall curve for Bybit + OKX and prints the F1-maximising threshold. Update `CNN_THRESHOLD` in `live/config.py` with the result.

**Recommended starting point:** Keep defaults (70 / 0.65) for the first run to understand the false-positive rate in your target market, then calibrate.

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
| `pump_detector_v3.pth` may not exist | Crash on startup with clear error message | Run `scripts/train_pump_detector.py` first; model is not committed to the repo |
| Pump precision ~47% at 0.5 threshold | Raise threshold for production use | Use `scripts/calibrate_threshold.py` to find the optimal value; default is 0.65 |
| Peak regression accuracy depends on training data | CNN peak estimate may be off by several candles | Quality improves as more synthetic data with known `peak_idx` is generated; heuristic detectors in `Detectors:` field serve as a cross-check |
| No persistence between runs | Cooldowns and buffers reset on restart | If you restart mid-pump, the buffer refills from scratch |
