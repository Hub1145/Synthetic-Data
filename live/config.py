"""
Paths and thresholds for the live inference pipeline.
Lives inside Synthetic Data/live/ — models/ folder is one level up.
"""

import sys
from pathlib import Path

# Project root = Synthetic Data/
ROOT = Path(__file__).parent.parent   # go up from live/ → Synthetic Data/
LIVE_DIR = Path(__file__).parent

# PUMPABLE COINS project is a sibling of Synthetic Data/ (local dev).
# On server deployments copy pumpable_coin_extractor.py into live/ — the fallback handles it.
PUMPABLE_ROOT = ROOT.parent / "PUMPABLE COINS"
if not PUMPABLE_ROOT.is_dir():
    PUMPABLE_ROOT = LIVE_DIR  # pumpable_coin_extractor.py lives alongside cascade.py

# Inject paths so models/ and PUMPABLE COINS are importable
sys.path.insert(0, str(ROOT))           # gives access to models/pump_detector.py
sys.path.insert(0, str(PUMPABLE_ROOT))  # gives access to pumpable_coin_extractor.py

# ── Model weights ──────────────────────────────────────────────────────────────
MODEL_PATH = ROOT / "models" / "pump_detector_v3.pth"

# ── Cascade decision thresholds ────────────────────────────────────────────────
PUMPABLE_THRESHOLD = 70.0   # PumpableCoinExtractor score required to proceed
CNN_THRESHOLD = 0.65         # PumpDetectorV3 probability required to fire alert

# ── Window configuration ───────────────────────────────────────────────────────
WINDOW_SIZE = 96             # 96 one-minute candles = 1h 36m of context
NUM_COIN_FEATURES = 6        # bid_price, ask_price, bid_size, ask_size, buy_ratio, aggressor_imbalance
NUM_MARKET_FEATURES = 6

# ── Market proxy ───────────────────────────────────────────────────────────────
MARKET_SYMBOL = "BTC/USDT"   # Market-context stream for PumpDetectorV3 dual-stream input

# ── Scan timing ────────────────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = 60

# ── Exchanges available for live scanning ──────────────────────────────────────
EXCHANGES = ["binance", "kucoin", "bybit", "okx", "mexc", "gateio", "bitget"]
