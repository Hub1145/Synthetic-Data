"""
Live detection cascade:

  Stage 1 — PumpableCoinExtractor snapshot score  (threshold: PUMPABLE_THRESHOLD = 70)
  Stage 2 — PumpDetectorV3 CNN probability         (threshold: CNN_THRESHOLD = 0.65)
  Stage 3 — Peak estimation                        (phase + minutes to peak)

A rolling deque of 96 one-minute feature vectors is kept per coin.
Every scan tick the latest candle is pushed in. Once the buffer is full the
three-stage cascade runs and an alert is printed to the terminal.
"""

import sys
import time
import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import ccxt.async_support as ccxt_async

from config import (
    MODEL_PATH, PUMPABLE_ROOT,
    PUMPABLE_THRESHOLD, CNN_THRESHOLD,
    WINDOW_SIZE, NUM_COIN_FEATURES, NUM_MARKET_FEATURES,
    MARKET_SYMBOL,
)
from peak_detector import estimate_peak

sys.path.insert(0, str(PUMPABLE_ROOT))
from pumpable_coin_extractor import PumpableCoinExtractor   # noqa: E402
from models.pump_detector import PumpDetectorV3              # noqa: E402

logger = logging.getLogger(__name__)


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model() -> PumpDetectorV3:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}\n"
            "Run the training pipeline in scripts/ first to generate pump_detector_v3.pth"
        )
    model = PumpDetectorV3(
        num_coin_features=NUM_COIN_FEATURES,
        num_market_features=NUM_MARKET_FEATURES,
    )
    state = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded PumpDetectorV3 from {MODEL_PATH}")
    return model


# ── Feature engineering ────────────────────────────────────────────────────────

def _ohlcv_to_features(candle: list) -> np.ndarray:
    """
    Convert [ts, open, high, low, close, volume] → 6 model features.

    buy_ratio is derived from the candle body position — the same formula
    used in generate_missing_trades.py during batch training:
        buy_ratio = (close - low) / (high - low + ε)

    bid/ask prices and sizes are approximated from close and volume so
    live OHLCV feeds can be used without a real L2 snapshot per tick.
    """
    ts, o, h, l, c, vol = candle[:6]
    denom = h - l + 1e-9
    buy_ratio = float(np.clip((c - l) / denom, 0.0, 1.0))
    agg_imb = 2.0 * buy_ratio - 1.0   # [0,1] → [-1,+1]

    bid_price = c * 0.9995
    ask_price = c * 1.0005
    bid_size = vol * buy_ratio
    ask_size = vol * (1.0 - buy_ratio)

    return np.array(
        [bid_price, ask_price, bid_size, ask_size, buy_ratio, agg_imb],
        dtype=np.float32,
    )


def _normalize_window(window: np.ndarray) -> np.ndarray:
    """
    Per-window normalization to match the training pipeline.
      - Prices  ÷ first mid-price  → ≈ 1.0 ± small %
      - Sizes   ÷ window mean size → relative depth
      - buy_ratio, aggressor_imbalance: no normalization
    """
    w = window.copy()
    mid0 = (w[0, 0] + w[0, 1]) / 2.0
    if mid0 > 0:
        w[:, 0:2] /= mid0
    mean_sz = w[:, 2:4].mean()
    if mean_sz > 0:
        w[:, 2:4] /= mean_sz
    return w


# ── Per-coin rolling buffer ────────────────────────────────────────────────────

class CoinMonitor:
    """Maintains a rolling 96-step feature buffer for a single coin."""

    def __init__(self, symbol: str, exchange: str):
        self.symbol = symbol
        self.exchange = exchange
        self.buffer: deque = deque(maxlen=WINDOW_SIZE)
        self.last_alert_ts: float = 0.0

    def push(self, features: np.ndarray) -> None:
        self.buffer.append(features)

    def ready(self) -> bool:
        return len(self.buffer) == WINDOW_SIZE

    def get_window(self) -> np.ndarray:
        return np.stack(list(self.buffer))   # [96, 6]


# ── Main cascade ───────────────────────────────────────────────────────────────

class LiveCascade:
    """
    Runs the 3-stage pump detection cascade across all monitored symbols.

    Usage:
        cascade = LiveCascade("binance", ["ETH/USDT", "SOL/USDT"])
        asyncio.run(cascade.run(interval=60))
    """

    def __init__(
        self,
        exchange_id: str,
        symbols: List[str],
        cooldown_minutes: int = 10,
    ):
        self.exchange_id = exchange_id
        self.symbols = symbols
        self.cooldown_seconds = cooldown_minutes * 60

        self.model = _load_model()
        self.extractor = PumpableCoinExtractor()

        self.monitors: Dict[str, CoinMonitor] = {
            s: CoinMonitor(s, exchange_id) for s in symbols
        }
        self.market_buffer: deque = deque(maxlen=WINDOW_SIZE)

        exc_class = getattr(ccxt_async, exchange_id)
        self.exchange = exc_class({"enableRateLimit": True})

    # ── Data fetching ──────────────────────────────────────────────────────────

    async def _fetch_candle(self, symbol: str) -> Optional[list]:
        """Return the most recent fully-closed 1-minute candle."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, "1m", limit=2)
            if not ohlcv:
                return None
            # Index -2 is the last completed candle; -1 may still be forming
            return ohlcv[-2] if len(ohlcv) >= 2 else ohlcv[-1]
        except Exception as e:
            logger.debug(f"OHLCV fetch [{symbol}]: {e}")
            return None

    async def _fetch_orderbook(self, symbol: str) -> Optional[Dict]:
        """Fetch a shallow orderbook snapshot for the PumpableCoinExtractor."""
        try:
            ob = await self.exchange.fetch_order_book(symbol, limit=20)
            return {"bids": ob.get("bids", []), "asks": ob.get("asks", [])}
        except Exception as e:
            logger.debug(f"Orderbook fetch [{symbol}]: {e}")
            return None

    # ── CNN inference ──────────────────────────────────────────────────────────

    def _run_cnn(self, coin_buf: np.ndarray, mkt_buf: np.ndarray) -> float:
        coin_norm = _normalize_window(coin_buf)
        mkt_norm = _normalize_window(mkt_buf)
        x_coin = torch.FloatTensor(coin_norm).unsqueeze(0)   # [1, 96, 6]
        x_mkt = torch.FloatTensor(mkt_norm).unsqueeze(0)     # [1, 96, 6]
        with torch.no_grad():
            return float(self.model(x_coin, x_mkt).item())

    # ── Per-coin scan ──────────────────────────────────────────────────────────

    async def _scan_coin(self, symbol: str) -> Optional[Dict]:
        if symbol == MARKET_SYMBOL:
            return None   # BTC is the market proxy, not a pump candidate

        monitor = self.monitors[symbol]

        candle = await self._fetch_candle(symbol)
        if candle is None:
            return None
        monitor.push(_ohlcv_to_features(candle))

        if not monitor.ready():
            return None   # buffer still warming up (need 96 candles first)

        # Stage 1: pumpable score
        orderbook = await self._fetch_orderbook(symbol)
        if orderbook is None:
            return None

        result = self.extractor.extract(symbol, orderbook)
        pump_score = float(result.get("pump_score", 0.0))
        if pump_score < PUMPABLE_THRESHOLD:
            return None

        # Stage 2: CNN (coin stream + market stream)
        coin_array = monitor.get_window()
        if len(self.market_buffer) == WINDOW_SIZE:
            mkt_array = np.stack(list(self.market_buffer))
        else:
            # Market buffer still warming — neutral fill
            mkt_array = np.zeros((WINDOW_SIZE, NUM_MARKET_FEATURES), dtype=np.float32)
            mkt_array[:, 0] = 1.0   # bid_price ≈ 1 post-normalization
            mkt_array[:, 1] = 1.0   # ask_price ≈ 1
            mkt_array[:, 4] = 0.5   # buy_ratio neutral

        prob = self._run_cnn(coin_array, mkt_array)
        if prob < CNN_THRESHOLD:
            return None

        # Cooldown — suppress repeat alerts for the same coin
        now = time.time()
        if now - monitor.last_alert_ts < self.cooldown_seconds:
            return None
        monitor.last_alert_ts = now

        # Stage 3: peak estimation
        mid_prices = (coin_array[:, 0] + coin_array[:, 1]) / 2.0
        peak_info = estimate_peak(
            mid_prices=mid_prices,
            buy_ratios=coin_array[:, 4],
            bid_sizes=coin_array[:, 2],
            ask_sizes=coin_array[:, 3],
            has_trades=True,
        )

        return {
            "symbol": symbol,
            "exchange": self.exchange_id,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "pump_score": round(pump_score, 2),
            "cnn_prob": round(prob, 4),
            "peak_idx": peak_info["peak_idx"],
            "phase": peak_info["phase"],
            "minutes_since_peak": peak_info["minutes_since_peak"],
            "methods": peak_info["methods"],
        }

    async def _update_market_buffer(self) -> None:
        candle = await self._fetch_candle(MARKET_SYMBOL)
        if candle is not None:
            self.market_buffer.append(_ohlcv_to_features(candle))

    # ── Full scan tick ─────────────────────────────────────────────────────────

    async def scan_once(self) -> List[Dict]:
        """Run one full scan over all symbols. Returns list of alerts."""
        await self._update_market_buffer()
        tasks = [self._scan_coin(s) for s in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        alerts = []
        for r in results:
            if isinstance(r, Exception):
                logger.debug(f"Error: {r}")
            elif r is not None:
                alerts.append(r)
        return alerts

    # ── Main loop ──────────────────────────────────────────────────────────────

    async def run(self, interval: int = 60) -> None:
        """Scan every `interval` seconds until Ctrl+C."""
        logger.info(
            f"Cascade started | exchange={self.exchange_id} "
            f"| symbols={len(self.symbols)} | interval={interval}s"
        )
        try:
            while True:
                t0 = time.time()
                alerts = await self.scan_once()
                elapsed = time.time() - t0

                for alert in alerts:
                    _print_alert(alert)

                wait = max(0.0, interval - elapsed)
                logger.debug(f"Scan done in {elapsed:.1f}s — next in {wait:.0f}s")
                await asyncio.sleep(wait)
        finally:
            await self.exchange.close()


# ── Alert printer ──────────────────────────────────────────────────────────────

def _print_alert(alert: Dict) -> None:
    if alert["phase"] == "peak":
        phase_note = f"AT THE PEAK — {alert['minutes_since_peak']} min ago"
    else:
        phase_note = f"Post-peak dump — {alert['minutes_since_peak']} min since peak"

    m = alert.get("methods", {})
    method_str = (
        f"price_max={m.get('price_max')}  "
        f"order_flow={m.get('order_flow')}  "
        f"imbalance_flip={m.get('imbalance_flip')}"
    )

    print(
        f"\n{'='*62}\n"
        f"  *** PUMP ALERT — {alert['symbol']} on {alert['exchange'].upper()} ***\n"
        f"  Time       : {alert['timestamp']} UTC\n"
        f"  Pumpable   : {alert['pump_score']:.1f} / 100  (threshold {PUMPABLE_THRESHOLD})\n"
        f"  CNN prob   : {alert['cnn_prob']:.4f}          (threshold {CNN_THRESHOLD})\n"
        f"  Phase      : {alert['phase'].upper()}\n"
        f"  Peak candle: {alert['peak_idx']} / {WINDOW_SIZE - 1}\n"
        f"  Status     : {phase_note}\n"
        f"  Detectors  : {method_str}\n"
        f"{'='*62}",
        flush=True,
    )
