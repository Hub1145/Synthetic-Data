"""
Peak estimation for a confirmed pump window.

Four independent methods combined into one consensus estimate.
Full documentation: Synthetic Data/detector.md §6 "Peak Estimation During a Pump"
"""

from typing import Optional
import numpy as np


def estimate_peak_price_max(
    mid_prices: np.ndarray,
    min_rise: float = 0.05,
    min_drop: float = 0.03,
) -> int:
    """
    Method 1: Highest mid-price satisfying a pre-rise and post-drop condition.

    Finds the first index where:
      - Price rose >= min_rise (5%) from the prior minimum
      - Price fell >= min_drop (3%) from that point to the subsequent minimum

    Falls back to argmax if no qualifying peak found.
    """
    prices = mid_prices
    for i in range(5, len(prices) - 5):
        prior_min = prices[:i].min()
        pre_gain = (prices[i] - prior_min) / prior_min if prior_min > 0 else 0.0
        post_drop = (prices[i] - prices[i:].min()) / prices[i] if prices[i] > 0 else 0.0
        if pre_gain >= min_rise and post_drop >= min_drop:
            return i
    return int(np.argmax(prices))


def estimate_peak_order_flow(buy_ratios: np.ndarray) -> Optional[int]:
    """
    Method 2: Order-flow reversal — buy_ratio drops from >0.60 to <0.45.

    Pre-peak: aggressive buyers cross the spread → buy_ratio ≈ 0.70–0.85.
    At peak: insiders flip to selling into retail → buy_ratio collapses to 0.20–0.40.
    Crossover typically leads or coincides with the price max by 1–3 candles.

    Returns None if no clear crossover is found (e.g. no trades data).
    """
    kernel = np.ones(3) / 3
    smooth = np.convolve(buy_ratios, kernel, mode="same")
    for i in range(1, len(smooth)):
        if smooth[i - 1] > 0.60 and smooth[i] < 0.45:
            return i
    return None


def estimate_peak_imbalance_flip(
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
) -> Optional[int]:
    """
    Method 4: Bid/ask size imbalance crosses below 1.0 — orderbook-only method.

    Pre-peak: bid_size inflated (accumulation) → imbalance > 1.
    Post-peak: ask_size inflated (distribution) → imbalance < 1.
    The crossing from >1 to <1 is the orderbook signature of the peak.

    Returns None if the flip never occurs in the window.
    """
    imbalance = bid_sizes / (ask_sizes + 1e-9)
    kernel = np.ones(5) / 5
    smooth = np.convolve(imbalance, kernel, mode="same")
    for i in range(1, len(smooth)):
        if smooth[i - 1] > 1.0 and smooth[i] < 1.0:
            return i
    return None


def estimate_peak(
    mid_prices: np.ndarray,
    buy_ratios: np.ndarray,
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
    has_trades: bool = True,
) -> dict:
    """
    Combined peak estimator — median of all available method estimates.

    Args:
        mid_prices:     [96] bid/ask midpoint series
        buy_ratios:     [96] buy volume fraction per candle (0–1)
        bid_sizes:      [96] bid depth at level 1
        ask_sizes:      [96] ask depth at level 1
        has_trades:     skip Method 2 if trade-flow features are unreliable

    Returns:
        peak_idx        consensus peak candle index (0-based within the 96-step window)
        phase           "accumulation" | "peak" | "dump"
        minutes_to_peak estimated minutes until peak (0 if at or past peak)
        methods         per-method raw estimates for diagnostics
    """
    peak_price = estimate_peak_price_max(mid_prices)
    peak_flow = estimate_peak_order_flow(buy_ratios) if has_trades else None
    peak_imb = estimate_peak_imbalance_flip(bid_sizes, ask_sizes)

    candidates = [p for p in [peak_price, peak_flow, peak_imb] if p is not None]
    peak_idx = int(np.median(candidates)) if candidates else len(mid_prices) - 1

    # All 96 candles are past data, so current_t is always the last index (95).
    # "accumulation" (peak hasn't happened yet) is impossible with a historical window.
    # Phase is determined by how recently the peak occurred within the window:
    #   ≤ 5 candles ago  → "peak"  (just happened, may still be actionable)
    #   > 5 candles ago  → "dump"  (distribution already underway)
    minutes_since_peak = (len(mid_prices) - 1) - peak_idx

    if minutes_since_peak <= 5:
        phase = "peak"
    else:
        phase = "dump"

    return {
        "peak_idx": peak_idx,
        "phase": phase,
        "minutes_since_peak": minutes_since_peak,
        "methods": {
            "price_max": peak_price,
            "order_flow": peak_flow,
            "imbalance_flip": peak_imb,
        },
    }
