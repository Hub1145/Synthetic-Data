"""
features/extractors/pumpable_coin_extractor.py

Pumpable Coin Detection Feature Extractor
Integrates with the Trade & Orderbook Pipeline to identify pump & dump candidates

This extractor analyzes orderbook imbalance, liquidity depth, and volume patterns
to score coins based on their susceptibility to price manipulation.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class PumpableCoinExtractor:
    """
    Feature extractor for identifying pumpable coins based on:
    1. Low liquidity (narrow orderbook depth)
    2. High order imbalance (bid/ask ratio extremes)
    3. Low recent volume (5-day average)
    4. High price impact (small orders move price significantly)
    5. Wide bid-ask spreads
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the pumpable coin extractor

        Args:
            config: Configuration dictionary with:
                - depth_levels: Number of orderbook levels to analyze (default: 20)
                - buy_threshold: Ratio threshold for buy-side imbalance (default: 5.0)
                - sell_threshold: Ratio threshold for sell-side imbalance (default: 0.2)
                - min_liquidity: Minimum total USD liquidity (default: 1000)
                - max_liquidity: Maximum total USD liquidity (default: 50000)
                - volume_window_days: Days for volume averaging (default: 5)
                - impact_notional_pct: Percentage of liquidity for impact calc (default: 0.05)
        """
        config = config or {}

        self.depth_levels = config.get('depth_levels', 20)
        self.buy_threshold = config.get('buy_threshold', 5.0)
        self.sell_threshold = config.get('sell_threshold', 0.2)
        self.min_liquidity = config.get('min_liquidity', 1000)
        self.max_liquidity = config.get('max_liquidity', 50000)
        self.volume_window_days = config.get('volume_window_days', 5)
        self.impact_notional_pct = config.get('impact_notional_pct', 0.05)

        # Historical data buffers for volume tracking
        self.volume_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.volume_window_days * 1440)  # ~1 min samples for 5 days
        )
        self.last_volume_update: Dict[str, datetime] = {}

        # Historical pump scores for trend analysis
        self.pump_score_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)  # Keep last 100 scores
        )

        logger.info(
            f"PumpableCoinExtractor initialized with depth={self.depth_levels}, "
            f"buy_threshold={self.buy_threshold}, sell_threshold={self.sell_threshold}, "
            f"liquidity_range=[{self.min_liquidity}, {self.max_liquidity}]"
        )

    def extract(
        self,
        symbol: str,
        orderbook: Dict[str, Any],
        trades: List[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Extract pumpable coin features from orderbook and trade data

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            orderbook: Normalized orderbook data with 'bids' and 'asks'
            trades: Recent trades for volume analysis (optional)
            timestamp: Current timestamp

        Returns:
            Dictionary of features with 'pump_' prefix
        """
        timestamp = timestamp or datetime.utcnow()
        features = {}

        try:
            # Extract orderbook metrics
            ob_features = self._extract_orderbook_features(symbol, orderbook)
            features.update(ob_features)

            # Extract volume metrics if trades available
            if trades:
                volume_features = self._extract_volume_features(symbol, trades, timestamp)
                features.update(volume_features)

            # Calculate composite pump score
            pump_score = self._calculate_pump_score(features)
            features['pump_score'] = pump_score
            features['is_pumpable'] = self._is_pumpable(features)

            # Store score for trend analysis
            # Store score for trend analysis
            exchange_id = orderbook.get('exchange') or orderbook.get('metadata', {}).get('exchange')
            self.pump_score_history[symbol].append({
                'timestamp': timestamp,
                'score': pump_score,
                'exchange': exchange_id
            })

            # Expose exchange in features for immediate API response
            features['exchange'] = exchange_id

            # Add trend features
            trend_features = self._extract_trend_features(symbol)
            features.update(trend_features)

        except Exception as e:
            logger.error(f"Error extracting pumpable features for {symbol}: {e}")
            features = self._get_default_features()

        return features

    def _extract_orderbook_features(
        self,
        symbol: str,
        orderbook: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract features from orderbook depth and imbalance"""
        features = {}

        try:
            # EXCLUSION CHECK (New based on PDF requirements)
            # symbol_metadata can be passed via orderbook or explicitly
            metadata = orderbook.get('metadata', {})
            if metadata.get('is_margin_enabled') or metadata.get('is_leveraged'):
                logger.debug(f"Filtering {symbol} - Margin: {metadata.get('is_margin_enabled')}, Leveraged: {metadata.get('is_leveraged')}")
                return self._get_default_features()

            bids = orderbook.get('bids', [])[:self.depth_levels]
            asks = orderbook.get('asks', [])[:self.depth_levels]

            if not bids or not asks:
                return self._get_default_features()

            # Convert to numpy arrays for vectorized operations
            bid_prices = np.array([float(b[0]) for b in bids])
            bid_amounts = np.array([float(b[1]) for b in bids])
            ask_prices = np.array([float(a[0]) for a in asks])
            ask_amounts = np.array([float(a[1]) for a in asks])

            # Liquidity metrics
            bid_liquidity = np.sum(bid_prices * bid_amounts)
            ask_liquidity = np.sum(ask_prices * ask_amounts)
            total_liquidity = bid_liquidity + ask_liquidity

            features['pump_bid_liquidity'] = float(bid_liquidity)
            features['pump_ask_liquidity'] = float(ask_liquidity)
            features['pump_total_liquidity'] = float(total_liquidity)

            # Order imbalance ratio
            if ask_liquidity > 0:
                buy_sell_ratio = bid_liquidity / ask_liquidity
            else:
                buy_sell_ratio = 999.0  # Extreme imbalance

            features['pump_buy_sell_ratio'] = float(buy_sell_ratio)

            # Normalized imbalance (-1 to +1 scale)
            features['pump_imbalance_normalized'] = float(
                (bid_liquidity - ask_liquidity) / max(total_liquidity, 1.0)
            )

            # Mid price and spread
            best_bid = bid_prices[0]
            best_ask = ask_prices[0]
            mid_price = (best_bid + best_ask) / 2
            spread_pct = ((best_ask - best_bid) / mid_price) * 100 if mid_price > 0 else 0

            features['pump_mid_price'] = float(mid_price)
            features['pump_spread_pct'] = float(spread_pct)

            # Price impact calculation (5% of liquidity on each side)
            target_bid = bid_liquidity * self.impact_notional_pct
            target_ask = ask_liquidity * self.impact_notional_pct

            buy_impact = self._calculate_price_impact(
                ask_prices, ask_amounts, target_ask, mid_price
            )
            sell_impact = self._calculate_price_impact(
                bid_prices, bid_amounts, target_bid, mid_price, direction='sell'
            )

            features['pump_buy_impact_pct'] = float(buy_impact)
            features['pump_sell_impact_pct'] = float(sell_impact)
            features['pump_max_impact_pct'] = float(max(buy_impact, sell_impact))

            # Depth concentration (what % of liquidity in top 5 levels)
            top5_bid_liq = np.sum(bid_prices[:5] * bid_amounts[:5])
            top5_ask_liq = np.sum(ask_prices[:5] * ask_amounts[:5])

            features['pump_depth_concentration_bid'] = float(
                top5_bid_liq / max(bid_liquidity, 1.0)
            )
            features['pump_depth_concentration_ask'] = float(
                top5_ask_liq / max(ask_liquidity, 1.0)
            )

            # Liquidity flag (within pumpable range)
            features['pump_liquidity_in_range'] = float(
                self.min_liquidity <= total_liquidity <= self.max_liquidity
            )

        except Exception as e:
            logger.error(f"Error extracting orderbook features: {e}")
            return self._get_default_features()

        return features

    def _calculate_price_impact(
        self,
        prices: np.ndarray,
        amounts: np.ndarray,
        target_notional: float,
        mid_price: float,
        direction: str = 'buy'
    ) -> float:
        """
        Calculate price impact for consuming target notional amount

        Args:
            prices: Price levels
            amounts: Amounts at each level
            target_notional: USD amount to consume
            mid_price: Current mid price
            direction: 'buy' or 'sell'

        Returns:
            Price impact as percentage
        """
        try:
            if mid_price <= 0:
                return 0.0

            notional_values = prices * amounts
            cumulative_notional = np.cumsum(notional_values)

            # Find where we exceed target
            idx = np.searchsorted(cumulative_notional, target_notional)

            if idx >= len(prices):
                # Would consume entire orderbook
                execution_price = prices[-1]
            else:
                execution_price = prices[idx]

            # Calculate percentage impact
            impact_pct = abs((execution_price / mid_price - 1.0) * 100)

            return impact_pct

        except Exception as e:
            logger.error(f"Error calculating price impact: {e}")
            return 0.0

    def _extract_volume_features(
        self,
        symbol: str,
        trades: List[Dict[str, Any]],
        timestamp: datetime
    ) -> Dict[str, float]:
        """Extract volume-based features from recent trades"""
        features = {}

        try:
            # Normalize trade input which might be a DataFrame or list of dicts/NormalizedTrade
            current_volume = 0.0
            if isinstance(trades, pd.DataFrame):
                 if not trades.empty:
                    # Assuming columns price and volume exist
                    current_volume = (trades['price'] * trades['volume']).sum()
            elif isinstance(trades, list):
                for t in trades:
                    # Handle both dictionary and object access
                    price = float(t.get('price', 0)) if isinstance(t, dict) else float(getattr(t, 'price', 0))
                    amount = float(t.get('amount', t.get('volume', 0))) if isinstance(t, dict) else float(getattr(t, 'volume', getattr(t, 'amount', 0)))
                    current_volume += price * amount

            self.volume_history[symbol].append({
                'timestamp': timestamp,
                'volume': current_volume
            })
            self.last_volume_update[symbol] = timestamp

            # Calculate average volume over window
            cutoff = timestamp - timedelta(days=self.volume_window_days)
            recent_volumes = [
                v['volume'] for v in self.volume_history[symbol]
                if v['timestamp'] >= cutoff
            ]

            if recent_volumes:
                avg_volume = float(np.mean(recent_volumes))
                volume_std = float(np.std(recent_volumes))

                features['pump_volume_5d_avg'] = avg_volume
                features['pump_volume_5d_std'] = volume_std
                features['pump_volume_current'] = float(current_volume)

                # Volume z-score (how unusual is current volume)
                if volume_std > 0:
                    features['pump_volume_zscore'] = float(
                        (current_volume - avg_volume) / volume_std
                    )
                else:
                    features['pump_volume_zscore'] = 0.0

                # Low volume flag (indicator of pumpability)
                # Use dynamic threshold if provided, else fallback to 10000
                volume_thresh = features.get('pump_volume_threshold', 10000)
                features['pump_low_volume_flag'] = float(avg_volume < volume_thresh)

        except Exception as e:
            logger.error(f"Error extracting volume features: {e}")

        return features

    def _extract_trend_features(self, symbol: str) -> Dict[str, float]:
        """Extract trend features from pump score history"""
        features = {}

        try:
            if len(self.pump_score_history[symbol]) < 2:
                features['pump_score_trend'] = 0.0
                features['pump_score_acceleration'] = 0.0
                return features

            scores = [s['score'] for s in self.pump_score_history[symbol]]

            # Simple linear trend
            x = np.arange(len(scores))
            if len(scores) > 1:
                trend = np.polyfit(x, scores, 1)[0]
            else:
                trend = 0.0

            features['pump_score_trend'] = float(trend)

            # Acceleration (second derivative)
            if len(scores) >= 3:
                acceleration = scores[-1] - 2*scores[-2] + scores[-3]
                features['pump_score_acceleration'] = float(acceleration)
            else:
                features['pump_score_acceleration'] = 0.0

        except Exception as e:
            logger.error(f"Error extracting trend features: {e}")

        return features

    def _calculate_pump_score(self, features: Dict[str, float]) -> float:
        """
        Calculate composite pump score (0-100 scale)

        Weighted combination of:
        - Order imbalance (40%)
        - Price impact (30%)
        - Spread (20%)
        - Liquidity range (10%)
        """
        try:
            # Normalize buy/sell ratio to score (higher = more extreme)
            ratio = features.get('pump_buy_sell_ratio', 1.0)
            if ratio > 1:
                imbalance_score = min(ratio / self.buy_threshold, 2.0) * 50
            else:
                imbalance_score = min((1/ratio) / (1/self.sell_threshold), 2.0) * 50

            # Price impact score (0-100, higher = more impact)
            impact = features.get('pump_max_impact_pct', 0.0)
            impact_score = min(impact / 5.0, 2.0) * 50  # 5% impact = 50 points

            # Spread score (0-100, higher = wider spread)
            spread = features.get('pump_spread_pct', 0.0)
            spread_score = min(spread / 1.0, 2.0) * 50  # 1% spread = 50 points

            # Liquidity score (100 if in range, 0 otherwise)
            liquidity_score = features.get('pump_liquidity_in_range', 0.0) * 100

            # Weighted composite (Updated per PDF: 50/30/20)
            # Liquidity score is now a multiplier/pre-filter
            pump_score = (
                imbalance_score * 0.5 +
                impact_score * 0.3 +
                spread_score * 0.2
            )

            # Apply liquidity range as a binary mask
            if not features.get('pump_liquidity_in_range', 0.0):
                pump_score = 0.0

            return min(pump_score, 100.0)

        except Exception as e:
            logger.error(f"Error calculating pump score: {e}")
            return 0.0

    def _is_pumpable(self, features: Dict[str, float]) -> float:
        """
        Binary classification: is this coin pumpable?

        Returns 1.0 if pumpable, 0.0 otherwise
        """
        try:
            # Criteria for pumpability
            ratio = features.get('pump_buy_sell_ratio', 1.0)
            impact = features.get('pump_max_impact_pct', 0.0)
            liquidity = features.get('pump_total_liquidity', float('inf'))

            # Strong imbalance OR high impact, BUT must be within liquidity range
            is_liquid = self.min_liquidity <= liquidity <= self.max_liquidity

            # Logic: Must be liquid AND have either high imbalance or high impact
            pumpable = is_liquid and (
                (ratio > self.buy_threshold or (ratio > 0 and ratio < self.sell_threshold)) or
                (impact > 3.0)
            )

            return 1.0 if pumpable else 0.0

        except Exception as e:
            logger.error(f"Error determining pumpability: {e}")
            return 0.0

    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when extraction fails"""
        return {
            'pump_bid_liquidity': 0.0,
            'pump_ask_liquidity': 0.0,
            'pump_total_liquidity': 0.0,
            'pump_buy_sell_ratio': 1.0,
            'pump_imbalance_normalized': 0.0,
            'pump_mid_price': 0.0,
            'pump_spread_pct': 0.0,
            'pump_buy_impact_pct': 0.0,
            'pump_sell_impact_pct': 0.0,
            'pump_max_impact_pct': 0.0,
            'pump_depth_concentration_bid': 0.0,
            'pump_depth_concentration_ask': 0.0,
            'pump_liquidity_in_range': 0.0,
            'pump_score': 0.0,
            'is_pumpable': 0.0,
            'pump_score_trend': 0.0,
            'pump_score_acceleration': 0.0
        }

    def get_top_pumpable(
        self,
        symbols: List[str],
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top N most pumpable coins based on recent scores

        Args:
            symbols: List of symbols to rank
            top_n: Number of top coins to return

        Returns:
            List of dicts with symbol and pump_score
        """
        rankings = []

        for symbol in symbols:
            if symbol in self.pump_score_history and self.pump_score_history[symbol]:
                latest_score = self.pump_score_history[symbol][-1]['score']
                rankings.append({
                    'symbol': symbol,
                    'pump_score': latest_score,
                    'exchange': self.pump_score_history[symbol][-1].get('exchange'),
                    'timestamp': self.pump_score_history[symbol][-1]['timestamp']
                })

        # Sort by score descending
        rankings.sort(key=lambda x: x['pump_score'], reverse=True)

        return rankings[:top_n]
