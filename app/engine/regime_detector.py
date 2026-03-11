"""Market regime detection module."""

from __future__ import annotations

import logging

import pandas as pd

from app.core.models import MarketRegime

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detect the current market regime from indicator data.

    Regimes:
        Trending:       ADX > 25, EMA20 slope > 0
        Range-bound:    ADX < 18, price range < 0.35% over 60 candles
        High Volatility: ATR rising, VIX rising
        Low Volatility:  ATR falling, VIX falling
    """

    ADX_TRENDING_THRESHOLD = 25
    ADX_RANGE_THRESHOLD = 18
    RANGE_PCT_THRESHOLD = 0.35
    RANGE_LOOKBACK = 60

    def detect(self, df: pd.DataFrame, vix_rising: bool = False) -> MarketRegime:
        """Determine market regime from the latest indicator DataFrame."""
        if df.empty or len(df) < self.RANGE_LOOKBACK:
            return MarketRegime.RANGE_BOUND

        last = df.iloc[-1]
        adx = last.get("adx", 20)
        ema20_slope = last.get("ema20_slope", 0)
        atr_slope = last.get("atr_slope", 0)

        # Check trending
        if adx is not None and adx > self.ADX_TRENDING_THRESHOLD and ema20_slope > 0:
            logger.info("Regime: TRENDING (ADX=%.1f, EMA20 slope=%.4f)", adx, ema20_slope)
            return MarketRegime.TRENDING

        # Check range-bound
        recent = df.tail(self.RANGE_LOOKBACK)
        price_range_pct = (
            (recent["high"].max() - recent["low"].min()) / recent["close"].iloc[0] * 100
        )
        if adx is not None and adx < self.ADX_RANGE_THRESHOLD and price_range_pct < self.RANGE_PCT_THRESHOLD:
            logger.info("Regime: RANGE_BOUND (ADX=%.1f, range=%.2f%%)", adx, price_range_pct)
            return MarketRegime.RANGE_BOUND

        # Check volatility regimes
        if atr_slope is not None and atr_slope > 0:
            if vix_rising:
                logger.info("Regime: HIGH_VOLATILITY (ATR slope=%.4f, VIX rising)", atr_slope)
                return MarketRegime.HIGH_VOLATILITY

        if atr_slope is not None and atr_slope < 0 and not vix_rising:
            logger.info("Regime: LOW_VOLATILITY (ATR slope=%.4f)", atr_slope)
            return MarketRegime.LOW_VOLATILITY

        return MarketRegime.RANGE_BOUND
