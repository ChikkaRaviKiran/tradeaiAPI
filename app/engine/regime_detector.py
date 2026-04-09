"""LOCKED v1.0 Market regime detection — 3 regimes only.

TREND:    ADX > 20
RANGE:    ADX < 18
VOLATILE: Between thresholds + VIX spike or large candles

DO NOT CHANGE for 10-15 trading days.
"""

from __future__ import annotations

import logging

import pandas as pd

from app.core.models import MarketRegime

logger = logging.getLogger(__name__)


class RegimeDetector:
    """LOCKED v1.0: Detect market regime — TREND, RANGE, or VOLATILE.

    Recalculated every 30 minutes by the orchestrator.
    """

    ADX_TREND_THRESHOLD = 20
    ADX_RANGE_THRESHOLD = 18
    MIN_CANDLES = 20

    def detect(self, df: pd.DataFrame, vix_rising: bool = False) -> MarketRegime:
        """Determine market regime from the latest indicator DataFrame."""
        if df.empty or len(df) < self.MIN_CANDLES:
            return MarketRegime.RANGE_BOUND  # Default to RANGE if insufficient data

        last = df.iloc[-1]
        adx = last.get("adx")

        # If ADX unavailable, default to RANGE
        if adx is None or (isinstance(adx, float) and pd.isna(adx)):
            return MarketRegime.RANGE_BOUND

        # TREND: ADX > 20
        if adx > self.ADX_TREND_THRESHOLD:
            logger.info("Regime: TREND (ADX=%.1f)", adx)
            return MarketRegime.TRENDING

        # RANGE: ADX < 18
        if adx < self.ADX_RANGE_THRESHOLD:
            logger.info("Regime: RANGE (ADX=%.1f)", adx)
            return MarketRegime.RANGE_BOUND

        # VOLATILE: ADX between 18-20 — check for VIX spike or large candles
        atr = last.get("atr")
        high = last.get("high", 0)
        low = last.get("low", 0)
        candle_range = high - low

        large_candle = False
        if atr and atr > 0 and candle_range > 0:
            large_candle = (candle_range / atr) > 1.5

        if vix_rising or large_candle:
            logger.info("Regime: VOLATILE (ADX=%.1f, VIX_rising=%s, large_candle=%s)", adx, vix_rising, large_candle)
            return MarketRegime.HIGH_VOLATILITY

        # Default: RANGE
        logger.info("Regime: RANGE (ADX=%.1f, between thresholds)", adx)
        return MarketRegime.RANGE_BOUND
