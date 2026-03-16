"""Market regime detection module."""

from __future__ import annotations

import logging

import pandas as pd

from app.core.models import MarketRegime

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detect the current market regime from indicator data.

    Regimes (ordered by priority):
        Trending:        ADX > 25, EMA20 slope confirms direction
        Range-bound:     ADX < 20, price range < 0.5% over lookback
        High Volatility: ATR rising AND VIX elevated
        Low Volatility:  ATR falling AND VIX calm
    """

    ADX_TRENDING_THRESHOLD = 22       # Lowered from 25 — catches trends earlier
    ADX_RANGE_THRESHOLD = 20          # Raised from 18 — more forgiving range detection
    RANGE_PCT_THRESHOLD = 0.50        # Raised from 0.35% — NIFTY ranges can be wider
    RANGE_LOOKBACK = 60
    MIN_CANDLES = 15                  # Lowered from 60 — system can start detecting sooner

    def detect(self, df: pd.DataFrame, vix_rising: bool = False) -> MarketRegime:
        """Determine market regime from the latest indicator DataFrame."""
        if df.empty or len(df) < self.MIN_CANDLES:
            return MarketRegime.INSUFFICIENT_DATA

        last = df.iloc[-1]
        adx = last.get("adx")
        ema20_slope = last.get("ema20_slope")
        atr_slope = last.get("atr_slope")

        # If core indicators are missing, can't determine regime
        if adx is None or (isinstance(adx, float) and pd.isna(adx)):
            return MarketRegime.INSUFFICIENT_DATA

        # Check trending — EMA slope confirms direction but isn't strictly required
        slope = ema20_slope if (ema20_slope is not None and not pd.isna(ema20_slope)) else 0
        if adx > self.ADX_TRENDING_THRESHOLD:
            if slope != 0:
                logger.info("Regime: TRENDING (ADX=%.1f, EMA20 slope=%.4f)", adx, slope)
            else:
                logger.info("Regime: TRENDING (ADX=%.1f, slope unavailable)", adx)
            return MarketRegime.TRENDING

        # Check range-bound — use available candles up to RANGE_LOOKBACK
        lookback = min(self.RANGE_LOOKBACK, len(df))
        recent = df.tail(lookback)
        price_range_pct = (
            (recent["high"].max() - recent["low"].min()) / recent["close"].iloc[0] * 100
        )
        if adx < self.ADX_RANGE_THRESHOLD and price_range_pct < self.RANGE_PCT_THRESHOLD:
            logger.info("Regime: RANGE_BOUND (ADX=%.1f, range=%.2f%%)", adx, price_range_pct)
            return MarketRegime.RANGE_BOUND

        # Check volatility regimes
        atr_s = atr_slope if (atr_slope is not None and not pd.isna(atr_slope)) else 0
        if atr_s > 0 and vix_rising:
            logger.info("Regime: HIGH_VOLATILITY (ATR slope=%.4f, VIX rising)", atr_s)
            return MarketRegime.HIGH_VOLATILITY

        if atr_s < 0 and not vix_rising:
            logger.info("Regime: LOW_VOLATILITY (ATR slope=%.4f)", atr_s)
            return MarketRegime.LOW_VOLATILITY

        # Between thresholds but has real data — default to range-bound
        return MarketRegime.RANGE_BOUND
