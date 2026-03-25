"""Strategy 5 — Range Breakout.

Range condition:
  ADX < 22, price range < 0.5% for 60 minutes

Breakout:
  Volume ≥ 1.5× avg
  RSI ≥ 55 (CALL), RSI ≤ 45 (PUT)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from app.core.models import OptionType, OptionsMetrics, StrategyName, StrategySignal
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

RANGE_LOOKBACK = 30  # Was 60 — 30 min range is sufficient
ADX_THRESHOLD = 25  # Was 22 — realistic range days often have ADX 20-25
RANGE_PCT_THRESHOLD = 0.80  # Was 0.65% — allow wider consolidation before breakout


class RangeBreakoutStrategy(BaseStrategy):
    """Range Breakout strategy."""

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
    ) -> Optional[StrategySignal]:
        if df.empty or len(df) < RANGE_LOOKBACK + 1:
            return None

        # Need enough candles for the lookback
        effective_lookback = min(RANGE_LOOKBACK, len(df) - 1)

        last = df.iloc[-1]
        adx = last.get("adx")
        rsi = last.get("rsi")
        close = last["close"]
        volume = last["volume"]
        avg_vol = last.get("avg_volume_10", volume)

        # Require real ADX and RSI data
        if any(v is None or (isinstance(v, float) and v != v) for v in [adx, rsi]):
            return None

        if avg_vol is None or avg_vol == 0:
            avg_vol = volume

        # For index data (volume=0), skip volume filter
        is_index = volume == 0 and avg_vol == 0

        # Check range condition in the lookback window (excluding current candle)
        range_window = df.iloc[-(effective_lookback + 1) : -1]
        range_high = range_window["high"].max()
        range_low = range_window["low"].min()
        range_pct = (range_high - range_low) / range_low * 100 if range_low > 0 else 999

        # Must be in a range (ADX < 22, range < 0.5%)
        if adx is None or adx >= ADX_THRESHOLD or range_pct >= RANGE_PCT_THRESHOLD:
            logger.debug(
                "RangeBreakout skip: ADX=%.1f (need <%.0f) range=%.2f%% (need <%.1f%%)",
                adx, ADX_THRESHOLD, range_pct, RANGE_PCT_THRESHOLD,
            )
            return None

        logger.debug(
            "RangeBreakout check: close=%.2f range=[%.2f,%.2f] RSI=%.1f ADX=%.1f range%%=%.2f",
            close, range_low, range_high, rsi, adx, range_pct,
        )

        # CALL breakout
        if close > range_high and (is_index or volume >= 1.5 * avg_vol) and rsi >= 55:
            return StrategySignal(
                strategy=StrategyName.RANGE_BREAKOUT,
                option_type=OptionType.CALL,
                strike_price=_nearest_strike(spot_price),
                details={
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_pct": round(range_pct, 2),
                    "adx": adx,
                    "rsi": rsi,
                    "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
                },
            )

        # PUT breakout
        if close < range_low and (is_index or volume >= 1.5 * avg_vol) and rsi <= 45:
            return StrategySignal(
                strategy=StrategyName.RANGE_BREAKOUT,
                option_type=OptionType.PUT,
                strike_price=_nearest_strike(spot_price),
                details={
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_pct": round(range_pct, 2),
                    "adx": adx,
                    "rsi": rsi,
                    "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
                },
            )

        return None


def _nearest_strike(price: float) -> float:
    return round(price / 50) * 50
