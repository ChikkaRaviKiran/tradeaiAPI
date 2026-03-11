"""Strategy 5 — Range Breakout.

Range condition:
  ADX < 18, price range < 0.35% for 60 minutes

Breakout:
  Volume ≥ 1.5× avg
  RSI ≥ 60 (CALL), RSI ≤ 40 (PUT)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from app.core.models import OptionType, OptionsMetrics, StrategyName, StrategySignal
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

RANGE_LOOKBACK = 60
ADX_THRESHOLD = 18
RANGE_PCT_THRESHOLD = 0.35


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

        last = df.iloc[-1]
        adx = last.get("adx", 20)
        rsi = last.get("rsi", 50)
        close = last["close"]
        volume = last["volume"]
        avg_vol = last.get("avg_volume_10", volume)

        if avg_vol is None or avg_vol == 0:
            avg_vol = volume

        # Check range condition in the lookback window (excluding current candle)
        range_window = df.iloc[-(RANGE_LOOKBACK + 1) : -1]
        range_high = range_window["high"].max()
        range_low = range_window["low"].min()
        range_pct = (range_high - range_low) / range_low * 100 if range_low > 0 else 999

        # Must be in a range (ADX < 18, range < 0.35%)
        if adx is None or adx >= ADX_THRESHOLD or range_pct >= RANGE_PCT_THRESHOLD:
            return None

        # CALL breakout
        if close > range_high and volume >= 1.5 * avg_vol and rsi >= 60:
            return StrategySignal(
                strategy=StrategyName.RANGE_BREAKOUT,
                option_type=OptionType.CALL,
                entry_price=close,
                strike_price=_nearest_strike(spot_price),
                stoploss=close * 0.72,
                target1=close * 1.5,
                target2=close * 2.0,
                details={
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_pct": round(range_pct, 2),
                    "adx": adx,
                    "rsi": rsi,
                    "volume_ratio": round(volume / avg_vol, 2),
                },
            )

        # PUT breakout
        if close < range_low and volume >= 1.5 * avg_vol and rsi <= 40:
            return StrategySignal(
                strategy=StrategyName.RANGE_BREAKOUT,
                option_type=OptionType.PUT,
                entry_price=close,
                strike_price=_nearest_strike(spot_price),
                stoploss=close * 0.72,
                target1=close * 1.5,
                target2=close * 2.0,
                details={
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_pct": round(range_pct, 2),
                    "adx": adx,
                    "rsi": rsi,
                    "volume_ratio": round(volume / avg_vol, 2),
                },
            )

        return None


def _nearest_strike(price: float) -> float:
    return round(price / 50) * 50
