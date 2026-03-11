"""Strategy 3 — Trend Pullback.

CALL:
  - EMA20 > EMA50 AND price > VWAP (uptrend)
  - Price pullback to EMA20
  - RSI 45–50
  - Bullish candle confirmation
  - Volume > 1.2× avg

PUT:
  - EMA20 < EMA50 (downtrend)
  - Price pullback to EMA20
  - RSI 50–55
  - Bearish candle confirmation
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from app.core.models import OptionType, OptionsMetrics, StrategyName, StrategySignal
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class TrendPullbackStrategy(BaseStrategy):
    """Trend Pullback strategy."""

    PULLBACK_TOLERANCE_PCT = 0.15  # price within 0.15% of EMA20

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
    ) -> Optional[StrategySignal]:
        if df.empty or len(df) < 50:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = last["close"]
        open_ = last["open"]
        vwap = last.get("vwap", close)
        ema20 = last.get("ema20", 0)
        ema50 = last.get("ema50", 0)
        rsi = last.get("rsi", 50)
        volume = last["volume"]
        avg_vol = last.get("avg_volume_10", volume)

        if avg_vol is None or avg_vol == 0:
            avg_vol = volume

        # Check pullback to EMA20 (price within tolerance)
        if ema20 == 0:
            return None
        pullback_distance_pct = abs(close - ema20) / ema20 * 100

        # CALL: uptrend pullback
        if (
            ema20 > ema50
            and close > vwap
            and pullback_distance_pct <= self.PULLBACK_TOLERANCE_PCT
            and 45 <= rsi <= 50
            and close > open_  # bullish candle
            and volume > 1.2 * avg_vol
        ):
            return StrategySignal(
                strategy=StrategyName.TREND_PULLBACK,
                option_type=OptionType.CALL,
                entry_price=close,
                strike_price=_nearest_strike(spot_price),
                stoploss=close * 0.72,
                target1=close * 1.5,
                target2=close * 2.0,
                details={"rsi": rsi, "ema20": ema20, "pullback_pct": round(pullback_distance_pct, 2)},
            )

        # PUT: downtrend pullback
        if (
            ema20 < ema50
            and pullback_distance_pct <= self.PULLBACK_TOLERANCE_PCT
            and 50 <= rsi <= 55
            and close < open_  # bearish candle
            and volume > 1.2 * avg_vol
        ):
            return StrategySignal(
                strategy=StrategyName.TREND_PULLBACK,
                option_type=OptionType.PUT,
                entry_price=close,
                strike_price=_nearest_strike(spot_price),
                stoploss=close * 0.72,
                target1=close * 1.5,
                target2=close * 2.0,
                details={"rsi": rsi, "ema20": ema20, "pullback_pct": round(pullback_distance_pct, 2)},
            )

        return None


def _nearest_strike(price: float) -> float:
    return round(price / 50) * 50
