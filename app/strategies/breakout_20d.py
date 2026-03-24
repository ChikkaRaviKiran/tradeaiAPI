"""Strategy: 20-Day High Breakout — SRS Strategy 2.

Conditions (BUY/CALL):
  - Price breaks above 20-day high
  - Volume > 2x 20-day average
  - Sector strength high (or instrument in uptrend)
  - RSI > 55 (momentum confirmation)
  - ADX > 20 (trending)

Conditions (SHORT/PUT):
  - Price breaks below 20-day low
  - Volume > 2x 20-day average
  - RSI < 45
  - ADX > 20
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import pytz

from app.core.models import OptionType, OptionsMetrics, StrategyName, StrategySignal
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")


class Breakout20DStrategy(BaseStrategy):
    """SRS Strategy 2: 20-day high/low breakout with volume confirmation."""

    LOOKBACK = 20
    VOLUME_MULTIPLIER = 1.8  # Slightly relaxed from 2x for intraday
    MIN_CANDLES = 25

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
    ) -> Optional[StrategySignal]:
        if len(df) < self.MIN_CANDLES:
            return None

        now = df.index[-1]
        if hasattr(now, "hour"):
            if now.hour < 9 or (now.hour == 9 and now.minute < 30):
                return None
            if now.hour >= 15 and now.minute > 15:
                return None

        last = df.iloc[-1]
        close = last["close"]
        volume = last.get("volume", 0)

        # 20-day high/low (excluding current candle)
        lookback = df.iloc[-self.LOOKBACK - 1 : -1]
        high_20d = lookback["high"].max()
        low_20d = lookback["low"].min()

        # Average volume
        avg_vol = lookback["volume"].mean() if "volume" in lookback.columns else 0
        vol_ratio = volume / avg_vol if avg_vol > 0 else 0

        # Technical filters
        rsi = last.get("rsi")
        adx = last.get("adx")

        if rsi is None or adx is None:
            return None

        # Bullish breakout
        if (
            close > high_20d
            and vol_ratio >= self.VOLUME_MULTIPLIER
            and rsi > 55
            and adx > 20
        ):
            return StrategySignal(
                strategy=StrategyName.BREAKOUT_20D,
                option_type=OptionType.CALL,
                strike_price=_nearest_strike(spot_price, "CE"),
                details={
                    "high_20d": round(high_20d, 2),
                    "close": round(close, 2),
                    "volume_ratio": round(vol_ratio, 2),
                    "rsi": round(rsi, 1),
                    "adx": round(adx, 1),
                },
            )

        # Bearish breakout
        if (
            close < low_20d
            and vol_ratio >= self.VOLUME_MULTIPLIER
            and rsi < 45
            and adx > 20
        ):
            return StrategySignal(
                strategy=StrategyName.BREAKOUT_20D,
                option_type=OptionType.PUT,
                strike_price=_nearest_strike(spot_price, "PE"),
                details={
                    "low_20d": round(low_20d, 2),
                    "close": round(close, 2),
                    "volume_ratio": round(vol_ratio, 2),
                    "rsi": round(rsi, 1),
                    "adx": round(adx, 1),
                },
            )

        return None


def _nearest_strike(price: float, opt_type: str = "CE", interval: float = 50) -> float:
    """Round to nearest strike. Uses default 50 for NIFTY, can be overridden."""
    base = round(price / interval) * interval
    if opt_type == "CE" and base < price:
        base += interval
    elif opt_type == "PE" and base > price:
        base -= interval
    return base
