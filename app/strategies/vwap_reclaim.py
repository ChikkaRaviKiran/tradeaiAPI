"""Strategy 2 — VWAP Reclaim.

Time window: 10:30–14:30

CALL: Price below VWAP for ≥10 min, closes above VWAP, volume > 1.3× avg, RSI > 55, EMA9 crosses EMA20
PUT:  Price above VWAP for ≥10 min, closes below VWAP, RSI < 45, EMA9 crosses below EMA20
"""

from __future__ import annotations

import logging
from datetime import time as dtime
from typing import Optional

import pandas as pd

from app.core.models import OptionType, OptionsMetrics, StrategyName, StrategySignal
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

WINDOW_START = dtime(10, 30)
WINDOW_END = dtime(14, 30)
MIN_BELOW_CANDLES = 10  # 10× 1-min candles = 10 minutes


class VWAPReclaimStrategy(BaseStrategy):
    """VWAP Reclaim strategy."""

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
    ) -> Optional[StrategySignal]:
        if df.empty or len(df) < 20:
            return None

        # Filter to time window
        window = df[(df.index.time >= WINDOW_START) & (df.index.time <= WINDOW_END)]
        if len(window) < MIN_BELOW_CANDLES + 1:
            return None

        last = window.iloc[-1]
        close = last["close"]
        vwap = last.get("vwap", close)
        volume = last["volume"]
        rsi = last.get("rsi", 50)
        ema9 = last.get("ema9", 0)
        ema20 = last.get("ema20", 0)
        avg_vol = last.get("avg_volume_10", volume)

        if avg_vol is None or avg_vol == 0:
            avg_vol = volume

        # Look back at last MIN_BELOW_CANDLES candles
        recent = window.iloc[-(MIN_BELOW_CANDLES + 1) :]

        # CALL: price was below VWAP for ≥10 candles, now closes above
        below_vwap = recent.iloc[:-1]["close"] < recent.iloc[:-1]["vwap"]
        if (
            below_vwap.sum() >= MIN_BELOW_CANDLES
            and close > vwap
            and volume > 1.3 * avg_vol
            and rsi > 55
            and _ema_cross_up(window, "ema9", "ema20")
        ):
            return StrategySignal(
                strategy=StrategyName.VWAP_RECLAIM,
                option_type=OptionType.CALL,
                entry_price=close,
                strike_price=_nearest_strike(spot_price),
                stoploss=close * 0.72,
                target1=close * 1.5,
                target2=close * 2.0,
                details={"rsi": rsi, "vwap": vwap, "volume_ratio": round(volume / avg_vol, 2)},
            )

        # PUT: price was above VWAP for ≥10 candles, now closes below
        above_vwap = recent.iloc[:-1]["close"] > recent.iloc[:-1]["vwap"]
        if (
            above_vwap.sum() >= MIN_BELOW_CANDLES
            and close < vwap
            and volume > 1.3 * avg_vol
            and rsi < 45
            and _ema_cross_down(window, "ema9", "ema20")
        ):
            return StrategySignal(
                strategy=StrategyName.VWAP_RECLAIM,
                option_type=OptionType.PUT,
                entry_price=close,
                strike_price=_nearest_strike(spot_price),
                stoploss=close * 0.72,
                target1=close * 1.5,
                target2=close * 2.0,
                details={"rsi": rsi, "vwap": vwap, "volume_ratio": round(volume / avg_vol, 2)},
            )

        return None


def _ema_cross_up(df: pd.DataFrame, fast: str, slow: str) -> bool:
    """Check if fast EMA crossed above slow EMA in last 3 candles."""
    if len(df) < 3:
        return False
    recent = df.tail(3)
    prev = recent.iloc[-2]
    curr = recent.iloc[-1]
    return prev.get(fast, 0) <= prev.get(slow, 0) and curr.get(fast, 0) > curr.get(slow, 0)


def _ema_cross_down(df: pd.DataFrame, fast: str, slow: str) -> bool:
    """Check if fast EMA crossed below slow EMA in last 3 candles."""
    if len(df) < 3:
        return False
    recent = df.tail(3)
    prev = recent.iloc[-2]
    curr = recent.iloc[-1]
    return prev.get(fast, 0) >= prev.get(slow, 0) and curr.get(fast, 0) < curr.get(slow, 0)


def _nearest_strike(price: float) -> float:
    return round(price / 50) * 50
