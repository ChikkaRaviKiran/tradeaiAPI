"""Strategy 6 — Momentum Breakout.

Catches strong directional moves with sustained momentum.
Designed for scenarios where price breaks out sharply from a consolidation zone.

Time window: 09:45–15:00

CALL:
  - Price breaks above 20-candle high
  - RSI > 60 (strong momentum)
  - EMA9 > EMA20 (short-term trend aligned)
  - ADX > 20 or ADX rising (directional strength)
  - Current candle body > 50% of range (strong close)

PUT:
  - Price breaks below 20-candle low
  - RSI < 40
  - EMA9 < EMA20
  - ADX > 20 or ADX rising
  - Current candle body > 50% of range
"""

from __future__ import annotations

import logging
from datetime import time as dtime
from typing import Optional

import pandas as pd

from app.core.models import OptionType, OptionsMetrics, StrategyName, StrategySignal
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

WINDOW_START = dtime(9, 45)
WINDOW_END = dtime(15, 0)
LOOKBACK = 20


class MomentumBreakoutStrategy(BaseStrategy):
    """Momentum Breakout strategy — catches strong directional moves."""

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
    ) -> Optional[StrategySignal]:
        if df.empty or len(df) < LOOKBACK + 2:
            return None

        # Time filter
        last_time = df.index[-1]
        if hasattr(last_time, "time"):
            t = last_time.time()
            if t < WINDOW_START or t > WINDOW_END:
                return None

        last = df.iloc[-1]
        close = last["close"]
        open_ = last["open"]
        high = last["high"]
        low = last["low"]
        rsi = last.get("rsi")
        ema9 = last.get("ema9")
        ema20 = last.get("ema20")
        adx = last.get("adx")

        # Require real indicator data
        if any(v is None or (isinstance(v, float) and v != v) for v in [rsi, ema9, ema20, adx]):
            return None

        # Candle body strength check
        candle_range = high - low
        body = abs(close - open_)
        if candle_range <= 0 or (body / candle_range) < 0.5:
            return None

        # Lookback window for high/low (excluding current candle)
        lookback_window = df.iloc[-(LOOKBACK + 1):-1]
        lookback_high = lookback_window["high"].max()
        lookback_low = lookback_window["low"].min()

        # ADX rising check (current > 5 candles ago)
        adx_prev = df.iloc[-6].get("adx") if len(df) >= 6 else None
        adx_rising = adx_prev is not None and not pd.isna(adx_prev) and adx > adx_prev

        logger.debug(
            "MomentumBreakout check: close=%.2f high20=%.2f low20=%.2f RSI=%.1f ADX=%.1f EMA9=%.1f EMA20=%.1f",
            close, lookback_high, lookback_low, rsi, adx, ema9, ema20,
        )

        # CALL: breakout above 20-candle high with momentum
        if (
            close > lookback_high
            and close > open_  # bullish candle
            and rsi > 60
            and ema9 > ema20
            and (adx > 20 or adx_rising)
        ):
            return StrategySignal(
                strategy=StrategyName.MOMENTUM_BREAKOUT,
                option_type=OptionType.CALL,
                strike_price=_nearest_strike(spot_price, "CE"),
                details={
                    "lookback_high": lookback_high,
                    "rsi": rsi,
                    "adx": adx,
                    "breakout_pct": round((close - lookback_high) / lookback_high * 100, 3),
                },
            )

        # PUT: breakdown below 20-candle low with momentum
        if (
            close < lookback_low
            and close < open_  # bearish candle
            and rsi < 40
            and ema9 < ema20
            and (adx > 20 or adx_rising)
        ):
            return StrategySignal(
                strategy=StrategyName.MOMENTUM_BREAKOUT,
                option_type=OptionType.PUT,
                strike_price=_nearest_strike(spot_price, "PE"),
                details={
                    "lookback_low": lookback_low,
                    "rsi": rsi,
                    "adx": adx,
                    "breakout_pct": round((lookback_low - close) / lookback_low * 100, 3),
                },
            )

        return None


def _nearest_strike(price: float, option_type: str = "CE") -> float:
    return round(price / 50) * 50
