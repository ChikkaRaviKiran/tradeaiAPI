"""Strategy 7 — EMA Breakout (SRS Strategy 2).

Conditions for CALL:
  - Price > EMA200 (long-term uptrend confirmed)
  - Price breaks above EMA50 from below (intermediate breakout)
  - RSI 50–70 (momentum without being overbought)
  - EMA9 > EMA20 (short-term aligned)
  - Volume confirmation via candle range vs ATR (since NIFTY index has no volume)

Conditions for PUT:
  - Price < EMA200
  - Price breaks below EMA50 from above
  - RSI 30–50
  - EMA9 < EMA20

Time window: 09:45–15:00
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


class EMABreakoutStrategy(BaseStrategy):
    """EMA Breakout strategy — catches price breaking key EMA levels with trend confirmation."""

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
    ) -> Optional[StrategySignal]:
        if df.empty or len(df) < 20:
            return None

        # Time filter
        last_time = df.index[-1]
        if hasattr(last_time, "time"):
            t = last_time.time()
            if t < WINDOW_START or t > WINDOW_END:
                return None

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else last
        close = last["close"]
        open_ = last["open"]
        high = last["high"]
        low = last["low"]
        ema9 = last.get("ema9")
        ema20 = last.get("ema20")
        ema50 = last.get("ema50")
        ema200 = last.get("ema200")
        rsi = last.get("rsi")
        atr = last.get("atr")

        # Require real indicator data
        if any(v is None or (isinstance(v, float) and v != v) for v in [rsi, ema9, ema20, ema50]):
            return None

        # EMA200 may not be available early — use EMA50 as fallback for trend
        has_ema200 = ema200 is not None and not (isinstance(ema200, float) and ema200 != ema200)

        # Previous candle's relationship to EMA50
        prev_close = prev["close"]
        prev_ema50 = prev.get("ema50")
        if prev_ema50 is None or (isinstance(prev_ema50, float) and prev_ema50 != prev_ema50):
            return None

        # Candle body strength — need decent body
        candle_range = high - low
        body = abs(close - open_)
        if candle_range <= 0 or (body / candle_range) < 0.4:
            return None

        # Activity check: candle range vs ATR (for index data without volume)
        if atr and atr > 0:
            range_ratio = candle_range / atr
            if range_ratio < 0.5:
                return None  # Very weak candle, skip

        logger.debug(
            "EMABreakout check: close=%.2f EMA50=%.1f EMA200=%s RSI=%.1f EMA9=%.1f EMA20=%.1f",
            close, ema50, f"{ema200:.1f}" if has_ema200 else "N/A", rsi, ema9, ema20,
        )

        # CALL: Price crosses above EMA50 with trend alignment
        if (
            prev_close <= prev_ema50          # was at or below EMA50
            and close > ema50                  # now above EMA50
            and close > open_                  # bullish candle
            and ema9 > ema20                   # short-term trend up
            and 50 <= rsi <= 70                # momentum sweet spot
            and (not has_ema200 or close > ema200)  # above long-term trend if available
        ):
            return StrategySignal(
                strategy=StrategyName.EMA_BREAKOUT,
                option_type=OptionType.CALL,
                strike_price=_nearest_strike(spot_price, "CE"),
                details={
                    "ema50": ema50,
                    "ema200": ema200 if has_ema200 else None,
                    "rsi": rsi,
                    "breakout_pct": round((close - ema50) / ema50 * 100, 3),
                },
            )

        # PUT: Price crosses below EMA50 with trend alignment
        if (
            prev_close >= prev_ema50          # was at or above EMA50
            and close < ema50                  # now below EMA50
            and close < open_                  # bearish candle
            and ema9 < ema20                   # short-term trend down
            and 30 <= rsi <= 50                # downward momentum
            and (not has_ema200 or close < ema200)  # below long-term trend if available
        ):
            return StrategySignal(
                strategy=StrategyName.EMA_BREAKOUT,
                option_type=OptionType.PUT,
                strike_price=_nearest_strike(spot_price, "PE"),
                details={
                    "ema50": ema50,
                    "ema200": ema200 if has_ema200 else None,
                    "rsi": rsi,
                    "breakout_pct": round((ema50 - close) / ema50 * 100, 3),
                },
            )

        return None


def _nearest_strike(price: float, option_type: str = "CE") -> float:
    return round(price / 50) * 50
