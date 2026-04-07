"""V2 Strategy — RSI Extreme Reversal.

Designed for VOLATILE days. Catches sharp mean-reversion bounces when RSI
reaches extreme levels (oversold/overbought) during high-volatility sessions.

Key principle: On volatile days, RSI extremes create predictable snap-back
moves as institutional algos rebalance positions at extreme levels.

Conditions for CALL (oversold reversal):
  1. VOLATILE context: Implied by day classification
  2. RSI < 25 at some point in recent candles (extreme oversold)
  3. RSI now rising (current RSI > prev RSI, momentum turning)
  4. Price below lower Bollinger Band or near it (statistical extreme)
  5. Bullish candle confirmation (close > open)
  6. PCR > 1.0 (more puts = potential short-covering fuel)

Conditions for PUT (overbought reversal):
  1. RSI > 75 at some point in recent candles
  2. RSI now falling
  3. Price above upper Bollinger Band or near it
  4. Bearish candle confirmation
  5. PCR < 0.7 (excess calls = potential unwinding)

Time window: 09:30–13:30 (volatile moves happen early, catch them fast)
"""

from __future__ import annotations

import logging
from datetime import time as dtime
from typing import Optional

import pandas as pd

from app.core.models import OptionType, OptionsMetrics, StrategyName, StrategySignal
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

WINDOW_START = dtime(9, 30)
WINDOW_END = dtime(13, 30)
RSI_OVERSOLD = 25              # Extreme oversold
RSI_OVERBOUGHT = 75            # Extreme overbought
RSI_RISING_MIN = 2.0           # RSI must have risen at least 2 points
LOOKBACK_CANDLES = 8           # Look for RSI extreme in last N candles
BB_PROXIMITY_PCT = 0.10        # Price within 0.10% of Bollinger band


class RSIExtremeStrategy(BaseStrategy):
    """V2: RSI extreme reversal on volatile days."""

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
        daily_levels: Optional[dict] = None,
        structure_data: Optional[dict] = None,
    ) -> Optional[StrategySignal]:
        if df.empty or len(df) < 15:
            return None

        # Time window filter
        window = df[(df.index.time >= WINDOW_START) & (df.index.time <= WINDOW_END)]
        if len(window) < 10:
            return None

        last = window.iloc[-1]
        close = last["close"]
        open_ = last["open"]
        rsi = last.get("rsi")
        bb_upper = last.get("bollinger_upper") if "bollinger_upper" in window.columns else None
        bb_lower = last.get("bollinger_lower") if "bollinger_lower" in window.columns else None

        if _invalid(rsi):
            return None

        # Check RSI history for extreme touch
        recent = window.tail(LOOKBACK_CANDLES)
        recent_rsi = recent["rsi"].dropna() if "rsi" in recent.columns else pd.Series(dtype=float)
        if len(recent_rsi) < 3:
            return None

        had_oversold = bool((recent_rsi < RSI_OVERSOLD).any())
        had_overbought = bool((recent_rsi > RSI_OVERBOUGHT).any())

        # RSI momentum direction
        prev_rsi = recent_rsi.iloc[-2] if len(recent_rsi) >= 2 else rsi
        rsi_rising = rsi > prev_rsi + RSI_RISING_MIN
        rsi_falling = rsi < prev_rsi - RSI_RISING_MIN

        # PCR filter
        pcr = options_metrics.pcr

        # CALL: Oversold extreme → reversal bounce
        if had_oversold and rsi_rising and close > open_:
            # Optional: Bollinger Band proximity
            near_bb_lower = True  # Default if BB not available
            if not _invalid(bb_lower) and bb_lower > 0:
                near_bb_lower = close <= bb_lower * (1 + BB_PROXIMITY_PCT / 100)

            # PCR filter: high PCR = lots of puts = short-covering fuel
            pcr_ok = pcr is None or pcr > 0.9  # Relaxed — PCR > 0.9 or unavailable

            if near_bb_lower and pcr_ok:
                logger.info(
                    "RSI_EXTREME CALL: rsi=%.1f (was <%.0f), close=%.2f, pcr=%s",
                    rsi, RSI_OVERSOLD, close,
                    f"{pcr:.2f}" if pcr else "N/A",
                )
                return StrategySignal(
                    strategy=StrategyName.RSI_EXTREME,
                    option_type=OptionType.CALL,
                    strike_price=_nearest_strike(spot_price),
                    details={
                        "rsi": rsi,
                        "prev_rsi": prev_rsi,
                        "min_rsi": float(recent_rsi.min()),
                        "pcr": pcr,
                        "bb_lower": bb_lower,
                        "close": close,
                    },
                )

        # PUT: Overbought extreme → reversal drop
        if had_overbought and rsi_falling and close < open_:
            near_bb_upper = True
            if not _invalid(bb_upper) and bb_upper > 0:
                near_bb_upper = close >= bb_upper * (1 - BB_PROXIMITY_PCT / 100)

            pcr_ok = pcr is None or pcr < 0.8  # Low PCR = excess calls

            if near_bb_upper and pcr_ok:
                logger.info(
                    "RSI_EXTREME PUT: rsi=%.1f (was >%.0f), close=%.2f, pcr=%s",
                    rsi, RSI_OVERBOUGHT, close,
                    f"{pcr:.2f}" if pcr else "N/A",
                )
                return StrategySignal(
                    strategy=StrategyName.RSI_EXTREME,
                    option_type=OptionType.PUT,
                    strike_price=_nearest_strike(spot_price),
                    details={
                        "rsi": rsi,
                        "prev_rsi": prev_rsi,
                        "max_rsi": float(recent_rsi.max()),
                        "pcr": pcr,
                        "bb_upper": bb_upper,
                        "close": close,
                    },
                )

        return None


def _invalid(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def _nearest_strike(price: float) -> float:
    return round(price / 50) * 50
