"""Strategy 3 — Trend Pullback (SRS Strategy 1).

CALL:
  - Price > EMA200 (long-term trend context, if available)
  - EMA20 > EMA50 AND price > VWAP (uptrend)
  - Price pullback to EMA20 or EMA50
  - RSI 40–55
  - Bullish candle confirmation

PUT:
  - Price < EMA200 (if available)
  - EMA20 < EMA50 (downtrend)
  - Price pullback to EMA20 or EMA50
  - RSI 45–60
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
    """Trend Pullback strategy per SRS specification."""

    PULLBACK_TOLERANCE_PCT = 0.40  # price within 0.40% of EMA (was 0.35%)

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
    ) -> Optional[StrategySignal]:
        if df.empty or len(df) < 20:
            return None

        last = df.iloc[-1]
        close = last["close"]
        open_ = last["open"]
        vwap = last.get("vwap")
        ema20 = last.get("ema20")
        ema50 = last.get("ema50")
        ema200 = last.get("ema200")
        rsi = last.get("rsi")
        volume = last["volume"]
        avg_vol = last.get("avg_volume_10", volume)

        # Require real indicators
        if any(v is None or (isinstance(v, float) and v != v) for v in [rsi, ema20, ema50]):
            return None
        if vwap is None or (isinstance(vwap, float) and vwap != vwap):
            vwap = close

        if avg_vol is None or avg_vol == 0:
            avg_vol = volume

        # For index data (volume=0), skip volume filter
        is_index = volume == 0 and avg_vol == 0

        # Check pullback to EMA20 or EMA50
        if ema20 == 0:
            return None
        pullback_to_ema20 = abs(close - ema20) / ema20 * 100
        pullback_to_ema50 = abs(close - ema50) / ema50 * 100 if ema50 > 0 else 999

        # Use the closer of EMA20/EMA50 for pullback
        pullback_distance_pct = min(pullback_to_ema20, pullback_to_ema50)

        # EMA200 availability check
        has_ema200 = ema200 is not None and not (isinstance(ema200, float) and ema200 != ema200)

        logger.debug(
            "TrendPullback check: close=%.2f EMA20=%.1f EMA50=%.1f EMA200=%s RSI=%.1f pullback=%.2f%%",
            close, ema20, ema50, f"{ema200:.1f}" if has_ema200 else "N/A", rsi, pullback_distance_pct,
        )

        # CALL: uptrend pullback
        if (
            ema20 > ema50
            and close > vwap
            and pullback_distance_pct <= self.PULLBACK_TOLERANCE_PCT
            and 38 <= rsi <= 60
            and close > open_  # bullish candle
            and (is_index or volume > 1.2 * avg_vol)
            and (not has_ema200 or close > ema200)  # above EMA200 if available
        ):
            return StrategySignal(
                strategy=StrategyName.TREND_PULLBACK,
                option_type=OptionType.CALL,
                strike_price=_nearest_strike(spot_price),
                details={
                    "rsi": rsi,
                    "ema20": ema20,
                    "ema50": ema50,
                    "ema200": ema200 if has_ema200 else None,
                    "pullback_pct": round(pullback_distance_pct, 2),
                },
            )

        # PUT: downtrend pullback
        if (
            ema20 < ema50
            and pullback_distance_pct <= self.PULLBACK_TOLERANCE_PCT
            and 45 <= rsi <= 60
            and close < open_  # bearish candle
            and (is_index or volume > 1.2 * avg_vol)
            and (not has_ema200 or close < ema200)  # below EMA200 if available
        ):
            return StrategySignal(
                strategy=StrategyName.TREND_PULLBACK,
                option_type=OptionType.PUT,
                strike_price=_nearest_strike(spot_price),
                details={
                    "rsi": rsi,
                    "ema20": ema20,
                    "ema50": ema50,
                    "ema200": ema200 if has_ema200 else None,
                    "pullback_pct": round(pullback_distance_pct, 2),
                },
            )

        return None


def _nearest_strike(price: float) -> float:
    return round(price / 50) * 50
