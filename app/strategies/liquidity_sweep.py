"""Strategy 4 — Liquidity Sweep.

Book references:
  - ICT/Smart Money Concepts — liquidity grab below swing lows/above swing highs
  - Nison, *Japanese Candlestick Charting* — wick rejection patterns
    (hammer for CALL, shooting star for PUT)
  - Wyckoff — spring (false break below support) / upthrust (false break above)

CALL (Wyckoff Spring):
  - Price breaks previous swing low by ≥0.03% (sweep the liquidity)
  - Reversal candle: bullish close, lower wick ≥ 40% (Nison: hammer)
  - Volume spike ≥ 1.4× avg (Wyckoff: volume on spring confirms)

PUT (Wyckoff Upthrust):
  - Price breaks swing high by ≥0.03%
  - Bearish rejection candle: upper wick ≥ 50% (Nison: shooting star)
  - Volume spike ≥ 1.4× avg
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from app.core.models import OptionType, OptionsMetrics, StrategyName, StrategySignal
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

SWING_LOOKBACK = 15  # Reduced from 20 — faster swing detection


class LiquiditySweepStrategy(BaseStrategy):
    """Liquidity Sweep strategy."""

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
        daily_levels: Optional[dict] = None,
        structure_data: Optional[dict] = None,
    ) -> Optional[StrategySignal]:
        if df.empty or len(df) < SWING_LOOKBACK + 2:
            return None

        last = df.iloc[-1]
        close = last["close"]
        open_ = last["open"]
        high = last["high"]
        low = last["low"]
        volume = last["volume"]
        avg_vol = last.get("avg_volume_10", volume)

        if avg_vol is None or avg_vol == 0:
            avg_vol = volume

        # For index data (volume=0), skip volume filter
        is_index = volume == 0 and avg_vol == 0

        # Micro-trigger: lower volume threshold, slightly relax wick requirement
        micro = (structure_data or {}).get("micro_trigger", {})
        micro_active = micro.get("active", False)
        vol_mult = 1.2 if micro_active else 1.4
        call_wick_min = 35 if micro_active else 40
        put_wick_min = 45 if micro_active else 50

        # Compute swing high/low from lookback window (excluding last 2 candles)
        lookback = df.iloc[-(SWING_LOOKBACK + 2) : -2]
        swing_low = lookback["low"].min()
        swing_high = lookback["high"].max()

        candle_range = high - low

        # CALL: sweep of swing low + bullish reversal
        sweep_threshold_low = swing_low * (1 - 0.0003)  # 0.03%
        lower_wick = min(close, open_) - low
        lower_wick_pct = (lower_wick / candle_range * 100) if candle_range > 0 else 0
        if (
            low <= sweep_threshold_low
            and close > open_  # bullish close
            and close > swing_low  # reclaim above swing low
            and lower_wick_pct >= call_wick_min
            and (is_index or volume >= vol_mult * avg_vol)
        ):
            return StrategySignal(
                strategy=StrategyName.LIQUIDITY_SWEEP,
                option_type=OptionType.CALL,
                strike_price=_nearest_strike(spot_price),
                details={
                    "swing_low": swing_low,
                    "sweep_depth_pct": round((swing_low - low) / swing_low * 100, 3),
                    "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
                    "micro_trigger": micro_active,
                },
            )

        # PUT: sweep of swing high + bearish rejection
        sweep_threshold_high = swing_high * (1 + 0.0003)
        upper_wick = high - max(close, open_)
        upper_wick_pct = (upper_wick / candle_range * 100) if candle_range > 0 else 0

        if (
            high >= sweep_threshold_high
            and close < open_  # bearish close
            and close < swing_high  # rejected back below
            and upper_wick_pct >= put_wick_min
            and (is_index or volume >= vol_mult * avg_vol)
        ):
            return StrategySignal(
                strategy=StrategyName.LIQUIDITY_SWEEP,
                option_type=OptionType.PUT,
                strike_price=_nearest_strike(spot_price),
                details={
                    "swing_high": swing_high,
                    "upper_wick_pct": round(upper_wick_pct, 1),
                    "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
                    "micro_trigger": micro_active,
                },
            )

        return None


def _nearest_strike(price: float) -> float:
    return round(price / 50) * 50
