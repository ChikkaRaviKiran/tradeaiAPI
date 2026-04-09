"""Strategy 3 — Trend Pullback.

Book references:
  - Raschke & Connors, *Street Smarts* — "Holy Grail" pattern:
    ADX > 30 on daily (> 20 adjusted for 1-min), pullback to EMA in trending market
  - Elder, *Trading for a Living* — buy pullbacks in uptrend using EMA alignment
  - Raschke — RSI 40-60 is the pullback sweet spot (not overbought, trend still intact)
  - Weinstein — EMA200 as long-term trend stage filter
  - Wyckoff — volume confirmation on bounce

CALL:
  - ADX > 20 (Raschke Holy Grail, adjusted from 30 for 1-min timeframe)
  - Price > EMA200 (Weinstein Stage 2, if available)
  - EMA20 > EMA50 AND price > VWAP (uptrend, Elder)
  - Price pullback to within 0.40% of EMA20 or EMA50
  - RSI 38–60 (Raschke: pullback sweet spot)
  - Bullish candle confirmation (Nison)
  - Volume > 1.2× avg (Wyckoff: bounce needs participation)

PUT:
  - ADX > 20
  - Price < EMA200 (Weinstein Stage 4, if available)
  - EMA20 < EMA50 (downtrend)
  - Price pullback to within 0.40% of EMA20 or EMA50
  - RSI 40–62 (Raschke: pullback sweet spot)
  - Bearish candle confirmation
  - Volume > 1.2× avg
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
        daily_levels: Optional[dict] = None,
        structure_data: Optional[dict] = None,
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
        adx = last.get("adx")
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

        # ADX trend strength — Raschke/Connors "Holy Grail": pullback must
        # be in a trending market (ADX > 30 on daily; relaxed to > 20 for 1-min)
        has_adx = adx is not None and not (isinstance(adx, float) and adx != adx)
        if not has_adx or adx < 20:
            return None

        # EMA200 availability check
        has_ema200 = ema200 is not None and not (isinstance(ema200, float) and ema200 != ema200)

        # Micro-trigger: widen pullback tolerance, lower volume requirement
        micro = (structure_data or {}).get("micro_trigger", {})
        micro_active = micro.get("active", False)
        pullback_tol = 0.55 if micro_active else self.PULLBACK_TOLERANCE_PCT
        vol_mult = 1.0 if micro_active else 1.2

        logger.debug(
            "TrendPullback check: close=%.2f EMA20=%.1f EMA50=%.1f EMA200=%s RSI=%.1f pullback=%.2f%% micro=%s",
            close, ema20, ema50, f"{ema200:.1f}" if has_ema200 else "N/A", rsi, pullback_distance_pct, micro_active,
        )

        # CALL: uptrend pullback
        if (
            ema20 > ema50
            and close > vwap
            and pullback_distance_pct <= pullback_tol
            and 38 <= rsi <= 60
            and close > open_  # bullish candle
            and (is_index or volume > vol_mult * avg_vol)
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
                    "micro_trigger": micro_active,
                },
            )

        # PUT: downtrend pullback
        if (
            ema20 < ema50
            and pullback_distance_pct <= pullback_tol
            and 40 <= rsi <= 62
            and close < open_  # bearish candle
            and (is_index or volume > vol_mult * avg_vol)
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
                    "micro_trigger": micro_active,
                },
            )

        return None


def _nearest_strike(price: float) -> float:
    return round(price / 50) * 50
