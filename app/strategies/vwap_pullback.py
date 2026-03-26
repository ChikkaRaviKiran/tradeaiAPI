"""V2 Strategy — VWAP Pullback.

Designed for TREND days. Buys pullbacks to VWAP during established intraday
trends, combining multiple confirmation signals:

  1. TREND context: ADX > 20, EMA20 > EMA50 (for CALL), price trending
  2. PULLBACK: Price touches or dips within 0.15% of VWAP
  3. BOUNCE confirmation: Bullish candle off VWAP with volume
  4. MOMENTUM: RSI in pullback sweet spot (40-60), not overbought/oversold

Time window: 10:00–14:00 (needs established trend, exits before 15:00)

CALL conditions:
  - EMA20 > EMA50 (uptrend structure, Elder)
  - ADX > 20 (trending, Raschke/Connors)
  - Price pulled back to within 0.15% of VWAP (institutional level)
  - Current candle closes above VWAP (bounce confirmed)
  - RSI 40-60 (pullback sweet spot, Raschke)
  - Bullish candle (close > open)

PUT conditions:
  - EMA20 < EMA50 (downtrend structure)
  - ADX > 20
  - Price rallied to within 0.15% of VWAP
  - Current candle closes below VWAP (rejection confirmed)
  - RSI 40-60
  - Bearish candle (close < open)
"""

from __future__ import annotations

import logging
from datetime import time as dtime
from typing import Optional

import pandas as pd

from app.core.models import OptionType, OptionsMetrics, StrategyName, StrategySignal
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

WINDOW_START = dtime(10, 0)
WINDOW_END = dtime(14, 0)
VWAP_TOUCH_PCT = 0.15          # Price within 0.15% of VWAP = "pullback to VWAP"
MIN_ADX = 20                   # Trending threshold
RSI_PULLBACK_LOW = 40
RSI_PULLBACK_HIGH = 60
LOOKBACK_CANDLES = 5           # Check last N candles for VWAP touch


class VWAPPullbackStrategy(BaseStrategy):
    """V2: VWAP pullback on trend days."""

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
        daily_levels: Optional[dict] = None,
    ) -> Optional[StrategySignal]:
        if df.empty or len(df) < 20:
            return None

        # Time window filter
        window = df[(df.index.time >= WINDOW_START) & (df.index.time <= WINDOW_END)]
        if len(window) < 10:
            return None

        last = window.iloc[-1]
        close = last["close"]
        open_ = last["open"]
        vwap = last.get("vwap")
        ema20 = last.get("ema20")
        ema50 = last.get("ema50")
        rsi = last.get("rsi")
        adx = last.get("adx")
        volume = last.get("volume", 0)
        avg_vol = last.get("avg_volume_10", volume)

        # Validate required indicators
        if any(_invalid(v) for v in [vwap, rsi, ema20, ema50]):
            return None
        if _invalid(adx) or adx < MIN_ADX:
            return None
        if avg_vol is None or avg_vol == 0:
            avg_vol = volume

        # Index data doesn't have volume — skip volume check
        is_index = volume == 0 and avg_vol == 0

        # Check if price recently touched/crossed VWAP (pullback detection)
        recent = window.tail(LOOKBACK_CANDLES)
        touched_vwap = False
        for _, row in recent.iloc[:-1].iterrows():
            row_vwap = row.get("vwap", vwap)
            if row_vwap and row_vwap > 0:
                distance_pct = abs(row["close"] - row_vwap) / row_vwap * 100
                if distance_pct <= VWAP_TOUCH_PCT:
                    touched_vwap = True
                    break
                # Also check if price crossed through VWAP (wick touch)
                if row["low"] <= row_vwap <= row["high"]:
                    touched_vwap = True
                    break

        if not touched_vwap:
            return None

        # CALL: uptrend + pullback to VWAP + bounce
        if (
            ema20 > ema50                           # Uptrend structure
            and close > vwap                        # Bounced above VWAP
            and close > open_                       # Bullish candle
            and RSI_PULLBACK_LOW <= rsi <= RSI_PULLBACK_HIGH  # Pullback zone
            and (is_index or volume > avg_vol)      # Volume confirmation
        ):
            logger.info(
                "VWAP_PULLBACK CALL: close=%.2f vwap=%.2f rsi=%.1f adx=%.1f",
                close, vwap, rsi, adx,
            )
            return StrategySignal(
                strategy=StrategyName.VWAP_PULLBACK,
                option_type=OptionType.CALL,
                strike_price=_nearest_strike(spot_price),
                details={
                    "rsi": rsi,
                    "adx": adx,
                    "vwap": vwap,
                    "ema20": ema20,
                    "ema50": ema50,
                    "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
                },
            )

        # PUT: downtrend + rally to VWAP + rejection
        if (
            ema20 < ema50                           # Downtrend structure
            and close < vwap                        # Rejected below VWAP
            and close < open_                       # Bearish candle
            and RSI_PULLBACK_LOW <= rsi <= RSI_PULLBACK_HIGH
            and (is_index or volume > avg_vol)
        ):
            logger.info(
                "VWAP_PULLBACK PUT: close=%.2f vwap=%.2f rsi=%.1f adx=%.1f",
                close, vwap, rsi, adx,
            )
            return StrategySignal(
                strategy=StrategyName.VWAP_PULLBACK,
                option_type=OptionType.PUT,
                strike_price=_nearest_strike(spot_price),
                details={
                    "rsi": rsi,
                    "adx": adx,
                    "vwap": vwap,
                    "ema20": ema20,
                    "ema50": ema50,
                    "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
                },
            )

        return None


def _invalid(v) -> bool:
    """Check if value is None or NaN."""
    return v is None or (isinstance(v, float) and v != v)


def _nearest_strike(price: float) -> float:
    return round(price / 50) * 50
