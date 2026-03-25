"""Strategy 6 — Momentum Breakout.

Book references:
  - Minervini, *Trade Like a Stock Market Wizard* — SEPA breakout criteria
  - O'Neil, *How to Make Money in Stocks* — volume ≥ 50% above avg on breakout
  - Donchian Channel (20-period high/low breakout)
  - Wilder, *New Concepts in Technical Trading* — ADX > 25 for trending

Time window: 09:45–15:00

CALL:
  - Price breaks above 20-candle high (Donchian)
  - Volume ≥ 1.5× avg (O'Neil: 40-50% above average minimum)
  - RSI > 60 (strong momentum, Wilder centerline + 10)
  - EMA9 > EMA20 (short-term trend aligned, Elder Triple Screen)
  - ADX > 25 or ADX rising (Wilder: > 25 = trending)
  - Current candle body > 50% of range (Nison, Japanese Candlestick Charting)

PUT:
  - Price breaks below 20-candle low
  - Volume ≥ 1.5× avg
  - RSI < 40
  - EMA9 < EMA20
  - ADX > 25 or ADX rising
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
        daily_levels: Optional[dict] = None,
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
        volume = last.get("volume", 0)
        avg_vol = last.get("avg_volume_10", volume)

        # Require real indicator data
        if any(v is None or (isinstance(v, float) and v != v) for v in [rsi, ema9, ema20, adx]):
            return None

        if avg_vol is None or avg_vol == 0:
            avg_vol = volume

        # For index data (volume=0), skip volume filter
        is_index = volume == 0 and avg_vol == 0

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
        # O'Neil: volume ≥ 50% above avg; Wilder: ADX > 25 = trending
        if (
            close > lookback_high
            and close > open_  # bullish candle (Nison)
            and (is_index or volume >= 1.5 * avg_vol)  # O'Neil minimum
            and rsi > 60  # Wilder: above centerline + momentum
            and ema9 > ema20  # Elder: short-term trend aligned
            and (adx > 25 or adx_rising)  # Wilder: ADX > 25 = trending
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
            and close < open_  # bearish candle (Nison)
            and (is_index or volume >= 1.5 * avg_vol)  # O'Neil minimum
            and rsi < 40  # Wilder: below centerline − momentum
            and ema9 < ema20  # Elder: short-term trend aligned
            and (adx > 25 or adx_rising)  # Wilder: ADX > 25 = trending
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
