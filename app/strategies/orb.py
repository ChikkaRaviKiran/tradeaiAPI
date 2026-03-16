"""Strategy 1 — Opening Range Breakout (ORB).

Opening range: 09:15–09:30
ORH = highest high, ORL = lowest low in that window.

CALL: Close > ORH + 0.05%, Volume > 1.5× avg, Price > VWAP, EMA9 > EMA20, RSI ≥ 50
PUT:  Close < ORL − 0.05%, Volume > 1.5× avg, Price < VWAP, EMA9 < EMA20, RSI ≤ 50

Invalidation: breakout candle wick > 60%, or next candle closes inside range.
"""

from __future__ import annotations

import logging
from datetime import time as dtime
from typing import Optional

import pandas as pd

from app.core.models import OptionType, OptionsMetrics, StrategyName, StrategySignal
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

ORB_START = dtime(9, 15)
ORB_END = dtime(9, 30)


class ORBStrategy(BaseStrategy):
    """Opening Range Breakout strategy."""

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
    ) -> Optional[StrategySignal]:
        if df.empty or len(df) < 20:
            return None

        # Identify opening range candles (09:15–09:30)
        or_candles = df[
            (df.index.time >= ORB_START) & (df.index.time <= ORB_END)
        ]
        if or_candles.empty:
            return None

        orh = or_candles["high"].max()
        orl = or_candles["low"].min()

        # Only evaluate candles after ORB window
        post_orb = df[df.index.time > ORB_END]
        if post_orb.empty:
            return None

        last = post_orb.iloc[-1]
        close = last["close"]
        volume = last["volume"]
        vwap = last.get("vwap")
        ema9 = last.get("ema9")
        ema20 = last.get("ema20")
        rsi = last.get("rsi")
        avg_vol = last.get("avg_volume_10", volume)
        high = last["high"]
        low = last["low"]
        open_ = last["open"]

        # Require real indicator data
        if any(v is None or (isinstance(v, float) and v != v) for v in [rsi, ema9, ema20]):
            return None
        if vwap is None or (isinstance(vwap, float) and vwap != vwap):
            vwap = close  # Fallback: VWAP unavailable, use close (VWAP checks won't earn score)

        if avg_vol is None or avg_vol == 0:
            avg_vol = volume

        # For index data (volume=0), skip volume filter
        is_index = volume == 0 and avg_vol == 0

        # Wick ratio check (invalidation)
        body = abs(close - open_)
        candle_range = high - low
        if candle_range > 0:
            wick_ratio = (candle_range - body) / candle_range
        else:
            wick_ratio = 0

        if wick_ratio > 0.6:
            return None

        breakout_buffer = 0.0005  # 0.05%

        # Log strategy evaluation details at DEBUG level
        logger.debug(
            "ORB check: close=%.2f ORH=%.2f ORL=%.2f RSI=%.1f EMA9=%.1f EMA20=%.1f VWAP=%.2f",
            close, orh, orl, rsi, ema9, ema20, vwap,
        )

        # CALL breakout
        if (
            close > orh * (1 + breakout_buffer)
            and (is_index or volume > 1.5 * avg_vol)
            and close > vwap
            and ema9 > ema20
            and rsi >= 50
        ):
            # Check next candle doesn't close inside range (if available)
            if len(post_orb) >= 2:
                next_candle = post_orb.iloc[-1]
                prev_candle = post_orb.iloc[-2]
                if prev_candle["close"] > orh * (1 + breakout_buffer) and next_candle["close"] < orh:
                    return None

            return StrategySignal(
                strategy=StrategyName.ORB,
                option_type=OptionType.CALL,
                strike_price=_nearest_strike(spot_price, "CE"),
                details={
                    "orh": orh,
                    "orl": orl,
                    "rsi": rsi,
                    "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
                },
            )

        # PUT breakout
        if (
            close < orl * (1 - breakout_buffer)
            and (is_index or volume > 1.5 * avg_vol)
            and close < vwap
            and ema9 < ema20
            and rsi <= 50
        ):
            if len(post_orb) >= 2:
                next_candle = post_orb.iloc[-1]
                prev_candle = post_orb.iloc[-2]
                if prev_candle["close"] < orl * (1 - breakout_buffer) and next_candle["close"] > orl:
                    return None

            return StrategySignal(
                strategy=StrategyName.ORB,
                option_type=OptionType.PUT,
                strike_price=_nearest_strike(spot_price, "PE"),
                details={
                    "orh": orh,
                    "orl": orl,
                    "rsi": rsi,
                    "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
                },
            )

        return None


def _nearest_strike(price: float, opt_type: str) -> float:
    """Round price to nearest 50 strike, ATM."""
    return round(price / 50) * 50
