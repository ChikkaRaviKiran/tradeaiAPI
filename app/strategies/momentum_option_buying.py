"""Momentum Option Buying Engine (Final v1).

Core objective: Catch strong intraday moves using simple rules.
→ Low trade count, small losses, occasional big wins.

Entry conditions:
  1. Market filter: avg range of last 10 candles must exceed threshold (no sideways)
  2. Momentum detection: current candle range > 1.5× avg last 10 candle range
  3. VWAP direction: price > VWAP → UP (CE), price < VWAP → DOWN (PE)
  4. Breakout/breakdown of recent 10-candle high/low
  5. Pullback confirmation: wait 1 candle pullback, next candle continues in direction

Time windows: 09:20–11:30, 14:30–15:15

Scoring (simple):
  - Momentum present: +1
  - Clear VWAP direction: +1
  - ATM option available: +1
  → score 3 = full qty, score 2 = half qty, else skip

Stop loss: 1R = max(recent swing distance, 12% of entry)
Target: +1R → move SL to cost, +2R → book 50%, rest trail previous candle
"""

from __future__ import annotations

import logging
from datetime import time as dtime
from typing import Optional

import numpy as np
import pandas as pd

from app.core.models import OptionType, OptionsMetrics, StrategyName, StrategySignal
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# Time windows
MORNING_START = dtime(9, 30)
MORNING_END = dtime(11, 30)
AFTERNOON_START = dtime(14, 30)
AFTERNOON_END = dtime(15, 15)

# Lookback for avg range & breakout level
RANGE_LOOKBACK = 10

# Momentum multiplier: current candle range must exceed this × avg range
MOMENTUM_MULTIPLIER = 1.3

# Minimum avg range as fraction of price (filters dead markets)
# Computed from typical NIFTY/SENSEX intraday movement
MIN_AVG_RANGE_PCT = 0.015  # 0.015% of price


class MomentumOptionBuyingStrategy(BaseStrategy):
    """Momentum Option Buying — catches strong directional moves with pullback confirmation."""

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
        daily_levels: Optional[dict] = None,
        structure_data: Optional[dict] = None,
    ) -> Optional[StrategySignal]:
        # Need enough bars for lookback + momentum + pullback + confirmation
        min_bars = RANGE_LOOKBACK + 4
        if df.empty or len(df) < min_bars:
            return None

        # ── Time filter: only trade in allowed windows ──
        last_time = df.index[-1]
        if hasattr(last_time, "time"):
            t = last_time.time()
            in_morning = MORNING_START <= t <= MORNING_END
            in_afternoon = AFTERNOON_START <= t <= AFTERNOON_END
            if not in_morning and not in_afternoon:
                return None

        # ── STEP 1: Market filter — skip sideways ──
        # Use the 10 candles BEFORE the momentum candle (which is at -3)
        # Momentum = df.iloc[-3], Pullback = df.iloc[-2], Confirmation = df.iloc[-1]
        lookback_end = -3  # exclude momentum candle, pullback, and confirmation
        lookback_start = lookback_end - RANGE_LOOKBACK
        lookback_candles = df.iloc[lookback_start:lookback_end]

        if len(lookback_candles) < RANGE_LOOKBACK:
            return None

        candle_ranges = lookback_candles["high"] - lookback_candles["low"]
        avg_range = float(candle_ranges.mean())

        # Dynamic threshold: skip if avg range is too small relative to price
        threshold = spot_price * MIN_AVG_RANGE_PCT / 100
        if avg_range < threshold:
            return None

        # ── STEP 2: Momentum detection ──
        # The "momentum candle" is 3 bars back (before pullback + confirmation)
        momentum_candle = df.iloc[-3]
        momentum_range = float(momentum_candle["high"]) - float(momentum_candle["low"])

        has_momentum = momentum_range > MOMENTUM_MULTIPLIER * avg_range
        if not has_momentum:
            return None

        # ── VWAP direction ──
        vwap = df.iloc[-1].get("vwap")
        close = float(df.iloc[-1]["close"])

        if vwap is None or pd.isna(vwap) or vwap <= 0:
            return None

        if close > vwap:
            direction = "UP"
        elif close < vwap:
            direction = "DOWN"
        else:
            return None

        # ── STEP 3: Breakout/breakdown of recent high/low ──
        recent_high = float(lookback_candles["high"].max())
        recent_low = float(lookback_candles["low"].min())

        momentum_close = float(momentum_candle["close"])
        momentum_open = float(momentum_candle["open"])

        # ── Pullback confirmation (STEP 3 continued) ──
        pullback_candle = df.iloc[-2]
        confirm_candle = df.iloc[-1]

        pb_close = float(pullback_candle["close"])
        pb_open = float(pullback_candle["open"])
        pb_high = float(pullback_candle["high"])
        pb_low = float(pullback_candle["low"])
        conf_close = float(confirm_candle["close"])
        conf_open = float(confirm_candle["open"])

        # ── CALL setup ──
        if (
            direction == "UP"
            and momentum_close > recent_high          # breakout
            and momentum_close > momentum_open         # bullish momentum candle
            and pb_close <= pb_open                    # pullback candle is bearish or doji
            and conf_close > conf_open                 # confirmation candle is bullish
        ):
            # ── Scoring ──
            score = 0
            if has_momentum:
                score += 1
            if direction == "UP":
                score += 1
            score += 1  # ATM option (always selecting ATM)

            if score < 2:
                return None

            return StrategySignal(
                strategy=StrategyName.MOMENTUM_OPTION_BUYING,
                option_type=OptionType.CALL,
                strike_price=_nearest_strike(spot_price, "CE"),
                details={
                    "avg_range": round(avg_range, 2),
                    "momentum_range": round(momentum_range, 2),
                    "momentum_ratio": round(momentum_range / avg_range, 2),
                    "vwap": round(float(vwap), 2),
                    "recent_high": round(recent_high, 2),
                    "breakout_level": round(recent_high, 2),
                    "mob_score": score,
                    "quantity_pct": 100 if score == 3 else 50,
                },
            )

        # ── PUT setup ──
        if (
            direction == "DOWN"
            and momentum_close < recent_low             # breakdown
            and momentum_close < momentum_open           # bearish momentum candle
            and pb_close >= pb_open                      # pullback candle is bullish or doji
            and conf_close < conf_open                   # confirmation candle is bearish
        ):
            score = 0
            if has_momentum:
                score += 1
            if direction == "DOWN":
                score += 1
            score += 1  # ATM option

            if score < 2:
                return None

            return StrategySignal(
                strategy=StrategyName.MOMENTUM_OPTION_BUYING,
                option_type=OptionType.PUT,
                strike_price=_nearest_strike(spot_price, "PE"),
                details={
                    "avg_range": round(avg_range, 2),
                    "momentum_range": round(momentum_range, 2),
                    "momentum_ratio": round(momentum_range / avg_range, 2),
                    "vwap": round(float(vwap), 2),
                    "recent_low": round(recent_low, 2),
                    "breakout_level": round(recent_low, 2),
                    "mob_score": score,
                    "quantity_pct": 100 if score == 3 else 50,
                },
            )

        return None


def _nearest_strike(price: float, option_type: str = "CE") -> float:
    return round(price / 50) * 50
