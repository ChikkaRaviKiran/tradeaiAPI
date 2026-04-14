"""Momentum Option Buying Engine (v2 — Enhanced).

Core objective: Catch strong intraday moves using confluence-based rules.
→ Low trade count, small losses, occasional big wins.
→ v2 adds: EMA trend alignment, RSI filter, volume confirmation,
   better scoring with momentum quality differentiation.

Entry conditions:
  1. Market filter: avg range of last 10 candles must exceed threshold (no sideways)
  2. Momentum detection: current candle range > 1.5× avg last 10 candle range
  3. VWAP direction: price > VWAP → UP (CE), price < VWAP → DOWN (PE)
  4. Breakout/breakdown of recent 10-candle high/low
  5. Pullback confirmation: wait 1 candle pullback, next candle continues in direction
  6. EMA trend alignment: EMA9 > EMA20 for calls, EMA9 < EMA20 for puts
  7. RSI filter: 35-70 for calls, 30-65 for puts (avoid extremes)
  8. Volume confirmation: momentum candle volume > 1.2× avg volume

Time windows: 09:30–11:30 (morning only — afternoon dropped for higher edge)

Scoring (enhanced 0-6):
  - Momentum present: +1
  - Clear VWAP direction: +1
  - EMA trend aligned: +1
  - Strong momentum (ratio > 1.8×): +1
  - RSI in sweet spot: +1
  - Volume confirmation: +1
  → score >= 4 = full qty, score 3 = half qty, else skip
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

# Time windows — morning only for higher edge
MORNING_START = dtime(9, 30)
MORNING_END = dtime(11, 30)
# Afternoon window disabled by default (configurable via settings)
AFTERNOON_START = dtime(14, 30)
AFTERNOON_END = dtime(15, 15)

# Lookback for avg range & breakout level
RANGE_LOOKBACK = 10

# Momentum multiplier: current candle range must exceed this × avg range
MOMENTUM_MULTIPLIER = 1.5  # Raised from 1.3 to filter weaker signals

# Strong momentum threshold for bonus score
STRONG_MOMENTUM_MULTIPLIER = 1.8

# Minimum avg range as fraction of price (filters dead markets)
MIN_AVG_RANGE_PCT = 0.015  # 0.015% of price

# RSI boundaries for entry
CALL_RSI_MIN = 35.0
CALL_RSI_MAX = 70.0
PUT_RSI_MIN = 30.0
PUT_RSI_MAX = 65.0

# Volume confirmation multiplier
VOLUME_CONFIRM_MULTIPLIER = 1.2


def is_range_bound_session(df: pd.DataFrame, spot_price: float,
                           adx_threshold: float = 18.0,
                           opening_range_pct: float = 0.4,
                           vwap_cross_limit: int = 4,
                           min_signals: int = 2,
                           check_bars: int = 30) -> bool:
    """Detect range-bound session using first N bars of the day.

    Checks 3 signals:
      1. ADX < threshold (no directional trend)
      2. Opening range < X% of spot (tight range)
      3. Price crossed VWAP >= N times (oscillating around mean)

    Returns True if `min_signals` of 3 say range-bound → skip MOB.
    """
    if len(df) < check_bars:
        return False  # Not enough data yet, allow trading

    window = df.iloc[:check_bars]
    range_signals = 0

    # Signal 1: ADX at check_bars mark
    adx_val = df.iloc[check_bars - 1].get("adx")
    if adx_val is not None and not pd.isna(adx_val) and adx_val < adx_threshold:
        range_signals += 1

    # Signal 2: Opening range as % of spot
    opening_high = float(window["high"].max())
    opening_low = float(window["low"].min())
    opening_range = (opening_high - opening_low) / spot_price * 100 if spot_price > 0 else 0
    if opening_range < opening_range_pct:
        range_signals += 1

    # Signal 3: VWAP crossings in the window
    vwap_series = window.get("vwap")
    if vwap_series is not None and not vwap_series.isna().all():
        closes = window["close"].values.astype(float)
        vwaps = vwap_series.values.astype(float)
        above = closes > vwaps
        crossings = int(np.sum(np.diff(above.astype(int)) != 0))
        if crossings >= vwap_cross_limit:
            range_signals += 1

    return range_signals >= min_signals


class MomentumOptionBuyingStrategy(BaseStrategy):
    """Momentum Option Buying v2 — catches strong directional moves with
    pullback confirmation and multi-factor confluence scoring."""

    def __init__(self, afternoon_enabled: bool = False):
        self.afternoon_enabled = afternoon_enabled

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
            in_afternoon = self.afternoon_enabled and (AFTERNOON_START <= t <= AFTERNOON_END)
            if not in_morning and not in_afternoon:
                return None

        # ── STEP 1: Market filter — skip sideways ──
        lookback_end = -3
        lookback_start = lookback_end - RANGE_LOOKBACK
        lookback_candles = df.iloc[lookback_start:lookback_end]

        if len(lookback_candles) < RANGE_LOOKBACK:
            return None

        candle_ranges = lookback_candles["high"] - lookback_candles["low"]
        avg_range = float(candle_ranges.mean())

        threshold = spot_price * MIN_AVG_RANGE_PCT / 100
        if avg_range < threshold:
            return None

        # ── STEP 2: Momentum detection ──
        momentum_candle = df.iloc[-3]
        momentum_range = float(momentum_candle["high"]) - float(momentum_candle["low"])
        momentum_ratio = momentum_range / avg_range if avg_range > 0 else 0

        has_momentum = momentum_ratio > MOMENTUM_MULTIPLIER
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

        # ── Pullback confirmation ──
        pullback_candle = df.iloc[-2]
        confirm_candle = df.iloc[-1]

        pb_close = float(pullback_candle["close"])
        pb_open = float(pullback_candle["open"])
        conf_close = float(confirm_candle["close"])
        conf_open = float(confirm_candle["open"])

        # ── Extract indicators for confluence (computed by FeatureEngine) ──
        ema9 = df.iloc[-1].get("ema9")
        ema20 = df.iloc[-1].get("ema20")
        rsi = df.iloc[-1].get("rsi")
        mom_volume = momentum_candle.get("volume", 0)
        avg_vol = df.iloc[-1].get("avg_volume_10", 0)

        # Safe numeric conversion
        ema9 = float(ema9) if ema9 is not None and not pd.isna(ema9) else None
        ema20 = float(ema20) if ema20 is not None and not pd.isna(ema20) else None
        rsi = float(rsi) if rsi is not None and not pd.isna(rsi) else None
        mom_volume = float(mom_volume) if mom_volume is not None and not pd.isna(mom_volume) else 0
        avg_vol = float(avg_vol) if avg_vol is not None and not pd.isna(avg_vol) else 0

        # ── CALL setup ──
        if (
            direction == "UP"
            and momentum_close > recent_high
            and momentum_close > momentum_open
            and pb_close <= pb_open
            and conf_close > conf_open
        ):
            # ── EMA trend alignment filter (hard gate) ──
            if ema9 is not None and ema20 is not None and ema9 < ema20:
                return None  # Short-term trend not aligned with call

            # ── Enhanced Scoring (0-6) ──
            score = 1  # Base: has_momentum (always true here)
            score += 1  # VWAP direction confirmed

            # +1 for EMA trend alignment
            ema_aligned = ema9 is not None and ema20 is not None and ema9 > ema20
            if ema_aligned:
                score += 1

            # +1 for strong momentum (ratio > 1.8)
            strong_momentum = momentum_ratio > STRONG_MOMENTUM_MULTIPLIER
            if strong_momentum:
                score += 1

            # +1 for RSI in sweet spot (not overbought)
            rsi_ok = rsi is not None and CALL_RSI_MIN <= rsi <= CALL_RSI_MAX
            if rsi_ok:
                score += 1

            # +1 for volume confirmation
            vol_confirmed = avg_vol > 0 and mom_volume > VOLUME_CONFIRM_MULTIPLIER * avg_vol
            if vol_confirmed:
                score += 1

            if score < 3:
                return None

            return StrategySignal(
                strategy=StrategyName.MOMENTUM_OPTION_BUYING,
                option_type=OptionType.CALL,
                strike_price=_nearest_strike(spot_price, "CE"),
                details={
                    "avg_range": round(avg_range, 2),
                    "momentum_range": round(momentum_range, 2),
                    "momentum_ratio": round(momentum_ratio, 2),
                    "vwap": round(float(vwap), 2),
                    "recent_high": round(recent_high, 2),
                    "breakout_level": round(recent_high, 2),
                    "mob_score": score,
                    "quantity_pct": 100 if score >= 4 else 50,
                    "ema_aligned": ema_aligned,
                    "rsi": round(rsi, 1) if rsi is not None else None,
                    "strong_momentum": strong_momentum,
                    "vol_confirmed": vol_confirmed,
                },
            )

        # ── PUT setup ──
        if (
            direction == "DOWN"
            and momentum_close < recent_low
            and momentum_close < momentum_open
            and pb_close >= pb_open
            and conf_close < conf_open
        ):
            # ── EMA trend alignment filter (hard gate) ──
            if ema9 is not None and ema20 is not None and ema9 > ema20:
                return None  # Short-term trend not aligned with put

            # ── Enhanced Scoring (0-6) ──
            score = 1  # Base: has_momentum
            score += 1  # VWAP direction confirmed

            ema_aligned = ema9 is not None and ema20 is not None and ema9 < ema20
            if ema_aligned:
                score += 1

            strong_momentum = momentum_ratio > STRONG_MOMENTUM_MULTIPLIER
            if strong_momentum:
                score += 1

            rsi_ok = rsi is not None and PUT_RSI_MIN <= rsi <= PUT_RSI_MAX
            if rsi_ok:
                score += 1

            vol_confirmed = avg_vol > 0 and mom_volume > VOLUME_CONFIRM_MULTIPLIER * avg_vol
            if vol_confirmed:
                score += 1

            if score < 3:
                return None

            return StrategySignal(
                strategy=StrategyName.MOMENTUM_OPTION_BUYING,
                option_type=OptionType.PUT,
                strike_price=_nearest_strike(spot_price, "PE"),
                details={
                    "avg_range": round(avg_range, 2),
                    "momentum_range": round(momentum_range, 2),
                    "momentum_ratio": round(momentum_ratio, 2),
                    "vwap": round(float(vwap), 2),
                    "recent_low": round(recent_low, 2),
                    "breakout_level": round(recent_low, 2),
                    "mob_score": score,
                    "quantity_pct": 100 if score >= 4 else 50,
                    "ema_aligned": ema_aligned,
                    "rsi": round(rsi, 1) if rsi is not None else None,
                    "strong_momentum": strong_momentum,
                    "vol_confirmed": vol_confirmed,
                },
            )

        return None


def _nearest_strike(price: float, option_type: str = "CE") -> float:
    return round(price / 50) * 50
