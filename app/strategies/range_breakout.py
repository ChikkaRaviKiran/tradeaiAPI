"""Strategy 5 — Range Breakout.

Book references:
  - Bollinger, *Bollinger on Bollinger Bands* — volatility squeeze → breakout
  - Wilder, *New Concepts in Technical Trading* — ADX < 20 = no trend (range)
  - O'Neil — volume ≥ 50% above avg on breakout
  - Bulkowski, *Encyclopedia of Chart Patterns* — body strength filter
  - Wilder — RSI > 50 for bullish bias

Range condition:
  ADX < 20 (Wilder: < 20 = no trend)
  Price range < 0.80% for 30 candles

Breakout:
  Volume ≥ 1.5× avg (O'Neil)
  RSI ≥ 55 (CALL), RSI ≤ 45 (PUT) — Wilder centerline + directional bias
  Candle body ≥ 40% (Bulkowski: reject doji/spinning tops)
"""

from __future__ import annotations

import logging
import os
from datetime import time as dtime
from typing import Optional

import pandas as pd

from app.core.models import OptionType, OptionsMetrics, StrategyName, StrategySignal
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


def _env_time(name: str, default: dtime) -> dtime:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        h, m = raw.split(":", 1)
        return dtime(int(h), int(m))
    except Exception:
        return default


WINDOW_START = _env_time("RB_WINDOW_START", dtime(9, 45))
WINDOW_END = _env_time("RB_WINDOW_END", dtime(15, 0))
RANGE_LOOKBACK = int(os.getenv("RB_LOOKBACK", "30"))  # Bollinger: squeeze measured over recent consolidation
ADX_THRESHOLD = float(os.getenv("RB_ADX_THRESHOLD", "20"))  # Wilder: ADX < 20 = no trend (ranging market)
RANGE_PCT_THRESHOLD = float(os.getenv("RB_RANGE_PCT_THRESHOLD", "0.80"))  # Bollinger: tight range before breakout
MIN_BODY_RATIO = float(os.getenv("RB_MIN_BODY_RATIO", "0.40"))
CALL_RSI_MIN = float(os.getenv("RB_CALL_RSI_MIN", "55"))
PUT_RSI_MAX = float(os.getenv("RB_PUT_RSI_MAX", "45"))
VOL_MULT_NORMAL = float(os.getenv("RB_VOL_MULT_NORMAL", "1.5"))
VOL_MULT_MICRO = float(os.getenv("RB_VOL_MULT_MICRO", "1.3"))
# Minimum index close value as proxy for meaningful range (avoids weak squeeze setups)
MIN_RANGE_HIGH = float(os.getenv("RB_MIN_RANGE_HIGH", "0"))   # skip if range_high < this (0 = disabled)
# Option entry quality filters (applied in engine at option candle level)
SL_PCT = float(os.getenv("RB_SL_PCT", "0.20"))           # stop-loss as fraction of entry price (default 20%)
MIN_ENTRY_PREMIUM = float(os.getenv("RB_MIN_ENTRY_PREMIUM", "0"))  # skip if option entry price < this (0 = disabled)


class RangeBreakoutStrategy(BaseStrategy):
    """Range Breakout strategy."""

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
        daily_levels: Optional[dict] = None,
        structure_data: Optional[dict] = None,
    ) -> Optional[StrategySignal]:
        if df.empty or len(df) < RANGE_LOOKBACK + 1:
            return None

        # Time filter
        last_time = df.index[-1]
        if hasattr(last_time, "time"):
            t = last_time.time()
            if t < WINDOW_START or t > WINDOW_END:
                return None

        # Need enough candles for the lookback
        effective_lookback = min(RANGE_LOOKBACK, len(df) - 1)

        last = df.iloc[-1]
        adx = last.get("adx")
        rsi = last.get("rsi")
        close = last["close"]
        volume = last["volume"]
        avg_vol = last.get("avg_volume_10", volume)

        # Require real ADX and RSI data
        if any(v is None or (isinstance(v, float) and v != v) for v in [adx, rsi]):
            return None

        if avg_vol is None or avg_vol == 0:
            avg_vol = volume

        # For index data (volume=0), skip volume filter
        is_index = volume == 0 and avg_vol == 0

        # Check range condition in the lookback window (excluding current candle)
        range_window = df.iloc[-(effective_lookback + 1) : -1]
        range_high = range_window["high"].max()
        range_low = range_window["low"].min()
        range_pct = (range_high - range_low) / range_low * 100 if range_low > 0 else 999

        # Must be in a range (ADX < threshold, range < threshold)
        if adx is None or adx >= ADX_THRESHOLD or range_pct >= RANGE_PCT_THRESHOLD:
            logger.debug(
                "RangeBreakout skip: ADX=%.1f (need <%.0f) range=%.2f%% (need <%.1f%%)",
                adx, ADX_THRESHOLD, range_pct, RANGE_PCT_THRESHOLD,
            )
            return None

        # Minimum range_high filter: skip if consolidation zone is too shallow
        if MIN_RANGE_HIGH > 0 and range_high < MIN_RANGE_HIGH:
            return None

        # Candle body strength — Bulkowski: breakout candles with weak bodies
        # (doji, spinning tops) have 2-3x higher failure rates
        open_ = last.get("open", close)
        high = last.get("high", close)
        low = last.get("low", close)
        candle_range = high - low
        body = abs(close - open_)
        if candle_range > 0 and (body / candle_range) < MIN_BODY_RATIO:
            return None

        logger.debug(
            "RangeBreakout check: close=%.2f range=[%.2f,%.2f] RSI=%.1f ADX=%.1f range%%=%.2f",
            close, range_low, range_high, rsi, adx, range_pct,
        )

        # Micro-trigger: use high/low for breakout detection, lower volume
        micro = (structure_data or {}).get("micro_trigger", {})
        micro_active = micro.get("active", False)
        vol_mult = VOL_MULT_MICRO if micro_active else VOL_MULT_NORMAL

        # CALL breakout
        call_break = close > range_high
        if micro_active and not call_break:
            call_break = last.get("high", close) > range_high
        if call_break and (is_index or volume >= vol_mult * avg_vol) and rsi >= CALL_RSI_MIN:
            return StrategySignal(
                strategy=StrategyName.RANGE_BREAKOUT,
                option_type=OptionType.CALL,
                strike_price=_nearest_strike(spot_price),
                details={
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_pct": round(range_pct, 2),
                    "adx": adx,
                    "rsi": rsi,
                    "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
                    "micro_trigger": micro_active,
                },
            )

        # PUT breakout
        put_break = close < range_low
        if micro_active and not put_break:
            put_break = last.get("low", close) < range_low
        if put_break and (is_index or volume >= vol_mult * avg_vol) and rsi <= PUT_RSI_MAX:
            return StrategySignal(
                strategy=StrategyName.RANGE_BREAKOUT,
                option_type=OptionType.PUT,
                strike_price=_nearest_strike(spot_price),
                details={
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_pct": round(range_pct, 2),
                    "adx": adx,
                    "rsi": rsi,
                    "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
                    "micro_trigger": micro_active,
                },
            )

        return None


def _nearest_strike(price: float) -> float:
    return round(price / 50) * 50
