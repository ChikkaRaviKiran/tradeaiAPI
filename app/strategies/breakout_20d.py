"""Strategy 8 — 20-Day Donchian Channel Breakout.

Book references:
  - Donchian Channel (Richard Donchian, 1960s) — 20-day high/low breakout,
    the foundation of the Turtle Trading system.
  - Curtis Faith, *Way of the Turtle* — buy on 20-day high breakout with
    volume confirmation, sell on 20-day low breakout.
  - O'Neil, *How to Make Money in Stocks* — breakout volume ≥ 50% above avg.
  - Minervini, *Trade Like a Stock Market Wizard* — volume ideally 2× on breakout.
  - Wilder — ADX > 25 for trending market, RSI > 55 for bullish momentum.

Conditions (CALL — bullish breakout):
  - Current price breaks above the 20-day high (Donchian upper band)
  - Intraday volume ratio ≥ 1.8× average (between O'Neil 1.5× and Minervini 2×)
  - RSI > 55 (Wilder: above centerline with momentum)
  - ADX > 25 (Wilder: trending market)

Conditions (PUT — bearish breakout):
  - Current price breaks below the 20-day low (Donchian lower band)
  - Intraday volume ratio ≥ 1.8× average
  - RSI < 45
  - ADX > 25

Implementation:
  The orchestrator pre-fetches daily candles (ONE_DAY interval) and computes
  20-day high, 20-day low, and 20-day average volume. These are passed as
  `daily_levels` dict to evaluate(). Intraday RSI, ADX, and volume ratios
  are computed from the 1-minute DataFrame.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import pytz

from app.core.models import OptionType, OptionsMetrics, StrategyName, StrategySignal
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")


class Breakout20DStrategy(BaseStrategy):
    """20-day Donchian Channel breakout with volume confirmation.

    Requires `daily_levels` dict with keys:
      - high_20d: float — 20-day high (Donchian upper band)
      - low_20d: float — 20-day low (Donchian lower band)
    """

    VOLUME_MULTIPLIER = 1.8  # Between O'Neil 1.5× and Minervini 2×

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
        daily_levels: Optional[dict] = None,
    ) -> Optional[StrategySignal]:
        # Require daily levels — cannot evaluate without real 20-day data
        if not daily_levels or "high_20d" not in daily_levels or "low_20d" not in daily_levels:
            return None

        if len(df) < 5:
            return None

        now = df.index[-1]
        if hasattr(now, "hour"):
            if now.hour < 9 or (now.hour == 9 and now.minute < 30):
                return None
            if now.hour >= 15 and now.minute > 15:
                return None

        last = df.iloc[-1]
        close = last["close"]
        volume = last.get("volume", 0)

        # 20-day Donchian levels from daily candles
        high_20d = daily_levels["high_20d"]
        low_20d = daily_levels["low_20d"]

        # Intraday volume ratio (using today's 1-min candles)
        avg_vol = df["volume"].mean() if "volume" in df.columns else 0
        vol_ratio = volume / avg_vol if avg_vol > 0 else 0

        # Technical filters from intraday indicators
        rsi = last.get("rsi")
        adx = last.get("adx")

        if rsi is None or adx is None:
            return None

        # Bullish breakout — price above 20-day high (Donchian upper band)
        if (
            close > high_20d
            and vol_ratio >= self.VOLUME_MULTIPLIER
            and rsi > 55   # Wilder: above centerline with momentum
            and adx > 25   # Wilder: ADX > 25 = trending
        ):
            return StrategySignal(
                strategy=StrategyName.BREAKOUT_20D,
                option_type=OptionType.CALL,
                strike_price=_nearest_strike(spot_price, "CE"),
                details={
                    "high_20d": round(high_20d, 2),
                    "low_20d": round(low_20d, 2),
                    "close": round(close, 2),
                    "volume_ratio": round(vol_ratio, 2),
                    "rsi": round(rsi, 1),
                    "adx": round(adx, 1),
                },
            )

        # Bearish breakout — price below 20-day low (Donchian lower band)
        if (
            close < low_20d
            and vol_ratio >= self.VOLUME_MULTIPLIER
            and rsi < 45   # Wilder: below centerline
            and adx > 25   # Wilder: ADX > 25 = trending
        ):
            return StrategySignal(
                strategy=StrategyName.BREAKOUT_20D,
                option_type=OptionType.PUT,
                strike_price=_nearest_strike(spot_price, "PE"),
                details={
                    "high_20d": round(high_20d, 2),
                    "low_20d": round(low_20d, 2),
                    "close": round(close, 2),
                    "volume_ratio": round(vol_ratio, 2),
                    "rsi": round(rsi, 1),
                    "adx": round(adx, 1),
                },
            )

        return None


def _nearest_strike(price: float, opt_type: str = "CE", interval: float = 50) -> float:
    """Round to nearest strike. Uses default 50 for NIFTY, can be overridden."""
    base = round(price / interval) * interval
    if opt_type == "CE" and base < price:
        base += interval
    elif opt_type == "PE" and base > price:
        base -= interval
    return base
