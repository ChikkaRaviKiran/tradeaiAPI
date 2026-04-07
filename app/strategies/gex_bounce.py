"""V2 Strategy — GEX Bounce.

Designed for RANGE days. Uses dealer Gamma Exposure levels as support/resistance
for mean-reversion bounce trades.

Theory: Market makers who sell options delta-hedge by buying/selling the underlying.
When dealers are long gamma (positive GEX), their hedging dampens moves — price
gravitates toward high-GEX strikes. On range days, these levels act as magnets.

CALL conditions (bounce off GEX support):
  1. RANGE day context
  2. Price near a high positive-GEX strike (within 0.3% of max GEX support)
  3. ADX < 22 (confirming range-bound)
  4. RSI 35-50 (not deeply oversold, but pulling back)
  5. Price shows bounce (bullish candle, close > open)
  6. Price above GEX flip level (dealers still net long gamma above)

PUT conditions (rejection off GEX resistance):
  1. Price near a high negative-GEX strike (within 0.3% of max GEX resistance)
  2. ADX < 22
  3. RSI 50-65
  4. Bearish candle (close < open)
  5. Price below GEX flip level

Time window: 10:00–13:30 (needs established range, early enough for exit)

Note: Requires GEXResult from GEXCalculator. If GEX data is unavailable,
the strategy returns None (no signal).
"""

from __future__ import annotations

import logging
from datetime import time as dtime
from typing import Optional

import pandas as pd

from app.core.models import OptionType, OptionsMetrics, StrategyName, StrategySignal
from app.engine.gex_calculator import GEXResult
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

WINDOW_START = dtime(10, 0)
WINDOW_END = dtime(13, 30)
GEX_PROXIMITY_PCT = 0.30      # Price within 0.30% of GEX level
MAX_ADX = 22                   # Range-bound threshold
RSI_CALL_LOW = 35
RSI_CALL_HIGH = 50
RSI_PUT_LOW = 50
RSI_PUT_HIGH = 65


class GEXBounceStrategy(BaseStrategy):
    """V2: GEX level bounce on range days.

    This strategy needs GEXResult injected via set_gex_result() before
    evaluate() is called each cycle.
    """

    def __init__(self) -> None:
        self._gex: Optional[GEXResult] = None

    def set_gex_result(self, gex: Optional[GEXResult]) -> None:
        """Inject the latest GEX calculation result (None clears it)."""
        self._gex = gex

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
        daily_levels: Optional[dict] = None,
        structure_data: Optional[dict] = None,
    ) -> Optional[StrategySignal]:
        if df.empty or len(df) < 15:
            return None
        if self._gex is None:
            return None

        # Time window filter
        window = df[(df.index.time >= WINDOW_START) & (df.index.time <= WINDOW_END)]
        if len(window) < 10:
            return None

        last = window.iloc[-1]
        close = last["close"]
        open_ = last["open"]
        rsi = last.get("rsi")
        adx = last.get("adx")

        if _invalid(rsi) or _invalid(adx):
            return None

        # Must be range-bound
        if adx > MAX_ADX:
            return None

        # Need GEX levels computed
        gex = self._gex
        if not gex.max_positive_strike and not gex.max_negative_strike:
            return None

        # CALL: Bounce off GEX support level
        if gex.max_positive_strike:
            support = gex.max_positive_strike
            distance_pct = abs(close - support) / support * 100 if support > 0 else 999

            if (
                distance_pct <= GEX_PROXIMITY_PCT
                and close > open_                          # Bullish candle
                and RSI_CALL_LOW <= rsi <= RSI_CALL_HIGH   # Pullback zone
                and (gex.gex_flip_strike is None or close >= gex.gex_flip_strike)
            ):
                logger.info(
                    "GEX_BOUNCE CALL: close=%.2f support=%.2f dist=%.2f%% rsi=%.1f adx=%.1f",
                    close, support, distance_pct, rsi, adx,
                )
                return StrategySignal(
                    strategy=StrategyName.GEX_BOUNCE,
                    option_type=OptionType.CALL,
                    strike_price=_nearest_strike(spot_price),
                    details={
                        "gex_support": support,
                        "gex_flip": gex.gex_flip_strike,
                        "gex_total": gex.total_gex,
                        "distance_pct": round(distance_pct, 2),
                        "rsi": rsi,
                        "adx": adx,
                    },
                )

        # PUT: Rejection off GEX resistance level
        if gex.max_negative_strike:
            resistance = gex.max_negative_strike
            distance_pct = abs(close - resistance) / resistance * 100 if resistance > 0 else 999

            if (
                distance_pct <= GEX_PROXIMITY_PCT
                and close < open_                          # Bearish candle
                and RSI_PUT_LOW <= rsi <= RSI_PUT_HIGH
                and (gex.gex_flip_strike is None or close <= gex.gex_flip_strike)
            ):
                logger.info(
                    "GEX_BOUNCE PUT: close=%.2f resistance=%.2f dist=%.2f%% rsi=%.1f adx=%.1f",
                    close, resistance, distance_pct, rsi, adx,
                )
                return StrategySignal(
                    strategy=StrategyName.GEX_BOUNCE,
                    option_type=OptionType.PUT,
                    strike_price=_nearest_strike(spot_price),
                    details={
                        "gex_resistance": resistance,
                        "gex_flip": gex.gex_flip_strike,
                        "gex_total": gex.total_gex,
                        "distance_pct": round(distance_pct, 2),
                        "rsi": rsi,
                        "adx": adx,
                    },
                )

        return None


def _invalid(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def _nearest_strike(price: float) -> float:
    return round(price / 50) * 50
