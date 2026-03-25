"""Strategy: Options Income — SRS Strategy 3.

AI determines the expected market range and suggests non-directional strategies:
  - Iron Condor: Sell OTM Call + Put spreads when range-bound
  - Short Strangle: Sell OTM Call + Put when low volatility expected
  - Bull/Bear spreads: When mild directional bias exists

This strategy generates INFO-level suggestions (not auto-traded) since
options income strategies require different execution logic (multi-leg orders).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pytz

from app.core.models import MarketRegime, OptionsMetrics

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")


@dataclass
class OptionsIncomeSignal:
    """Non-directional options strategy suggestion."""

    strategy_type: str = ""       # iron_condor, short_strangle, bull_spread, bear_spread
    instrument: str = "NIFTY"
    expected_range_low: float = 0
    expected_range_high: float = 0
    sell_call_strike: float = 0
    sell_put_strike: float = 0
    buy_call_strike: float = 0     # Protection leg (0 = no protection / strangle)
    buy_put_strike: float = 0
    max_profit: float = 0
    max_loss: float = 0
    confidence: float = 0           # 0-100
    reason: str = ""


class OptionsIncomeStrategy:
    """Generates options income strategy suggestions when market is range-bound."""

    ATR_MULTIPLIER = 1.5   # Expected range = spot ± ATR * multiplier
    MIN_PCR = 0.7          # Minimum PCR to consider put selling
    MAX_PCR = 1.3          # Maximum PCR to consider call selling

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
        daily_levels: Optional[dict] = None,
        regime: MarketRegime = MarketRegime.RANGE_BOUND,
        strike_interval: float = 50,
    ) -> Optional[OptionsIncomeSignal]:
        """Evaluate whether an options income strategy is appropriate.

        Only triggers in range-bound or low-volatility regimes.
        """
        if len(df) < 30:
            return None

        # Only in range-bound / low volatility
        if regime not in (MarketRegime.RANGE_BOUND, MarketRegime.LOW_VOLATILITY):
            return None

        last = df.iloc[-1]
        atr = last.get("atr")
        adx = last.get("adx")

        if atr is None or atr <= 0:
            return None

        # ADX must be low (non-trending)
        if adx is not None and adx > 22:
            return None

        # Compute expected range
        range_width = atr * self.ATR_MULTIPLIER
        range_low = round(spot_price - range_width, 2)
        range_high = round(spot_price + range_width, 2)

        # Round to strike intervals
        sell_call = self._round_to_strike(range_high + strike_interval, strike_interval)
        sell_put = self._round_to_strike(range_low - strike_interval, strike_interval)
        buy_call = sell_call + 2 * strike_interval  # Protection leg
        buy_put = sell_put - 2 * strike_interval

        pcr = options_metrics.pcr

        # Determine strategy type
        if pcr is not None and self.MIN_PCR <= pcr <= self.MAX_PCR:
            strategy_type = "iron_condor"
            reason = f"Range-bound market (ADX={adx:.0f}), balanced PCR={pcr:.2f}"
            confidence = 70
        elif regime == MarketRegime.LOW_VOLATILITY:
            strategy_type = "short_strangle"
            buy_call = 0  # No protection in strangle
            buy_put = 0
            reason = f"Low volatility regime, ATR={atr:.1f}"
            confidence = 60
        else:
            strategy_type = "iron_condor"
            reason = f"Range-bound, ADX={adx:.0f if adx else 'N/A'}"
            confidence = 55

        signal = OptionsIncomeSignal(
            strategy_type=strategy_type,
            expected_range_low=range_low,
            expected_range_high=range_high,
            sell_call_strike=sell_call,
            sell_put_strike=sell_put,
            buy_call_strike=buy_call,
            buy_put_strike=buy_put,
            confidence=confidence,
            reason=reason,
        )

        logger.info(
            "Options Income: %s | Range: %.0f-%.0f | Sell %dCE/%dPE | Confidence: %d%%",
            strategy_type,
            range_low,
            range_high,
            int(sell_call),
            int(sell_put),
            confidence,
        )

        return signal

    @staticmethod
    def _round_to_strike(price: float, interval: float) -> float:
        return round(price / interval) * interval
