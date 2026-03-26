"""GEX (Gamma Exposure) Calculator.

Computes net dealer gamma exposure across the options chain to identify key
price levels where dealer hedging creates support/resistance.

Since AngelOne API doesn't provide Greeks, we estimate gamma from option
prices using the Black-Scholes model:
  1. Estimate IV from option LTP via Newton-Raphson on BS formula
  2. Compute gamma per strike using BS gamma formula
  3. GEX per strike = gamma × OI × contract_multiplier × spot²/100
  4. Net GEX = sum(call_gex) - sum(put_gex)  (dealer perspective)

Key outputs:
  - gex_levels: GEX at each strike (for bounce/magnet identification)
  - gex_flip_strike: Strike where net GEX flips sign (support ↔ resistance)
  - total_gex: Net dealer gamma (positive = dealers long gamma = mean-reverting)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from app.core.models import OptionsChainRow

logger = logging.getLogger(__name__)

# Black-Scholes constants
_SQRT_2PI = math.sqrt(2 * math.pi)
_RISK_FREE_RATE = 0.07  # India 10Y yield approx


@dataclass
class GEXLevel:
    """GEX value at a single strike."""

    strike: float
    call_gex: float = 0.0
    put_gex: float = 0.0
    net_gex: float = 0.0  # call_gex - put_gex (dealer perspective)
    call_gamma: float = 0.0
    put_gamma: float = 0.0
    call_iv: float = 0.0
    put_iv: float = 0.0


@dataclass
class GEXResult:
    """Aggregated GEX analysis."""

    levels: list[GEXLevel] = field(default_factory=list)
    total_gex: float = 0.0                 # Net dealer gamma
    gex_flip_strike: Optional[float] = None  # Where GEX flips sign
    max_positive_strike: Optional[float] = None  # Strongest support
    max_negative_strike: Optional[float] = None  # Strongest resistance
    positive_gex_zone: tuple[float, float] = (0.0, 0.0)  # Support zone range
    negative_gex_zone: tuple[float, float] = (0.0, 0.0)  # Resistance zone range


class GEXCalculator:
    """Calculate Gamma Exposure from options chain data."""

    def __init__(self, lot_size: int = 50):
        self.lot_size = lot_size

    def compute(
        self,
        chain: list[OptionsChainRow],
        spot: float,
        expiry_date: Optional[date] = None,
    ) -> GEXResult:
        """Compute GEX across the options chain.

        Args:
            chain: Options chain rows with strike, OI, LTP, etc.
            spot: Current underlying spot price.
            expiry_date: Options expiry date (for time-to-expiry calculation).
                         If None, assumes weekly expiry (next Thursday).

        Returns:
            GEXResult with per-strike and aggregate GEX data.
        """
        if not chain or spot <= 0:
            return GEXResult()

        tte = self._time_to_expiry(expiry_date)
        if tte <= 0:
            tte = 1 / 365  # Minimum 1 day

        levels: list[GEXLevel] = []
        for row in chain:
            if row.strike_price <= 0:
                continue

            level = GEXLevel(strike=row.strike_price)

            # Estimate IV and gamma for calls
            if row.call_ltp and row.call_ltp > 0 and row.call_oi > 0:
                call_iv = self._implied_vol(
                    spot, row.strike_price, tte, row.call_ltp, is_call=True
                )
                if call_iv and call_iv > 0.01:
                    level.call_iv = call_iv
                    level.call_gamma = self._bs_gamma(spot, row.strike_price, tte, call_iv)
                    # GEX = gamma × OI × lot_size × spot² / 100
                    level.call_gex = level.call_gamma * row.call_oi * self.lot_size * spot * spot / 100

            # Estimate IV and gamma for puts
            if row.put_ltp and row.put_ltp > 0 and row.put_oi > 0:
                put_iv = self._implied_vol(
                    spot, row.strike_price, tte, row.put_ltp, is_call=False
                )
                if put_iv and put_iv > 0.01:
                    level.put_iv = put_iv
                    level.put_gamma = self._bs_gamma(spot, row.strike_price, tte, put_iv)
                    # Put GEX: computed as positive; subtracted from call GEX in net_gex
                    level.put_gex = level.put_gamma * row.put_oi * self.lot_size * spot * spot / 100

            level.net_gex = level.call_gex - level.put_gex
            levels.append(level)

        if not levels:
            return GEXResult()

        # Aggregate
        total_gex = sum(lv.net_gex for lv in levels)

        # Find GEX flip point (where cumulative GEX changes sign)
        gex_flip = self._find_flip_strike(levels, spot)

        # Find max positive and negative GEX strikes
        max_pos = max(levels, key=lambda lv: lv.net_gex, default=None)
        max_neg = min(levels, key=lambda lv: lv.net_gex, default=None)

        # Support/resistance zones: top 3 strikes by GEX magnitude
        sorted_pos = sorted([lv for lv in levels if lv.net_gex > 0], key=lambda lv: lv.net_gex, reverse=True)
        sorted_neg = sorted([lv for lv in levels if lv.net_gex < 0], key=lambda lv: lv.net_gex)

        pos_zone = (0.0, 0.0)
        neg_zone = (0.0, 0.0)
        if len(sorted_pos) >= 2:
            pos_strikes = [lv.strike for lv in sorted_pos[:3]]
            pos_zone = (min(pos_strikes), max(pos_strikes))
        if len(sorted_neg) >= 2:
            neg_strikes = [lv.strike for lv in sorted_neg[:3]]
            neg_zone = (min(neg_strikes), max(neg_strikes))

        result = GEXResult(
            levels=levels,
            total_gex=total_gex,
            gex_flip_strike=gex_flip,
            max_positive_strike=max_pos.strike if max_pos and max_pos.net_gex > 0 else None,
            max_negative_strike=max_neg.strike if max_neg and max_neg.net_gex < 0 else None,
            positive_gex_zone=pos_zone,
            negative_gex_zone=neg_zone,
        )

        logger.info(
            "GEX: total=%.0f | flip=%.0f | support=%.0f | resistance=%.0f",
            total_gex,
            gex_flip or 0,
            max_pos.strike if max_pos and max_pos.net_gex > 0 else 0,
            max_neg.strike if max_neg and max_neg.net_gex < 0 else 0,
        )
        return result

    # ── Black-Scholes helpers ────────────────────────────────────────────

    @staticmethod
    def _norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / _SQRT_2PI

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _bs_price(
        self, s: float, k: float, t: float, sigma: float, is_call: bool
    ) -> float:
        """Black-Scholes European option price."""
        if sigma <= 0 or t <= 0:
            return 0.0
        d1 = (math.log(s / k) + (_RISK_FREE_RATE + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        if is_call:
            return s * self._norm_cdf(d1) - k * math.exp(-_RISK_FREE_RATE * t) * self._norm_cdf(d2)
        else:
            return k * math.exp(-_RISK_FREE_RATE * t) * self._norm_cdf(-d2) - s * self._norm_cdf(-d1)

    def _bs_gamma(self, s: float, k: float, t: float, sigma: float) -> float:
        """Black-Scholes gamma (same for calls and puts)."""
        if sigma <= 0 or t <= 0 or s <= 0:
            return 0.0
        d1 = (math.log(s / k) + (_RISK_FREE_RATE + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
        return self._norm_pdf(d1) / (s * sigma * math.sqrt(t))

    def _bs_vega(self, s: float, k: float, t: float, sigma: float) -> float:
        """BS vega for Newton-Raphson IV solver."""
        if sigma <= 0 or t <= 0 or s <= 0:
            return 0.0
        d1 = (math.log(s / k) + (_RISK_FREE_RATE + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
        return s * self._norm_pdf(d1) * math.sqrt(t)

    def _implied_vol(
        self,
        s: float,
        k: float,
        t: float,
        market_price: float,
        is_call: bool,
        max_iter: int = 50,
        tol: float = 0.001,
    ) -> Optional[float]:
        """Estimate implied volatility using Newton-Raphson.

        Returns None if convergence fails.
        """
        if market_price <= 0 or s <= 0 or k <= 0 or t <= 0:
            return None

        # Quick intrinsic check
        intrinsic = max(s - k, 0) if is_call else max(k - s, 0)
        if market_price < intrinsic * 0.5:
            return None  # Price too low, likely bad data

        # Initial guess based on ATM approximation
        sigma = 0.20  # Start at 20% vol

        for _ in range(max_iter):
            price = self._bs_price(s, k, t, sigma, is_call)
            vega = self._bs_vega(s, k, t, sigma)

            if vega < 1e-10:
                break  # Can't improve

            diff = price - market_price
            if abs(diff) < tol:
                return sigma

            sigma -= diff / vega
            if sigma <= 0.001:
                sigma = 0.001
            elif sigma > 5.0:
                return None  # Unrealistic vol

        # Final check: if we got close enough, return it
        if abs(self._bs_price(s, k, t, sigma, is_call) - market_price) < tol * 5:
            return sigma
        return None

    def _time_to_expiry(self, expiry_date: Optional[date]) -> float:
        """Time to expiry in years."""
        if expiry_date is None:
            # Default: assume next Thursday (weekly expiry)
            today = date.today()
            days_ahead = (3 - today.weekday()) % 7  # Thursday = 3
            if days_ahead == 0 and datetime.now().hour >= 15:
                days_ahead = 7
            return max(days_ahead, 1) / 365

        today = date.today()
        days = (expiry_date - today).days
        return max(days, 1) / 365

    def _find_flip_strike(
        self, levels: list[GEXLevel], spot: float
    ) -> Optional[float]:
        """Find strike closest to spot where net GEX flips sign."""
        sorted_levels = sorted(levels, key=lambda lv: lv.strike)

        for i in range(1, len(sorted_levels)):
            prev = sorted_levels[i - 1]
            curr = sorted_levels[i]
            # Flip: one positive, one negative
            if prev.net_gex * curr.net_gex < 0:
                # Return the one closer to spot
                if abs(prev.strike - spot) < abs(curr.strike - spot):
                    return prev.strike
                return curr.strike

        return None
