"""Day Classifier — Categorise the trading day within the first 30-45 minutes.

Uses the first N minutes of 1-min candle data + pre-market context to classify
the day as TREND / RANGE / VOLATILE / UNCLEAR. The classification gates which
V2 strategies are eligible to trade:

  TREND    → VWAP Pullback
  RANGE    → GEX Bounce
  VOLATILE → RSI Extreme
  UNCLEAR  → No V2 trades (configurable via v2_skip_unclear_days)

The classifier re-evaluates at configurable intervals (default: once at 10:00)
but can be called mid-session if market character shifts dramatically.

Inputs required:
  - 1-min OHLCV DataFrame (at least 30 candles from 9:15)
  - MarketSnapshot with indicators (ADX, ATR, VWAP, Bollinger)
  - VIX value (from global indices)
  - Previous day levels (open, high, low, close)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from app.core.models import DayType, MarketSnapshot, TechnicalIndicators

logger = logging.getLogger(__name__)


class DayClassifier:
    """Classify trading day type from intraday data and market context."""

    # Thresholds — tuned for NIFTY 1-min data
    ADX_TREND_THRESHOLD = 22        # ADX above this → trending
    ADX_RANGE_CEILING = 20          # ADX below this → range-bound
    VIX_VOLATILE_THRESHOLD = 18.0   # India VIX above this → volatile
    GAP_VOLATILE_PCT = 0.70         # Opening gap > 0.70% → volatile
    VWAP_DISTANCE_TREND = 0.20      # Price > 0.20% from VWAP → trend confirmation
    ATR_EXPANSION_FACTOR = 1.5      # ATR > 1.5× 20-period avg → volatile
    CANDLE_RANGE_RATIO = 0.65       # % of candles with body > 50% of range → trend

    def classify(
        self,
        df: pd.DataFrame,
        snap: MarketSnapshot,
        vix: Optional[float] = None,
    ) -> DayType:
        """Classify the day using multiple signals and voting.

        Args:
            df: 1-min OHLCV data from 9:15 to current time.
            snap: Latest market snapshot with indicators.
            vix: India VIX value (None if unavailable).

        Returns:
            DayType enum value.
        """
        if df is None or df.empty or len(df) < 15:
            return DayType.UNCLEAR

        votes: dict[DayType, float] = {
            DayType.TREND: 0.0,
            DayType.RANGE: 0.0,
            DayType.VOLATILE: 0.0,
        }

        # 1. ADX signal (strongest individual classifier)
        adx = snap.indicators.adx
        if adx is not None:
            if adx > self.ADX_TREND_THRESHOLD:
                votes[DayType.TREND] += 2.0
            elif adx < self.ADX_RANGE_CEILING:
                votes[DayType.RANGE] += 2.0
            else:
                # ADX in no-man's-land (20-22)
                votes[DayType.RANGE] += 0.5

        # 2. VIX signal
        if vix is not None:
            if vix > self.VIX_VOLATILE_THRESHOLD:
                votes[DayType.VOLATILE] += 2.0
            elif vix > 15:
                votes[DayType.VOLATILE] += 0.5
            else:
                votes[DayType.RANGE] += 0.5  # Low VIX → calm / range

        # 3. Opening gap
        gap_pct = self._gap_pct(snap)
        if gap_pct > self.GAP_VOLATILE_PCT:
            votes[DayType.VOLATILE] += 1.5
        elif gap_pct > 0.3:
            votes[DayType.TREND] += 0.5  # Moderate gap may lead to trend

        # 4. Price distance from VWAP
        vwap_dist = self._vwap_distance_pct(snap)
        if vwap_dist > self.VWAP_DISTANCE_TREND:
            votes[DayType.TREND] += 1.5
        else:
            votes[DayType.RANGE] += 1.0  # Price hugging VWAP → range

        # 5. Candle character analysis (directional bodies vs dojis)
        trend_score = self._candle_trend_score(df)
        if trend_score > self.CANDLE_RANGE_RATIO:
            votes[DayType.TREND] += 1.0
        else:
            votes[DayType.RANGE] += 0.5

        # 6. ATR expansion (current ATR vs average of last 20 candles)
        atr_expanded = self._atr_expanding(df)
        if atr_expanded:
            votes[DayType.VOLATILE] += 1.0

        # 7. Bollinger Band width relative to price
        bb_signal = self._bollinger_squeeze(snap)
        if bb_signal == "squeeze":
            votes[DayType.RANGE] += 1.0
        elif bb_signal == "expansion":
            votes[DayType.TREND] += 0.5

        # Winner takes all — but require minimum conviction
        winner = max(votes, key=votes.get)  # type: ignore[arg-type]
        winner_score = votes[winner]
        runner_up = sorted(votes.values(), reverse=True)[1]

        logger.info(
            "DayClassifier: TREND=%.1f RANGE=%.1f VOLATILE=%.1f → %s (margin=%.1f)",
            votes[DayType.TREND], votes[DayType.RANGE], votes[DayType.VOLATILE],
            winner.value, winner_score - runner_up,
        )

        # Need at least 1.0 margin of victory, otherwise UNCLEAR
        if winner_score - runner_up < 1.0:
            return DayType.UNCLEAR

        return winner

    # ── Helper signals ───────────────────────────────────────────────────

    def _gap_pct(self, snap: MarketSnapshot) -> float:
        if snap.day_open and snap.prev_day_close and snap.prev_day_close > 0:
            return abs(snap.day_open - snap.prev_day_close) / snap.prev_day_close * 100
        return 0.0

    def _vwap_distance_pct(self, snap: MarketSnapshot) -> float:
        price = snap.price or snap.nifty_price
        vwap = snap.indicators.vwap
        if vwap and price and vwap > 0:
            return abs(price - vwap) / vwap * 100
        return 0.0

    def _candle_trend_score(self, df: pd.DataFrame) -> float:
        """Fraction of candles that have directional bodies (> 50% of total range)."""
        if len(df) < 5:
            return 0.0
        recent = df.tail(30) if len(df) >= 30 else df
        body = (recent["close"] - recent["open"]).abs()
        full_range = recent["high"] - recent["low"]
        # Avoid division by zero for doji candles
        full_range = full_range.replace(0, float("inf"))
        ratio = body / full_range
        return float((ratio > 0.5).mean())

    def _atr_expanding(self, df: pd.DataFrame) -> bool:
        """Check if recent ATR is expanding vs earlier period."""
        if len(df) < 30:
            return False
        recent_range = (df["high"].tail(10) - df["low"].tail(10)).mean()
        early_range = (df["high"].iloc[:20] - df["low"].iloc[:20]).mean()
        if early_range <= 0:
            return False
        return recent_range > early_range * self.ATR_EXPANSION_FACTOR

    def _bollinger_squeeze(self, snap: MarketSnapshot) -> str:
        """Detect Bollinger Band squeeze or expansion."""
        bb_upper = snap.indicators.bollinger_upper
        bb_lower = snap.indicators.bollinger_lower
        price = snap.price or snap.nifty_price
        if not all([bb_upper, bb_lower, price]) or price <= 0:
            return "neutral"
        bb_width = (bb_upper - bb_lower) / price * 100
        if bb_width < 0.3:
            return "squeeze"
        if bb_width > 0.8:
            return "expansion"
        return "neutral"
