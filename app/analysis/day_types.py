"""Enhanced 6-type day classifier for both backtest analysis and live trading.

Day Types:
  TREND        — Strong directional move, close near day's extreme
  RANGE        — Sideways, close near midpoint
  VOLATILE     — Wide swings, multiple direction changes
  NARROW_RANGE — Ultra-tight range (inside day)
  GAP_AND_GO   — Large gap that continues in gap direction
  REVERSAL     — Gap or early move that reverses

Two classification modes:
  classify_realtime()   — Uses first 30-45 min of data (for live at ~10:00 AM)
  classify_hindsight()  — Uses full day data (ground truth for analysis)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from app.core.models import DayType

logger = logging.getLogger(__name__)


class EnhancedDayClassifier:
    """6-type day classifier usable in both backtest and live contexts."""

    # ── Thresholds ────────────────────────────────────────────────────
    GAP_LARGE = 0.50
    GAP_MEDIUM = 0.30
    NARROW_RANGE_PCT = 0.25
    RANGE_TIGHT_PCT = 0.40
    VIX_HIGH = 18.0
    ADX_TREND = 22
    ADX_LOW = 18

    # ── Realtime classification (first 30-45 candles) ─────────────────

    def classify_realtime(
        self,
        df: pd.DataFrame,
        prev_close: Optional[float] = None,
        vix: Optional[float] = None,
    ) -> DayType:
        """Classify using first 30-45 min of 1-min data.

        Used at ~10:00 AM in both backtest simulation and live trading.
        Requires enriched DataFrame (with indicators from FeatureEngine).
        """
        if df is None or df.empty or len(df) < 15:
            return DayType.UNCLEAR

        sample = df.iloc[: min(45, len(df))]

        open_price = float(sample.iloc[0]["open"])
        current_close = float(sample.iloc[-1]["close"])
        high = float(sample["high"].max())
        low = float(sample["low"].min())

        if open_price <= 0:
            return DayType.UNCLEAR

        range_pct = (high - low) / open_price * 100

        # Gap
        gap_pct = 0.0
        gap_direction = "flat"
        if prev_close and prev_close > 0:
            gap_pct = abs(open_price - prev_close) / prev_close * 100
            gap_direction = "up" if open_price > prev_close else "down"

        gap_continuing = (
            (gap_direction == "up" and current_close > open_price)
            or (gap_direction == "down" and current_close < open_price)
        )
        gap_reversing = False
        if prev_close and prev_close > 0:
            gap_reversing = (
                (gap_direction == "up" and current_close < prev_close)
                or (gap_direction == "down" and current_close > prev_close)
            )

        # ADX from enriched data
        adx = self._safe_float(sample.iloc[-1].get("adx") if "adx" in sample.columns else None)

        # Candle character
        bodies = (sample["close"] - sample["open"]).abs()
        ranges = (sample["high"] - sample["low"]).replace(0, float("inf"))
        body_ratio = float((bodies / ranges).mean())

        # Direction consistency
        overall_dir = 1 if current_close > open_price else -1
        candle_dirs = np.sign(sample["close"].values - sample["open"].values)
        dir_consistency = float((candle_dirs == overall_dir).mean())

        # Close position within range
        close_pos = (current_close - low) / (high - low) if high > low else 0.5

        # ── Voting ────────────────────────────────────────────────────
        votes = {
            DayType.TREND: 0.0,
            DayType.RANGE: 0.0,
            DayType.VOLATILE: 0.0,
            DayType.NARROW_RANGE: 0.0,
            DayType.GAP_AND_GO: 0.0,
            DayType.REVERSAL: 0.0,
        }

        # 1. Gap signals
        if gap_pct > self.GAP_LARGE and gap_continuing:
            votes[DayType.GAP_AND_GO] += 3.0
        elif gap_pct > self.GAP_MEDIUM and gap_reversing:
            votes[DayType.REVERSAL] += 3.0
        elif gap_pct > self.GAP_MEDIUM and gap_continuing:
            votes[DayType.GAP_AND_GO] += 1.5
            votes[DayType.TREND] += 1.0
        elif gap_pct > self.GAP_MEDIUM:
            votes[DayType.VOLATILE] += 1.0
        elif gap_pct < 0.15:
            votes[DayType.RANGE] += 0.5
            votes[DayType.NARROW_RANGE] += 0.5

        # 2. Range width
        if range_pct < 0.20:
            votes[DayType.NARROW_RANGE] += 3.0
        elif range_pct < self.NARROW_RANGE_PCT:
            votes[DayType.NARROW_RANGE] += 2.0
            votes[DayType.RANGE] += 1.0
        elif range_pct < self.RANGE_TIGHT_PCT:
            votes[DayType.RANGE] += 2.0
        elif range_pct > 0.80:
            votes[DayType.VOLATILE] += 2.0
        elif range_pct > 0.60:
            votes[DayType.TREND] += 1.0

        # 3. Direction consistency
        if dir_consistency > 0.65:
            votes[DayType.TREND] += 2.0
        elif dir_consistency < 0.40:
            votes[DayType.RANGE] += 1.5

        # 4. Body strength
        if body_ratio > 0.60:
            votes[DayType.TREND] += 1.0
        elif body_ratio < 0.35:
            votes[DayType.RANGE] += 1.0

        # 5. Close position
        if close_pos > 0.80 or close_pos < 0.20:
            votes[DayType.TREND] += 1.5
        elif 0.40 < close_pos < 0.60:
            votes[DayType.RANGE] += 1.0

        # 6. VIX
        if vix is not None:
            if vix > self.VIX_HIGH:
                votes[DayType.VOLATILE] += 2.0
            elif vix > 15:
                votes[DayType.VOLATILE] += 0.5
            else:
                votes[DayType.RANGE] += 0.5

        # 7. ADX
        if adx is not None:
            if adx > self.ADX_TREND:
                votes[DayType.TREND] += 2.0
            elif adx < self.ADX_LOW:
                votes[DayType.RANGE] += 1.5
                votes[DayType.NARROW_RANGE] += 0.5

        # Winner with margin requirement
        winner = max(votes, key=votes.get)
        sorted_scores = sorted(votes.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

        if margin < 1.0:
            return DayType.UNCLEAR

        return winner

    # ── Hindsight classification (full day) ───────────────────────────

    def classify_hindsight(
        self,
        df: pd.DataFrame,
        prev_close: Optional[float] = None,
    ) -> DayType:
        """Ground-truth classification using full day's data."""
        if df is None or df.empty or len(df) < 30:
            return DayType.UNCLEAR

        open_price = float(df.iloc[0]["open"])
        close_price = float(df.iloc[-1]["close"])
        day_high = float(df["high"].max())
        day_low = float(df["low"].min())

        if open_price <= 0 or day_high <= day_low:
            return DayType.UNCLEAR

        day_range_pct = (day_high - day_low) / open_price * 100

        # Gap
        gap_pct = 0.0
        gap_direction = "flat"
        if prev_close and prev_close > 0:
            gap_pct = abs(open_price - prev_close) / prev_close * 100
            gap_direction = "up" if open_price > prev_close else "down"

        gap_extended = (
            (gap_direction == "up" and close_price > open_price)
            or (gap_direction == "down" and close_price < open_price)
        )
        gap_reversed = False
        if prev_close and prev_close > 0:
            gap_reversed = (
                (gap_direction == "up" and close_price < prev_close)
                or (gap_direction == "down" and close_price > prev_close)
            )

        # Close position
        close_pos = (close_price - day_low) / (day_high - day_low)
        near_extreme = close_pos > 0.75 or close_pos < 0.25
        near_midpoint = 0.35 < close_pos < 0.65

        # Direction changes from 15-min returns
        direction_changes = self._count_direction_changes(df)

        # Classification rules (most specific first)
        if day_range_pct < 0.35:
            return DayType.NARROW_RANGE

        if gap_pct > 0.40 and gap_extended and near_extreme:
            return DayType.GAP_AND_GO

        if gap_pct > 0.30 and gap_reversed:
            return DayType.REVERSAL

        if day_range_pct > 1.2 and direction_changes >= 4:
            return DayType.VOLATILE

        if day_range_pct > 0.60 and near_extreme and direction_changes < 4:
            return DayType.TREND

        if near_midpoint or day_range_pct < 0.60:
            return DayType.RANGE

        return DayType.TREND if near_extreme else DayType.RANGE

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _count_direction_changes(df: pd.DataFrame) -> int:
        """Count turning points in 15-min resampled closes."""
        try:
            df_15m = df.resample("15min").agg({"close": "last"}).dropna()
            if len(df_15m) < 3:
                return 0
            returns = df_15m["close"].pct_change().dropna()
            signs = np.sign(returns.values)
            return sum(
                1 for i in range(1, len(signs))
                if signs[i] != signs[i - 1] and signs[i] != 0
            )
        except Exception:
            return 0

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        if val is None:
            return None
        try:
            f = float(val)
            return None if np.isnan(f) else f
        except (ValueError, TypeError):
            return None
