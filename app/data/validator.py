"""Data validation layer — ensures data quality before processing."""

from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates market data before it enters the processing pipeline."""

    MAX_CANDLE_GAP_MINUTES = 2
    MAX_PRICE_CHANGE_PCT = 3.0

    def validate_candles(self, df: pd.DataFrame, is_index: bool = False) -> pd.DataFrame:
        """Run all validations and return cleaned DataFrame.

        Checks:
            1. OHLCV consistency (high >= low, high >= open/close, etc.)
            2. Missing data (candle gap > 2 minutes)
            3. Price spike (> 3% change in 1 minute)
            4. Volume must be > 0 (skipped for index data where volume is always 0)
        """
        if df.empty:
            return df

        df = self._validate_ohlcv(df)
        if not is_index:
            df = self._remove_zero_volume(df)
        df = self._remove_price_spikes(df)
        return df

    def has_data_gap(self, df: pd.DataFrame) -> bool:
        """Check if there is a candle gap > MAX_CANDLE_GAP_MINUTES in today's data."""
        if len(df) < 2:
            return False

        # Only check today's candles — overnight/weekend gaps are expected
        today = df.index[-1].strftime("%Y-%m-%d")
        today_df = df[df.index.strftime("%Y-%m-%d") == today]
        if len(today_df) < 2:
            return False

        # Check ALL today's candles (not just last 10)
        timestamps = today_df.index.to_series()
        diffs = timestamps.diff().dropna()
        gaps = diffs[diffs > timedelta(minutes=self.MAX_CANDLE_GAP_MINUTES)]
        if len(gaps) > 0:
            max_gap = gaps.max()
            logger.warning(
                "Data gap detected: %s (%d gap(s), max allowed: %d min)",
                max_gap,
                len(gaps),
                self.MAX_CANDLE_GAP_MINUTES,
            )
            return True
        return False

    def _remove_price_spikes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove candles where price change > 3% in one minute."""
        if len(df) < 2:
            return df

        pct_change = df["close"].pct_change().abs() * 100
        spike_mask = pct_change > self.MAX_PRICE_CHANGE_PCT
        n_spikes = spike_mask.sum()
        if n_spikes > 0:
            logger.warning("Removed %d price spike candles (>%.1f%%)", n_spikes, self.MAX_PRICE_CHANGE_PCT)
        return df[~spike_mask]

    def _remove_zero_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove candles with zero volume."""
        zero_vol = df["volume"] <= 0
        n_zero = zero_vol.sum()
        if n_zero > 0:
            logger.warning("Removed %d zero-volume candles", n_zero)
        return df[~zero_vol]

    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reject candles with invalid OHLC relationships.

        Ensures: high >= low, high >= open, high >= close,
                 low <= open, low <= close, all prices > 0.
        """
        if df.empty:
            return df

        invalid = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
            | (df["close"] <= 0)
            | (df["open"] <= 0)
        )
        n_invalid = invalid.sum()
        if n_invalid > 0:
            logger.error("Rejected %d invalid OHLCV candles", n_invalid)
        return df[~invalid]

    def is_valid_for_trading(self, df: pd.DataFrame) -> bool:
        """Check if the validated data is sufficient for trading."""
        if df.empty:
            logger.warning("No data available")
            return False
        if len(df) < 5:
            logger.warning("Not enough candles for analysis: %d", len(df))
            return False
        if self.has_data_gap(df):
            return False
        return True
