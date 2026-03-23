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
            1. Missing data (candle gap > 2 minutes)
            2. Price spike (> 3% change in 1 minute)
            3. Volume must be > 0 (skipped for index data where volume is always 0)
        """
        if df.empty:
            return df

        if not is_index:
            df = self._remove_zero_volume(df)
        df = self._remove_price_spikes(df)
        return df

    def has_data_gap(self, df: pd.DataFrame) -> bool:
        """Check if there is a candle gap > MAX_CANDLE_GAP_MINUTES in recent data."""
        if len(df) < 2:
            return False

        # Only check today's candles — overnight/weekend gaps are expected
        # when previous-day data is included for indicator warmup
        today = df.index[-1].strftime("%Y-%m-%d")
        today_df = df[df.index.strftime("%Y-%m-%d") == today]
        if len(today_df) < 2:
            return False

        recent = today_df.tail(10)
        timestamps = recent.index.to_series()
        diffs = timestamps.diff().dropna()
        max_gap = diffs.max()
        if max_gap > timedelta(minutes=self.MAX_CANDLE_GAP_MINUTES):
            logger.warning(
                "Data gap detected: %s (max allowed: %d min)",
                max_gap,
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
