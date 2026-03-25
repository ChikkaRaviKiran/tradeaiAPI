"""Base strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from app.core.models import OptionsMetrics, StrategySignal


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @abstractmethod
    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
        daily_levels: Optional[dict] = None,
    ) -> Optional[StrategySignal]:
        """Evaluate the strategy and return a signal if conditions are met.

        Args:
            df: OHLCV DataFrame with technical indicators.
            options_metrics: Current options chain metrics.
            spot_price: Current NIFTY spot price.
            daily_levels: Optional dict with pre-computed daily levels
                          (e.g. {'high_20d': ..., 'low_20d': ..., 'avg_volume_20d': ...}).

        Returns:
            StrategySignal if conditions met, else None.
        """
        ...
