"""Abstract base class for market data sources.

Any data provider (AngelOne, Yahoo Finance, Zerodha, etc.) implements this interface.
The orchestrator works against the interface, not the concrete provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from app.core.instruments import InstrumentConfig
from app.core.models import Candle, OptionsChainRow


class BaseDataSource(ABC):
    """Interface all market data providers must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        ...

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the data provider. Returns True on success."""
        ...

    @abstractmethod
    def get_candles(
        self,
        instrument: InstrumentConfig,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> list[Candle]:
        """Fetch OHLCV candle data for an instrument.

        Args:
            instrument: The instrument to fetch data for.
            interval: Candle interval (ONE_MINUTE, FIVE_MINUTE, etc.)
            from_date: Start date (YYYY-MM-DD HH:MM)
            to_date: End date (YYYY-MM-DD HH:MM)
        """
        ...

    @abstractmethod
    def get_futures_candles(
        self,
        instrument: InstrumentConfig,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> list[Candle]:
        """Fetch futures candle data (has real volume for indices)."""
        ...

    @abstractmethod
    def get_ltp(
        self,
        instrument: InstrumentConfig,
        trading_symbol: str,
        token: str,
    ) -> Optional[float]:
        """Fetch last traded price for a specific contract."""
        ...

    @abstractmethod
    def get_option_chain(
        self,
        instrument: InstrumentConfig,
        expiry_date: str,
    ) -> list[OptionsChainRow]:
        """Fetch the full options chain for an instrument around its spot price."""
        ...

    @abstractmethod
    def get_nearest_expiry(
        self,
        instrument: InstrumentConfig,
    ) -> Optional[str]:
        """Find the nearest weekly/monthly expiry for the instrument."""
        ...

    @abstractmethod
    def get_option_token(
        self,
        instrument: InstrumentConfig,
        expiry: str,
        strike: float,
        option_type: str,
    ) -> Optional[dict]:
        """Get broker-specific token info for a specific option contract."""
        ...

    def candles_to_dataframe(self, candles: list[Candle]) -> pd.DataFrame:
        """Convert candle list to timestamp-indexed DataFrame."""
        if not candles:
            return pd.DataFrame()
        data = [c.model_dump() for c in candles]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        return df
