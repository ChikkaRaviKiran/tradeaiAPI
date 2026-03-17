"""Yahoo Finance data source — stocks, global indices, and supplementary data.

Used for:
  - Stock OHLCV data (equity instruments)
  - Global market indices (already in global_markets.py, this wraps yfinance)
  - FII/DII flow data (from NSE)
  - Market breadth (advance-decline)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pytz
import yfinance as yf

from app.core.instruments import InstrumentConfig, InstrumentType
from app.core.models import Candle, OptionsChainRow
from app.data.base import BaseDataSource

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

# Yahoo Finance symbol mapping for NSE stocks
_NSE_SUFFIX = ".NS"
_INDEX_MAP = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "NIFTY_MID_SELECT.NS",
}


class YahooFinanceClient(BaseDataSource):
    """Yahoo Finance data source for stocks and indices."""

    @property
    def name(self) -> str:
        return "YahooFinance"

    def authenticate(self) -> bool:
        """Yahoo Finance is public — no auth needed."""
        return True

    def _get_yf_symbol(self, instrument: InstrumentConfig) -> str:
        """Map instrument to Yahoo Finance ticker symbol."""
        if instrument.symbol in _INDEX_MAP:
            return _INDEX_MAP[instrument.symbol]
        if instrument.instrument_type == InstrumentType.EQUITY:
            return f"{instrument.symbol}{_NSE_SUFFIX}"
        return instrument.symbol

    def get_candles(
        self,
        instrument: InstrumentConfig,
        interval: str = "1m",
        from_date: str = "",
        to_date: str = "",
    ) -> list[Candle]:
        """Fetch OHLCV from Yahoo Finance.

        Args:
            interval: yfinance interval — '1m', '5m', '15m', '1h', '1d'
            from_date/to_date: YYYY-MM-DD format
        """
        yf_symbol = self._get_yf_symbol(instrument)
        try:
            # yfinance interval mapping
            yf_interval = self._map_interval(interval)

            ticker = yf.Ticker(yf_symbol)
            if from_date and to_date:
                df = ticker.history(
                    start=from_date.split(" ")[0],
                    end=to_date.split(" ")[0],
                    interval=yf_interval,
                )
            else:
                # Default: today's data
                df = ticker.history(period="1d", interval=yf_interval)

            if df.empty:
                logger.warning("No data from Yahoo Finance for %s", yf_symbol)
                return []

            candles = []
            for ts, row in df.iterrows():
                candles.append(
                    Candle(
                        symbol=instrument.symbol,
                        timestamp=ts.to_pydatetime(),
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=int(row.get("Volume", 0)),
                    )
                )
            logger.info("Yahoo Finance: %d candles for %s", len(candles), yf_symbol)
            return candles

        except Exception:
            logger.exception("Yahoo Finance error for %s", yf_symbol)
            return []

    def get_futures_candles(
        self,
        instrument: InstrumentConfig,
        interval: str = "1m",
        from_date: str = "",
        to_date: str = "",
    ) -> list[Candle]:
        """Yahoo Finance doesn't have NSE futures. Return empty."""
        return []

    def get_ltp(
        self,
        instrument: InstrumentConfig,
        trading_symbol: str = "",
        token: str = "",
    ) -> Optional[float]:
        """Get last traded price from Yahoo Finance."""
        yf_symbol = self._get_yf_symbol(instrument)
        try:
            ticker = yf.Ticker(yf_symbol)
            info = ticker.fast_info
            return float(info.get("lastPrice", 0) or info.get("last_price", 0))
        except Exception:
            logger.exception("Yahoo Finance LTP error for %s", yf_symbol)
            return None

    def get_option_chain(
        self,
        instrument: InstrumentConfig,
        expiry_date: str = "",
    ) -> list[OptionsChainRow]:
        """Yahoo Finance option chain — available for NSE stocks."""
        yf_symbol = self._get_yf_symbol(instrument)
        try:
            ticker = yf.Ticker(yf_symbol)
            expirations = ticker.options
            if not expirations:
                return []

            # Use nearest expiry
            exp = expirations[0]
            chain = ticker.option_chain(exp)

            rows = []
            calls = chain.calls.set_index("strike") if not chain.calls.empty else pd.DataFrame()
            puts = chain.puts.set_index("strike") if not chain.puts.empty else pd.DataFrame()
            all_strikes = sorted(set(calls.index.tolist() + puts.index.tolist()))

            for strike in all_strikes:
                row = OptionsChainRow(strike_price=float(strike))
                if strike in calls.index:
                    c = calls.loc[strike]
                    row.call_ltp = float(c.get("lastPrice", 0))
                    row.call_oi = int(c.get("openInterest", 0))
                    row.call_volume = int(c.get("volume", 0))
                    row.implied_volatility = float(c.get("impliedVolatility", 0))
                if strike in puts.index:
                    p = puts.loc[strike]
                    row.put_ltp = float(p.get("lastPrice", 0))
                    row.put_oi = int(p.get("openInterest", 0))
                    row.put_volume = int(p.get("volume", 0))
                rows.append(row)

            return rows

        except Exception:
            logger.exception("Yahoo Finance option chain error for %s", yf_symbol)
            return []

    def get_nearest_expiry(self, instrument: InstrumentConfig) -> Optional[str]:
        """Get nearest expiry from Yahoo Finance."""
        yf_symbol = self._get_yf_symbol(instrument)
        try:
            ticker = yf.Ticker(yf_symbol)
            expirations = ticker.options
            return expirations[0] if expirations else None
        except Exception:
            return None

    def get_option_token(
        self,
        instrument: InstrumentConfig,
        expiry: str,
        strike: float,
        option_type: str,
    ) -> Optional[dict]:
        """Not applicable for Yahoo Finance."""
        return None

    def get_daily_ohlcv(self, instrument: InstrumentConfig, period: str = "6mo") -> pd.DataFrame:
        """Fetch daily OHLCV for backtesting and trend analysis."""
        yf_symbol = self._get_yf_symbol(instrument)
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval="1d")
            if df.empty:
                return pd.DataFrame()
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception:
            logger.exception("Yahoo Finance daily data error for %s", yf_symbol)
            return pd.DataFrame()

    @staticmethod
    def _map_interval(interval: str) -> str:
        """Map generic interval names to yfinance format."""
        mapping = {
            "ONE_MINUTE": "1m",
            "FIVE_MINUTE": "5m",
            "FIFTEEN_MINUTE": "15m",
            "ONE_HOUR": "1h",
            "ONE_DAY": "1d",
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "1d": "1d",
        }
        return mapping.get(interval, "1m")
