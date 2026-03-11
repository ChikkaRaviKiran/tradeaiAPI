"""Feature engineering engine — computes technical indicators and options metrics."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import ta

from app.core.models import OptionsChainRow, OptionsMetrics, TechnicalIndicators

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Compute technical indicators and options metrics from raw data."""

    # ── Technical Indicators ──────────────────────────────────────────────

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicator columns to the DataFrame.

        Expects OHLCV DataFrame indexed by timestamp.
        """
        if df.empty or len(df) < 14:
            return df

        # Trend: EMAs
        df["ema9"] = ta.trend.ema_indicator(df["close"], window=9)
        df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
        df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)

        # VWAP
        df["vwap"] = self._compute_vwap(df)

        # Momentum: RSI
        df["rsi"] = ta.momentum.rsi(df["close"], window=14)

        # Momentum: MACD
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # Volatility: ATR
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=14
        )

        # Volatility: Bollinger Bands
        bb = ta.volatility.BollingerBands(df["close"])
        df["bollinger_upper"] = bb.bollinger_hband()
        df["bollinger_middle"] = bb.bollinger_mavg()
        df["bollinger_lower"] = bb.bollinger_lband()

        # Trend: ADX
        adx_indicator = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
        df["adx"] = adx_indicator.adx()

        # Derived helpers
        df["ema20_slope"] = df["ema20"].diff(5)
        df["atr_slope"] = df["atr"].diff(5)
        df["avg_volume_10"] = df["volume"].rolling(window=10).mean()

        return df

    def get_latest_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Extract the latest indicator values into a TechnicalIndicators model."""
        if df.empty:
            return TechnicalIndicators()

        last = df.iloc[-1]
        return TechnicalIndicators(
            ema9=_safe(last, "ema9"),
            ema20=_safe(last, "ema20"),
            ema50=_safe(last, "ema50"),
            vwap=_safe(last, "vwap"),
            rsi=_safe(last, "rsi"),
            macd=_safe(last, "macd"),
            macd_signal=_safe(last, "macd_signal"),
            macd_hist=_safe(last, "macd_hist"),
            atr=_safe(last, "atr"),
            bollinger_upper=_safe(last, "bollinger_upper"),
            bollinger_middle=_safe(last, "bollinger_middle"),
            bollinger_lower=_safe(last, "bollinger_lower"),
            adx=_safe(last, "adx"),
        )

    @staticmethod
    def _compute_vwap(df: pd.DataFrame) -> pd.Series:
        """Compute intraday VWAP."""
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_vol = df["volume"].cumsum()
        cum_tp_vol = (typical * df["volume"]).cumsum()
        vwap = cum_tp_vol / cum_vol
        return vwap

    # ── Options Metrics ──────────────────────────────────────────────────

    def compute_options_metrics(
        self, chain: list[OptionsChainRow], spot_price: float
    ) -> OptionsMetrics:
        """Compute PCR, max pain, OI clusters from options chain data."""
        if not chain:
            return OptionsMetrics()

        total_call_oi = sum(r.call_oi for r in chain)
        total_put_oi = sum(r.put_oi for r in chain)

        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

        # Max Pain: strike where total losses (for option writers) are minimized
        max_pain = self._compute_max_pain(chain)

        # OI clusters: strike with highest CE OI and PE OI
        call_oi_cluster = max(chain, key=lambda r: r.call_oi).strike_price if chain else 0
        put_oi_cluster = max(chain, key=lambda r: r.put_oi).strike_price if chain else 0

        total_oi_change = sum(r.change_in_oi for r in chain)

        return OptionsMetrics(
            pcr=round(pcr, 2),
            max_pain=max_pain,
            call_oi_cluster=call_oi_cluster,
            put_oi_cluster=put_oi_cluster,
            oi_change=total_oi_change,
        )

    @staticmethod
    def _compute_max_pain(chain: list[OptionsChainRow]) -> float:
        """Compute max pain strike price.

        Max pain is the strike at which the total value of all outstanding
        options (calls + puts) that expire in-the-money is minimized.
        """
        if not chain:
            return 0.0

        strikes = [r.strike_price for r in chain]
        min_pain_value = float("inf")
        max_pain_strike = 0.0

        for test_strike in strikes:
            total_pain = 0.0
            for r in chain:
                # Call pain: if test_strike > strike, calls are ITM
                if test_strike > r.strike_price:
                    total_pain += (test_strike - r.strike_price) * r.call_oi
                # Put pain: if test_strike < strike, puts are ITM
                if test_strike < r.strike_price:
                    total_pain += (r.strike_price - test_strike) * r.put_oi

            if total_pain < min_pain_value:
                min_pain_value = total_pain
                max_pain_strike = test_strike

        return max_pain_strike


def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    """Safely retrieve a float value from a DataFrame row."""
    val = row.get(col, default)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)
