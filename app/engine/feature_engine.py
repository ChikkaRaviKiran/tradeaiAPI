"""Feature engineering engine — computes technical indicators and options metrics."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import ta

from app.core.models import OptionsChainRow, OptionsMetrics, TechnicalIndicators

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Compute technical indicators and options metrics from raw data."""

    def __init__(self) -> None:
        self._prev_total_oi: int = 0  # Track OI change between fetches

    # ── Technical Indicators ──────────────────────────────────────────────

    def compute_indicators(self, df: pd.DataFrame, today_date: str | None = None) -> pd.DataFrame:
        """Add all technical indicator columns to the DataFrame.

        Expects OHLCV DataFrame indexed by timestamp.
        Gracefully handles insufficient candles for each indicator group.
        If today_date is provided (YYYY-MM-DD), VWAP is computed intraday-only.
        """
        if df.empty or len(df) < 2:
            return df

        n = len(df)
        _nan = pd.Series([float("nan")] * n, index=df.index)

        # Trend: EMAs (always computable)
        df["ema9"] = ta.trend.ema_indicator(df["close"], window=9)
        df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
        df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
        # EMA200 — only reliable with 200+ candles; mark as NaN otherwise
        if len(df) >= 200:
            df["ema200"] = ta.trend.ema_indicator(df["close"], window=200)
        else:
            # Not enough data for stable EMA200 — leave as NaN
            # Strategies and scorers will see None and skip EMA200 checks
            df["ema200"] = float("nan")

        # VWAP (intraday only — reset each day)
        if today_date:
            today_mask = df.index.strftime("%Y-%m-%d") == today_date
            df["vwap"] = float("nan")
            if today_mask.any():
                df.loc[today_mask, "vwap"] = self._compute_vwap(df[today_mask]).values
        else:
            df["vwap"] = self._compute_vwap(df)

        # Indicators that need ≥14 candles (RSI, ATR, MACD, Bollinger)
        if n >= 14:
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
        else:
            for col in ("rsi", "macd", "macd_signal", "macd_hist", "atr",
                        "bollinger_upper", "bollinger_middle", "bollinger_lower"):
                df[col] = _nan

        # Trend: ADX (requires 2×window candles internally; guard against crash)
        try:
            adx_indicator = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
            df["adx"] = adx_indicator.adx()
        except (IndexError, Exception):
            df["adx"] = pd.Series([float("nan")] * len(df), index=df.index)

        # Derived helpers
        df["ema20_slope"] = df["ema20"].diff(5)
        df["atr_slope"] = df["atr"].diff(5)
        df["avg_volume_10"] = df["volume"].rolling(window=10).mean()

        # Price momentum — rate of change over 10 candles
        df["roc_10"] = df["close"].pct_change(10) * 100

        # Trend strength composite: EMA alignment score
        # +1 if ema9>ema20, +1 if ema20>ema50, +1 if ema50>ema200
        df["trend_strength"] = (
            (df["ema9"] > df["ema20"]).astype(int)
            + (df["ema20"] > df["ema50"]).astype(int)
            + (df["ema50"] > df["ema200"]).astype(int)
        )

        return df

    def get_latest_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Extract the latest indicator values into a TechnicalIndicators model."""
        if df.empty:
            return TechnicalIndicators()

        last = df.iloc[-1]
        vol_sum = df["volume"].sum() if "volume" in df.columns else 0
        return TechnicalIndicators(
            ema9=_safe(last, "ema9"),
            ema20=_safe(last, "ema20"),
            ema50=_safe(last, "ema50"),
            ema200=_safe(last, "ema200"),
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
            trend_strength=int(last["trend_strength"]) if "trend_strength" in last.index and pd.notna(last.get("trend_strength")) else None,
            vwap_is_volume_weighted=(vol_sum > 0),
        )

    @staticmethod
    def _compute_vwap(df: pd.DataFrame) -> pd.Series:
        """Compute intraday VWAP.

        Uses real volume if available (futures data merged in).
        Falls back to tick-weighted typical price for pure index data,
        but marks it as non-volume-weighted so scoring can distinguish.
        """
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"]

        if vol.sum() == 0:
            # No volume data — simple cumulative average (NOT real VWAP)
            cum_count = pd.Series(range(1, len(df) + 1), index=df.index, dtype=float)
            vwap = typical.cumsum() / cum_count
        else:
            cum_vol = vol.cumsum()
            cum_tp_vol = (typical * vol).cumsum()
            vwap = cum_tp_vol / cum_vol

        return vwap

    def merge_futures_volume(self, spot_df: pd.DataFrame, futures_df: pd.DataFrame) -> pd.DataFrame:
        """Replace spot NIFTY volume (always 0) with NIFTY Futures volume.

        This enables real volume-weighted VWAP and meaningful volume analysis.
        Both DataFrames must be timestamp-indexed 1-min candles.
        """
        if futures_df.empty:
            return spot_df

        # Align futures volume to spot timestamps
        fut_vol = futures_df[["volume"]].rename(columns={"volume": "fut_volume"})
        merged = spot_df.join(fut_vol, how="left")
        merged["fut_volume"] = merged["fut_volume"].fillna(0).astype(int)

        # Replace zero spot volume with futures volume
        merged["volume"] = merged["fut_volume"]
        merged.drop(columns=["fut_volume"], inplace=True)

        logger.info(
            "Merged futures volume: %d candles, total vol=%d",
            len(merged), merged["volume"].sum(),
        )
        return merged

    # ── Options Metrics ──────────────────────────────────────────────────

    def compute_options_metrics(
        self, chain: list[OptionsChainRow], spot_price: float
    ) -> OptionsMetrics:
        """Compute PCR, max pain, OI clusters, and volume from options chain."""
        if not chain:
            return OptionsMetrics()

        total_call_oi = sum(r.call_oi for r in chain)
        total_put_oi = sum(r.put_oi for r in chain)
        total_call_volume = sum(r.call_volume for r in chain)
        total_put_volume = sum(r.put_volume for r in chain)

        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else None

        # Max Pain
        max_pain = self._compute_max_pain(chain)

        # OI clusters: strike with highest CE OI and PE OI
        call_oi_cluster = max(chain, key=lambda r: r.call_oi).strike_price if chain else None
        put_oi_cluster = max(chain, key=lambda r: r.put_oi).strike_price if chain else None

        # ATM option volume (±100 points around spot)
        atm_volume = sum(
            r.call_volume + r.put_volume
            for r in chain
            if abs(r.strike_price - spot_price) <= 100
        )

        # OI change: compare total OI against previous fetch
        current_total_oi = total_call_oi + total_put_oi
        oi_change = current_total_oi - self._prev_total_oi if self._prev_total_oi > 0 else 0
        self._prev_total_oi = current_total_oi

        return OptionsMetrics(
            pcr=round(pcr, 2) if pcr is not None else None,
            max_pain=max_pain,
            call_oi_cluster=call_oi_cluster,
            put_oi_cluster=put_oi_cluster,
            oi_change=oi_change,
            total_call_volume=total_call_volume,
            total_put_volume=total_put_volume,
            atm_option_volume=atm_volume,
        )

    @staticmethod
    def _compute_max_pain(chain: list[OptionsChainRow]) -> Optional[float]:
        """Compute max pain strike price.

        Max pain is the strike at which the total value of all outstanding
        options (calls + puts) that expire in-the-money is minimized.
        """
        if not chain:
            return None

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


def _safe(row: pd.Series, col: str) -> Optional[float]:
    """Safely retrieve a float value from a DataFrame row. Returns None if missing/NaN."""
    val = row.get(col)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return float(val)
