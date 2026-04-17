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

    def compute_htf_bias(self, df_5min: pd.DataFrame) -> str:
        """Compute higher-timeframe trend bias from 5-minute candles.

        Uses EMA9/EMA20 alignment and EMA20 slope on the 5-min chart
        to determine the dominant intraday trend direction.

        Returns: 'bullish', 'bearish', or 'neutral'
        """
        if df_5min.empty or len(df_5min) < 12:
            return "neutral"

        ema9 = ta.trend.ema_indicator(df_5min["close"], window=9)
        ema20 = ta.trend.ema_indicator(df_5min["close"], window=20)

        last_ema9 = ema9.iloc[-1]
        last_ema20 = ema20.iloc[-1]

        if pd.isna(last_ema9) or pd.isna(last_ema20):
            return "neutral"

        # EMA20 slope over last 3 bars for trend momentum
        ema20_slope = ema20.iloc[-1] - ema20.iloc[-4] if len(ema20) >= 4 else 0
        if pd.isna(ema20_slope):
            ema20_slope = 0

        if last_ema9 > last_ema20 and ema20_slope > 0:
            return "bullish"
        elif last_ema9 < last_ema20 and ema20_slope < 0:
            return "bearish"
        return "neutral"

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


# ── Market Structure Detection ───────────────────────────────────────────────


def compute_market_structure(df: pd.DataFrame, swing_lookback: int = 5) -> dict:
    """Detect swing points, trend structure, BOS and CHoCH from OHLCV data.

    Args:
        df: OHLCV DataFrame with at least 20 rows.
        swing_lookback: Number of candles each side to confirm a swing (default 5).

    Returns:
        dict with keys:
          - swing_highs: list of (index, price) tuples
          - swing_lows: list of (index, price) tuples
          - bias: "bullish" | "bearish" | "neutral"
          - last_bos: dict or None — last Break of Structure event
          - last_choch: dict or None — last Change of Character event
          - hh_hl: bool — Higher High / Higher Low sequence present
          - lh_ll: bool — Lower High / Lower Low sequence present
    """
    result = {
        "swing_highs": [],
        "swing_lows": [],
        "bias": "neutral",
        "last_bos": None,
        "last_choch": None,
        "hh_hl": False,
        "lh_ll": False,
    }

    if df.empty or len(df) < swing_lookback * 2 + 1:
        return result

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    idx = df.index

    # Detect swing highs / swing lows (pivot points)
    swing_highs = []  # (position, price)
    swing_lows = []

    for i in range(swing_lookback, len(df) - swing_lookback):
        # Swing high: high[i] > all neighbors within lookback
        if all(highs[i] >= highs[i - j] for j in range(1, swing_lookback + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, swing_lookback + 1)):
            swing_highs.append((i, float(highs[i])))

        # Swing low: low[i] < all neighbors within lookback
        if all(lows[i] <= lows[i - j] for j in range(1, swing_lookback + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, swing_lookback + 1)):
            swing_lows.append((i, float(lows[i])))

    result["swing_highs"] = swing_highs
    result["swing_lows"] = swing_lows

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return result

    # Determine HH/HL (bullish) or LH/LL (bearish) structure
    last_two_highs = swing_highs[-2:]
    last_two_lows = swing_lows[-2:]

    hh = last_two_highs[1][1] > last_two_highs[0][1]
    hl = last_two_lows[1][1] > last_two_lows[0][1]
    lh = last_two_highs[1][1] < last_two_highs[0][1]
    ll = last_two_lows[1][1] < last_two_lows[0][1]

    result["hh_hl"] = hh and hl
    result["lh_ll"] = lh and ll

    if hh and hl:
        result["bias"] = "bullish"
    elif lh and ll:
        result["bias"] = "bearish"

    # Detect BOS: price CLOSES beyond the previous swing high/low
    # BOS bullish: close > prev swing high (body, not just wick)
    # BOS bearish: close < prev swing low
    last_close = float(closes[-1])
    prev_swing_high = swing_highs[-1][1]
    prev_swing_low = swing_lows[-1][1]
    second_last_high = swing_highs[-2][1] if len(swing_highs) >= 2 else None
    second_last_low = swing_lows[-2][1] if len(swing_lows) >= 2 else None

    if last_close > prev_swing_high:
        result["last_bos"] = {
            "type": "bullish",
            "level": prev_swing_high,
            "close": last_close,
        }
    elif last_close < prev_swing_low:
        result["last_bos"] = {
            "type": "bearish",
            "level": prev_swing_low,
            "close": last_close,
        }

    # Detect CHoCH: first structural shift
    # Bullish CHoCH: was making LH/LL, now makes a higher high
    # Bearish CHoCH: was making HH/HL, now makes a lower low
    if len(swing_highs) >= 3 and len(swing_lows) >= 3:
        h3, h2, h1 = [x[1] for x in swing_highs[-3:]]
        l3, l2, l1 = [x[1] for x in swing_lows[-3:]]

        # Was bearish (h3>h2 = LH), now bullish (h1>h2 = HH)
        if h2 < h3 and h1 > h2:
            result["last_choch"] = {"type": "bullish", "level": h2}

        # Was bullish (l2>l3 = HL), now bearish (l1<l2 = LL)
        if l2 > l3 and l1 < l2:
            result["last_choch"] = {"type": "bearish", "level": l2}

    return result


def compute_key_levels(
    df: pd.DataFrame,
    options_metrics=None,
    daily_levels: Optional[dict] = None,
    gex_data: Optional[dict] = None,
) -> list[dict]:
    """Build a priority-ranked key level map from multiple sources.

    Each level is {price, type, strength (1-5), source}.

    Sources:
      - Previous day high/low/close
      - Opening range high/low
      - Swing highs/lows from market structure
      - GEX flip / support / resistance
      - Max pain
      - OI clusters
    """
    levels: list[dict] = []

    if df.empty:
        return levels

    today_str = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], "strftime") else None

    # 1. Previous day high/low/close
    if daily_levels:
        if "prev_high" in daily_levels:
            levels.append({"price": daily_levels["prev_high"], "type": "resistance", "strength": 4, "source": "prev_day_high"})
        if "prev_low" in daily_levels:
            levels.append({"price": daily_levels["prev_low"], "type": "support", "strength": 4, "source": "prev_day_low"})
        if "prev_close" in daily_levels:
            levels.append({"price": daily_levels["prev_close"], "type": "pivot", "strength": 3, "source": "prev_day_close"})

    # 2. Opening range high/low (09:15-09:30)
    if today_str:
        or_mask = (df.index.strftime("%Y-%m-%d") == today_str) & \
                  (df.index.time >= pd.Timestamp("09:15").time()) & \
                  (df.index.time <= pd.Timestamp("09:30").time())
        or_candles = df[or_mask]
        if not or_candles.empty:
            orh = float(or_candles["high"].max())
            orl = float(or_candles["low"].min())
            levels.append({"price": orh, "type": "resistance", "strength": 4, "source": "orb_high"})
            levels.append({"price": orl, "type": "support", "strength": 4, "source": "orb_low"})

    # 3. Swing points
    structure = compute_market_structure(df)
    for _, price in structure["swing_highs"][-3:]:
        levels.append({"price": price, "type": "resistance", "strength": 3, "source": "swing_high"})
    for _, price in structure["swing_lows"][-3:]:
        levels.append({"price": price, "type": "support", "strength": 3, "source": "swing_low"})

    # 4. GEX levels
    if gex_data:
        if gex_data.get("flip_strike"):
            levels.append({"price": gex_data["flip_strike"], "type": "pivot", "strength": 5, "source": "gex_flip"})
        for s in gex_data.get("support_zones", [])[:2]:
            levels.append({"price": s, "type": "support", "strength": 4, "source": "gex_support"})
        for r in gex_data.get("resistance_zones", [])[:2]:
            levels.append({"price": r, "type": "resistance", "strength": 4, "source": "gex_resistance"})

    # 5. Options metrics
    if options_metrics:
        if options_metrics.max_pain:
            levels.append({"price": options_metrics.max_pain, "type": "pivot", "strength": 3, "source": "max_pain"})
        if options_metrics.call_oi_cluster:
            levels.append({"price": options_metrics.call_oi_cluster, "type": "resistance", "strength": 3, "source": "call_oi_wall"})
        if options_metrics.put_oi_cluster:
            levels.append({"price": options_metrics.put_oi_cluster, "type": "support", "strength": 3, "source": "put_oi_wall"})

    # Sort by strength descending
    levels.sort(key=lambda x: x["strength"], reverse=True)
    return levels


def analyze_candle_character(df: pd.DataFrame) -> dict:
    """Analyze the character of the most recent candle.

    Returns:
        dict with keys:
          - type: "momentum" | "rejection" | "absorption" | "indecision" | "neutral"
          - direction: "bullish" | "bearish" | "neutral"
          - body_ratio: float (body / full range, 0-1)
          - upper_wick_ratio: float
          - lower_wick_ratio: float
          - volume_character: "climax" | "acceleration" | "declining" | "normal"
    """
    result = {
        "type": "neutral",
        "direction": "neutral",
        "body_ratio": 0.0,
        "upper_wick_ratio": 0.0,
        "lower_wick_ratio": 0.0,
        "volume_character": "normal",
    }

    if df.empty or len(df) < 2:
        return result

    last = df.iloc[-1]
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    vol = float(last.get("volume", 0))
    avg_vol = float(df["volume"].tail(10).mean()) if len(df) >= 10 else vol

    candle_range = h - l
    if candle_range <= 0:
        return result

    body = abs(c - o)
    body_ratio = body / candle_range
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    upper_wick_ratio = upper_wick / candle_range
    lower_wick_ratio = lower_wick / candle_range

    result["body_ratio"] = round(body_ratio, 3)
    result["upper_wick_ratio"] = round(upper_wick_ratio, 3)
    result["lower_wick_ratio"] = round(lower_wick_ratio, 3)
    result["direction"] = "bullish" if c > o else ("bearish" if c < o else "neutral")

    # Classify candle type
    if body_ratio > 0.65:
        result["type"] = "momentum"  # Strong directional candle
    elif upper_wick_ratio > 0.5 and lower_wick_ratio < 0.15:
        result["type"] = "rejection"  # Shooting star / bearish rejection
        result["direction"] = "bearish"
    elif lower_wick_ratio > 0.5 and upper_wick_ratio < 0.15:
        result["type"] = "rejection"  # Hammer / bullish rejection
        result["direction"] = "bullish"
    elif body_ratio < 0.25:
        result["type"] = "indecision"  # Doji-like
    elif upper_wick_ratio > 0.3 and lower_wick_ratio > 0.3:
        result["type"] = "absorption"  # Both sides tested, body small

    # Volume character
    if avg_vol > 0 and vol > 0:
        vol_ratio = vol / avg_vol
        if vol_ratio >= 2.5:
            result["volume_character"] = "climax"
        elif vol_ratio >= 1.5:
            result["volume_character"] = "acceleration"
        elif vol_ratio < 0.7:
            result["volume_character"] = "declining"

    return result


# ── Candlestick Pattern Detection ─────────────────────────────────────

def _candle_parts(row):
    """Extract OHLCV components and derived ratios for one candle."""
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    vol = float(row.get("volume", 0))
    rng = h - l
    if rng <= 0:
        return None
    body = abs(c - o)
    return {
        "o": o, "h": h, "l": l, "c": c, "vol": vol,
        "range": rng, "body": body,
        "body_ratio": body / rng,
        "upper_wick": (h - max(o, c)) / rng,
        "lower_wick": (min(o, c) - l) / rng,
        "bullish": c > o, "bearish": c < o,
    }


def detect_candlestick_patterns(df: pd.DataFrame) -> list[dict]:
    """Detect classic candlestick patterns on the last 3 candles.

    Returns list of dicts, each: {pattern, direction, strength (1-5)}.
    Strength accounts for body size, wick quality, and volume.
    """
    patterns: list[dict] = []
    if df is None or len(df) < 2:
        return patterns

    avg_vol = float(df["volume"].tail(10).mean()) if len(df) >= 10 else 1.0
    avg_vol = max(avg_vol, 1.0)

    cur = _candle_parts(df.iloc[-1])
    if cur is None:
        return patterns

    vol_boost = min(cur["vol"] / avg_vol, 3.0)  # Cap at 3x for strength calc

    # ── Single-candle patterns ───────────────────────────────

    # Marubozu — virtually no wicks, decisive direction
    if cur["body_ratio"] > 0.95:
        direction = "bullish" if cur["bullish"] else "bearish"
        strength = min(5, int(2 + vol_boost))
        patterns.append({"pattern": "marubozu", "direction": direction, "strength": strength})

    # Pin Bar (refined hammer / shooting star)
    if cur["body_ratio"] < 0.33:
        if cur["lower_wick"] >= 0.60 and cur["upper_wick"] < 0.10:
            strength = min(5, int(2 + vol_boost))
            patterns.append({"pattern": "pin_bar_hammer", "direction": "bullish", "strength": strength})
        elif cur["upper_wick"] >= 0.60 and cur["lower_wick"] < 0.10:
            strength = min(5, int(2 + vol_boost))
            patterns.append({"pattern": "pin_bar_shooting_star", "direction": "bearish", "strength": strength})

    # Spinning Top — small body, roughly equal wicks
    if 0.20 <= cur["body_ratio"] <= 0.40 and cur["upper_wick"] >= 0.20 and cur["lower_wick"] >= 0.20:
        patterns.append({"pattern": "spinning_top", "direction": "neutral", "strength": 1})

    # Doji variants — tiny body
    if cur["body_ratio"] < 0.05:
        if cur["upper_wick"] > 0.30 and cur["lower_wick"] > 0.30:
            patterns.append({"pattern": "doji_long_legged", "direction": "neutral", "strength": 2})
        elif cur["lower_wick"] > 0.60:
            patterns.append({"pattern": "doji_dragonfly", "direction": "bullish", "strength": 2})
        elif cur["upper_wick"] > 0.60:
            patterns.append({"pattern": "doji_gravestone", "direction": "bearish", "strength": 2})
        else:
            patterns.append({"pattern": "doji", "direction": "neutral", "strength": 1})

    # ── Two-candle patterns ──────────────────────────────────

    if len(df) >= 3:
        prev = _candle_parts(df.iloc[-2])
        if prev is not None:
            prev_body_top = max(prev["o"], prev["c"])
            prev_body_bot = min(prev["o"], prev["c"])
            cur_body_top = max(cur["o"], cur["c"])
            cur_body_bot = min(cur["o"], cur["c"])

            # Bullish Engulfing: prev bearish + cur bullish body engulfs prev body
            if prev["bearish"] and cur["bullish"] and cur_body_bot <= prev_body_bot and cur_body_top >= prev_body_top and cur["body"] > prev["body"]:
                strength = min(5, int(3 + vol_boost * 0.5))
                patterns.append({"pattern": "bullish_engulfing", "direction": "bullish", "strength": strength})

            # Bearish Engulfing
            if prev["bullish"] and cur["bearish"] and cur_body_bot <= prev_body_bot and cur_body_top >= prev_body_top and cur["body"] > prev["body"]:
                strength = min(5, int(3 + vol_boost * 0.5))
                patterns.append({"pattern": "bearish_engulfing", "direction": "bearish", "strength": strength})

            # Bullish Harami: prev bearish large + cur bullish small inside
            if prev["bearish"] and cur["bullish"] and prev["body_ratio"] > 0.50 and cur_body_top <= prev_body_top and cur_body_bot >= prev_body_bot and cur["body"] < prev["body"] * 0.5:
                patterns.append({"pattern": "bullish_harami", "direction": "bullish", "strength": 2})

            # Bearish Harami
            if prev["bullish"] and cur["bearish"] and prev["body_ratio"] > 0.50 and cur_body_top <= prev_body_top and cur_body_bot >= prev_body_bot and cur["body"] < prev["body"] * 0.5:
                patterns.append({"pattern": "bearish_harami", "direction": "bearish", "strength": 2})

            # Piercing Line: prev bearish + cur bullish opens below prev low, closes above 50% of prev body
            prev_mid = (prev_body_top + prev_body_bot) / 2
            if prev["bearish"] and cur["bullish"] and cur["o"] <= prev["l"] and cur["c"] > prev_mid and cur["c"] < prev_body_top:
                strength = min(5, int(2 + vol_boost * 0.5))
                patterns.append({"pattern": "piercing_line", "direction": "bullish", "strength": strength})

            # Dark Cloud Cover: prev bullish + cur bearish opens above prev high, closes below 50%
            if prev["bullish"] and cur["bearish"] and cur["o"] >= prev["h"] and cur["c"] < prev_mid and cur["c"] > prev_body_bot:
                strength = min(5, int(2 + vol_boost * 0.5))
                patterns.append({"pattern": "dark_cloud_cover", "direction": "bearish", "strength": strength})

            # Tweezer Bottom: matching lows within 0.05%
            low_tol = prev["l"] * 0.0005
            if abs(prev["l"] - cur["l"]) <= low_tol and cur["bullish"] and prev["bearish"]:
                patterns.append({"pattern": "tweezer_bottom", "direction": "bullish", "strength": 2})

            # Tweezer Top: matching highs within 0.05%
            high_tol = prev["h"] * 0.0005
            if abs(prev["h"] - cur["h"]) <= high_tol and cur["bearish"] and prev["bullish"]:
                patterns.append({"pattern": "tweezer_top", "direction": "bearish", "strength": 2})

    # ── Three-candle patterns ────────────────────────────────

    if len(df) >= 4:
        c1 = _candle_parts(df.iloc[-3])
        c2 = _candle_parts(df.iloc[-2])
        c3 = cur  # already computed
        if c1 is not None and c2 is not None:
            c1_mid = (c1["o"] + c1["c"]) / 2

            # Morning Star: c1 bearish large, c2 small body, c3 bullish closes above c1 midpoint
            if (c1["bearish"] and c1["body_ratio"] > 0.50
                    and c2["body_ratio"] < 0.30
                    and c3["bullish"] and c3["c"] > c1_mid):
                strength = min(5, int(3 + vol_boost * 0.5))
                patterns.append({"pattern": "morning_star", "direction": "bullish", "strength": strength})

            # Evening Star: c1 bullish large, c2 small body, c3 bearish closes below c1 midpoint
            if (c1["bullish"] and c1["body_ratio"] > 0.50
                    and c2["body_ratio"] < 0.30
                    and c3["bearish"] and c3["c"] < c1_mid):
                strength = min(5, int(3 + vol_boost * 0.5))
                patterns.append({"pattern": "evening_star", "direction": "bearish", "strength": strength})

            # Three White Soldiers: 3 consecutive bullish, higher closes, decent bodies
            if (c1["bullish"] and c2["bullish"] and c3["bullish"]
                    and c2["c"] > c1["c"] and c3["c"] > c2["c"]
                    and c1["body_ratio"] > 0.50 and c2["body_ratio"] > 0.50 and c3["body_ratio"] > 0.50):
                patterns.append({"pattern": "three_white_soldiers", "direction": "bullish", "strength": 4})

            # Three Black Crows: 3 consecutive bearish, lower closes, decent bodies
            if (c1["bearish"] and c2["bearish"] and c3["bearish"]
                    and c2["c"] < c1["c"] and c3["c"] < c2["c"]
                    and c1["body_ratio"] > 0.50 and c2["body_ratio"] > 0.50 and c3["body_ratio"] > 0.50):
                patterns.append({"pattern": "three_black_crows", "direction": "bearish", "strength": 4})

    return patterns


# ── Move Detection Engine ─────────────────────────────────────────────
# Detects real market BEHAVIOR (expansion → hold → continuation) rather
# than candle shapes.  Primary signal layer for trend capture.
# ──────────────────────────────────────────────────────────────────────

def detect_move(df: pd.DataFrame) -> dict | None:
    """Detect an actionable move starting: expansion + hold + continuation.

    Requires a DataFrame with OHLCV + vwap columns (at least 20 rows).
    Returns a dict describing the move, or None if no move detected.

    Detection layers:
      1. EXPANSION  — current candle range > 1.5× avg_range_10
      2. HOLDING    — next 3 candles hold above/below 50% of expansion body
      3. CONTINUATION — price breaks the expansion candle high/low after hold
      4. VWAP SUPPORT — price stays on the correct side of VWAP

    Output dict keys:
      direction, expansion_idx, expansion_range, expansion_ratio,
      hold_candles, hold_quality, continuation_break, vwap_aligned,
      confidence (0-100), entry_price, move_type
    """
    if df is None or len(df) < 20:
        return None

    # Precompute rolling averages
    ranges = (df["high"] - df["low"]).astype(float)
    avg_range = ranges.rolling(10).mean()
    bodies = (df["close"] - df["open"]).abs().astype(float)
    avg_body = bodies.rolling(10).mean()

    # Scan from candle 14 onward (need 10 avg + room for hold/continuation)
    # Check the LAST viable expansion in the recent window for real-time use
    best_move = None

    scan_start = max(14, len(df) - 30)  # Last 30 candles only
    scan_end = len(df) - 4              # Need 3 hold + 1 continuation candle after

    for i in range(scan_start, scan_end):
        ar = avg_range.iloc[i]
        ab = avg_body.iloc[i]
        if ar is None or pd.isna(ar) or ar <= 0:
            continue

        c_range = float(ranges.iloc[i])
        c_body = float(bodies.iloc[i])
        o_i = float(df.iloc[i]["open"])
        h_i = float(df.iloc[i]["high"])
        l_i = float(df.iloc[i]["low"])
        c_i = float(df.iloc[i]["close"])
        v_i = float(df.iloc[i].get("volume", 0))

        expansion_ratio = c_range / ar

        # ── Layer 1: EXPANSION ───────────────────────────────
        if expansion_ratio < 1.5:
            continue

        # Must be a decisive candle (body > 50% of range)
        body_ratio = c_body / c_range if c_range > 0 else 0
        if body_ratio < 0.50:
            continue

        is_bullish = c_i > o_i
        direction = "bullish" if is_bullish else "bearish"

        # Expansion body midpoint (50% retracement level)
        body_top = max(o_i, c_i)
        body_bot = min(o_i, c_i)
        body_mid = (body_top + body_bot) / 2

        # ── Layer 2: HOLDING ─────────────────────────────────
        # Next 3 candles must hold above body_mid (bullish) or below (bearish)
        hold_candles = 0
        hold_quality = 0.0  # Average: how much of each hold candle stays in zone

        for j in range(1, 4):
            if i + j >= len(df):
                break
            hj = float(df.iloc[i + j]["high"])
            lj = float(df.iloc[i + j]["low"])
            cj = float(df.iloc[i + j]["close"])

            if is_bullish:
                # Bullish hold: low must stay above body midpoint
                if lj >= body_mid:
                    hold_candles += 1
                    # Quality: how far above mid did the close stay?
                    hold_quality += (cj - body_mid) / c_body if c_body > 0 else 0
                else:
                    break
            else:
                # Bearish hold: high must stay below body midpoint
                if hj <= body_mid:
                    hold_candles += 1
                    hold_quality += (body_mid - cj) / c_body if c_body > 0 else 0
                else:
                    break

        if hold_candles < 2:  # Need at least 2 of 3 candles holding
            continue

        hold_quality = hold_quality / hold_candles if hold_candles > 0 else 0

        # ── Layer 3: CONTINUATION ────────────────────────────
        # After hold period, check if price breaks the expansion high/low
        cont_idx = i + hold_candles + 1
        continuation_break = False

        if cont_idx < len(df):
            cont_high = float(df.iloc[cont_idx]["high"])
            cont_low = float(df.iloc[cont_idx]["low"])

            if is_bullish and cont_high > h_i:
                continuation_break = True
            elif not is_bullish and cont_low < l_i:
                continuation_break = True

        # Also check the candle after that for delayed continuation
        if not continuation_break and cont_idx + 1 < len(df):
            cont_high2 = float(df.iloc[cont_idx + 1]["high"])
            cont_low2 = float(df.iloc[cont_idx + 1]["low"])
            if is_bullish and cont_high2 > h_i:
                continuation_break = True
            elif not is_bullish and cont_low2 < l_i:
                continuation_break = True

        # ── Layer 4: VWAP ALIGNMENT ──────────────────────────
        vwap_aligned = False
        vwap_val = df.iloc[i].get("vwap") if "vwap" in df.columns else None
        if vwap_val is not None and not pd.isna(vwap_val):
            vwap_f = float(vwap_val)
            if is_bullish and c_i > vwap_f:
                vwap_aligned = True
            elif not is_bullish and c_i < vwap_f:
                vwap_aligned = True

        # ── EMA alignment check ──────────────────────────────
        ema_aligned = False
        if "ema9" in df.columns and "ema20" in df.columns:
            e9 = df.iloc[i].get("ema9")
            e20 = df.iloc[i].get("ema20")
            if e9 is not None and e20 is not None and not pd.isna(e9) and not pd.isna(e20):
                if is_bullish and float(e9) > float(e20):
                    ema_aligned = True
                elif not is_bullish and float(e9) < float(e20):
                    ema_aligned = True

        # ── Confidence scoring ───────────────────────────────
        confidence = 0
        confidence += 20 if expansion_ratio >= 1.5 else 0       # Expansion exists
        confidence += 10 if expansion_ratio >= 2.0 else 0       # Strong expansion
        confidence += 5  if expansion_ratio >= 2.5 else 0       # Very strong
        confidence += 15 if hold_candles >= 3 else (10 if hold_candles >= 2 else 0)
        confidence += 10 if hold_quality >= 0.5 else 5          # Quality of hold
        confidence += 20 if continuation_break else 0           # Continuation confirmed
        confidence += 10 if vwap_aligned else 0                 # VWAP support
        confidence += 10 if ema_aligned else 0                  # EMA trend support

        # Minimum viable move: expansion + hold (continuation is bonus)
        if confidence < 40:
            continue

        # Classify move type
        if continuation_break and hold_candles >= 3:
            move_type = "confirmed_trend_start"
        elif continuation_break and hold_candles >= 2:
            move_type = "probable_trend_start"
        elif hold_candles >= 3 and not continuation_break:
            move_type = "holding_pre_breakout"
        else:
            move_type = "expansion_hold"

        # Entry price: for bullish, above expansion high; for bearish, below low
        entry_price = h_i if is_bullish else l_i

        move = {
            "direction": direction,
            "expansion_idx": i,
            "expansion_range": round(c_range, 2),
            "expansion_ratio": round(expansion_ratio, 2),
            "body_ratio": round(body_ratio, 3),
            "hold_candles": hold_candles,
            "hold_quality": round(hold_quality, 3),
            "continuation_break": continuation_break,
            "vwap_aligned": vwap_aligned,
            "ema_aligned": ema_aligned,
            "confidence": confidence,
            "entry_price": round(entry_price, 2),
            "move_type": move_type,
        }

        # Keep the highest-confidence move
        if best_move is None or confidence > best_move["confidence"]:
            best_move = move

    return best_move


def scan_all_moves(df: pd.DataFrame) -> list[dict]:
    """Scan an entire day's DataFrame and return ALL detected moves.

    Unlike detect_move() which returns only the best recent move,
    this scans every candle and collects every qualifying expansion+hold.
    Used for backtesting.
    """
    if df is None or len(df) < 20:
        return []

    ranges = (df["high"] - df["low"]).astype(float)
    avg_range = ranges.rolling(10).mean()
    bodies = (df["close"] - df["open"]).abs().astype(float)

    moves: list[dict] = []
    cooldown_until = -1  # Avoid overlapping moves

    for i in range(14, len(df) - 4):
        if i <= cooldown_until:
            continue

        ar = avg_range.iloc[i]
        if ar is None or pd.isna(ar) or ar <= 0:
            continue

        c_range = float(ranges.iloc[i])
        c_body = float(bodies.iloc[i])
        o_i = float(df.iloc[i]["open"])
        h_i = float(df.iloc[i]["high"])
        l_i = float(df.iloc[i]["low"])
        c_i = float(df.iloc[i]["close"])

        expansion_ratio = c_range / ar
        if expansion_ratio < 1.5:
            continue

        body_ratio = c_body / c_range if c_range > 0 else 0
        if body_ratio < 0.50:
            continue

        is_bullish = c_i > o_i
        body_top = max(o_i, c_i)
        body_bot = min(o_i, c_i)
        body_mid = (body_top + body_bot) / 2

        # Hold check
        hold_candles = 0
        hold_quality = 0.0

        for j in range(1, 4):
            if i + j >= len(df):
                break
            hj = float(df.iloc[i + j]["high"])
            lj = float(df.iloc[i + j]["low"])
            cj = float(df.iloc[i + j]["close"])

            if is_bullish:
                if lj >= body_mid:
                    hold_candles += 1
                    hold_quality += (cj - body_mid) / c_body if c_body > 0 else 0
                else:
                    break
            else:
                if hj <= body_mid:
                    hold_candles += 1
                    hold_quality += (body_mid - cj) / c_body if c_body > 0 else 0
                else:
                    break

        if hold_candles < 2:
            continue

        hold_quality = hold_quality / hold_candles

        # Continuation check
        cont_idx = i + hold_candles + 1
        continuation_break = False
        if cont_idx < len(df):
            if is_bullish and float(df.iloc[cont_idx]["high"]) > h_i:
                continuation_break = True
            elif not is_bullish and float(df.iloc[cont_idx]["low"]) < l_i:
                continuation_break = True
        if not continuation_break and cont_idx + 1 < len(df):
            if is_bullish and float(df.iloc[cont_idx + 1]["high"]) > h_i:
                continuation_break = True
            elif not is_bullish and float(df.iloc[cont_idx + 1]["low"]) < l_i:
                continuation_break = True

        # VWAP
        vwap_aligned = False
        vwap_val = df.iloc[i].get("vwap") if "vwap" in df.columns else None
        if vwap_val is not None and not pd.isna(vwap_val):
            vwap_f = float(vwap_val)
            if is_bullish and c_i > vwap_f:
                vwap_aligned = True
            elif not is_bullish and c_i < vwap_f:
                vwap_aligned = True

        # EMA
        ema_aligned = False
        if "ema9" in df.columns and "ema20" in df.columns:
            e9 = df.iloc[i].get("ema9")
            e20 = df.iloc[i].get("ema20")
            if e9 is not None and e20 is not None and not pd.isna(e9) and not pd.isna(e20):
                if is_bullish and float(e9) > float(e20):
                    ema_aligned = True
                elif not is_bullish and float(e9) < float(e20):
                    ema_aligned = True

        # Confidence
        confidence = 0
        confidence += 20 if expansion_ratio >= 1.5 else 0
        confidence += 10 if expansion_ratio >= 2.0 else 0
        confidence += 5  if expansion_ratio >= 2.5 else 0
        confidence += 15 if hold_candles >= 3 else (10 if hold_candles >= 2 else 0)
        confidence += 10 if hold_quality >= 0.5 else 5
        confidence += 20 if continuation_break else 0
        confidence += 10 if vwap_aligned else 0
        confidence += 10 if ema_aligned else 0

        if confidence < 40:
            continue

        # Pattern confirmation (secondary — boosts confidence)
        pattern_confirms = False
        if len(df) >= i + 1:
            window = df.iloc[max(0, i - 3): i + 1]
            pats = detect_candlestick_patterns(window)
            direction = "bullish" if is_bullish else "bearish"
            for p in pats:
                if p["direction"] == direction:
                    pattern_confirms = True
                    confidence = min(100, confidence + 5)
                    break

        if continuation_break and hold_candles >= 3:
            move_type = "confirmed_trend_start"
        elif continuation_break and hold_candles >= 2:
            move_type = "probable_trend_start"
        elif hold_candles >= 3:
            move_type = "holding_pre_breakout"
        else:
            move_type = "expansion_hold"

        moves.append({
            "idx": i,
            "direction": "bullish" if is_bullish else "bearish",
            "expansion_ratio": round(expansion_ratio, 2),
            "body_ratio": round(body_ratio, 3),
            "hold_candles": hold_candles,
            "hold_quality": round(hold_quality, 3),
            "continuation_break": continuation_break,
            "vwap_aligned": vwap_aligned,
            "ema_aligned": ema_aligned,
            "pattern_confirms": pattern_confirms,
            "confidence": confidence,
            "entry_price": round(h_i if is_bullish else l_i, 2),
            "move_type": move_type,
        })

        # Cooldown: skip next 5 candles to avoid duplicate detections
        cooldown_until = i + 5

    return moves


# ── Production Trade Selection ────────────────────────────────────────


def assess_day_quality(df: pd.DataFrame, check_candles: int = 45) -> dict:
    """Assess whether the day is tradeable using first N candles (~10:00 AM).

    Checks:
      1. Range expansion — are candles moving or flat?
      2. VWAP stability — is price trending or oscillating around VWAP?
      3. Directional bias — does the first hour show clear direction?

    Returns dict with:
      tradeable (bool), reason (str), volatility_score (0-100),
      direction_score (0-100), vwap_trend (str)
    """
    if df is None or len(df) < check_candles:
        return {"tradeable": False, "reason": "insufficient_data",
                "volatility_score": 0, "direction_score": 0, "vwap_trend": "unknown"}

    window = df.iloc[:check_candles]
    ranges = (window["high"] - window["low"]).astype(float)
    closes = window["close"].astype(float).values

    # 1. Range expansion: compare last-15 avg range to first-15 avg range
    first_avg = float(ranges.iloc[:15].mean()) if len(ranges) >= 15 else float(ranges.mean())
    last_avg = float(ranges.iloc[-15:].mean()) if len(ranges) >= 30 else float(ranges.mean())
    range_ratio = last_avg / first_avg if first_avg > 0 else 1.0

    # 2. VWAP stability: how many times does price cross VWAP?
    vwap_crosses = 0
    if "vwap" in df.columns:
        vwap_vals = window["vwap"].astype(float).values
        above = closes > vwap_vals
        for k in range(1, len(above)):
            if above[k] != above[k - 1]:
                vwap_crosses += 1

    # 3. Direction: net move from open to current close
    first_close = float(closes[0])
    last_close = float(closes[-1])
    net_pct = abs(last_close - first_close) / first_close * 100 if first_close > 0 else 0

    # Score: higher = more tradeable
    volatility_score = min(100, int(range_ratio * 30 + net_pct * 40))
    direction_score = max(0, min(100, int(net_pct * 80 - vwap_crosses * 3)))

    # VWAP trend
    if vwap_crosses <= 4:
        vwap_trend = "trending"
    elif vwap_crosses <= 8:
        vwap_trend = "mild_chop"
    else:
        vwap_trend = "choppy"

    # Decision — relaxed: only block truly choppy days
    tradeable = (
        vwap_trend != "choppy"
        and (direction_score >= 5 or volatility_score >= 40)
    )

    reason = "ok" if tradeable else (
        "choppy_vwap" if vwap_trend == "choppy"
        else "low_direction" if direction_score < 15
        else "low_volatility"
    )

    return {
        "tradeable": tradeable,
        "reason": reason,
        "volatility_score": volatility_score,
        "direction_score": direction_score,
        "vwap_trend": vwap_trend,
        "vwap_crosses": vwap_crosses,
        "net_move_pct": round(net_pct, 4),
        "range_ratio": round(range_ratio, 2),
    }


def detect_micro_pullback(df: pd.DataFrame, move: dict,
                          max_wait: int = 10) -> dict | None:
    """After a move is detected, wait for a micro-pullback into the move body.

    Instead of entering at the expansion high/low (aggressive), wait for price
    to pull back toward the expansion body midpoint, then re-break out.

    Args:
        df: Full day DataFrame.
        move: Move dict from scan_all_moves / detect_move.
        max_wait: Max candles to wait after move detection for a pullback.

    Returns:
        dict with pullback_entry, pullback_idx, pullback_depth, or None.
    """
    idx = move["idx"]
    direction = move["direction"]
    hold_end = idx + move["hold_candles"]

    # Get expansion candle metrics
    o_i = float(df.iloc[idx]["open"])
    h_i = float(df.iloc[idx]["high"])
    l_i = float(df.iloc[idx]["low"])
    c_i = float(df.iloc[idx]["close"])
    body_top = max(o_i, c_i)
    body_bot = min(o_i, c_i)
    body_mid = (body_top + body_bot) / 2

    # Scan candles after the continuation break
    scan_start = hold_end + 1
    scan_end = min(scan_start + max_wait, len(df) - 1)

    for j in range(scan_start, scan_end):
        hj = float(df.iloc[j]["high"])
        lj = float(df.iloc[j]["low"])
        cj = float(df.iloc[j]["close"])

        if direction == "bullish":
            # Pullback: low dips toward body_top but doesn't break body_mid
            if lj <= body_top and lj >= body_mid:
                # Now check if the NEXT candle re-breaks upward
                if j + 1 < len(df):
                    next_h = float(df.iloc[j + 1]["high"])
                    if next_h > hj:
                        depth = (body_top - lj) / (body_top - body_bot) if body_top > body_bot else 0
                        return {
                            "pullback_entry": round(hj, 2),
                            "pullback_idx": j + 1,
                            "pullback_depth": round(depth, 3),
                            "candles_waited": j - hold_end,
                        }
        else:
            # Pullback: high rises toward body_bot but doesn't break body_mid
            if hj >= body_bot and hj <= body_mid:
                if j + 1 < len(df):
                    next_l = float(df.iloc[j + 1]["low"])
                    if next_l < lj:
                        depth = (hj - body_bot) / (body_top - body_bot) if body_top > body_bot else 0
                        return {
                            "pullback_entry": round(lj, 2),
                            "pullback_idx": j + 1,
                            "pullback_depth": round(depth, 3),
                            "candles_waited": j - hold_end,
                        }

    return None


def select_production_trade(df: pd.DataFrame, moves: list[dict],
                            min_confidence: int = 70,
                            min_confidence_bullish: int = 80,
                            earliest_candle: int = 45,
                            deadline_candle: int = 135,
                            max_trades: int = 1) -> list[dict]:
    """Apply production filters to select only the best trade(s) of the day.

    Rules:
      1. Only confirmed_trend_start moves
      2. Confidence ≥ min_confidence (bearish) or min_confidence_bullish (bullish)
      3. Not before earliest_candle (~10:00 AM, 45 candles from 09:15)
      4. If no move by deadline_candle (~11:30 AM), skip the day
      5. Take first N qualifying moves (default: 1)

    Returns list of selected moves (with pullback info attached if found).
    """
    selected: list[dict] = []

    for move in moves:
        if move["move_type"] != "confirmed_trend_start":
            continue

        # Asymmetric confidence: bullish requires stronger signal
        required_conf = min_confidence_bullish if move["direction"] == "bullish" else min_confidence
        if move["confidence"] < required_conf:
            continue

        if move["idx"] < earliest_candle:
            continue
        if move["idx"] > deadline_candle:
            break  # Past deadline — no more trades

        # Default: aggressive entry at expansion break
        move_with_pb = {**move}
        move_with_pb["pullback"] = None
        move_with_pb["final_entry"] = move["entry_price"]
        move_with_pb["final_entry_idx"] = move["idx"] + move["hold_candles"] + 1

        # Optional: if a clean pullback appears, improve the entry
        pullback = detect_micro_pullback(df, move)
        if pullback and pullback["pullback_depth"] >= 0.3:
            # Only use pullback if depth ≥ 30% of body (clean pullback)
            move_with_pb["pullback"] = pullback
            move_with_pb["final_entry"] = pullback["pullback_entry"]
            move_with_pb["final_entry_idx"] = pullback["pullback_idx"]

        selected.append(move_with_pb)
        if len(selected) >= max_trades:
            break

    return selected


def select_weekly_best(daily_trades: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """From daily trade selections, pick only ONE trade per week.

    Takes the FIRST qualifying signal of each ISO week (earliest date).
    This avoids lookahead bias — in live trading you don't know future signals.
    Returns filtered dict: {date: [trade]} for only the selected days.
    """
    from datetime import datetime

    weekly_buckets: dict[str, list[tuple[str, dict]]] = {}

    for dt_str, trades in daily_trades.items():
        if not trades:
            continue
        dt = datetime.strptime(dt_str, "%Y-%m-%d")
        week_key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
        for trade in trades:
            weekly_buckets.setdefault(week_key, []).append((dt_str, trade))

    result: dict[str, list[dict]] = {}
    for week_key, candidates in weekly_buckets.items():
        # Take the FIRST signal of the week (earliest date) — no lookahead bias
        candidates.sort(key=lambda x: x[0])  # sort by date string
        first_dt, first_trade = candidates[0]
        result[first_dt] = [first_trade]

    return result


def find_nearest_levels(
    spot_price: float,
    levels: list[dict],
    max_distance_pct: float = 1.0,
) -> dict:
    """Find nearest support and resistance from key levels relative to spot.

    Returns:
        dict with nearest_support, nearest_resistance, at_key_level (bool),
        distance_to_support_pct, distance_to_resistance_pct.
    """
    nearest_support = None
    nearest_resistance = None
    min_sup_dist = float("inf")
    min_res_dist = float("inf")

    for lvl in levels:
        price = lvl["price"]
        dist = abs(price - spot_price) / spot_price * 100
        if dist > max_distance_pct:
            continue

        if lvl["type"] in ("support", "pivot") and price <= spot_price:
            if spot_price - price < min_sup_dist:
                min_sup_dist = spot_price - price
                nearest_support = lvl
        elif lvl["type"] in ("resistance", "pivot") and price >= spot_price:
            if price - spot_price < min_res_dist:
                min_res_dist = price - spot_price
                nearest_resistance = lvl

    sup_dist_pct = (min_sup_dist / spot_price * 100) if nearest_support else None
    res_dist_pct = (min_res_dist / spot_price * 100) if nearest_resistance else None

    return {
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "at_key_level": (sup_dist_pct is not None and sup_dist_pct < 0.15)
                        or (res_dist_pct is not None and res_dist_pct < 0.15),
        "distance_to_support_pct": round(sup_dist_pct, 3) if sup_dist_pct else None,
        "distance_to_resistance_pct": round(res_dist_pct, 3) if res_dist_pct else None,
    }
