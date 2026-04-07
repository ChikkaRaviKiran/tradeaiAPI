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
