"""Adaptive Market Structure Strategy — context-aware entry logic.

Instead of fixed mechanical rules, this strategy reads market structure and
enters on CONFIRMED reactions at key levels.

Three setups:
  1. Trend continuation after pullback to structure level
  2. Reversal at liquidity sweep + failed breakout
  3. Breakout with structure confirmation (wait for confirmation candle)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from app.core.models import OptionType, OptionsMetrics, StrategyName, StrategySignal
from app.engine.feature_engine import (
    analyze_candle_character,
    compute_key_levels,
    compute_market_structure,
    find_nearest_levels,
)
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class AdaptiveStrategy(BaseStrategy):
    """Context-aware adaptive strategy with 3 trade setups."""

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
        daily_levels: Optional[dict] = None,
        structure_data: Optional[dict] = None,
    ) -> Optional[StrategySignal]:
        if df.empty or len(df) < 20:
            return None

        # Compute structure if not provided
        structure = structure_data or compute_market_structure(df)
        candle = analyze_candle_character(df)

        # Build key levels (pass daily levels as prev day levels if available)
        prev_levels = {}
        if daily_levels:
            prev_levels = daily_levels
        key_levels = compute_key_levels(df, options_metrics, prev_levels)
        level_info = find_nearest_levels(spot_price, key_levels)

        last = df.iloc[-1]
        rsi = last.get("rsi")
        ema9 = last.get("ema9")
        ema20 = last.get("ema20")
        ema50 = last.get("ema50")
        vwap = last.get("vwap")
        adx = last.get("adx")
        volume = last.get("volume", 0)
        avg_vol = last.get("avg_volume_10", 0) or 1

        # Require basic indicator data
        if any(v is None or (isinstance(v, float) and v != v) for v in [rsi, ema9, ema20]):
            return None

        # Try each setup in priority order
        signal = self._setup_trend_continuation(
            df, structure, candle, level_info, spot_price, rsi, ema9, ema20, ema50, vwap, adx, volume, avg_vol
        )
        if signal:
            return signal

        signal = self._setup_liquidity_sweep(
            df, structure, candle, level_info, key_levels, spot_price, rsi, volume, avg_vol
        )
        if signal:
            return signal

        signal = self._setup_confirmed_breakout(
            df, structure, candle, level_info, spot_price, rsi, ema9, ema20, volume, avg_vol
        )
        if signal:
            return signal

        return None

    def _setup_trend_continuation(
        self, df, structure, candle, level_info, spot_price,
        rsi, ema9, ema20, ema50, vwap, adx, volume, avg_vol,
    ) -> Optional[StrategySignal]:
        """Setup 1: Trend continuation after pullback to structure level.

        Conditions:
        - Market structure = uptrend (HH/HL) or downtrend (LH/LL)
        - Price has pulled back to key level / EMA / BOS level
        - Candle shows rejection at that level (hammer/engulfing)
        - Volume declining during pullback, increasing on bounce
        """
        bias = structure.get("bias", "neutral")
        if bias == "neutral":
            return None

        at_level = level_info.get("at_key_level", False)
        has_ema50 = ema50 is not None and not (isinstance(ema50, float) and ema50 != ema50)

        if bias == "bullish" and structure.get("hh_hl"):
            # Bullish trend — look for pullback to support
            pullback_to_ema = (abs(spot_price - ema20) / spot_price * 100) < 0.40
            pullback_to_ema50 = has_ema50 and (abs(spot_price - ema50) / spot_price * 100) < 0.40
            at_bos = structure.get("last_bos") and structure["last_bos"]["type"] == "bullish" and \
                     abs(spot_price - structure["last_bos"]["level"]) / spot_price * 100 < 0.30

            pullback_confirmed = pullback_to_ema or pullback_to_ema50 or at_level or at_bos

            if not pullback_confirmed:
                return None

            # Candle must show bullish rejection or momentum
            if candle["direction"] != "bullish":
                return None
            if candle["type"] not in ("rejection", "momentum", "absorption"):
                return None

            # RSI should not be overbought (pullback should reset RSI)
            if rsi > 68:
                return None

            # Volume: ideally declining before and increasing on this candle
            vol_ratio = volume / avg_vol if avg_vol > 0 else 1
            if vol_ratio < 0.5:
                return None  # Too low even for pullback bounce

            # Structure stoploss: below nearest swing low
            sl_level = None
            if structure["swing_lows"]:
                sl_level = structure["swing_lows"][-1][1]
            target_level = None
            if structure["swing_highs"]:
                target_level = structure["swing_highs"][-1][1]

            return StrategySignal(
                strategy=StrategyName.ADAPTIVE,
                option_type=OptionType.CALL,
                strike_price=_nearest_strike(spot_price, "CE"),
                details={
                    "setup": "trend_continuation",
                    "structure_bias": bias,
                    "pullback_to": "ema20" if pullback_to_ema else ("ema50" if pullback_to_ema50 else "key_level"),
                    "candle_type": candle["type"],
                    "rsi": rsi,
                    "volume_ratio": round(vol_ratio, 2),
                    "structure_sl": sl_level,
                    "structure_target": target_level,
                },
            )

        elif bias == "bearish" and structure.get("lh_ll"):
            # Bearish trend — look for pullback to resistance
            pullback_to_ema = (abs(spot_price - ema20) / spot_price * 100) < 0.40
            pullback_to_ema50 = has_ema50 and (abs(spot_price - ema50) / spot_price * 100) < 0.40
            at_bos = structure.get("last_bos") and structure["last_bos"]["type"] == "bearish" and \
                     abs(spot_price - structure["last_bos"]["level"]) / spot_price * 100 < 0.30

            pullback_confirmed = pullback_to_ema or pullback_to_ema50 or at_level or at_bos

            if not pullback_confirmed:
                return None

            if candle["direction"] != "bearish":
                return None
            if candle["type"] not in ("rejection", "momentum", "absorption"):
                return None

            if rsi < 32:
                return None

            vol_ratio = volume / avg_vol if avg_vol > 0 else 1
            if vol_ratio < 0.5:
                return None

            sl_level = None
            if structure["swing_highs"]:
                sl_level = structure["swing_highs"][-1][1]
            target_level = None
            if structure["swing_lows"]:
                target_level = structure["swing_lows"][-1][1]

            return StrategySignal(
                strategy=StrategyName.ADAPTIVE,
                option_type=OptionType.PUT,
                strike_price=_nearest_strike(spot_price, "PE"),
                details={
                    "setup": "trend_continuation",
                    "structure_bias": bias,
                    "pullback_to": "ema20" if pullback_to_ema else ("ema50" if pullback_to_ema50 else "key_level"),
                    "candle_type": candle["type"],
                    "rsi": rsi,
                    "volume_ratio": round(vol_ratio, 2),
                    "structure_sl": sl_level,
                    "structure_target": target_level,
                },
            )

        return None

    def _setup_liquidity_sweep(
        self, df, structure, candle, level_info, key_levels, spot_price, rsi, volume, avg_vol,
    ) -> Optional[StrategySignal]:
        """Setup 2: Reversal at liquidity sweep + failed breakout.

        Conditions:
        - Price swept a key level (went through it) in previous candles
        - Now CLOSED BACK inside the range (failed breakout by close)
        - Volume spike on the sweep then declining
        - Reversal candle (opposite direction to sweep)
        """
        if len(df) < 5:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(last["close"])
        prev_close = float(prev["close"])
        prev_high = float(prev["high"])
        prev_low = float(prev["low"])
        prev_vol = float(prev.get("volume", 0))

        # Look for a sweep above resistance then close back below
        for lvl in key_levels:
            if lvl["type"] != "resistance" or lvl["strength"] < 3:
                continue
            level_price = lvl["price"]

            # Previous candle swept above the level (wick poked above)
            if prev_high > level_price and prev_close < level_price:
                # Current candle also closed below = confirmed failed breakout
                if close < level_price:
                    # Bearish reversal candle
                    if candle["direction"] != "bearish":
                        continue
                    if candle["type"] not in ("momentum", "rejection"):
                        continue
                    # Volume spike on sweep candle
                    vol_ratio = prev_vol / avg_vol if avg_vol > 0 else 1
                    if vol_ratio < 1.2:
                        continue

                    sl_level = prev_high  # Beyond the sweep extreme

                    return StrategySignal(
                        strategy=StrategyName.ADAPTIVE,
                        option_type=OptionType.PUT,
                        strike_price=_nearest_strike(spot_price, "PE"),
                        details={
                            "setup": "liquidity_sweep",
                            "sweep_level": level_price,
                            "sweep_source": lvl["source"],
                            "sweep_high": prev_high,
                            "candle_type": candle["type"],
                            "rsi": rsi,
                            "volume_ratio": round(vol_ratio, 2),
                            "structure_sl": sl_level,
                        },
                    )

        # Look for a sweep below support then close back above
        for lvl in key_levels:
            if lvl["type"] != "support" or lvl["strength"] < 3:
                continue
            level_price = lvl["price"]

            if prev_low < level_price and prev_close > level_price:
                if close > level_price:
                    if candle["direction"] != "bullish":
                        continue
                    if candle["type"] not in ("momentum", "rejection"):
                        continue
                    vol_ratio = prev_vol / avg_vol if avg_vol > 0 else 1
                    if vol_ratio < 1.2:
                        continue

                    sl_level = prev_low

                    return StrategySignal(
                        strategy=StrategyName.ADAPTIVE,
                        option_type=OptionType.CALL,
                        strike_price=_nearest_strike(spot_price, "CE"),
                        details={
                            "setup": "liquidity_sweep",
                            "sweep_level": level_price,
                            "sweep_source": lvl["source"],
                            "sweep_low": prev_low,
                            "candle_type": candle["type"],
                            "rsi": rsi,
                            "volume_ratio": round(vol_ratio, 2),
                            "structure_sl": sl_level,
                        },
                    )

        return None

    def _setup_confirmed_breakout(
        self, df, structure, candle, level_info, spot_price, rsi, ema9, ema20, volume, avg_vol,
    ) -> Optional[StrategySignal]:
        """Setup 3: Breakout WITH structure confirmation.

        Conditions:
        - Price broke a key level with full-body candle (not wick poke)
        - Volume > 2x average on breakout
        - Previous candle was the breakout; current candle holds above level
        - Enter on confirmation candle close
        """
        if len(df) < 3:
            return None

        last = df.iloc[-1]  # Confirmation candle
        prev = df.iloc[-2]  # Breakout candle
        close = float(last["close"])
        prev_close = float(prev["close"])
        prev_open = float(prev["open"])
        prev_vol = float(prev.get("volume", 0))
        prev_body_ratio = abs(prev_close - prev_open) / max(float(prev["high"]) - float(prev["low"]), 0.01)

        # Breakout candle must have strong body (>55%)
        if prev_body_ratio < 0.55:
            return None

        # Volume on breakout candle > 2x average
        vol_ratio = prev_vol / avg_vol if avg_vol > 0 else 1

        # For index data with no volume, use range vs ATR as proxy
        is_index = prev_vol == 0 and (avg_vol is None or avg_vol <= 1)
        if is_index:
            atr = last.get("atr")
            if atr and atr > 0:
                prev_range = float(prev["high"]) - float(prev["low"])
                vol_ratio = prev_range / atr
            else:
                vol_ratio = 1.5  # Pass if ATR unavailable

        if vol_ratio < 1.5:
            return None

        # Check for resistance breakout: prev close broke above resistance
        nearest_res = level_info.get("nearest_resistance")
        nearest_sup = level_info.get("nearest_support")

        # Bullish breakout above resistance
        if nearest_res and prev_close > nearest_res["price"]:
            # Confirmation: current candle holds above the broken level
            if close > nearest_res["price"]:
                if ema9 <= ema20:
                    return None  # Need EMA alignment

                sl_level = nearest_res["price"]  # Below breakout level

                return StrategySignal(
                    strategy=StrategyName.ADAPTIVE,
                    option_type=OptionType.CALL,
                    strike_price=_nearest_strike(spot_price, "CE"),
                    details={
                        "setup": "confirmed_breakout",
                        "breakout_level": nearest_res["price"],
                        "breakout_source": nearest_res["source"],
                        "candle_type": candle["type"],
                        "rsi": rsi,
                        "volume_ratio": round(vol_ratio, 2),
                        "confirmed_hold": True,
                        "structure_sl": sl_level,
                    },
                )

        # Bearish breakout below support
        if nearest_sup and prev_close < nearest_sup["price"]:
            if close < nearest_sup["price"]:
                if ema9 >= ema20:
                    return None

                sl_level = nearest_sup["price"]

                return StrategySignal(
                    strategy=StrategyName.ADAPTIVE,
                    option_type=OptionType.PUT,
                    strike_price=_nearest_strike(spot_price, "PE"),
                    details={
                        "setup": "confirmed_breakout",
                        "breakout_level": nearest_sup["price"],
                        "breakout_source": nearest_sup["source"],
                        "candle_type": candle["type"],
                        "rsi": rsi,
                        "volume_ratio": round(vol_ratio, 2),
                        "confirmed_hold": True,
                        "structure_sl": sl_level,
                    },
                )

        return None


def _nearest_strike(price: float, opt_type: str) -> float:
    """Round price to nearest 50 strike, ATM."""
    return round(price / 50) * 50
