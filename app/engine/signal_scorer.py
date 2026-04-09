"""LOCKED v1.0 Signal Scoring System.

5-factor confluence scoring (max 100):
  Strategy Strength:    30  (trigger quality, RSI, candle body, MACD)
  Market Alignment:     25  (VWAP alignment, EMA trend, regime fit)
  Volume Confirmation:  20  (futures volume, ATM option volume, candle range)
  Options OI Signal:    15  (PCR, OI change, OI buildup)
  Volatility Context:   10  (ATR ratio, VIX alignment)

DO NOT CHANGE for 10-15 trading days.
"""

from __future__ import annotations

import logging

import pandas as pd

from app.core.models import (
    GlobalBias,
    OptionType,
    OptionsMetrics,
    SignalScore,
    StrategySignal,
)

logger = logging.getLogger(__name__)


class SignalScorer:
    """LOCKED v1.0: Score a strategy signal across 5 factors (max 100)."""

    def __init__(self) -> None:
        self._prev_atm_volume: int = 0

    def score(
        self,
        signal: StrategySignal,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        global_bias: GlobalBias,
        structure_data: dict | None = None,
    ) -> SignalScore:
        """Compute 5-factor score for a strategy signal."""
        scores = SignalScore()

        # Factor 1: Strategy Strength (0-30)
        scores.strategy_strength = self._score_strategy_strength(signal, df)

        # Factor 2: Market Alignment (0-25)
        scores.market_alignment = self._score_market_alignment(signal, df)

        # Factor 3: Volume Confirmation (0-20)
        scores.volume_confirmation = self._score_volume(signal, df, options_metrics)

        # Factor 4: Options OI Signal (0-15)
        scores.options_oi_signal = self._score_options_oi(signal, options_metrics)

        # Factor 5: Volatility Context (0-10)
        scores.volatility_context = self._score_volatility(signal, df)

        scores.total = (
            scores.strategy_strength
            + scores.market_alignment
            + scores.volume_confirmation
            + scores.options_oi_signal
            + scores.volatility_context
        )

        logger.info(
            "Signal score: %.0f (strength=%.0f align=%.0f vol=%.0f oi=%.0f vola=%.0f) %s %s",
            scores.total,
            scores.strategy_strength,
            scores.market_alignment,
            scores.volume_confirmation,
            scores.options_oi_signal,
            scores.volatility_context,
            signal.strategy.value,
            signal.option_type.value,
        )
        return scores

    # ── Factor 1: Strategy Strength (0-30) ───────────────────────────────

    def _score_strategy_strength(self, signal: StrategySignal, df: pd.DataFrame) -> float:
        """Score based on strategy trigger quality (0-30).

        Components:
          - RSI directional confirmation (+10)
          - Strong candle body (+7)
          - Strategy-specific quality metrics (+7)
          - MACD histogram confirmation (+6)
        """
        if df.empty:
            return 0.0

        score = 0.0
        last = df.iloc[-1]

        # RSI confirmation (+10)
        rsi = last.get("rsi")
        has_rsi = rsi is not None and not (isinstance(rsi, float) and pd.isna(rsi))
        is_pullback = signal.strategy.value in ("TREND_PULLBACK", "VWAP_RECLAIM")

        if has_rsi:
            if signal.option_type == OptionType.CALL:
                if is_pullback:
                    if 40 <= rsi <= 55:
                        score += 10
                    elif 35 <= rsi <= 60:
                        score += 6
                else:
                    if 55 <= rsi <= 70:
                        score += 10
                    elif 50 <= rsi <= 75:
                        score += 6
            elif signal.option_type == OptionType.PUT:
                if is_pullback:
                    if 45 <= rsi <= 60:
                        score += 10
                    elif 40 <= rsi <= 65:
                        score += 6
                else:
                    if 30 <= rsi <= 45:
                        score += 10
                    elif 25 <= rsi <= 50:
                        score += 6

        # Strong candle body (+7)
        open_ = last.get("open", 0)
        close = last.get("close", 0)
        high = last.get("high", 0)
        low = last.get("low", 0)
        candle_range = high - low
        body = abs(close - open_)
        if candle_range > 0 and (body / candle_range) > 0.6:
            score += 7
        elif candle_range > 0 and (body / candle_range) > 0.5:
            score += 4

        # Strategy-specific quality (+7)
        details = signal.details
        if "orh" in details and close > 0:
            ref = details["orh"] if signal.option_type == OptionType.CALL else details.get("orl", details["orh"])
            margin_pct = abs(close - ref) / close * 100
            if margin_pct > 0.1:
                score += 7
            elif margin_pct > 0.05:
                score += 4
        elif "sweep_depth_pct" in details:
            if details["sweep_depth_pct"] > 0.03:
                score += 7
            elif details["sweep_depth_pct"] > 0.01:
                score += 4
        elif "breakout_pct" in details:
            bp = details["breakout_pct"]
            if bp > 0.1:
                score += 7
            elif bp > 0.05:
                score += 5
            elif bp > 0.02:
                score += 3
        elif "adx" in details and details["adx"] > 20:
            score += 7

        # MACD histogram (+6)
        macd_hist = last.get("macd_hist")
        if macd_hist is not None and not pd.isna(macd_hist):
            if signal.option_type == OptionType.CALL and macd_hist > 0:
                score += 6
            elif signal.option_type == OptionType.PUT and macd_hist < 0:
                score += 6

        return min(score, 30.0)

    # ── Factor 2: Market Alignment (0-25) ────────────────────────────────

    def _score_market_alignment(self, signal: StrategySignal, df: pd.DataFrame) -> float:
        """Score based on trend and VWAP alignment (0-25).

        Components:
          - VWAP alignment (+10)
          - EMA trend alignment (+8)
          - ADX trend strength (+4)
          - Bollinger position (+3)
        """
        if df.empty:
            return 0.0

        score = 0.0
        last = df.iloc[-1]
        close = last.get("close", 0)

        # VWAP alignment (+10)
        vwap = last.get("vwap")
        if vwap is not None and not pd.isna(vwap) and vwap > 0:
            if signal.option_type == OptionType.CALL and close > vwap:
                pct_above = (close - vwap) / vwap * 100
                if pct_above > 1.5:
                    score -= 3  # Extended — mean reversion risk
                elif pct_above > 0.1:
                    score += 10
                else:
                    score += 7
            elif signal.option_type == OptionType.PUT and close < vwap:
                pct_below = (vwap - close) / vwap * 100
                if pct_below > 1.5:
                    score -= 3
                elif pct_below > 0.1:
                    score += 10
                else:
                    score += 7

        # EMA trend alignment (+8)
        ema9 = last.get("ema9")
        ema20 = last.get("ema20")
        ema50 = last.get("ema50")

        if ema9 is not None and ema20 is not None and not pd.isna(ema9) and not pd.isna(ema20):
            has_ema50 = ema50 is not None and not pd.isna(ema50)
            if signal.option_type == OptionType.CALL:
                if has_ema50 and ema9 > ema20 > ema50:
                    score += 8
                elif ema9 > ema20:
                    score += 4
            else:
                if has_ema50 and ema9 < ema20 < ema50:
                    score += 8
                elif ema9 < ema20:
                    score += 4

        # ADX trend strength (+4)
        adx = last.get("adx")
        if adx is not None and not pd.isna(adx):
            if adx > 25:
                score += 4
            elif adx > 20:
                score += 2

        # Bollinger position (+3)
        bb_upper = last.get("bollinger_upper")
        bb_lower = last.get("bollinger_lower")
        if (bb_upper is not None and bb_lower is not None
                and not pd.isna(bb_upper) and not pd.isna(bb_lower)
                and bb_upper > bb_lower):
            bb_pct = (close - bb_lower) / (bb_upper - bb_lower)
            if signal.option_type == OptionType.CALL:
                if bb_pct >= 0.8:
                    score += 3
                elif bb_pct >= 0.6:
                    score += 2
            else:
                if bb_pct <= 0.2:
                    score += 3
                elif bb_pct <= 0.4:
                    score += 2

        return max(min(score, 25.0), 0.0)

    # ── Factor 3: Volume Confirmation (0-20) ─────────────────────────────

    def _score_volume(
        self, signal: StrategySignal, df: pd.DataFrame, options_metrics: OptionsMetrics,
    ) -> float:
        """Score based on volume confirmation (0-20)."""
        if df.empty:
            return 0.0

        score = 0.0
        last = df.iloc[-1]
        vol = last.get("volume", 0)
        avg = last.get("avg_volume_10", 0)

        # Path 1: Futures volume
        if vol > 0 and avg and avg > 0:
            ratio = vol / avg
            if ratio >= 2.0:
                score = 20
            elif ratio >= 1.5:
                score = 15
            elif ratio >= 1.2:
                score = 8
            elif ratio >= 1.0:
                score = 4

        # Path 2: ATM option volume
        elif options_metrics.atm_option_volume > 0:
            atm_vol = options_metrics.atm_option_volume
            if self._prev_atm_volume > 0:
                opt_ratio = atm_vol / self._prev_atm_volume
                if opt_ratio >= 2.0:
                    score = 16
                elif opt_ratio >= 1.5:
                    score = 12
                elif opt_ratio >= 1.2:
                    score = 8
            else:
                if atm_vol > 50000:
                    score = 8
                elif atm_vol > 20000:
                    score = 4
            self._prev_atm_volume = atm_vol

        # Path 3: Candle range vs ATR
        if score == 0:
            atr = last.get("atr")
            high = last.get("high", 0)
            low = last.get("low", 0)
            candle_range = high - low
            if atr and atr > 0 and candle_range > 0:
                range_ratio = candle_range / atr
                if range_ratio >= 2.0:
                    score = 16
                elif range_ratio >= 1.5:
                    score = 12
                elif range_ratio >= 1.0:
                    score = 8
                elif range_ratio >= 0.7:
                    score = 4

        return min(score, 20.0)

    # ── Factor 4: Options OI Signal (0-15) ───────────────────────────────

    def _score_options_oi(self, signal: StrategySignal, metrics: OptionsMetrics) -> float:
        """Score based on options OI and PCR (0-15).

        Components:
          - PCR directional confirmation (+8)
          - OI change/buildup (+7)
        """
        if metrics.pcr is None:
            return 0.0

        score = 0.0
        if signal.option_type == OptionType.CALL:
            # PCR > 1 = put heavy = contrarian bullish
            if metrics.pcr > 1.5:
                score += 8
            elif metrics.pcr > 1.2:
                score += 6
            elif metrics.pcr > 1.0:
                score += 3
            # OI buildup confirms participation
            if metrics.oi_change > 0:
                score += 7
            elif metrics.oi_change is not None:
                score += 2
        else:
            # PCR < 0.7 = call heavy = contrarian bearish
            if metrics.pcr < 0.6:
                score += 8
            elif metrics.pcr < 0.8:
                score += 6
            elif metrics.pcr < 1.0:
                score += 3
            if metrics.oi_change is not None and metrics.oi_change < 0:
                score += 7
            elif metrics.oi_change is not None:
                score += 2

        return min(score, 15.0)

    # ── Factor 5: Volatility Context (0-10) ──────────────────────────────

    def _score_volatility(self, signal: StrategySignal, df: pd.DataFrame) -> float:
        """Score based on volatility environment (0-10).

        Components:
          - ATR trend (expanding/contracting appropriately) (+5)
          - Candle range vs ATR ratio (+5)
        """
        if df.empty or len(df) < 5:
            return 0.0

        score = 0.0
        last = df.iloc[-1]
        atr = last.get("atr")

        if atr is None or pd.isna(atr) or atr <= 0:
            return 0.0

        # ATR trend: expanding = good for breakouts, contracting = good for pullbacks
        is_breakout = signal.strategy.value in ("ORB", "RANGE_BREAKOUT")
        if len(df) >= 10:
            atr_5_ago = df.iloc[-5].get("atr")
            if atr_5_ago is not None and not pd.isna(atr_5_ago) and atr_5_ago > 0:
                atr_change = (atr - atr_5_ago) / atr_5_ago
                if is_breakout:
                    if atr_change > 0.1:
                        score += 5
                    elif atr_change > 0:
                        score += 3
                else:
                    if atr_change < 0:
                        score += 5
                    elif atr_change < 0.1:
                        score += 3

        # Current candle activity vs ATR (+5)
        high = last.get("high", 0)
        low = last.get("low", 0)
        candle_range = high - low
        if candle_range > 0:
            range_ratio = candle_range / atr
            if 0.8 <= range_ratio <= 1.5:
                score += 5
            elif 0.5 <= range_ratio <= 2.0:
                score += 3

        return min(score, 10.0)
