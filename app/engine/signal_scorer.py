"""Signal scoring system.

Scoring factors (max 100):
  Strategy trigger:    25
  Volume confirmation: 20
  VWAP alignment:      15
  Options OI signal:   15
  Global bias:         10
  Historical pattern:  15

Trade allowed if score ≥ 70.
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

MIN_SCORE = 70


class SignalScorer:
    """Score a strategy signal across multiple confirmation factors."""

    def score(
        self,
        signal: StrategySignal,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        global_bias: GlobalBias,
    ) -> SignalScore:
        """Compute multi-factor score for a strategy signal."""
        scores = SignalScore()

        # 1. Strategy trigger (25 pts) — signal already triggered
        scores.strategy_trigger = 25.0

        # 2. Volume confirmation (20 pts)
        scores.volume_confirmation = self._score_volume(signal, df)

        # 3. VWAP alignment (15 pts)
        scores.vwap_alignment = self._score_vwap(signal, df)

        # 4. Options OI signal (15 pts)
        scores.options_oi_signal = self._score_options(signal, options_metrics)

        # 5. Global bias alignment (10 pts)
        scores.global_bias_score = self._score_global_bias(signal, global_bias)

        # 6. Historical pattern (15 pts)
        scores.historical_pattern = self._score_historical(signal, df)

        scores.total = (
            scores.strategy_trigger
            + scores.volume_confirmation
            + scores.vwap_alignment
            + scores.options_oi_signal
            + scores.global_bias_score
            + scores.historical_pattern
        )

        logger.info(
            "Signal score: %.0f (strat=%.0f vol=%.0f vwap=%.0f oi=%.0f global=%.0f hist=%.0f)",
            scores.total,
            scores.strategy_trigger,
            scores.volume_confirmation,
            scores.vwap_alignment,
            scores.options_oi_signal,
            scores.global_bias_score,
            scores.historical_pattern,
        )
        return scores

    def _score_volume(self, signal: StrategySignal, df: pd.DataFrame) -> float:
        """Score based on volume confirmation (0–20)."""
        if df.empty:
            return 0
        last = df.iloc[-1]
        vol = last["volume"]
        avg = last.get("avg_volume_10", vol)
        if avg is None or avg == 0:
            return 0
        ratio = vol / avg
        if ratio >= 2.0:
            return 20
        if ratio >= 1.5:
            return 15
        if ratio >= 1.3:
            return 10
        return 5

    def _score_vwap(self, signal: StrategySignal, df: pd.DataFrame) -> float:
        """Score based on VWAP alignment (0–15)."""
        if df.empty:
            return 0
        last = df.iloc[-1]
        close = last["close"]
        vwap = last.get("vwap", close)
        if signal.option_type == OptionType.CALL and close > vwap:
            return 15
        if signal.option_type == OptionType.PUT and close < vwap:
            return 15
        return 0

    def _score_options(self, signal: StrategySignal, metrics: OptionsMetrics) -> float:
        """Score based on options chain data (0–15)."""
        score = 0.0
        if signal.option_type == OptionType.CALL:
            # Bullish: high PCR (more puts = support), price near put OI cluster
            if metrics.pcr > 1.2:
                score += 8
            if metrics.oi_change > 0:
                score += 7
        else:
            # Bearish: low PCR (more calls = resistance)
            if metrics.pcr < 0.8:
                score += 8
            if metrics.oi_change < 0:
                score += 7
        return min(score, 15)

    def _score_global_bias(self, signal: StrategySignal, bias: GlobalBias) -> float:
        """Score based on global market bias alignment (0–10)."""
        if signal.option_type == OptionType.CALL and bias == GlobalBias.BULLISH:
            return 10
        if signal.option_type == OptionType.PUT and bias == GlobalBias.BEARISH:
            return 10
        if bias == GlobalBias.NEUTRAL:
            return 5
        return 0

    def _score_historical(self, signal: StrategySignal, df: pd.DataFrame) -> float:
        """Score based on price action patterns (0–15)."""
        if df.empty or len(df) < 5:
            return 0

        score = 0.0
        recent = df.tail(5)

        if signal.option_type == OptionType.CALL:
            # Higher lows pattern
            lows = recent["low"].values
            higher_lows = all(lows[i] <= lows[i + 1] for i in range(len(lows) - 1))
            if higher_lows:
                score += 8
            # RSI not overbought
            rsi = recent.iloc[-1].get("rsi", 50)
            if 55 <= rsi <= 70:
                score += 7
        else:
            # Lower highs pattern
            highs = recent["high"].values
            lower_highs = all(highs[i] >= highs[i + 1] for i in range(len(highs) - 1))
            if lower_highs:
                score += 8
            rsi = recent.iloc[-1].get("rsi", 50)
            if 30 <= rsi <= 45:
                score += 7

        return min(score, 15)
