"""Signal scoring system.

Scoring factors (max 100) — every point must come from real market data:
  Strategy trigger quality:  25  (RSI confirmation, candle strength, breakout margin)
  Volume confirmation:       20  (NIFTY futures volume OR options ATM volume)
  VWAP alignment:            15  (only if VWAP is volume-weighted from futures data)
  Options OI signal:         15  (PCR, OI change — only if real data present)
  Global bias:               10  (only if real global data fetched)
  Historical pattern:        15  (EMA alignment, swing structure, ADX)

Trade allowed if score ≥ 70.
No free points. Missing data = 0 points for that factor.
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
    """Score a strategy signal across multiple confirmation factors.

    Zero-baseline: no free points. Every point is earned from real data.
    """

    def __init__(self) -> None:
        self._prev_atm_volume: int = 0  # Track ATM option volume between cycles

    def score(
        self,
        signal: StrategySignal,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        global_bias: GlobalBias,
    ) -> SignalScore:
        """Compute multi-factor score for a strategy signal."""
        scores = SignalScore()

        # 1. Strategy trigger (25 pts) — earned from signal quality
        scores.strategy_trigger = self._score_strategy_trigger(signal, df)

        # 2. Volume confirmation (20 pts) — from futures volume or ATM option volume
        scores.volume_confirmation = self._score_volume(signal, df, options_metrics)

        # 3. VWAP alignment (15 pts) — only if real volume-weighted VWAP
        scores.vwap_alignment = self._score_vwap(signal, df)

        # 4. Options OI signal (15 pts) — only if real OI data
        scores.options_oi_signal = self._score_options(signal, options_metrics)

        # 5. Global bias alignment (10 pts) — only if real global data
        scores.global_bias_score = self._score_global_bias(signal, global_bias)

        # 6. Historical pattern (15 pts) — from real price/indicator data
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

    # ── Component Scorers ────────────────────────────────────────────────

    def _score_strategy_trigger(self, signal: StrategySignal, df: pd.DataFrame) -> float:
        """Score based on strategy trigger quality (0–25).

        No free base points. Points earned from:
          - RSI in directional sweet spot (+8)
          - Strong candle body > 60% of range (+7)
          - Breakout margin / sweep depth from details (+5)
          - MACD histogram confirming direction (+5)
        """
        if df.empty:
            return 0

        score = 0.0
        last = df.iloc[-1]

        rsi = last.get("rsi")
        if rsi is None or (isinstance(rsi, float) and pd.isna(rsi)):
            return 0  # No RSI = can't evaluate trigger quality

        # RSI directional confirmation (+8)
        if signal.option_type == OptionType.CALL and 55 <= rsi <= 70:
            score += 8
        elif signal.option_type == OptionType.PUT and 30 <= rsi <= 45:
            score += 8
        elif signal.option_type == OptionType.CALL and rsi > 50:
            score += 4  # Partial credit for positive momentum
        elif signal.option_type == OptionType.PUT and rsi < 50:
            score += 4

        # Strong candle body (+7)
        open_ = last.get("open", 0)
        close = last.get("close", 0)
        high = last.get("high", 0)
        low = last.get("low", 0)
        candle_range = high - low
        body = abs(close - open_)
        if candle_range > 0 and (body / candle_range) > 0.6:
            score += 7
        elif candle_range > 0 and (body / candle_range) > 0.4:
            score += 3  # Partial for decent body

        # Strategy-specific quality metrics from details (+5)
        details = signal.details
        # ORB: breakout margin beyond the range
        if "orh" in details and close > 0:
            margin_pct = abs(close - details["orh"]) / close * 100
            if margin_pct > 0.1:
                score += 5
            elif margin_pct > 0.05:
                score += 2
        # Liquidity sweep depth
        elif "sweep_depth_pct" in details:
            if details["sweep_depth_pct"] > 0.03:
                score += 5
            elif details["sweep_depth_pct"] > 0.01:
                score += 2
        # Range breakout: ADX from details
        elif "adx" in details and details["adx"] > 22:
            score += 5
        # VWAP reclaim: just passing through, RSI + body carry it
        else:
            # No specific depth metric — candle/RSI must carry the full weight
            pass

        # MACD histogram confirmation (+5)
        macd_hist = last.get("macd_hist")
        if macd_hist is not None and not pd.isna(macd_hist):
            if signal.option_type == OptionType.CALL and macd_hist > 0:
                score += 5
            elif signal.option_type == OptionType.PUT and macd_hist < 0:
                score += 5

        return min(score, 25.0)

    def _score_volume(
        self, signal: StrategySignal, df: pd.DataFrame, options_metrics: OptionsMetrics,
    ) -> float:
        """Score based on volume confirmation (0–20).

        Two data sources:
          1. NIFTY Futures volume (merged into df by orchestrator) — preferred
          2. ATM option volume from options chain — fallback for participation
        If neither is available, returns 0 (no free points).
        """
        if df.empty:
            return 0

        score = 0.0
        last = df.iloc[-1]
        vol = last.get("volume", 0)
        avg = last.get("avg_volume_10", 0)

        # Path 1: Futures volume is available (vol > 0 and avg > 0)
        if vol > 0 and avg and avg > 0:
            ratio = vol / avg
            if ratio >= 2.0:
                score = 20
            elif ratio >= 1.5:
                score = 15
            elif ratio >= 1.3:
                score = 10
            elif ratio >= 1.0:
                score = 5
            # Below average volume = 0

        # Path 2: No futures volume — try ATM option volume from chain
        elif options_metrics.atm_option_volume > 0:
            atm_vol = options_metrics.atm_option_volume
            # Compare against previous cycle's ATM volume
            if self._prev_atm_volume > 0:
                opt_ratio = atm_vol / self._prev_atm_volume
                if opt_ratio >= 2.0:
                    score = 15  # Slightly capped vs futures (less direct)
                elif opt_ratio >= 1.5:
                    score = 12
                elif opt_ratio >= 1.2:
                    score = 8
            else:
                # First fetch — can't compare, grant partial if volume is substantial
                if atm_vol > 50000:
                    score = 8
                elif atm_vol > 20000:
                    score = 5

            self._prev_atm_volume = atm_vol
        # Path 3: No volume data at all — 0 points
        else:
            score = 0

        return score

    def _score_vwap(self, signal: StrategySignal, df: pd.DataFrame) -> float:
        """Score based on VWAP alignment (0–15).

        Only awards full points if VWAP is calculated from real volume data.
        Non-volume-weighted VWAP (index fallback) gets 0 points.
        """
        if df.empty:
            return 0

        last = df.iloc[-1]
        close = last.get("close", 0)
        vwap = last.get("vwap")

        # VWAP must exist
        if vwap is None or pd.isna(vwap):
            return 0

        # Check if volume data exists (futures volume was merged)
        vol_sum = df["volume"].sum() if "volume" in df.columns else 0
        if vol_sum == 0:
            # No real volume — VWAP is just cumulative typical price. Not trustworthy.
            return 0

        # Real volume-weighted VWAP — award points for alignment
        if signal.option_type == OptionType.CALL and close > vwap:
            # More points for stronger alignment
            pct_above = (close - vwap) / vwap * 100 if vwap > 0 else 0
            if pct_above > 0.1:
                return 15
            return 10
        if signal.option_type == OptionType.PUT and close < vwap:
            pct_below = (vwap - close) / vwap * 100 if vwap > 0 else 0
            if pct_below > 0.1:
                return 15
            return 10
        return 0

    def _score_options(self, signal: StrategySignal, metrics: OptionsMetrics) -> float:
        """Score based on options chain data (0–15).

        Only awards points if real OI data is present (pcr is not None).
        """
        # No data fetched yet — 0 points
        if metrics.pcr is None:
            return 0

        score = 0.0
        if signal.option_type == OptionType.CALL:
            # Bullish: high PCR (more puts written = market makers see support)
            if metrics.pcr > 1.5:
                score += 8
            elif metrics.pcr > 1.2:
                score += 5
            elif metrics.pcr > 1.0:
                score += 2
            # OI increasing confirms participation
            if metrics.oi_change > 0:
                score += 7
            elif metrics.oi_change == 0:
                score += 0  # No change, no confirmation
        else:
            # Bearish: low PCR (more calls written = resistance)
            if metrics.pcr < 0.6:
                score += 8
            elif metrics.pcr < 0.8:
                score += 5
            elif metrics.pcr < 1.0:
                score += 2
            if metrics.oi_change < 0:
                score += 7

        return min(score, 15)

    def _score_global_bias(self, signal: StrategySignal, bias: GlobalBias) -> float:
        """Score based on global market bias alignment (0–10).

        UNAVAILABLE/NEUTRAL = 0 points. Only directional alignment earns points.
        """
        if bias == GlobalBias.UNAVAILABLE:
            return 0
        if bias == GlobalBias.NEUTRAL:
            return 0  # No directional confirmation

        if signal.option_type == OptionType.CALL and bias == GlobalBias.BULLISH:
            return 10
        if signal.option_type == OptionType.PUT and bias == GlobalBias.BEARISH:
            return 10
        # Signal direction opposes global bias — this is a negative signal
        # but we don't subtract; we just return 0
        return 0

    def _score_historical(self, signal: StrategySignal, df: pd.DataFrame) -> float:
        """Score based on price action and trend structure (0–15).

        Uses 30-candle lookback for swing structure and EMA trend alignment.
        Requires real indicator data (None = 0 points).
        """
        if df.empty or len(df) < 30:
            return 0

        score = 0.0
        recent = df.tail(30)
        last = recent.iloc[-1]

        ema9 = last.get("ema9")
        ema20 = last.get("ema20")
        ema50 = last.get("ema50")
        adx = last.get("adx")

        # If core indicators are missing, can't score
        if any(v is None or (isinstance(v, float) and pd.isna(v)) for v in [ema9, ema20, ema50]):
            return 0

        if signal.option_type == OptionType.CALL:
            # EMA alignment: 9 > 20 > 50 (strong uptrend)
            if ema9 > ema20 > ema50 and ema50 > 0:
                score += 5
            # Higher lows over 30 candles (check 5 evenly-spaced lows)
            lows = recent["low"].values
            sampled = [lows[i] for i in range(0, 30, 6)]
            if all(sampled[i] <= sampled[i + 1] for i in range(len(sampled) - 1)):
                score += 5
            # ADX confirming trend (must be real, not default)
            if adx is not None and not pd.isna(adx) and adx > 20:
                score += 5
        else:
            # EMA alignment: 9 < 20 < 50 (strong downtrend)
            if ema9 < ema20 < ema50 and ema50 > 0:
                score += 5
            # Lower highs over 30 candles
            highs = recent["high"].values
            sampled = [highs[i] for i in range(0, 30, 6)]
            if all(sampled[i] >= sampled[i + 1] for i in range(len(sampled) - 1)):
                score += 5
            if adx is not None and not pd.isna(adx) and adx > 20:
                score += 5

        return min(score, 15)
