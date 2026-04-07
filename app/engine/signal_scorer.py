"""Signal scoring system.

Multi-factor confluence scoring inspired by systematic trading literature:
  - Elder, *Trading for a Living* — Triple Screen multi-factor confirmation
  - O'Neil, *How to Make Money in Stocks* — CANSLIM multi-factor checklist
  - Minervini, *Trade Like a Stock Market Wizard* — SEPA criteria alignment
  - Van Tharp, *Trade Your Way to Financial Freedom* — expectancy framework

Scoring factors (max 100) — every point must come from real market data:
  Strategy trigger quality:  22  (RSI — Wilder, candle body — Nison, MACD — Appel)
  Volume confirmation:       18  (O'Neil: vol ≥ 50% above avg; Minervini: 2× ideal)
  Historical pattern:        16  (EMA alignment — Elder #1 screen; swing structure — Dow;
                                   ADX — Wilder; EMA200 — Weinstein stages;
                                   Bollinger position — Bollinger)
  VWAP alignment:            13  (Shannon: institutional price anchor)
  Options OI signal:          8  (McMillan: PCR is "secondary" indicator;
                                   intraday PCR noisy vs daily/weekly)
  Global bias:                8  (global market correlation — academic finance)
  FII/DII institutional:      6  (Minervini/O'Neil: institutional sponsorship)
  Market breadth:              5  (Zweig: advance/decline as market health)
  News sentiment:              4  (Tetlock 2007: quantitative news sentiment)

Adaptive threshold (Van Tharp): adjusts required score based on data availability.
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

# Optional: InsightManager reference (set by orchestrator)
_insight_manager = None


def set_insight_manager(manager) -> None:
    """Set the insight manager for scoring with intelligence data."""
    global _insight_manager
    _insight_manager = manager

MIN_SCORE = 48  # Balanced: requires multi-factor confluence but not extreme

# Session-locked threshold: computed once per day, not per-cycle
_session_min_score: int | None = None


def lock_session_threshold(
    options_available: bool,
    global_available: bool,
    fii_dii_available: bool = False,
    breadth_available: bool = False,
    news_available: bool = False,
) -> int:
    """Lock the adaptive threshold once at session start, not per-cycle.

    Called once during pre-market setup. The threshold stays constant for the
    entire trading day so the same signal quality is required regardless of
    mid-day data fetch failures.

    Deductions are proportional to each factor's max contribution:
      Options (max 8): -6 if unavailable
      Global (max 8): -6 if unavailable
      FII/DII (max 6): -5 if unavailable
      Breadth (max 5): -4 if unavailable
      News (max 4): -3 if unavailable
    """
    global _session_min_score
    unavailable_points = 0
    if not options_available:
        unavailable_points += 6
    if not global_available:
        unavailable_points += 6
    if not fii_dii_available:
        unavailable_points += 5
    if not breadth_available:
        unavailable_points += 4
    if not news_available:
        unavailable_points += 3
    _session_min_score = max(MIN_SCORE - unavailable_points, 35)  # Floor at 35
    return _session_min_score


def compute_adaptive_min_score(options_metrics: OptionsMetrics, global_bias: GlobalBias) -> int:
    """Return the session-locked threshold, or compute a conservative one.

    If lock_session_threshold() was called, always returns that locked value.
    Otherwise falls back to per-cycle computation (backward compat).
    """
    if _session_min_score is not None:
        return _session_min_score

    # Fallback: per-cycle computation (only if session lock wasn't called)
    unavailable_points = 0
    if options_metrics.pcr is None:
        unavailable_points += 6
    if global_bias in (GlobalBias.UNAVAILABLE, GlobalBias.NEUTRAL):
        unavailable_points += 6
    if _insight_manager is None or not _insight_manager.has_insight:
        unavailable_points += 12  # FII(5) + breadth(4) + news(3)
    return max(MIN_SCORE - unavailable_points, 35)  # Floor at 35


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
        structure_data: dict | None = None,
    ) -> SignalScore:
        """Compute multi-factor score for a strategy signal."""
        scores = SignalScore()
        is_call = signal.option_type == OptionType.CALL

        # 1. Strategy trigger (15 pts — reduced from 22, less weight on mechanical rules)
        scores.strategy_trigger = self._score_strategy_trigger(signal, df)

        # 2. Volume confirmation (18 pts) — from futures volume or ATM option volume
        scores.volume_confirmation = self._score_volume(signal, df, options_metrics)

        # 3. Historical pattern (16 pts) — Elder's #1 screen: trend is king
        scores.historical_pattern = self._score_historical(signal, df)

        # 4. VWAP alignment (13 pts) — only if real volume-weighted VWAP
        scores.vwap_alignment = self._score_vwap(signal, df)

        # 5. Options OI signal (8 pts) — McMillan: secondary indicator
        scores.options_oi_signal = self._score_options(signal, options_metrics)

        # 6. Global bias alignment (8 pts) — only if real global data
        scores.global_bias_score = self._score_global_bias(signal, global_bias)

        # 7-9. Intelligence factors from InsightManager
        fii_score = 0.0
        breadth_score = 0.0
        news_score = 0.0
        if _insight_manager and _insight_manager.has_insight:
            fii_score = _insight_manager.get_fii_dii_score(is_call)
            breadth_score = _insight_manager.get_breadth_score(is_call)
            news_score = _insight_manager.get_news_sentiment_score(is_call)

        # 10. Structure alignment (10 pts — NEW)
        structure_score = self._score_structure(signal, structure_data)

        # 11. Pre-market directional penalty (-8 if opposing pre-market bias)
        premarket_penalty = 0.0
        if _insight_manager and _insight_manager.has_insight:
            pm_bias = _insight_manager.get_market_bias()
            if pm_bias == "bullish" and signal.option_type == OptionType.PUT:
                premarket_penalty = -8
            elif pm_bias == "bearish" and signal.option_type == OptionType.CALL:
                premarket_penalty = -8

        scores.total = (
            scores.strategy_trigger
            + scores.volume_confirmation
            + scores.vwap_alignment
            + scores.options_oi_signal
            + scores.global_bias_score
            + scores.historical_pattern
            + fii_score
            + breadth_score
            + news_score
            + structure_score
            + premarket_penalty
        )

        logger.info(
            "Signal score: %.0f (strat=%.0f vol=%.0f vwap=%.0f oi=%.0f global=%.0f hist=%.0f fii=%.0f breadth=%.0f news=%.0f struct=%.0f)",
            scores.total,
            scores.strategy_trigger,
            scores.volume_confirmation,
            scores.vwap_alignment,
            scores.options_oi_signal,
            scores.global_bias_score,
            scores.historical_pattern,
            fii_score,
            breadth_score,
            news_score,
            structure_score,
        )
        return scores

    # ── Component Scorers ────────────────────────────────────────────────

    def _score_strategy_trigger(self, signal: StrategySignal, df: pd.DataFrame) -> float:
        """Score based on strategy trigger quality (0–22).

        Book references:
          - Wilder, RSI: momentum sweet spots per strategy type
          - Raschke, *Street Smarts*: pullback RSI 40-60 sweet spot
          - Nison, *Japanese Candlestick Charting*: body > 60% = strong candle
          - Appel, *Technical Analysis*: MACD histogram for momentum

        Points earned from:
          - RSI in directional sweet spot (+7) or acceptable range (+4)
          - Strong candle body > 60% of range (+5) (Nison)
          - Breakout margin / sweep depth from details (+5)
          - MACD histogram confirming direction (+5) (Appel)
        """
        if df.empty:
            return 0

        score = 0.0
        last = df.iloc[-1]

        rsi = last.get("rsi")
        has_rsi = rsi is not None and not (isinstance(rsi, float) and pd.isna(rsi))

        # RSI directional confirmation (+7)
        # Pullback strategies (TrendPullback, VWAPReclaim) have different RSI
        # sweet spots — pullback entries occur at intermediate RSI by design
        # (Linda Raschke, Street Smarts).
        is_pullback = signal.strategy.value in ("TREND_PULLBACK", "VWAP_RECLAIM")
        if has_rsi:
            if signal.option_type == OptionType.CALL:
                if is_pullback:
                    if 40 <= rsi <= 55:
                        score += 7  # Pullback sweet spot
                    elif 35 <= rsi <= 60:
                        score += 4
                else:
                    if 55 <= rsi <= 70:
                        score += 7  # Momentum sweet spot
                    elif 50 <= rsi <= 75:
                        score += 4
            elif signal.option_type == OptionType.PUT:
                if is_pullback:
                    if 45 <= rsi <= 60:
                        score += 7  # Pullback sweet spot
                    elif 40 <= rsi <= 65:
                        score += 4
                else:
                    if 30 <= rsi <= 45:
                        score += 7  # Momentum sweet spot
                    elif 25 <= rsi <= 50:
                        score += 4

        # Strong candle body (+5) — at least 50% body required
        open_ = last.get("open", 0)
        close = last.get("close", 0)
        high = last.get("high", 0)
        low = last.get("low", 0)
        candle_range = high - low
        body = abs(close - open_)
        if candle_range > 0 and (body / candle_range) > 0.6:
            score += 5
        elif candle_range > 0 and (body / candle_range) > 0.5:
            score += 3

        # Strategy-specific quality metrics from details (+5)
        details = signal.details
        # ORB: breakout margin beyond the range
        if "orh" in details and close > 0:
            ref = details["orh"] if signal.option_type == OptionType.CALL else details.get("orl", details["orh"])
            margin_pct = abs(close - ref) / close * 100
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
        # Range breakout or EMA breakout: breakout percentage
        elif "breakout_pct" in details:
            bp = details["breakout_pct"]
            if bp > 0.1:
                score += 5
            elif bp > 0.05:
                score += 3
            elif bp > 0.02:
                score += 2
        # Range breakout: ADX from details
        elif "adx" in details and details["adx"] > 20:
            score += 5
        # VWAP reclaim: just passing through, RSI + body carry it
        else:
            pass

        # MACD histogram confirmation (+5)
        macd_hist = last.get("macd_hist")
        if macd_hist is not None and not pd.isna(macd_hist):
            if signal.option_type == OptionType.CALL and macd_hist > 0:
                score += 5
            elif signal.option_type == OptionType.PUT and macd_hist < 0:
                score += 5

        return min(score, 15.0)

    def _score_volume(
        self, signal: StrategySignal, df: pd.DataFrame, options_metrics: OptionsMetrics,
    ) -> float:
        """Score based on volume confirmation (0–18).

        Two data sources:
          1. NIFTY Futures volume (merged into df by orchestrator) — preferred
          2. ATM option volume from options chain — fallback for participation
        If neither is available, check candle range vs ATR for implied activity.
        """
        if df.empty:
            return 0

        score = 0.0
        last = df.iloc[-1]
        vol = last.get("volume", 0)
        avg = last.get("avg_volume_10", 0)

        # Check if this is index data with no volume
        is_index = vol == 0 and (avg is None or avg == 0)

        # Path 1: Futures volume is available (vol > 0 and avg > 0)
        # O'Neil: breakout volume should be ≥ 50% above avg (1.5×)
        # Minervini: ideal breakout volume ≥ 2×
        if vol > 0 and avg and avg > 0:
            ratio = vol / avg
            if ratio >= 2.0:
                score = 18  # Minervini ideal
            elif ratio >= 1.5:
                score = 14  # O'Neil minimum for breakout
            elif ratio >= 1.2:
                score = 7   # Above average but below O'Neil threshold
            elif ratio >= 1.0:
                score = 4   # Average volume — minimal confirmation
            # Below average volume = 0

        # Path 2: No futures volume — try ATM option volume from chain
        elif options_metrics.atm_option_volume > 0:
            atm_vol = options_metrics.atm_option_volume
            # Compare against previous cycle's ATM volume
            if self._prev_atm_volume > 0:
                opt_ratio = atm_vol / self._prev_atm_volume
                if opt_ratio >= 2.0:
                    score = 14  # Slightly capped vs futures (less direct)
                elif opt_ratio >= 1.5:
                    score = 10
                elif opt_ratio >= 1.2:
                    score = 7
            else:
                # First fetch — can't compare, grant partial if volume is substantial
                if atm_vol > 50000:
                    score = 7
                elif atm_vol > 20000:
                    score = 4

            self._prev_atm_volume = atm_vol
        # Path 3: No volume data — infer activity from candle range vs ATR
        # For indices, futures volume is often unavailable; ATR proxy should
        # not be capped too low or indices are systematically penalised.
        if score == 0:
            atr = last.get("atr")
            high = last.get("high", 0)
            low = last.get("low", 0)
            candle_range = high - low
            if atr and atr > 0 and candle_range > 0:
                range_ratio = candle_range / atr
                if range_ratio >= 2.0:
                    score = 14
                elif range_ratio >= 1.5:
                    score = 11
                elif range_ratio >= 1.0:
                    score = 7
                elif range_ratio >= 0.7:
                    score = 4

        return score

    def _score_vwap(self, signal: StrategySignal, df: pd.DataFrame) -> float:
        """Score based on VWAP alignment (0–13).

        Shannon, *Technical Analysis Using Multiple Timeframes*:
        VWAP = institutional average fill price. Trading on the right side of
        VWAP aligns with the institutional flow direction.

        Full points if VWAP from real volume data (futures).
        Partial points (max 7) if price-only VWAP (no volume weight).
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
        has_real_volume = vol_sum > 0

        # Score multiplier: full credit with real volume, reduced without
        max_pts = 13 if has_real_volume else 7  # Non-volume VWAP still has value for price alignment

        if signal.option_type == OptionType.CALL and close > vwap:
            pct_above = (close - vwap) / vwap * 100 if vwap > 0 else 0
            # Extended move penalty: price too far above VWAP → mean reversion risk
            if pct_above > 1.5:
                return -5
            if pct_above > 0.1:
                return max_pts
            return round(max_pts * 0.67)
        if signal.option_type == OptionType.PUT and close < vwap:
            pct_below = (vwap - close) / vwap * 100 if vwap > 0 else 0
            # Extended move penalty: price too far below VWAP → mean reversion risk
            if pct_below > 1.5:
                return -5
            if pct_below > 0.1:
                return max_pts
            return round(max_pts * 0.67)
        return 0

    def _score_options(self, signal: StrategySignal, metrics: OptionsMetrics) -> float:
        """Score based on options chain data (0–8).

        McMillan, *Options as a Strategic Investment*:
        - PCR is explicitly a "secondary indicator" — confirms, doesn't lead
        - PCR > 1.0 = above-average puts = contrarian bullish (support)
        - PCR > 1.5 = extreme put writing = strong contrarian bullish
        - PCR < 0.6 = extreme call writing = contrarian bearish (resistance)
        - OI increase confirms market participation

        Reduced from 13 to 8 pts: intraday PCR is noisier than the daily/weekly
        PCR that McMillan's research validated.
        """
        # No data fetched yet — 0 points
        if metrics.pcr is None:
            return 0

        score = 0.0
        if signal.option_type == OptionType.CALL:
            if metrics.pcr > 1.5:
                score += 5
            elif metrics.pcr > 1.2:
                score += 3
            elif metrics.pcr > 1.0:
                score += 1
            if metrics.oi_change > 0:
                score += 3
        else:
            if metrics.pcr < 0.6:
                score += 5
            elif metrics.pcr < 0.8:
                score += 3
            elif metrics.pcr < 1.0:
                score += 1
            if metrics.oi_change < 0:
                score += 3

        return min(score, 8)

    def _score_global_bias(self, signal: StrategySignal, bias: GlobalBias) -> float:
        """Score based on global market bias alignment (0–8).

        Academic finance: high correlation between US/Asian markets and Indian
        open. UNAVAILABLE/NEUTRAL = 0 pts. Opposing direction = mild -1 penalty
        (avoids triple-penalizing with breadth and FII — Van Tharp principle of
        not over-weighting correlated factors).
        """
        if bias == GlobalBias.UNAVAILABLE:
            return 0
        if bias == GlobalBias.NEUTRAL:
            return 0  # No directional confirmation

        if signal.option_type == OptionType.CALL and bias == GlobalBias.BULLISH:
            return 8
        if signal.option_type == OptionType.PUT and bias == GlobalBias.BEARISH:
            return 8
        # Signal direction opposes global bias — mild penalty (breadth + FII
        # already reduce score for opposing direction; avoid triple-penalizing)
        if signal.option_type == OptionType.CALL and bias == GlobalBias.BEARISH:
            return -1
        if signal.option_type == OptionType.PUT and bias == GlobalBias.BULLISH:
            return -1
        return 0

    def _score_historical(self, signal: StrategySignal, df: pd.DataFrame) -> float:
        """Score based on price action and trend structure (0–16).

        Elder's #1 screen — trend is the most important factor:
          - EMA stacking (EMA9 > EMA20 > EMA50) = healthy trend (+4)
          - Dow Theory: higher lows / lower highs (+4)
          - Wilder: ADX > 25 = trending market (+3)
          - Weinstein: price vs EMA200 = Stage 2/4 (+2)
          - Bollinger: price near upper band = momentum (+3)
        """
        if df.empty or len(df) < 20:
            return 0

        score = 0.0
        lookback = min(30, len(df))
        recent = df.tail(lookback)
        last = recent.iloc[-1]

        ema9 = last.get("ema9")
        ema20 = last.get("ema20")
        ema50 = last.get("ema50")
        ema200 = last.get("ema200")
        adx = last.get("adx")
        close = last.get("close", 0)
        bb_upper = last.get("bollinger_upper")
        bb_lower = last.get("bollinger_lower")
        bb_middle = last.get("bollinger_middle")

        # If core short-term indicators are missing, can't score
        if any(v is None or (isinstance(v, float) and pd.isna(v)) for v in [ema9, ema20]):
            return 0

        has_ema50 = ema50 is not None and not (isinstance(ema50, float) and pd.isna(ema50))
        has_ema200 = ema200 is not None and not (isinstance(ema200, float) and pd.isna(ema200))
        has_bb = all(
            v is not None and not (isinstance(v, float) and pd.isna(v))
            for v in [bb_upper, bb_lower, bb_middle]
        )

        if signal.option_type == OptionType.CALL:
            # EMA alignment: short > medium > long (+4) — Elder
            if has_ema50 and ema9 > ema20 > ema50:
                score += 4
            elif ema9 > ema20:
                score += 2

            # Higher lows over lookback — Dow Theory (+4)
            lows = recent["low"].values
            n_samples = min(5, len(lows) // 5)
            if n_samples >= 2:
                step = max(1, len(lows) // n_samples)
                sampled = [lows[i] for i in range(0, len(lows), step)][:n_samples]
                if all(sampled[i] <= sampled[i + 1] for i in range(len(sampled) - 1)):
                    score += 4

            # ADX trending (+3) — Wilder: ADX > 25 = trending
            if adx is not None and not pd.isna(adx) and adx > 25:
                score += 3

            # Weinstein Stage 2: above EMA200 (+2)
            if has_ema200 and close > ema200:
                score += 2

            # Bollinger: price in upper half of bands = bullish momentum (+3)
            if has_bb and bb_upper > bb_lower:
                bb_pct = (close - bb_lower) / (bb_upper - bb_lower)
                if bb_pct >= 0.8:
                    score += 3  # Near upper band — strong momentum
                elif bb_pct >= 0.6:
                    score += 2  # Upper half
                elif bb_pct >= 0.5:
                    score += 1  # Above middle
        else:
            # PUT direction
            if has_ema50 and ema9 < ema20 < ema50:
                score += 4
            elif ema9 < ema20:
                score += 2

            highs = recent["high"].values
            n_samples = min(5, len(highs) // 5)
            if n_samples >= 2:
                step = max(1, len(highs) // n_samples)
                sampled = [highs[i] for i in range(0, len(highs), step)][:n_samples]
                if all(sampled[i] >= sampled[i + 1] for i in range(len(sampled) - 1)):
                    score += 4

            # Wilder: ADX > 25 = trending
            if adx is not None and not pd.isna(adx) and adx > 25:
                score += 3

            # Weinstein Stage 4: below EMA200
            if has_ema200 and close < ema200:
                score += 2

            # Bollinger: price in lower half = bearish momentum
            if has_bb and bb_upper > bb_lower:
                bb_pct = (close - bb_lower) / (bb_upper - bb_lower)
                if bb_pct <= 0.2:
                    score += 3  # Near lower band
                elif bb_pct <= 0.4:
                    score += 2
                elif bb_pct <= 0.5:
                    score += 1

        return min(score, 16)

    def _score_structure(self, signal: StrategySignal, structure_data: dict | None) -> float:
        """Score based on market structure alignment (0–10).

        +5 if signal direction matches structural bias (HH/HL = bullish, LH/LL = bearish)
        +3 if BOS confirms direction
        +2 if signal has confirmation candle (from adaptive strategy details)
        """
        if not structure_data:
            return 0.0

        score = 0.0
        bias = structure_data.get("bias", "neutral")
        is_call = signal.option_type == OptionType.CALL

        # Direction matches structural bias (+5)
        if (is_call and bias == "bullish") or (not is_call and bias == "bearish"):
            score += 5.0
        elif bias == "neutral":
            pass  # No penalty, no bonus
        else:
            score -= 2.0  # Opposing structure = mild penalty

        # BOS confirms direction (+3)
        bos = structure_data.get("last_bos")
        if bos:
            if (is_call and bos["type"] == "bullish") or (not is_call and bos["type"] == "bearish"):
                score += 3.0

        # Confirmation candle present (+2) — from adaptive strategy
        if signal.details.get("confirmed_hold") or signal.details.get("setup"):
            score += 2.0

        return max(min(score, 10.0), 0.0)
