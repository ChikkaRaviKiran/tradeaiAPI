"""AI Decision Engine — validates signals using OpenAI LLM."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

import pytz
from openai import AsyncOpenAI

_IST = pytz.timezone("Asia/Kolkata")

from app.core.config import settings
from app.core.models import (
    AIDecision,
    GlobalBias,
    MarketRegime,
    MarketSnapshot,
    SignalScore,
    StrategySignal,
    TechnicalIndicators,
    OptionsMetrics,
)

logger = logging.getLogger(__name__)

# Optional: InsightManager reference (set by orchestrator)
_insight_manager = None


def set_ai_insight_manager(manager) -> None:
    """Set the insight manager for enriching AI prompts."""
    global _insight_manager
    _insight_manager = manager


SYSTEM_PROMPT = """You are an expert Indian market options day-trading analyst for NIFTY/BANKNIFTY/FINNIFTY.
You receive comprehensive real-time market data and a strategy signal. Decide whether to APPROVE or REJECT.

Respond ONLY with valid JSON (no markdown, no extra text):
{
    "trade_decision": true/false,
    "confidence_score": 0-100,
    "entry_price": float,
    "stoploss": float,
    "target1": float,
    "target2": float,
    "reason": "2-3 sentences citing specific data points"
}

=== CONFIDENCE SCORING RUBRIC (use graduated scores — NEVER default to 55 or 75) ===
90-100: Perfect — all 5 confirmations aligned, trending regime, strong ADX, structure supports
80-89:  Strong — 4/5 confirmations, one minor concern
70-79:  Good — 3/5 confirmations, acceptable risk-reward
60-69:  Marginal — mixed signals, only approve if 2+ STRONG confirmations
50-59:  Weak — more risk than reward, reject
40-49:  Poor — clear opposing signals, reject
<40:    Dangerous — multiple red flags, strong reject

The 5 core confirmations:
1. TREND ALIGNMENT: EMA hierarchy + htf_trend + structure_bias all match signal direction
2. MOMENTUM: ADX > 20 + RSI in sweet zone (55-70 for CE, 30-45 for PE) + positive ema20_slope
3. VWAP ALIGNMENT: price above VWAP for CALL, below for PUT
4. OPTIONS/VOLUME: PCR supports direction (>0.9 for CE, <1.1 for PE), OI clusters favorable
5. CLEAN ENTRY: not at key level, no rejection candle, adequate R:R, bid-ask spread < 1.5%

=== HARD REJECT RULES (confidence < 40) ===
- ADX < 20 AND regime is "range_bound" — no directional momentum
- After 14:30 IST — chop/reversal zone
- ORB strategy after 11:30 — stale breakout, momentum exhausted
- Signal opposes BOTH htf_trend AND structure_bias — fighting the trend
- RSI > 78 for CE or RSI < 22 for PE — exhaustion zone
- 2+ losing trades already today (check session_context.consecutive_losses)
- Last trade hit SL < 15 min ago in same direction — revenge trade pattern
- last_candle.type is "rejection" against signal direction

=== STRATEGY-SPECIFIC RULES ===
- RANGE_BREAKOUT: Low ADX is EXPECTED pre-breakout — don't penalize. But regime MUST NOT be "range_bound" at signal time.
- ORB: Only valid before 11:30. Strong volume + clean break above ORH/below ORL required.
- TREND_PULLBACK: Needs EMA hierarchy aligned + RSI pullback from extreme + structure_bias matching.
- MOMENTUM_BREAKOUT: Needs ADX > 25 + strong ROC + EMA20 slope positive.
- For all breakout strategies, ADX rises AFTER breakout — evaluate the breakout quality, not current ADX.

=== DATA INTERPRETATION ===
- entry_price, stoploss, target1, target2 are OPTION PREMIUM values, not spot
- PCR > 1.0 = bullish (put writers provide support), PCR < 0.8 = bearish
- structure_bias (from swing HH/HL vs LH/LL analysis) is MORE reliable than regime classification
- BOS (Break of Structure) in signal direction = strong confirmation
- CHoCH (Change of Character) against signal direction = strong red flag
- ema20_slope > 0 confirms uptrend momentum; < 0 confirms downtrend
- day_type: TREND = wider targets ok; RANGE = tighter; VOLATILE = wider SL needed
- vix > 20 needs wider SL; vix < 14 = complacent, breakouts may fail
- Trend strength: 3 = strong uptrend (ema9>20>50>200), 0 = strong downtrend
- Null values mean data unavailable — skip that check, don't penalize
- Missing options/volume data should NOT block trades if technicals are strong

=== SESSION AWARENESS ===
- If today_pnl is positive after 13:00 — be more selective (protect gains)
- If 1 loss today — require 4/5 confirmations
- If 2+ losses — reject unless near-perfect setup (confidence 85+)
- Don't approve same direction trade if last SL was recent (< 15 min)

=== CRITICAL INSTRUCTIONS ===
- Your confidence MUST be your OWN independent assessment — use the full rubric above
- Evaluate EACH signal individually — never use the same confidence for different setups
- Cite specific data points in your reason (e.g., "ADX=24.7, RSI=61.2 in sweet zone, bullish BOS at 23880")
- Do NOT anchor on any pre-computed score — assess independently from all provided data
"""


class AIDecisionEngine:
    """Validates strategy signals using OpenAI GPT model."""

    def __init__(self) -> None:
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    async def evaluate(
        self,
        signal: StrategySignal,
        snapshot: MarketSnapshot,
        score: float,
        score_breakdown: Optional[SignalScore] = None,
        extra_context: Optional[dict] = None,
    ) -> AIDecision:
        """Send signal + market context to AI for validation.

        Args:
            extra_context: Optional dict with keys:
                - structure_data: market structure from compute_market_structure()
                - day_type: DayType classification string
                - session_context: dict with trades_today, today_pnl, etc.
                - global_indices: list of GlobalIndex objects
                - candle_data: dict from analyze_candle_character()
                - key_levels: list from compute_key_levels()
                - bid_ask_spread_pct: float
                - extra_scores: dict with fii_dii, breadth, news, structure scores
        """
        try:
            prompt = self._build_prompt(signal, snapshot, score, score_breakdown, extra_context)

            # Append intelligence context if available
            intelligence_context = ""
            news_context = ""
            if _insight_manager:
                intelligence_context = _insight_manager.get_ai_context_block()
                # Fetch recent news for AI context
                try:
                    from app.data.telegram_news import get_recent_news
                    recent_news = await get_recent_news(days=1)
                    news_context = _insight_manager.get_recent_news_summary(recent_news)
                except Exception:
                    pass

            client = self._get_client()

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            if intelligence_context:
                ctx = f"Pre-market intelligence:\n{intelligence_context}"
                if news_context:
                    ctx += f"\n\n{news_context}"
                messages.insert(1, {
                    "role": "system",
                    "content": ctx,
                })

            response = await client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )

            content = response.choices[0].message.content
            if content is None:
                return AIDecision(trade_decision=False, reason="Empty AI response")

            return self._parse_response(content, signal)

        except Exception as e:
            logger.exception("AI decision engine error")
            return AIDecision(trade_decision=False, reason=f"AI error: {str(e)}")

    def _build_prompt(
        self,
        signal: StrategySignal,
        snapshot: MarketSnapshot,
        score: float,
        score_breakdown: Optional[SignalScore] = None,
        extra_context: Optional[dict] = None,
    ) -> str:
        """Build enriched structured prompt for AI evaluation."""
        now = datetime.now(_IST)
        ctx = extra_context or {}

        # Score breakdown — all components including hidden ones
        score_detail = {}
        if score_breakdown:
            score_detail = {
                "strategy_trigger": round(score_breakdown.strategy_trigger, 1),
                "volume_confirmation": round(score_breakdown.volume_confirmation, 1),
                "vwap_alignment": round(score_breakdown.vwap_alignment, 1),
                "options_oi_signal": round(score_breakdown.options_oi_signal, 1),
                "global_bias_score": round(score_breakdown.global_bias_score, 1),
                "historical_pattern": round(score_breakdown.historical_pattern, 1),
            }
        # Add extra score components that the scorer computes but SignalScore model doesn't store
        extra_scores = ctx.get("extra_scores", {})
        if extra_scores:
            score_detail.update({
                "fii_dii_score": round(extra_scores.get("fii_dii", 0), 1),
                "breadth_score": round(extra_scores.get("breadth", 0), 1),
                "news_sentiment_score": round(extra_scores.get("news", 0), 1),
                "structure_alignment_score": round(extra_scores.get("structure", 0), 1),
            })

        # Compute today's open gap context
        open_gap_pct = None
        if snapshot.day_open and snapshot.prev_day_close and snapshot.prev_day_close > 0:
            open_gap_pct = round(
                ((snapshot.day_open - snapshot.prev_day_close) / snapshot.prev_day_close) * 100, 2
            )

        # Market structure from swing analysis
        structure_data = ctx.get("structure_data", {})
        market_structure = {
            "regime": snapshot.regime.value,
            "htf_trend": snapshot.htf_trend,
        }
        if structure_data:
            market_structure.update({
                "structure_bias": structure_data.get("bias", "neutral"),
                "hh_hl": structure_data.get("hh_hl", False),
                "lh_ll": structure_data.get("lh_ll", False),
            })
            if structure_data.get("last_bos"):
                bos = structure_data["last_bos"]
                market_structure["last_bos"] = {"type": bos.get("type"), "level": round(bos.get("level", 0), 2)}
            if structure_data.get("last_choch"):
                choch = structure_data["last_choch"]
                market_structure["last_choch"] = {"type": choch.get("type"), "level": round(choch.get("level", 0), 2)}
        # Day type
        day_type = ctx.get("day_type")
        if day_type:
            market_structure["day_type"] = day_type

        # Last candle character
        candle_data = ctx.get("candle_data")
        if candle_data and candle_data.get("type") != "neutral":
            market_structure["last_candle"] = {
                "type": candle_data.get("type"),
                "direction": candle_data.get("direction"),
                "body_ratio": candle_data.get("body_ratio"),
            }

        # Key levels — nearest support/resistance
        key_levels_section = {}
        key_levels = ctx.get("key_levels", [])
        if key_levels and snapshot.price:
            supports = [lv for lv in key_levels if lv.get("type") == "support"]
            resistances = [lv for lv in key_levels if lv.get("type") == "resistance"]
            if supports:
                nearest_sup = min(supports, key=lambda x: abs(x["price"] - snapshot.price))
                key_levels_section["nearest_support"] = round(nearest_sup["price"], 2)
                key_levels_section["support_source"] = nearest_sup.get("source", "")
                if snapshot.price > 0:
                    key_levels_section["distance_to_support_pct"] = round(
                        (snapshot.price - nearest_sup["price"]) / snapshot.price * 100, 2
                    )
            if resistances:
                nearest_res = min(resistances, key=lambda x: abs(x["price"] - snapshot.price))
                key_levels_section["nearest_resistance"] = round(nearest_res["price"], 2)
                key_levels_section["resistance_source"] = nearest_res.get("source", "")
                if snapshot.price > 0:
                    key_levels_section["distance_to_resistance_pct"] = round(
                        (nearest_res["price"] - snapshot.price) / snapshot.price * 100, 2
                    )

        # Global index details
        global_section: dict = {"bias": snapshot.global_bias.value}
        global_indices = ctx.get("global_indices", [])
        if global_indices:
            idx_details = {}
            vix_value = None
            for idx in global_indices:
                if hasattr(idx, "symbol") and hasattr(idx, "change_pct") and idx.last_price > 0:
                    if "VIX" in idx.symbol:
                        vix_value = round(idx.last_price, 2)
                    elif "NSEI" in idx.symbol:
                        idx_details["nifty_prev_close_change_pct"] = round(idx.change_pct, 2)
                    elif "DJI" in idx.symbol:
                        idx_details["dow_change_pct"] = round(idx.change_pct, 2)
                    elif "IXIC" in idx.symbol:
                        idx_details["nasdaq_change_pct"] = round(idx.change_pct, 2)
                    elif "GSPC" in idx.symbol:
                        idx_details["sp500_change_pct"] = round(idx.change_pct, 2)
                    elif "N225" in idx.symbol:
                        idx_details["nikkei_change_pct"] = round(idx.change_pct, 2)
            if idx_details:
                global_section["index_changes"] = idx_details
            if vix_value is not None:
                global_section["vix"] = vix_value

        # Session context
        session_section = ctx.get("session_context", {})

        # Bid-ask spread
        spread_pct = ctx.get("bid_ask_spread_pct")

        # Compute risk-reward ratio
        rr_ratio = None
        if signal.entry_price and signal.stoploss and signal.target1:
            risk = abs(signal.entry_price - signal.stoploss)
            reward = abs(signal.target1 - signal.entry_price)
            if risk > 0:
                rr_ratio = round(reward / risk, 2)

        # Get DataFrame-derived indicators if available
        df_indicators = ctx.get("df_indicators", {})

        return json.dumps(
            {
                "market_snapshot": {
                    "instrument": snapshot.instrument or "NIFTY",
                    "spot_price": snapshot.price,
                    "vwap": snapshot.vwap,
                    "vwap_is_volume_weighted": snapshot.indicators.vwap_is_volume_weighted,
                    "timestamp": now.strftime("%H:%M:%S"),
                    "day_open": snapshot.day_open,
                    "open_gap_pct": open_gap_pct,
                    "prev_day_high": snapshot.prev_day_high,
                    "prev_day_low": snapshot.prev_day_low,
                    "prev_day_close": snapshot.prev_day_close,
                },
                "technical_indicators": {
                    "ema9": snapshot.indicators.ema9,
                    "ema20": snapshot.indicators.ema20,
                    "ema50": snapshot.indicators.ema50,
                    "ema200": snapshot.indicators.ema200,
                    "ema20_slope": df_indicators.get("ema20_slope"),
                    "rsi": snapshot.indicators.rsi,
                    "roc_10": df_indicators.get("roc_10"),
                    "macd": snapshot.indicators.macd,
                    "macd_signal": snapshot.indicators.macd_signal,
                    "macd_histogram": snapshot.indicators.macd_hist,
                    "adx": snapshot.indicators.adx,
                    "atr": snapshot.indicators.atr,
                    "atr_slope": df_indicators.get("atr_slope"),
                    "bollinger_upper": snapshot.indicators.bollinger_upper,
                    "bollinger_middle": snapshot.indicators.bollinger_middle,
                    "bollinger_lower": snapshot.indicators.bollinger_lower,
                    "trend_strength": snapshot.indicators.trend_strength,
                },
                "market_structure": market_structure,
                "key_levels": key_levels_section or {"status": "unavailable"},
                "options_data": {
                    k: v for k, v in {
                        "pcr": snapshot.options_metrics.pcr,
                        "max_pain": snapshot.options_metrics.max_pain,
                        "call_oi_cluster": snapshot.options_metrics.call_oi_cluster,
                        "put_oi_cluster": snapshot.options_metrics.put_oi_cluster,
                        "oi_change": snapshot.options_metrics.oi_change,
                        "total_call_volume": snapshot.options_metrics.total_call_volume,
                        "total_put_volume": snapshot.options_metrics.total_put_volume,
                        "atm_option_volume": snapshot.options_metrics.atm_option_volume,
                    }.items() if v is not None
                } or {"status": "unavailable"},
                "global_context": global_section,
                "signal_score_breakdown": score_detail,
                "strategy_signal": {
                    "strategy": signal.strategy.value,
                    "option_type": signal.option_type.value,
                    "strike_price": signal.strike_price,
                    "option_premium_ltp": signal.entry_price,
                    "suggested_stoploss": signal.stoploss,
                    "suggested_target1": signal.target1,
                    "suggested_target2": signal.target2,
                    "risk_reward_ratio": rr_ratio,
                    "bid_ask_spread_pct": spread_pct,
                    "details": signal.details,
                },
                "session_context": session_section or {"status": "no_trades_yet"},
                "time_context": {
                    "current_time": now.strftime("%H:%M"),
                    "minutes_to_close": max(0, (15 * 60 + 30) - (now.hour * 60 + now.minute)),
                    "is_expiry_day": snapshot.is_expiry_day,
                    "day_of_week": now.strftime("%A"),
                },
            },
            indent=2,
        )

    def _parse_response(self, content: str, signal: StrategySignal) -> AIDecision:
        """Parse AI JSON response into AIDecision model."""
        try:
            # Strip markdown code fences if present
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines)

            data = json.loads(cleaned)
            return AIDecision(
                trade_decision=data.get("trade_decision", False),
                confidence_score=float(data.get("confidence_score", 0)),
                entry_price=float(data.get("entry_price", signal.entry_price)),
                stoploss=float(data.get("stoploss", signal.stoploss)),
                target1=float(data.get("target1", signal.target1)),
                target2=float(data.get("target2", signal.target2)),
                reason=str(data.get("reason", "")),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to parse AI response: %s — %s", content[:200], e)
            return AIDecision(
                trade_decision=False,
                reason=f"Parse error: {e}",
            )
