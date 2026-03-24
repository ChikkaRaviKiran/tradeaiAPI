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


SYSTEM_PROMPT = """You are an expert Indian market options trading analyst. You receive real-time market data and a strategy signal with the actual option premium (LTP).
Your job is to validate the trade signal and provide a final decision.

You must respond ONLY with a valid JSON object (no markdown, no extra text) in this exact format:
{
    "trade_decision": true/false,
    "confidence_score": 0-100,
    "entry_price": float,
    "stoploss": float,
    "target1": float,
    "target2": float,
    "reason": "brief explanation"
}

Rules:
- Only approve trades with confidence >= 70
- Consider market regime, volatility, and time of day
- A good setup needs 3+ confirmations — don't approve marginal setups
- entry_price, stoploss, target1, target2 are all OPTION PREMIUM values (not NIFTY spot)
- The signal includes ATR-based suggested SL/targets — evaluate whether they are reasonable given the ATR value
- Use the ATR value to judge if SL is too tight or too wide for current volatility
- If ATR is null, reject the trade (volatility data unavailable)
- Consider PCR, OI data, and option volumes for flow confirmation — if PCR is null, options data is unavailable (don't penalize)
- When PCR or global bias data is missing, focus more on technical indicators (RSI, EMAs, ADX)
- EMA200 provides long-term trend context — price above EMA200 favors CALL, below favors PUT
- vwap_is_volume_weighted indicates if VWAP is computed from real futures volume (true) or just price average (false)
- Consider global market bias — if "unavailable", ignore global context (don't reject because of it)
- REJECT trades after 14:30 — last hour has highest chop/reversal rate
- REJECT when regime is "insufficient_data" — not enough market data yet
- REJECT when price is within 0.1% of previous day high/low — these are reversal zones
- Be extra cautious on expiry days (is_expiry_day=true) — gamma risk increases sharply, pin risk near ATM strikes
- prev_day_high, prev_day_low are key support/resistance levels — price near these levels often reverses
- If spot is within 0.15% of prev_day_high for CALL or prev_day_low for PUT, be cautious (reversal zone)
- prev_day_close is the reference — opening gap direction matters (gap up favors CALL momentum, gap down favors PUT)
- day_open and open_gap_pct show today's opening gap — large gap ups (>0.5%) favor CALL continuation, large gap downs favor PUT
- signal_score_breakdown shows how the pre-computed signal scored across multiple factors (each factor's max points):
  strategy_trigger (22): RSI, candle body, breakout margin, MACD confirmation
  volume_confirmation (18): futures/options volume vs average
  vwap_alignment (13): price alignment with VWAP (higher if real volume-weighted)
  options_oi_signal (13): PCR direction + OI change confirmation
  global_bias_score (8): alignment with global market direction
  historical_pattern (11): EMA hierarchy, ADX, price structure
  Use the breakdown to understand WHERE confirmation is strong vs weak — a high total from just strategy_trigger with no volume/options confirmation is fragile
- Bollinger bands help assess if price is extended — reject if price is at extreme bands without reversal signal
- Any null indicator means that data point is genuinely unavailable — do not assume a default value
- Trend strength score: 3=strong uptrend (ema9>20>50>200), 0=strong downtrend
- If pre-market intelligence data is provided, consider FII/DII flows, market breadth, and news sentiment
- FII selling > 1000cr is bearish; DII buying > 1000cr supports market
- Market breadth A/D ratio > 1.5 is bullish, < 0.7 is bearish
- Apply lessons from past trade patterns mentioned in the intelligence context
- If htf_trend is provided ("bullish"/"bearish"/"neutral"), it represents the 5-minute chart EMA trend direction. Signals opposing this trend are lower probability — require extra confirmation.
- IMPORTANT: Your confidence_score MUST be your OWN independent assessment based on the market data, indicators, and setup quality. Do NOT copy or mirror any pre-computed score. Assess the setup freshly using the technical indicators, market regime, options data, and timing.
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
    ) -> AIDecision:
        """Send signal + market context to AI for validation."""
        try:
            prompt = self._build_prompt(signal, snapshot, score, score_breakdown)

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
                temperature=0.2,
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
    ) -> str:
        """Build structured prompt for AI evaluation."""
        now = datetime.now(_IST)

        # Score breakdown tells AI where confirmation is strong vs weak
        score_detail = {"total": round(score, 1)}
        if score_breakdown:
            score_detail.update({
                "strategy_trigger": round(score_breakdown.strategy_trigger, 1),
                "volume_confirmation": round(score_breakdown.volume_confirmation, 1),
                "vwap_alignment": round(score_breakdown.vwap_alignment, 1),
                "options_oi_signal": round(score_breakdown.options_oi_signal, 1),
                "global_bias_score": round(score_breakdown.global_bias_score, 1),
                "historical_pattern": round(score_breakdown.historical_pattern, 1),
            })

        # Compute today's open gap context
        open_gap_pct = None
        if snapshot.day_open and snapshot.prev_day_close and snapshot.prev_day_close > 0:
            open_gap_pct = round(
                ((snapshot.day_open - snapshot.prev_day_close) / snapshot.prev_day_close) * 100, 2
            )

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
                    "rsi": snapshot.indicators.rsi,
                    "macd": snapshot.indicators.macd,
                    "macd_histogram": snapshot.indicators.macd_hist,
                    "adx": snapshot.indicators.adx,
                    "atr": snapshot.indicators.atr,
                    "bollinger_upper": snapshot.indicators.bollinger_upper,
                    "bollinger_lower": snapshot.indicators.bollinger_lower,
                    "trend_strength": snapshot.indicators.trend_strength,
                },
                "market_structure": {
                    "regime": snapshot.regime.value,
                    "htf_trend": snapshot.htf_trend,
                },
                "options_data": {
                    "pcr": snapshot.options_metrics.pcr,
                    "max_pain": snapshot.options_metrics.max_pain,
                    "call_oi_cluster": snapshot.options_metrics.call_oi_cluster,
                    "put_oi_cluster": snapshot.options_metrics.put_oi_cluster,
                    "oi_change": snapshot.options_metrics.oi_change,
                    "total_call_volume": snapshot.options_metrics.total_call_volume,
                    "total_put_volume": snapshot.options_metrics.total_put_volume,
                    "atm_option_volume": snapshot.options_metrics.atm_option_volume,
                },
                "global_context": {
                    "bias": snapshot.global_bias.value,
                },
                "signal_score_breakdown": score_detail,
                "strategy_signal": {
                    "strategy": signal.strategy.value,
                    "option_type": signal.option_type.value,
                    "strike_price": signal.strike_price,
                    "option_premium_ltp": signal.entry_price,
                    "suggested_stoploss": signal.stoploss,
                    "suggested_target1": signal.target1,
                    "suggested_target2": signal.target2,
                    "details": signal.details,
                },
                "volatility_regime": snapshot.regime.value,
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
