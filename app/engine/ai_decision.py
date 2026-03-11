"""AI Decision Engine — validates signals using OpenAI LLM."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.models import (
    AIDecision,
    GlobalBias,
    MarketRegime,
    MarketSnapshot,
    StrategySignal,
    TechnicalIndicators,
    OptionsMetrics,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert NIFTY options trading analyst. You receive market data and a strategy signal. 
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
- Be conservative — prefer no trade over a bad trade
- Stoploss should be 25-30% of option premium
- Target1 = 50% profit, Target2 = 100% profit
- Consider options OI data for confirmation
- Avoid trades in last 30 minutes of market (after 15:00)
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
    ) -> AIDecision:
        """Send signal + market context to AI for validation."""
        try:
            prompt = self._build_prompt(signal, snapshot, score)
            client = self._get_client()

            response = await client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
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
    ) -> str:
        """Build structured prompt for AI evaluation."""
        now = datetime.now()
        return json.dumps(
            {
                "market_snapshot": {
                    "nifty_price": snapshot.nifty_price,
                    "vwap": snapshot.vwap,
                    "timestamp": now.strftime("%H:%M:%S"),
                },
                "technical_indicators": {
                    "ema9": snapshot.indicators.ema9,
                    "ema20": snapshot.indicators.ema20,
                    "ema50": snapshot.indicators.ema50,
                    "rsi": snapshot.indicators.rsi,
                    "macd": snapshot.indicators.macd,
                    "adx": snapshot.indicators.adx,
                    "atr": snapshot.indicators.atr,
                },
                "market_structure": {
                    "regime": snapshot.regime.value,
                },
                "options_data": {
                    "pcr": snapshot.options_metrics.pcr,
                    "max_pain": snapshot.options_metrics.max_pain,
                    "call_oi_cluster": snapshot.options_metrics.call_oi_cluster,
                    "put_oi_cluster": snapshot.options_metrics.put_oi_cluster,
                },
                "global_context": {
                    "bias": snapshot.global_bias.value,
                },
                "strategy_signal": {
                    "strategy": signal.strategy.value,
                    "option_type": signal.option_type.value,
                    "entry_price": signal.entry_price,
                    "strike_price": signal.strike_price,
                    "score": score,
                    "details": signal.details,
                },
                "volatility_regime": snapshot.regime.value,
                "time_context": {
                    "current_time": now.strftime("%H:%M"),
                    "minutes_to_close": max(0, (15 * 60 + 30) - (now.hour * 60 + now.minute)),
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
