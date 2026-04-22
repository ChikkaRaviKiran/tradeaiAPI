"""AI-GPT 3-stage decision pipeline (Interpreter → Reasoning → Validator).

Used by AIGPTScanner. Each stage is a standalone JSON-mode LLM call so a
failure in one stage cleanly aborts the cycle (per spec: "skip trade if AI
fails"). All calls share the same `AsyncOpenAI` client.

Output shapes (verbatim from spec):

INTERPRETER:
  {trend, strength, market_phase, momentum, volume_behavior,
   oi_behavior, premium_behavior}

REASONING:
  {bias, setup_quality, confidence (int 0-100), expected_move, trade}
  trade ∈ {"CE", "PE", "NONE"}

VALIDATOR:
  {decision: "APPROVED"|"REJECTED", risk: "low"|"moderate"|"high",
   reason: str}

MONITOR (separate function — used after entry):
  {action: "HOLD"|"EXIT"|"PARTIAL_EXIT", confidence (int 0-100), reason: str}
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Per-call timeout. AI calls must complete fast (spec: < 2s ideal, hard cap 30s).
_CALL_TIMEOUT = 30.0


class AIGPTPipeline:
    """Wraps the 3 LLM stages plus the in-trade monitor call."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("AIGPTPipeline requires an OpenAI API key")
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key)

    # ── Stage 1: INTERPRETER ────────────────────────────────────────
    async def interpret(self, payload: dict) -> Optional[dict]:
        sys_prompt = (
            "You are a market structure interpreter for an Indian index "
            "(NIFTY) intraday options system. Given raw 5-minute market "
            "data, output a structured reading of trend, strength, phase, "
            "momentum, volume, OI and premium behaviour. Respond ONLY with "
            "a JSON object using EXACTLY these keys: trend, strength, "
            "market_phase, momentum, volume_behavior, oi_behavior, "
            "premium_behavior. Allowed values: trend ∈ {bullish, bearish, "
            "sideways}; strength ∈ {weak, moderate, strong}; market_phase ∈ "
            "{accumulation, breakout, trend, distribution, consolidation, "
            "exhaustion}; momentum ∈ {increasing, decreasing, flat}; "
            "volume_behavior ∈ {high, normal, low}; oi_behavior ∈ "
            "{bullish buildup, bearish buildup, long unwinding, short "
            "covering, neutral}; premium_behavior ∈ {expanding, "
            "contracting, flat}."
        )
        user_prompt = "Market data:\n" + json.dumps(payload, default=str)
        return await self._call(sys_prompt, user_prompt, "interpret")

    # ── Stage 2: REASONING ──────────────────────────────────────────
    async def reason(self, payload: dict, interpretation: dict) -> Optional[dict]:
        sys_prompt = (
            "You are an intraday options trade selector. Given raw market "
            "data plus a structured interpretation, decide if a high-"
            "conviction options trade exists right now. Respond ONLY with a "
            "JSON object using EXACTLY these keys: bias, setup_quality, "
            "confidence, expected_move, trade. Allowed values: bias ∈ "
            "{bullish, bearish, neutral}; setup_quality ∈ {A, B, C, D}; "
            "confidence is an INTEGER 0-100; expected_move is a short "
            "string like '30-50 points'; trade ∈ {CE, PE, NONE}. Be strict "
            "— output trade=NONE if there is no clear high-probability "
            "setup. Target moves of 30-50 NIFTY points."
        )
        user_prompt = (
            "Market data:\n" + json.dumps(payload, default=str)
            + "\n\nInterpretation:\n" + json.dumps(interpretation, default=str)
        )
        return await self._call(sys_prompt, user_prompt, "reason")

    # ── Stage 3: VALIDATOR ──────────────────────────────────────────
    async def validate(
        self, payload: dict, interpretation: dict, reasoning: dict
    ) -> Optional[dict]:
        sys_prompt = (
            "You are a risk validator for an intraday options trade about "
            "to be placed. Reject any setup that is misaligned, low-"
            "conviction, or carries excessive risk (e.g. fighting the "
            "trend, choppy structure, weak volume, contradictory OI). "
            "Respond ONLY with a JSON object using EXACTLY these keys: "
            "decision, risk, reason. Allowed values: decision ∈ {APPROVED, "
            "REJECTED}; risk ∈ {low, moderate, high}; reason is a short "
            "string."
        )
        user_prompt = (
            "Market data:\n" + json.dumps(payload, default=str)
            + "\n\nInterpretation:\n" + json.dumps(interpretation, default=str)
            + "\n\nReasoning:\n" + json.dumps(reasoning, default=str)
        )
        return await self._call(sys_prompt, user_prompt, "validate")

    # ── In-trade monitor ────────────────────────────────────────────
    async def monitor(self, payload: dict, trade_context: dict) -> Optional[dict]:
        sys_prompt = (
            "You are an intraday options trade monitor. A trade is OPEN. "
            "Given the latest market snapshot and the trade context, decide "
            "whether to HOLD, EXIT, or PARTIAL_EXIT. Be quick to EXIT if "
            "the original thesis breaks (trend reversal, momentum loss, "
            "premium collapse against direction). Respond ONLY with a JSON "
            "object using EXACTLY these keys: action, confidence, reason. "
            "Allowed values: action ∈ {HOLD, EXIT, PARTIAL_EXIT}; "
            "confidence is an INTEGER 0-100; reason is a short string."
        )
        user_prompt = (
            "Market data:\n" + json.dumps(payload, default=str)
            + "\n\nTrade context:\n" + json.dumps(trade_context, default=str)
        )
        return await self._call(sys_prompt, user_prompt, "monitor")

    # ── Internal: single JSON-mode call with timeout/error guards ──
    async def _call(self, system: str, user: str, stage: str) -> Optional[dict]:
        try:
            resp = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                ),
                timeout=_CALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("[AI-GPT] %s stage timed out after %ds", stage, _CALL_TIMEOUT)
            return None
        except Exception:
            logger.exception("[AI-GPT] %s stage failed", stage)
            return None

        try:
            content = resp.choices[0].message.content or "{}"
            data: Any = json.loads(content)
            if not isinstance(data, dict):
                logger.warning("[AI-GPT] %s returned non-object JSON: %r", stage, content)
                return None
            return data
        except (json.JSONDecodeError, IndexError, AttributeError):
            logger.exception("[AI-GPT] %s returned malformed JSON", stage)
            return None
