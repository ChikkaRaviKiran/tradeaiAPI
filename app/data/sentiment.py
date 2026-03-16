"""News & Sentiment Analysis Engine — SRS Module 3.4.

Uses GPT-based classification to score news sentiment for stocks.
Sources: Financial news via web search, earnings data.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pytz
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.models import SentimentScore

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

SENTIMENT_SYSTEM_PROMPT = """You are a financial news sentiment analyzer for Indian stock markets.

Given a stock symbol and recent news headlines, analyze the sentiment and return a JSON response:
{
    "symbol": "SYMBOL",
    "score": float between -1.0 (very bearish) and +1.0 (very bullish),
    "summary": "one-line summary of sentiment drivers",
    "key_factors": ["factor1", "factor2"]
}

Scoring guide:
  +0.7 to +1.0: Strong positive (earnings beat, major contract win, upgrade)
  +0.3 to +0.7: Moderate positive (decent results, sector tailwind)
  -0.3 to +0.3: Neutral (mixed signals, no clear direction)
  -0.7 to -0.3: Moderate negative (earnings miss, sector headwind)
  -1.0 to -0.7: Strong negative (fraud, regulatory action, downgrade)

Respond ONLY with valid JSON. No markdown, no extra text.
"""


class SentimentAnalyzer:
    """Analyzes news sentiment using GPT."""

    def __init__(self) -> None:
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    async def analyze_headlines(
        self,
        symbol: str,
        headlines: list[str],
    ) -> Optional[SentimentScore]:
        """Analyze a list of news headlines for a stock.

        Args:
            symbol: Stock symbol (e.g. "TCS", "RELIANCE")
            headlines: List of recent news headlines

        Returns:
            SentimentScore or None on failure
        """
        if not headlines:
            return None

        try:
            client = self._get_client()
            prompt = (
                f"Stock: {symbol}\n"
                f"Headlines:\n" + "\n".join(f"- {h}" for h in headlines[:10])
            )

            response = await client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=200,
            )

            content = response.choices[0].message.content
            if not content:
                return None

            import json
            # Strip markdown fences
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines)

            data = json.loads(cleaned)
            return SentimentScore(
                symbol=symbol,
                score=float(data.get("score", 0)),
                source="news",
                headline=data.get("summary", ""),
                timestamp=datetime.now(_IST),
            )

        except Exception:
            logger.exception("Sentiment analysis failed for %s", symbol)
            return None

    async def analyze_symbol(self, symbol: str) -> Optional[SentimentScore]:
        """Full pipeline: fetch news + analyze sentiment for a symbol.

        Currently uses a placeholder — to be connected to a news API.
        """
        # Placeholder: return neutral until news API is integrated
        logger.debug("Sentiment analysis for %s — news API not yet connected", symbol)
        return SentimentScore(
            symbol=symbol,
            score=0.0,
            source="pending",
            headline="News API integration pending",
        )
