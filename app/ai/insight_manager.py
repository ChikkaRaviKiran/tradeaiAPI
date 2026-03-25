"""Insight Manager — provides AI insights to signal scorer and AI decision engine.

Acts as a bridge between PreMarketAnalyst's output and the trading pipeline:
  - Provides score modifiers based on pre-market analysis
  - Enriches AI decision prompts with market context
  - Tracks insight accuracy over time
"""

from __future__ import annotations

import logging
from typing import Optional

from app.ai.pre_market_analyst import PreMarketAnalyst
from app.data.institutional import InstitutionalFlow
from app.data.market_breadth import MarketBreadth

logger = logging.getLogger(__name__)

# Clamp range for score modifiers from AI insights
MAX_SCORE_MODIFIER = 15
MIN_SCORE_MODIFIER = -15


class InsightManager:
    """Manages AI insights and provides them to trading components."""

    def __init__(self, analyst: PreMarketAnalyst) -> None:
        self._analyst = analyst

    @property
    def has_insight(self) -> bool:
        return self._analyst.latest_insight is not None

    @property
    def insight(self) -> Optional[dict]:
        return self._analyst.latest_insight

    @property
    def institutional_flow(self) -> Optional[InstitutionalFlow]:
        return self._analyst.institutional_flow

    @property
    def market_breadth(self) -> Optional[MarketBreadth]:
        return self._analyst.market_breadth

    def get_score_modifier(self) -> int:
        """Get the AI-recommended score modifier for signal scoring.

        Clamped to [-15, +15]. Positive = more selective, negative = more aggressive.
        """
        if not self.has_insight:
            return 0
        mod = self.insight.get("score_modifier", 0)
        return max(MIN_SCORE_MODIFIER, min(MAX_SCORE_MODIFIER, int(mod)))

    def get_market_bias(self) -> str:
        """Get today's market bias: bullish/bearish/neutral."""
        if not self.has_insight:
            return "neutral"
        return self.insight.get("market_bias", "neutral")

    def get_bias_confidence(self) -> float:
        """Get confidence in the market bias (0-100)."""
        if not self.has_insight:
            return 0.0
        return float(self.insight.get("confidence", 0))

    def get_risk_advice(self) -> str:
        """Get risk advice: conservative/normal/aggressive."""
        if not self.has_insight:
            return "normal"
        return self.insight.get("risk_advice", "normal")

    def get_fii_dii_score(self, is_call: bool = True) -> float:
        """Score FII/DII data for signal scoring (0-6 pts).

        Directional: institutional buying supports CALL, selling supports PUT.
        """
        flow = self.institutional_flow
        if not flow:
            return 0.0

        signal = flow.signal
        if is_call:
            if signal == "strong_buy":
                return 6.0
            elif signal == "buy":
                return 4.0
            elif signal == "sell":
                return 1.0
            elif signal == "strong_sell":
                return 0.0
            return 3.0  # neutral
        else:
            # PUT: institutional selling supports bearish trades
            if signal == "strong_sell":
                return 6.0
            elif signal == "sell":
                return 4.0
            elif signal == "buy":
                return 1.0
            elif signal == "strong_buy":
                return 0.0
            return 3.0  # neutral

    def get_breadth_score(self, is_call: bool) -> float:
        """Score market breadth for signal scoring (0-5 pts).

        Directional: bullish breadth supports CALL, bearish supports PUT.
        """
        breadth = self.market_breadth
        if not breadth:
            return 0.0

        signal = breadth.breadth_signal
        if is_call:
            if signal == "strong_bullish":
                return 5.0
            elif signal == "bullish":
                return 3.5
            elif signal == "bearish":
                return 1.0
            elif signal == "strong_bearish":
                return 0.0
            return 2.5  # neutral
        else:
            if signal == "strong_bearish":
                return 5.0
            elif signal == "bearish":
                return 3.5
            elif signal == "bullish":
                return 1.0
            elif signal == "strong_bullish":
                return 0.0
            return 2.5  # neutral

    def get_news_sentiment_score(self, is_call: bool) -> float:
        """Score news sentiment for signal scoring (0-4 pts).

        Directional: positive sentiment supports CALL, negative supports PUT.
        """
        if not self.has_insight:
            return 0.0

        sentiment = self.insight.get("news_sentiment", 0)
        if isinstance(sentiment, dict):
            sentiment = sentiment.get("score", 0)

        if is_call:
            if sentiment > 0.5:
                return 4.0
            elif sentiment > 0.2:
                return 3.0
            elif sentiment > 0:
                return 2.0
            elif sentiment < -0.3:
                return 0.0
            return 1.0
        else:
            if sentiment < -0.5:
                return 4.0
            elif sentiment < -0.2:
                return 3.0
            elif sentiment < 0:
                return 2.0
            elif sentiment > 0.3:
                return 0.0
            return 1.0

    def get_ai_context_block(self) -> str:
        """Build a context block for the AI decision engine prompt.

        Provides learning context from insights so GTP can make better decisions.
        """
        if not self.has_insight:
            return ""

        insight = self.insight
        parts = []

        # Market bias
        bias = insight.get("market_bias", "neutral")
        conf = insight.get("confidence", 0)
        parts.append(f"Pre-market AI bias: {bias} (confidence: {conf}%)")

        # FII/DII
        fii_info = insight.get("fii_dii", {})
        if fii_info and fii_info.get("signal") != "unavailable":
            parts.append(
                f"FII/DII: FII net={fii_info.get('fii_net', 'N/A')}cr, "
                f"DII net={fii_info.get('dii_net', 'N/A')}cr, "
                f"signal={fii_info.get('signal', 'N/A')}"
            )

        # Breadth
        breadth_info = insight.get("breadth", {})
        if breadth_info and breadth_info.get("signal") != "unavailable":
            parts.append(
                f"Market breadth: A/D ratio={breadth_info.get('advance_decline_ratio', 'N/A')}, "
                f"signal={breadth_info.get('signal', 'N/A')}"
            )
            strong = breadth_info.get("strong_sectors", [])
            weak = breadth_info.get("weak_sectors", [])
            if strong:
                parts.append(f"Strong sectors: {', '.join(strong[:5])}")
            if weak:
                parts.append(f"Weak sectors: {', '.join(weak[:5])}")

        # Live news sentiment
        news_sentiment = insight.get("news_sentiment", 0)
        if isinstance(news_sentiment, dict):
            news_sentiment = news_sentiment.get("score", 0)
        if news_sentiment:
            direction = "bullish" if news_sentiment > 0.2 else "bearish" if news_sentiment < -0.2 else "neutral"
            parts.append(f"Live news sentiment: {news_sentiment:.2f} ({direction})")

        # AI plan
        plan = insight.get("trading_plan", "")
        if plan:
            parts.append(f"Trading plan: {plan}")

        # Lessons
        lessons = insight.get("lessons_applied", [])
        if lessons:
            parts.append(f"Lessons from past trades: {'; '.join(lessons[:3])}")

        return "\n".join(parts)

    def get_recent_news_summary(self, recent_news: list[dict]) -> str:
        """Build a compact news summary for AI decision context.

        Args:
            recent_news: List of news items from DB (last 2 hours).

        Returns:
            Formatted string with recent headlines and sentiment.
        """
        if not recent_news:
            return ""

        # Pick the 10 most recent items
        items = recent_news[:10]
        lines = ["Recent market news (last 2 hours):"]
        for item in items:
            sentiment = item.get("sentiment", "neutral")
            score = item.get("sentiment_score", 0)
            text = item.get("extracted_text", "")[:120]
            source = item.get("source", "unknown")
            lines.append(f"- [{source}] {text} (sentiment: {sentiment}, score: {score:.1f})")

        return "\n".join(lines)
