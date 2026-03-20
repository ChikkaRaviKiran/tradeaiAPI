"""Pre-Market Analyst — AI-powered daily market intelligence.

Runs at 08:45 AM IST (pre-market), collects all data sources and generates:
  1. Market bias prediction (bullish/bearish/neutral)
  2. Key trading levels (support/resistance)
  3. Sector strength assessment
  4. Trading plan for the day
  5. Score modifiers for the signal scorer

Data sources:
  - FII/DII institutional flow (NSE)
  - Market breadth (advance/decline + sectors from NSE)
  - Telegram news (GPT Vision extracted)
  - Rolling 30-day trade history (self-learning)
  - Global market indices
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import pytz
from openai import AsyncOpenAI

from app.core.config import settings
from app.data.institutional import InstitutionalFlow, fetch_fii_dii_data
from app.data.market_breadth import MarketBreadth, fetch_market_breadth
from app.data.telegram_news import collect_telegram_news, save_news_to_db, get_recent_news
from app.data.global_markets import fetch_global_indices

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

PRE_MARKET_SYSTEM_PROMPT = """You are an expert Indian stock market analyst preparing a pre-market trading plan.

You receive comprehensive market data including:
- FII/DII institutional flows
- Market breadth (advance/decline, sector performance)
- NIFTY 50 spot price (previous close, last price, day high/low) — USE THIS for accurate key_levels
- Recent news from financial channels
- Rolling 30-day trade performance history
- Global market indices

Your job is to produce a structured trading plan for the day. Respond ONLY with valid JSON:
{
    "market_bias": "bullish" | "bearish" | "neutral",
    "confidence": 0-100,
    "summary": "2-3 sentence market outlook",
    "key_observations": ["observation1", "observation2", "observation3"],
    "trading_plan": "Detailed trading plan for the day (3-5 bullet points)",
    "sectors_to_watch": ["SECTOR1", "SECTOR2"],
    "avoid_sectors": ["SECTOR1"],
    "score_modifier": -15 to +15,
    "risk_advice": "conservative" | "normal" | "aggressive",
    "key_levels": {
        "nifty_support": [level1, level2],
        "nifty_resistance": [level1, level2]
    },
    "lessons_applied": ["lesson from past trades being applied today"]
}

Rules:
- score_modifier adjusts the signal scoring threshold: positive = more selective, negative = more aggressive
- Base your analysis on DATA, not assumptions
- If FII heavily selling (>1000cr), lean bearish unless strong DII buying offsets
- Breadth A/D ratio > 1.5 is bullish, < 0.7 is bearish
- News sentiment aggregated from channel analysis
- Past trade patterns: note repeated losses in specific conditions and adjust
- Be specific about entry/exit conditions in the trading plan
- Confidence reflects how clear the market direction is (< 40 means uncertain)
- key_levels MUST be based on actual NIFTY spot price data provided — support should be below and resistance above the current/previous close
"""


class PreMarketAnalyst:
    """Generates AI-driven pre-market intelligence from multiple data sources."""

    def __init__(self) -> None:
        self._client: Optional[AsyncOpenAI] = None
        self._latest_insight: Optional[dict] = None
        self._institutional_flow: Optional[InstitutionalFlow] = None
        self._market_breadth: Optional[MarketBreadth] = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    @property
    def latest_insight(self) -> Optional[dict]:
        return self._latest_insight

    @property
    def institutional_flow(self) -> Optional[InstitutionalFlow]:
        return self._institutional_flow

    @property
    def market_breadth(self) -> Optional[MarketBreadth]:
        return self._market_breadth

    async def run_analysis(self) -> Optional[dict]:
        """Execute full pre-market analysis pipeline.

        1. Collect data from all sources
        2. Send to GPT for analysis
        3. Save insight to DB
        4. Return structured insight
        """
        logger.info("Starting pre-market analysis...")
        today = datetime.now(_IST).strftime("%Y-%m-%d")

        # 1. Collect all data concurrently
        import asyncio
        fii_task = fetch_fii_dii_data()
        breadth_task = fetch_market_breadth()
        news_task = collect_telegram_news()
        global_task = fetch_global_indices()

        results = await asyncio.gather(
            fii_task, breadth_task, news_task, global_task,
            return_exceptions=True,
        )

        fii_data = results[0] if not isinstance(results[0], Exception) else None
        breadth_data = results[1] if not isinstance(results[1], Exception) else None
        news_items = results[2] if not isinstance(results[2], Exception) else []
        global_indices = results[3] if not isinstance(results[3], Exception) else []

        # Store for orchestrator access
        self._institutional_flow = fii_data
        self._market_breadth = breadth_data

        # Log data collection results
        logger.info(
            "Pre-market data collected: FII/DII=%s, Breadth=%s, News images=%d, Global=%d",
            "yes" if fii_data else "NO",
            "yes" if breadth_data else "NO",
            len(news_items) if isinstance(news_items, list) else 0,
            len(global_indices) if isinstance(global_indices, list) else 0,
        )

        # Save news to DB
        if news_items:
            saved = await save_news_to_db(news_items)
            logger.info("Telegram news: %d items scraped, %d saved to DB", len(news_items), saved)
        else:
            logger.warning(
                "No Telegram news collected — channel=%s (set TELEGRAM_NEWS_CHANNEL in .env)",
                settings.telegram_news_channel or 'NOT SET',
            )

        # 2. Fetch 30-day trade history for learning
        trade_history = await self._get_rolling_trade_history(days=30)

        # 3. Fetch recent news from DB (includes today's + recent days)
        recent_news = await get_recent_news(days=2)
        logger.info("Recent news from DB (last 2 days): %d items", len(recent_news))

        # 4. Build analysis prompt
        prompt_data = self._build_prompt_data(
            fii_data, breadth_data, recent_news, trade_history, global_indices,
        )

        # 5. Get AI analysis
        insight = await self._get_ai_analysis(prompt_data)
        if not insight:
            logger.warning("Pre-market AI analysis returned empty")
            return None

        # Enrich insight with raw data
        insight["fii_dii"] = {
            "fii_net": fii_data.fii_net if fii_data else None,
            "dii_net": fii_data.dii_net if fii_data else None,
            "signal": fii_data.signal if fii_data else "unavailable",
        }
        insight["breadth"] = {
            "advance_decline_ratio": breadth_data.advance_decline_ratio if breadth_data else None,
            "signal": breadth_data.breadth_signal if breadth_data else "unavailable",
            "strong_sectors": breadth_data.strong_sectors if breadth_data else [],
            "weak_sectors": breadth_data.weak_sectors if breadth_data else [],
            "sectors": [
                {"name": s.name, "change_pct": s.change_pct}
                for s in (breadth_data.sectors if breadth_data else [])
            ],
        }
        insight["news_count"] = len(recent_news)
        insight["date"] = today

        self._latest_insight = insight

        # 6. Save to DB
        await self._save_insight(insight, fii_data, breadth_data, recent_news)

        logger.info(
            "Pre-market analysis complete: bias=%s, confidence=%d, modifier=%+d",
            insight.get("market_bias", "?"),
            insight.get("confidence", 0),
            insight.get("score_modifier", 0),
        )
        return insight

    def _build_prompt_data(
        self,
        fii: Optional[InstitutionalFlow],
        breadth: Optional[MarketBreadth],
        news: list[dict],
        trades: list[dict],
        global_indices: list,
    ) -> str:
        """Build structured prompt data for GPT analysis."""
        data = {}

        # FII/DII
        if fii:
            data["institutional_flows"] = {
                "fii_net_crores": fii.fii_net,
                "dii_net_crores": fii.dii_net,
                "net_combined": fii.net_institutional,
                "signal": fii.signal,
            }
        else:
            data["institutional_flows"] = "unavailable"

        # Market breadth
        if breadth:
            data["market_breadth"] = {
                "advancing": breadth.total_advancing,
                "declining": breadth.total_declining,
                "ad_ratio": breadth.advance_decline_ratio,
                "signal": breadth.breadth_signal,
                "strong_sectors": breadth.strong_sectors,
                "weak_sectors": breadth.weak_sectors,
                "all_sectors": [
                    {"name": s.name, "change_pct": s.change_pct}
                    for s in breadth.sectors
                ],
            }
        else:
            data["market_breadth"] = "unavailable"

        # News (last 2 days)
        if news:
            data["recent_news"] = [
                {
                    "text": n.get("extracted_text", ""),
                    "symbols": n.get("symbols", ""),
                    "sentiment": n.get("sentiment", ""),
                    "score": n.get("sentiment_score", 0),
                    "date": n.get("date", ""),
                }
                for n in news[:20]  # Limit to 20 most recent
            ]
        else:
            data["recent_news"] = "no news available"

        # Rolling 30-day trade history
        if trades:
            win_count = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
            loss_count = sum(1 for t in trades if (t.get("pnl") or 0) < 0)
            total_pnl = sum(t.get("pnl") or 0 for t in trades)

            # Find patterns in losses
            loss_strategies = {}
            for t in trades:
                if (t.get("pnl") or 0) < 0:
                    strat = t.get("strategy", "unknown")
                    loss_strategies[strat] = loss_strategies.get(strat, 0) + 1

            data["trade_history_30d"] = {
                "total_trades": len(trades),
                "winners": win_count,
                "losers": loss_count,
                "win_rate": round(win_count / len(trades) * 100, 1) if trades else 0,
                "total_pnl": round(total_pnl, 2),
                "loss_pattern_by_strategy": loss_strategies,
                "recent_5_trades": [
                    {
                        "date": t.get("date"),
                        "strategy": t.get("strategy"),
                        "instrument": t.get("instrument"),
                        "pnl": t.get("pnl"),
                        "option_type": t.get("option_type"),
                    }
                    for t in trades[:5]
                ],
            }
        else:
            data["trade_history_30d"] = "no trade history"

        # Global indices
        if global_indices:
            data["global_indices"] = [
                {
                    "symbol": idx.symbol if hasattr(idx, "symbol") else str(idx),
                    "change_pct": idx.change_pct if hasattr(idx, "change_pct") else 0,
                }
                for idx in global_indices
            ]
        else:
            data["global_indices"] = "unavailable"

        # NIFTY 50 spot price for accurate key level generation
        if breadth and breadth.nifty_prev_close:
            data["nifty_spot"] = {
                "previous_close": breadth.nifty_prev_close,
                "last_price": breadth.nifty_last_price,
                "day_high": breadth.nifty_day_high,
                "day_low": breadth.nifty_day_low,
            }
        else:
            data["nifty_spot"] = "unavailable"

        data["current_date"] = datetime.now(_IST).strftime("%Y-%m-%d")
        data["current_time"] = datetime.now(_IST).strftime("%H:%M IST")

        return json.dumps(data, indent=2, default=str)

    async def _get_ai_analysis(self, prompt_data: str) -> Optional[dict]:
        """Send data to GPT and get structured analysis."""
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": PRE_MARKET_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_data},
                ],
                temperature=0.2,
                max_tokens=1000,
            )

            content = response.choices[0].message.content
            if not content:
                return None

            # Clean markdown fences
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines)

            return json.loads(cleaned)

        except Exception:
            logger.exception("Pre-market AI analysis failed")
            return None

    async def _get_rolling_trade_history(self, days: int = 30) -> list[dict]:
        """Fetch trade history for the last N days from DB."""
        from app.db.models import TradeRecord, AsyncSessionLocal
        from sqlalchemy import select

        cutoff = (datetime.now(_IST) - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(TradeRecord)
                    .where(TradeRecord.date >= cutoff)
                    .order_by(TradeRecord.created_at.desc())
                )
                records = result.scalars().all()
                return [
                    {
                        "date": r.date,
                        "instrument": r.instrument,
                        "strategy": r.strategy,
                        "option_type": r.option_type,
                        "pnl": r.pnl,
                        "entry_price": r.entry_price,
                        "exit_price": r.exit_price,
                        "confidence": r.confidence,
                        "status": r.status,
                    }
                    for r in records
                ]
        except Exception:
            logger.exception("Error fetching trade history")
            return []

    async def _save_insight(
        self,
        insight: dict,
        fii: Optional[InstitutionalFlow],
        breadth: Optional[MarketBreadth],
        news: list[dict],
    ) -> None:
        """Save insight to database."""
        from app.db.models import DailyAIInsight, AsyncSessionLocal

        today = datetime.now(_IST).strftime("%Y-%m-%d")

        try:
            async with AsyncSessionLocal() as session:
                record = DailyAIInsight(
                    date=today,
                    insight_type="pre_market",
                    market_bias=insight.get("market_bias"),
                    confidence=insight.get("confidence", 0),
                    fii_dii_signal=fii.signal if fii else None,
                    fii_net=fii.fii_net if fii else None,
                    dii_net=fii.dii_net if fii else None,
                    breadth_signal=breadth.breadth_signal if breadth else None,
                    advance_decline_ratio=breadth.advance_decline_ratio if breadth else None,
                    news_sentiment=_avg_news_sentiment(news),
                    strong_sectors=",".join(breadth.strong_sectors) if breadth else None,
                    weak_sectors=",".join(breadth.weak_sectors) if breadth else None,
                    key_levels=json.dumps(insight.get("key_levels", {})),
                    ai_summary=insight.get("summary", ""),
                    trading_plan=insight.get("trading_plan", ""),
                    raw_data=json.dumps(insight, default=str),
                )
                session.add(record)
                await session.commit()
                logger.info("Pre-market insight saved to DB")
        except Exception:
            logger.exception("Error saving pre-market insight")

    async def get_insight_from_db(self, date: str = "") -> Optional[dict]:
        """Load today's insight from DB (cold start recovery)."""
        from app.db.models import DailyAIInsight, AsyncSessionLocal
        from sqlalchemy import select

        target = date or datetime.now(_IST).strftime("%Y-%m-%d")

        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(DailyAIInsight)
                    .where(DailyAIInsight.date == target)
                    .order_by(DailyAIInsight.created_at.desc())
                    .limit(1)
                )
                record = result.scalar_one_or_none()
                if not record:
                    return None

                insight = {
                    "date": record.date,
                    "insight_type": record.insight_type,
                    "market_bias": record.market_bias,
                    "confidence": record.confidence,
                    "ai_summary": record.ai_summary,
                    "trading_plan": record.trading_plan,
                    "fii_dii": {
                        "fii_net": record.fii_net,
                        "dii_net": record.dii_net,
                        "signal": record.fii_dii_signal,
                    },
                    "breadth": {
                        "advance_decline_ratio": record.advance_decline_ratio,
                        "signal": record.breadth_signal,
                        "strong_sectors": record.strong_sectors.split(",") if record.strong_sectors else [],
                        "weak_sectors": record.weak_sectors.split(",") if record.weak_sectors else [],
                    },
                    "key_levels": json.loads(record.key_levels) if record.key_levels else {},
                    "news_sentiment": record.news_sentiment,
                }

                # Try to load full data from raw_data
                if record.raw_data:
                    try:
                        full = json.loads(record.raw_data)
                        insight.update(full)
                    except json.JSONDecodeError:
                        pass

                return insight
        except Exception:
            logger.exception("Error loading insight from DB")
            return None


def _avg_news_sentiment(news: list[dict]) -> float:
    """Compute average sentiment score from news items."""
    scores = [n.get("sentiment_score", 0) for n in news if n.get("sentiment_score") is not None]
    return round(sum(scores) / len(scores), 3) if scores else 0.0
