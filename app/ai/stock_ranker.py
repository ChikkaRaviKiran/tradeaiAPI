"""Stock Ranking AI — SRS Module 3.3.

Ranks stocks using weighted multi-factor scoring:
  Trend strength:      25%
  Institutional buying: 20%
  Volume breakout:     20%
  Earnings growth:     20%
  News sentiment:      15%
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pytz

from app.core.instruments import InstrumentConfig, get_equities
from app.core.models import SentimentScore, StockRanking

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

# Factor weights per SRS
WEIGHTS = {
    "trend_strength": 0.25,
    "institutional": 0.20,
    "volume_breakout": 0.20,
    "earnings_growth": 0.20,
    "sentiment": 0.15,
}


class StockRanker:
    """Rank stocks by composite AI score."""

    def rank_stocks(
        self,
        stock_data: dict[str, pd.DataFrame],
        sentiment_scores: dict[str, SentimentScore],
        fii_dii_signal: str = "neutral",
    ) -> list[StockRanking]:
        """Compute rankings for all stocks with available data.

        Args:
            stock_data: symbol → daily OHLCV DataFrame (min 50 rows)
            sentiment_scores: symbol → SentimentScore
            fii_dii_signal: institutional flow signal (strong_buy/buy/neutral/sell/strong_sell)
        """
        rankings: list[StockRanking] = []

        for symbol, df in stock_data.items():
            if df.empty or len(df) < 20:
                continue

            try:
                ranking = self._score_stock(symbol, df, sentiment_scores.get(symbol), fii_dii_signal)
                rankings.append(ranking)
            except Exception:
                logger.exception("Error ranking %s", symbol)

        # Sort by composite score descending
        rankings.sort(key=lambda r: r.composite_score, reverse=True)

        # Assign ranks
        for i, r in enumerate(rankings, 1):
            r.rank = i

        logger.info(
            "Stock rankings: Top 3 = %s",
            [(r.symbol, f"{r.composite_score:.1f}") for r in rankings[:3]],
        )
        return rankings

    def _score_stock(
        self,
        symbol: str,
        df: pd.DataFrame,
        sentiment: Optional[SentimentScore],
        fii_dii_signal: str,
    ) -> StockRanking:
        """Compute all factor scores for a single stock."""
        ranking = StockRanking(
            symbol=symbol,
            timestamp=datetime.now(_IST),
        )

        # 1. Trend Strength (25%) — EMA alignment + ADX
        ranking.trend_strength_score = self._score_trend(df)

        # 2. Institutional (20%) — uses FII/DII signal as proxy
        ranking.institutional_score = self._score_institutional(fii_dii_signal)

        # 3. Volume Breakout (20%) — recent volume vs average
        ranking.volume_breakout_score = self._score_volume(df)

        # 4. Earnings Growth (20%) — price momentum as proxy
        ranking.earnings_growth_score = self._score_momentum(df)

        # 5. Sentiment (15%) — from news analysis
        ranking.sentiment_score = self._score_sentiment(sentiment)

        # Weighted composite
        ranking.composite_score = round(
            ranking.trend_strength_score * WEIGHTS["trend_strength"]
            + ranking.institutional_score * WEIGHTS["institutional"]
            + ranking.volume_breakout_score * WEIGHTS["volume_breakout"]
            + ranking.earnings_growth_score * WEIGHTS["earnings_growth"]
            + ranking.sentiment_score * WEIGHTS["sentiment"],
            2,
        )

        return ranking

    @staticmethod
    def _score_trend(df: pd.DataFrame) -> float:
        """Score trend strength 0-100 based on EMA alignment and ADX."""
        close = df["close"] if "close" in df.columns else df["Close"]
        score = 0.0

        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()

        last_close = close.iloc[-1]
        last_ema20 = ema20.iloc[-1]
        last_ema50 = ema50.iloc[-1]

        # Price above EMAs
        if last_close > last_ema20:
            score += 30
        if last_close > last_ema50:
            score += 20

        # EMA alignment
        if last_ema20 > last_ema50:
            score += 20

        # EMA200 if enough data
        if len(close) >= 200:
            ema200 = close.ewm(span=200).mean().iloc[-1]
            if last_close > ema200:
                score += 15
            if last_ema20 > ema200:
                score += 15
        else:
            score += 15  # Partial credit

        return min(score, 100)

    @staticmethod
    def _score_institutional(fii_dii_signal: str) -> float:
        """Convert FII/DII signal to 0-100 score."""
        signal_map = {
            "strong_buy": 100,
            "buy": 75,
            "neutral": 50,
            "sell": 25,
            "strong_sell": 0,
        }
        return signal_map.get(fii_dii_signal, 50)

    @staticmethod
    def _score_volume(df: pd.DataFrame) -> float:
        """Score volume breakout 0-100."""
        volume = df["volume"] if "volume" in df.columns else df.get("Volume", pd.Series([0]))
        if volume.empty or volume.sum() == 0:
            return 50  # No volume data, neutral

        avg_vol = volume.rolling(20).mean().iloc[-1]
        last_vol = volume.iloc[-1]

        if avg_vol <= 0:
            return 50

        ratio = last_vol / avg_vol
        if ratio > 3.0:
            return 100
        elif ratio > 2.0:
            return 85
        elif ratio > 1.5:
            return 70
        elif ratio > 1.0:
            return 55
        return 35

    @staticmethod
    def _score_momentum(df: pd.DataFrame) -> float:
        """Score price momentum as earnings growth proxy (0-100)."""
        close = df["close"] if "close" in df.columns else df["Close"]
        if len(close) < 20:
            return 50

        # 20-day return
        ret_20d = (close.iloc[-1] / close.iloc[-20] - 1) * 100

        if ret_20d > 10:
            return 100
        elif ret_20d > 5:
            return 80
        elif ret_20d > 2:
            return 65
        elif ret_20d > 0:
            return 50
        elif ret_20d > -5:
            return 35
        return 15

    @staticmethod
    def _score_sentiment(sentiment: Optional[SentimentScore]) -> float:
        """Convert sentiment score (-1 to +1) to 0-100."""
        if sentiment is None or sentiment.source == "pending":
            return 50  # Neutral when unavailable
        # Map -1..+1 → 0..100
        return round((sentiment.score + 1) * 50, 1)
