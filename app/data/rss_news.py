"""RSS news fetcher — polls MoneyControl and Economic Times for market news.

Fetches RSS feeds every 30 minutes during market hours, extracts headlines,
and uses GPT-4o-mini to analyze sentiment for trading decisions.
"""

from __future__ import annotations

import hashlib
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Optional

import httpx
import pytz
from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/xml, text/xml, */*",
}

# RSS feed sources
RSS_FEEDS = [
    {
        "name": "MoneyControl Top News",
        "url": "https://www.moneycontrol.com/rss/MCtopnews.xml",
        "source": "moneycontrol",
    },
    {
        "name": "MoneyControl Market Reports",
        "url": "https://www.moneycontrol.com/rss/marketreports.xml",
        "source": "moneycontrol",
    },
    {
        "name": "Economic Times Markets",
        "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "source": "economictimes",
    },
]

# Track seen article IDs to avoid re-processing
_seen_article_ids: set[str] = set()


def _article_id(title: str, link: str) -> str:
    """Generate a unique ID for an article based on title and link."""
    return hashlib.md5(f"{title}|{link}".encode()).hexdigest()[:16]


async def fetch_rss_feeds() -> list[dict]:
    """Fetch all configured RSS feeds and return new (unseen) articles.

    Returns list of dicts with: title, description, link, source, pub_date.
    """
    all_articles = []

    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        for feed_info in RSS_FEEDS:
            try:
                resp = await client.get(feed_info["url"], headers=_HEADERS)
                if resp.status_code != 200:
                    logger.warning(
                        "RSS feed %s returned %d", feed_info["name"], resp.status_code
                    )
                    continue

                articles = _parse_rss_xml(resp.text, feed_info["source"])
                all_articles.extend(articles)
                logger.debug(
                    "RSS %s: fetched %d articles", feed_info["name"], len(articles)
                )

            except Exception:
                logger.warning("Failed to fetch RSS feed: %s", feed_info["name"])
                continue

    # Filter out already-seen articles
    new_articles = []
    for article in all_articles:
        aid = _article_id(article["title"], article.get("link", ""))
        if aid not in _seen_article_ids:
            _seen_article_ids.add(aid)
            article["article_id"] = aid
            new_articles.append(article)

    # Cap seen set to prevent unbounded growth (keep last 500)
    if len(_seen_article_ids) > 500:
        excess = len(_seen_article_ids) - 500
        for _ in range(excess):
            _seen_article_ids.pop()

    if new_articles:
        logger.info("RSS: %d new articles from %d total fetched", len(new_articles), len(all_articles))
    else:
        logger.debug("RSS: no new articles (%d total seen)", len(_seen_article_ids))

    return new_articles


def _parse_rss_xml(xml_text: str, source: str) -> list[dict]:
    """Parse RSS XML and extract articles."""
    articles = []
    try:
        root = ET.fromstring(xml_text)

        # Handle both RSS 2.0 (<channel><item>) and Atom (<entry>) formats
        items = root.findall(".//item")
        if not items:
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

        for item in items:
            title = _get_text(item, "title") or ""
            description = _get_text(item, "description") or ""
            link = _get_text(item, "link") or ""
            pub_date = _get_text(item, "pubDate") or _get_text(item, "published") or ""

            if not title:
                continue

            # Clean HTML from description
            description = _strip_html(description)
            # Truncate long descriptions
            if len(description) > 300:
                description = description[:300] + "..."

            articles.append({
                "title": title.strip(),
                "description": description.strip(),
                "link": link.strip(),
                "source": source,
                "pub_date": pub_date.strip(),
            })

    except ET.ParseError:
        logger.warning("Failed to parse RSS XML from %s", source)

    return articles


def _get_text(element: ET.Element, tag: str) -> Optional[str]:
    """Get text from an XML element, handling namespaces."""
    child = element.find(tag)
    if child is None:
        # Try with Atom namespace
        child = element.find(f"{{http://www.w3.org/2005/Atom}}{tag}")
    return child.text if child is not None and child.text else None


def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    import re
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"&\w+;", " ", clean)
    return re.sub(r"\s+", " ", clean).strip()


async def analyze_news_sentiment(articles: list[dict]) -> list[dict]:
    """Use GPT-4o-mini to analyze sentiment of news articles.

    Returns the same articles enriched with sentiment, sentiment_score, and symbols.
    """
    if not articles:
        return []

    client = AsyncOpenAI(api_key=settings.openai_api_key)

    # Batch all articles into one GPT call for efficiency
    headlines = []
    for i, article in enumerate(articles[:20]):  # Cap at 20 per batch
        headlines.append(f"{i+1}. [{article['source']}] {article['title']}")
        if article.get("description"):
            headlines.append(f"   {article['description'][:150]}")

    headlines_text = "\n".join(headlines)

    prompt = f"""Analyze these Indian stock market news headlines for trading sentiment.
For each headline, provide:
- sentiment: "bullish", "bearish", or "neutral"
- score: -1.0 (very bearish) to +1.0 (very bullish)
- symbols: comma-separated stock/index symbols mentioned (NIFTY, BANKNIFTY, RELIANCE, etc.) or empty

Respond ONLY with a JSON array, one object per headline in order:
[{{"sentiment": "bullish", "score": 0.6, "symbols": "NIFTY,BANKNIFTY"}}, ...]

Headlines:
{headlines_text}"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial sentiment analyst for Indian stock markets. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )

        content = response.choices[0].message.content or "[]"
        # Strip markdown fences if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        import json
        sentiments = json.loads(content)

        # Merge sentiment data back into articles
        for i, article in enumerate(articles[:20]):
            if i < len(sentiments):
                s = sentiments[i]
                article["sentiment"] = s.get("sentiment", "neutral")
                article["sentiment_score"] = float(s.get("score", 0))
                article["symbols"] = s.get("symbols", "")
            else:
                article["sentiment"] = "neutral"
                article["sentiment_score"] = 0.0
                article["symbols"] = ""

        logger.info(
            "RSS sentiment: analyzed %d articles, avg_score=%.2f",
            len(articles[:20]),
            sum(a.get("sentiment_score", 0) for a in articles[:20]) / max(len(articles[:20]), 1),
        )

    except Exception:
        logger.exception("GPT sentiment analysis failed — marking all as neutral")
        for article in articles:
            article.setdefault("sentiment", "neutral")
            article.setdefault("sentiment_score", 0.0)
            article.setdefault("symbols", "")

    return articles


async def fetch_and_analyze() -> list[dict]:
    """Full pipeline: fetch RSS feeds → analyze sentiment → return enriched articles.

    Only processes NEW articles (not previously seen).
    """
    articles = await fetch_rss_feeds()
    if not articles:
        return []

    analyzed = await analyze_news_sentiment(articles)

    # Prepare for DB save (compatible with TelegramNewsRecord schema)
    today = datetime.now(_IST).strftime("%Y-%m-%d")
    for article in analyzed:
        article["date"] = today
        article["extracted_text"] = f"{article['title']}. {article.get('description', '')}"
        article["message_id"] = None  # RSS doesn't have message_id
        article["image_url"] = article.get("link", "")

    return analyzed
