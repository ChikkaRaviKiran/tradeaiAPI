"""Telegram news channel scraper — extracts financial news from image posts.

Scrapes the public web preview of @daytradertelugu (or configured channel)
and uses GPT-4o-mini Vision to extract text from news images.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Optional

import httpx
import pytz
from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
}

# Max images to process per scrape (cost control)
MAX_IMAGES_PER_SCRAPE = 15
# Batch size for GPT Vision calls (images per call)
VISION_BATCH_SIZE = 4


async def scrape_channel_images(
    channel: str = "",
    max_pages: int = 3,
) -> list[dict]:
    """Scrape recent image posts from Telegram channel web preview.

    Args:
        channel: Channel username (without @). Uses config if empty.
        max_pages: Max pagination pages to fetch.

    Returns:
        List of dicts with 'image_url' and 'message_id'.
    """
    channel = channel or settings.telegram_news_channel.lstrip("@")
    if not channel:
        logger.warning("No Telegram news channel configured")
        return []

    base_url = f"https://t.me/s/{channel}"
    images: list[dict] = []
    url = base_url

    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            for page in range(max_pages):
                resp = await client.get(url, headers=_HEADERS)
                if resp.status_code != 200:
                    logger.warning("Channel scrape returned %d for %s", resp.status_code, url)
                    break

                html = resp.text
                page_images = _extract_images(html, channel)
                if not page_images:
                    break

                images.extend(page_images)

                # Pagination: find earliest message ID for ?before= param
                msg_ids = [img["message_id"] for img in page_images if img.get("message_id")]
                if msg_ids:
                    earliest = min(msg_ids)
                    url = f"{base_url}?before={earliest}"
                else:
                    break

    except Exception:
        logger.exception("Error scraping Telegram channel %s", channel)

    # Deduplicate by message_id and limit
    seen = set()
    unique = []
    for img in images:
        mid = img.get("message_id")
        if mid and mid not in seen:
            seen.add(mid)
            unique.append(img)

    logger.info("Scraped %d unique images from %s", len(unique), channel)
    return unique[:MAX_IMAGES_PER_SCRAPE]


def _extract_images(html: str, channel: str) -> list[dict]:
    """Extract image URLs and message IDs from Telegram web preview HTML."""
    results = []

    # Pattern for message containers with data-post attribute
    msg_pattern = re.compile(
        r'data-post="' + re.escape(channel) + r'/(\d+)"'
    )
    # Pattern for background-image style (Telegram uses this for post images)
    img_pattern = re.compile(
        r"background-image:\s*url\('(https://cdn[^']+)'\)"
    )
    # Also match <img> tags with src pointing to CDN
    img_tag_pattern = re.compile(
        r'<img[^>]+src="(https://cdn[^"]+)"'
    )

    # Split by message blocks
    blocks = html.split('tgme_widget_message_wrap')
    for block in blocks:
        msg_match = msg_pattern.search(block)
        if not msg_match:
            continue
        msg_id = int(msg_match.group(1))

        # Find image URL in this block
        img_match = img_pattern.search(block)
        if not img_match:
            img_match = img_tag_pattern.search(block)
        if img_match:
            results.append({
                "message_id": msg_id,
                "image_url": img_match.group(1),
            })

    return results


async def extract_news_from_images(
    images: list[dict],
) -> list[dict]:
    """Use GPT-4o-mini Vision to extract text/news from image posts.

    Processes images in batches to reduce API costs.

    Returns:
        List of dicts with 'message_id', 'image_url', 'extracted_text',
        'symbols', 'sentiment', 'sentiment_score'.
    """
    if not images or not settings.openai_api_key:
        return []

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    results = []

    # Process in batches
    for i in range(0, len(images), VISION_BATCH_SIZE):
        batch = images[i:i + VISION_BATCH_SIZE]
        try:
            batch_results = await _process_image_batch(client, batch)
            results.extend(batch_results)
        except Exception:
            logger.exception("Error processing image batch %d", i // VISION_BATCH_SIZE)

    logger.info("Extracted news from %d/%d images", len(results), len(images))
    return results


async def _process_image_batch(
    client: AsyncOpenAI,
    batch: list[dict],
) -> list[dict]:
    """Process a batch of images with a single GPT Vision call."""
    content_parts = [
        {
            "type": "text",
            "text": (
                "Extract the key financial news from these images. "
                "For each image, provide:\n"
                "1. A brief summary of the news (1-2 sentences)\n"
                "2. Stock symbols mentioned (NSE/BSE tickers)\n"
                "3. Sentiment: bullish/bearish/neutral\n"
                "4. Sentiment score: -1.0 to +1.0\n\n"
                "Respond as a JSON array with one object per image:\n"
                '[{"summary": "...", "symbols": ["SYM1"], "sentiment": "bullish", "score": 0.7}]\n'
                "If an image is not financial news, return null for that entry.\n"
                "Respond ONLY with valid JSON array. No markdown."
            ),
        },
    ]

    for img in batch:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": img["image_url"], "detail": "low"},
        })

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": content_parts}],
        temperature=0.1,
        max_tokens=800,
    )

    text = response.choices[0].message.content or "[]"
    # Clean markdown fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse Vision response: %s", cleaned[:200])
        return []

    if not isinstance(parsed, list):
        parsed = [parsed]

    results = []
    for idx, item in enumerate(parsed):
        if item is None or idx >= len(batch):
            continue
        img = batch[idx]
        results.append({
            "message_id": img.get("message_id"),
            "image_url": img.get("image_url"),
            "extracted_text": item.get("summary", ""),
            "symbols": ",".join(item.get("symbols", [])),
            "sentiment": item.get("sentiment", "neutral"),
            "sentiment_score": float(item.get("score", 0)),
        })

    return results


async def collect_telegram_news() -> list[dict]:
    """Full pipeline: scrape channel + extract news via Vision.

    Returns list of news items ready to be saved to DB.
    """
    images = await scrape_channel_images()
    if not images:
        return []

    news_items = await extract_news_from_images(images)
    today = datetime.now(_IST).strftime("%Y-%m-%d")

    for item in news_items:
        item["date"] = today
        item["source"] = settings.telegram_news_channel.lstrip("@") or "daytradertelugu"

    return news_items


async def save_news_to_db(news_items: list[dict]) -> int:
    """Save extracted news items to the database.

    Returns the number of items saved.
    """
    if not news_items:
        return 0

    from app.db.models import TelegramNewsRecord, AsyncSessionLocal

    saved = 0
    try:
        async with AsyncSessionLocal() as session:
            for item in news_items:
                record = TelegramNewsRecord(
                    date=item.get("date", ""),
                    message_id=item.get("message_id"),
                    image_url=item.get("image_url"),
                    extracted_text=item.get("extracted_text"),
                    symbols=item.get("symbols"),
                    sentiment=item.get("sentiment"),
                    sentiment_score=item.get("sentiment_score", 0),
                    source=item.get("source", "daytradertelugu"),
                )
                session.add(record)
                saved += 1
            await session.commit()
    except Exception:
        logger.exception("Error saving telegram news to DB")

    logger.info("Saved %d news items to DB", saved)
    return saved


async def get_recent_news(days: int = 1) -> list[dict]:
    """Fetch recent news from DB for the last N days."""
    from app.db.models import TelegramNewsRecord, AsyncSessionLocal
    from sqlalchemy import select
    from datetime import timedelta

    cutoff = (datetime.now(_IST) - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(TelegramNewsRecord)
                .where(TelegramNewsRecord.date >= cutoff)
                .order_by(TelegramNewsRecord.created_at.desc())
            )
            records = result.scalars().all()
            return [
                {
                    "id": r.id,
                    "date": r.date,
                    "message_id": r.message_id,
                    "extracted_text": r.extracted_text,
                    "symbols": r.symbols,
                    "sentiment": r.sentiment,
                    "sentiment_score": r.sentiment_score,
                    "source": r.source,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in records
            ]
    except Exception:
        logger.exception("Error fetching recent news")
        return []
