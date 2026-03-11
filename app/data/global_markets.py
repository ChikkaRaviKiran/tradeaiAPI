"""Global market data fetcher with retry logic using httpx."""

from __future__ import annotations

import logging
from datetime import datetime, time as dtime
from typing import Optional

import httpx

from app.core.models import GlobalBias, GlobalIndex

logger = logging.getLogger(__name__)

# Yahoo Finance Chart API (unofficial) for global indices
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Comprehensive global index coverage
GLOBAL_SYMBOLS = {
    # US Markets
    "^DJI": "Dow Jones",
    "^IXIC": "Nasdaq",
    "^GSPC": "S&P 500",
    # Europe
    "^FTSE": "FTSE 100",
    "^GDAXI": "DAX",
    # Asia-Pacific
    "^N225": "Nikkei 225",
    "^HSI": "Hang Seng",
    "^STI": "SGX Straits Times",
    "^KS11": "KOSPI",
    # Volatility
    "^VIX": "CBOE VIX",
    # SGX Nifty proxy (Singapore-listed Nifty futures ETF)
    "^NSEI": "NIFTY 50",
}

# Max retries per symbol
_MAX_RETRIES = 2


async def _fetch_single_index(
    client: httpx.AsyncClient, symbol: str, name: str
) -> GlobalIndex:
    """Fetch a single index with retry logic."""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = await client.get(
                YAHOO_QUOTE_URL.format(symbol=symbol),
                params={"range": "1d", "interval": "1d"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if resp.status_code == 200:
                data = resp.json()
                result = data.get("chart", {}).get("result")
                if not result:
                    logger.warning("[Global] %s (%s): empty chart result", name, symbol)
                    break
                meta = result[0].get("meta", {})
                prev_close = meta.get("chartPreviousClose", meta.get("previousClose", 0))
                current = meta.get("regularMarketPrice", 0)
                if prev_close and prev_close > 0:
                    change_pct = ((current - prev_close) / prev_close) * 100
                else:
                    change_pct = 0.0
                return GlobalIndex(
                    symbol=name, change_pct=round(change_pct, 2), last_price=current
                )
            else:
                logger.warning(
                    "[Global] %s (%s): HTTP %d (attempt %d/%d)",
                    name, symbol, resp.status_code, attempt + 1, _MAX_RETRIES + 1,
                )
        except Exception as exc:
            logger.warning(
                "[Global] %s (%s): fetch error (attempt %d/%d): %s",
                name, symbol, attempt + 1, _MAX_RETRIES + 1, exc,
            )
    # All retries exhausted
    return GlobalIndex(symbol=name, change_pct=0.0, last_price=0.0)


async def fetch_global_indices() -> list[GlobalIndex]:
    """Fetch all global index data with retry logic."""
    indices: list[GlobalIndex] = []
    success_count = 0

    async with httpx.AsyncClient(timeout=10.0) as client:
        for symbol, name in GLOBAL_SYMBOLS.items():
            idx = await _fetch_single_index(client, symbol, name)
            indices.append(idx)
            if idx.last_price > 0:
                success_count += 1

    logger.info(
        "[Global] Fetched %d/%d indices | %s",
        success_count,
        len(GLOBAL_SYMBOLS),
        " | ".join(
            f"{i.symbol}: {i.change_pct:+.2f}%" for i in indices if i.last_price > 0
        ),
    )

    if success_count == 0:
        logger.error("[Global] ALL index fetches failed — data may be unavailable")

    return indices


def compute_global_bias(indices: list[GlobalIndex]) -> GlobalBias:
    """Determine global market bias.

    Rules (only counts indices with actual data):
        If ≥3 indices > +1% → bullish
        If ≥3 indices < −1% → bearish
        Else neutral
    """
    # Only consider indices that actually returned data
    valid = [idx for idx in indices if idx.last_price > 0]
    if not valid:
        logger.warning("[Global] No valid index data — returning UNAVAILABLE")
        return GlobalBias.UNAVAILABLE

    bullish_count = sum(1 for idx in valid if idx.change_pct > 1.0)
    bearish_count = sum(1 for idx in valid if idx.change_pct < -1.0)

    # VIX spike check — if VIX > 25, adds bearish weight
    vix = next((idx for idx in valid if "VIX" in idx.symbol), None)
    if vix and vix.last_price > 25:
        bearish_count += 1
        logger.info("[Global] VIX elevated at %.1f — adding bearish weight", vix.last_price)

    if bullish_count >= 3:
        return GlobalBias.BULLISH
    if bearish_count >= 3:
        return GlobalBias.BEARISH
    return GlobalBias.NEUTRAL
