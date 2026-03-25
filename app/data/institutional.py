"""Institutional flow data — FII/DII daily activity from NSE.

SRS Module 3.2: Institutional Flow feature.
Fetches FII/DII buy/sell data and computes net flow signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import httpx
import pytz

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

NSE_FII_URLS = [
    "https://www.nseindia.com/api/fiidiiTradeReact",
    "https://www.nseindia.com/api/fiidiiActivity",
]
NSE_BASE = "https://www.nseindia.com"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class InstitutionalFlow:
    """Daily FII/DII activity summary."""

    date: str = ""
    fii_buy: float = 0.0  # in crores
    fii_sell: float = 0.0
    fii_net: float = 0.0
    dii_buy: float = 0.0
    dii_sell: float = 0.0
    dii_net: float = 0.0

    @property
    def net_institutional(self) -> float:
        """Combined FII + DII net flow."""
        return self.fii_net + self.dii_net

    @property
    def signal(self) -> str:
        """Institutional bias: strong_buy, buy, neutral, sell, strong_sell."""
        net = self.net_institutional
        if net > 2000:
            return "strong_buy"
        elif net > 500:
            return "buy"
        elif net < -2000:
            return "strong_sell"
        elif net < -500:
            return "sell"
        return "neutral"


async def fetch_fii_dii_data(max_retries: int = 3) -> Optional[InstitutionalFlow]:
    """Fetch today's FII/DII data from NSE website.

    NSE requires a session cookie, so we first hit the homepage.
    Retries up to max_retries times on failure (NSE API is unreliable).
    """
    import asyncio

    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                # Get session cookie
                await client.get(NSE_BASE, headers=_HEADERS)

                # Fetch FII/DII data (try multiple endpoints)
                data = None
                for url in NSE_FII_URLS:
                    resp = await client.get(url, headers=_HEADERS)
                    if resp.status_code == 200:
                        data = resp.json()
                        break
                    logger.warning("NSE FII/DII API %s returned %d", url, resp.status_code)
                if data is None:
                    if attempt < max_retries:
                        logger.warning("FII/DII attempt %d/%d failed — retrying in %ds", attempt, max_retries, attempt * 5)
                        await asyncio.sleep(attempt * 5)
                        continue
                    logger.warning("All NSE FII/DII endpoints failed after %d attempts", max_retries)
                    return None

                flow = InstitutionalFlow(
                    date=datetime.now(_IST).strftime("%Y-%m-%d"),
                )

                # Parse FII and DII entries
                for entry in data:
                    category = entry.get("category", "").upper()
                    buy_val = _parse_crore(entry.get("buyValue", "0"))
                    sell_val = _parse_crore(entry.get("sellValue", "0"))
                    net_val = _parse_crore(entry.get("netValue", "0"))

                    if "FII" in category or "FPI" in category:
                        flow.fii_buy = buy_val
                        flow.fii_sell = sell_val
                        flow.fii_net = net_val
                    elif "DII" in category:
                        flow.dii_buy = buy_val
                        flow.dii_sell = sell_val
                        flow.dii_net = net_val

                logger.info(
                    "FII/DII: FII_net=%.0f cr, DII_net=%.0f cr, Signal=%s",
                    flow.fii_net,
                    flow.dii_net,
                    flow.signal,
                )
                return flow

        except Exception:
            if attempt < max_retries:
                logger.warning("FII/DII attempt %d/%d error — retrying in %ds", attempt, max_retries, attempt * 5)
                await asyncio.sleep(attempt * 5)
            else:
                logger.exception("Error fetching FII/DII data after %d attempts", max_retries)

    return None


def _parse_crore(value: str) -> float:
    """Parse NSE value string (may have commas) to float."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0
