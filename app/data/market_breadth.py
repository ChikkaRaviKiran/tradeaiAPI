"""Market breadth analysis — Advance/Decline ratio and sector strength.

SRS Module 3.2: Market Breadth features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx
import pytz

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

NSE_MARKET_STATUS_URL = "https://www.nseindia.com/api/marketStatus"
NSE_SECTOR_URL = "https://www.nseindia.com/api/equity-stockIndices?index="
NSE_BASE = "https://www.nseindia.com"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

# NIFTY sector indices
SECTORS = [
    "NIFTY BANK",
    "NIFTY IT",
    "NIFTY PHARMA",
    "NIFTY AUTO",
    "NIFTY FMCG",
    "NIFTY METAL",
    "NIFTY REALTY",
    "NIFTY ENERGY",
    "NIFTY INFRA",
    "NIFTY MEDIA",
]


@dataclass
class SectorData:
    """Single sector performance."""

    name: str = ""
    change_pct: float = 0.0
    advancing: int = 0
    declining: int = 0


@dataclass
class MarketBreadth:
    """Market-wide breadth data."""

    total_advancing: int = 0
    total_declining: int = 0
    total_unchanged: int = 0
    advance_decline_ratio: float = 1.0
    sectors: list[SectorData] = field(default_factory=list)

    @property
    def breadth_signal(self) -> str:
        """Interpret breadth: strong_bullish, bullish, neutral, bearish, strong_bearish."""
        ratio = self.advance_decline_ratio
        if ratio > 2.0:
            return "strong_bullish"
        elif ratio > 1.3:
            return "bullish"
        elif ratio < 0.5:
            return "strong_bearish"
        elif ratio < 0.75:
            return "bearish"
        return "neutral"

    @property
    def strong_sectors(self) -> list[str]:
        """Sectors with > 1% positive change."""
        return [s.name for s in self.sectors if s.change_pct > 1.0]

    @property
    def weak_sectors(self) -> list[str]:
        """Sectors with > 1% negative change."""
        return [s.name for s in self.sectors if s.change_pct < -1.0]


async def fetch_market_breadth() -> Optional[MarketBreadth]:
    """Fetch market breadth from NSE.

    Gets advance/decline data and sector performance.
    """
    try:
        breadth = MarketBreadth()

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            # Session cookie
            await client.get(NSE_BASE, headers=_HEADERS)

            # Fetch NIFTY 50 index data for A/D
            resp = await client.get(
                f"{NSE_SECTOR_URL}NIFTY%2050", headers=_HEADERS
            )
            if resp.status_code == 200:
                data = resp.json()
                _parse_advance_decline(data, breadth)

            # Fetch sector indices
            for sector in SECTORS:
                try:
                    encoded = sector.replace(" ", "%20")
                    resp = await client.get(
                        f"{NSE_SECTOR_URL}{encoded}", headers=_HEADERS
                    )
                    if resp.status_code == 200:
                        sector_data = resp.json()
                        metadata = sector_data.get("metadata", {})
                        change = float(metadata.get("percentChange", 0))
                        breadth.sectors.append(
                            SectorData(name=sector, change_pct=change)
                        )
                except Exception:
                    continue

        logger.info(
            "Market breadth: A/D=%d/%d (%.2f), Strong=%s, Weak=%s",
            breadth.total_advancing,
            breadth.total_declining,
            breadth.advance_decline_ratio,
            breadth.strong_sectors[:3],
            breadth.weak_sectors[:3],
        )
        return breadth

    except Exception:
        logger.exception("Error fetching market breadth")
        return None


def _parse_advance_decline(data: dict, breadth: MarketBreadth) -> None:
    """Extract advance/decline counts from NSE index data."""
    advance = data.get("advance", {})
    if isinstance(advance, dict):
        breadth.total_advancing = int(advance.get("advances", 0))
        breadth.total_declining = int(advance.get("declines", 0))
        breadth.total_unchanged = int(advance.get("unchanged", 0))
    elif isinstance(data.get("data"), list):
        # Count from individual stock data
        for stock in data["data"]:
            change = float(stock.get("pChange", 0))
            if change > 0:
                breadth.total_advancing += 1
            elif change < 0:
                breadth.total_declining += 1
            else:
                breadth.total_unchanged += 1

    if breadth.total_declining > 0:
        breadth.advance_decline_ratio = round(
            breadth.total_advancing / breadth.total_declining, 2
        )
