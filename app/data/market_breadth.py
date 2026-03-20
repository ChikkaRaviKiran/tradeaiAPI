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
    # NIFTY 50 spot price data (extracted from index fetch)
    nifty_prev_close: Optional[float] = None
    nifty_last_price: Optional[float] = None
    nifty_day_high: Optional[float] = None
    nifty_day_low: Optional[float] = None

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
                        # Try metadata first, then look at the index row in data array
                        metadata = sector_data.get("metadata", {})
                        change = float(metadata.get("percentChange", 0))
                        if change == 0 and sector_data.get("data"):
                            # First entry in data array is the index itself
                            for row in sector_data["data"]:
                                if row.get("symbol") == sector or row.get("priority") == 0:
                                    change = float(row.get("pChange", 0))
                                    break
                        breadth.sectors.append(
                            SectorData(name=sector, change_pct=round(change, 2))
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
    """Extract advance/decline counts and NIFTY spot price from NSE index data."""
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

    # Extract NIFTY 50 spot price from the index row (first entry in data array)
    if isinstance(data.get("data"), list) and data["data"]:
        idx_row = data["data"][0]
        try:
            breadth.nifty_prev_close = float(idx_row.get("previousClose", 0)) or None
            breadth.nifty_last_price = float(idx_row.get("lastPrice", 0)) or None
            breadth.nifty_day_high = float(idx_row.get("dayHigh", 0)) or None
            breadth.nifty_day_low = float(idx_row.get("dayLow", 0)) or None
        except (ValueError, TypeError):
            pass
