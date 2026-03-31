"""NSE market holiday calendar — fetched dynamically from NSE India API.

Fetches the official holiday list from https://www.nseindia.com once per year
and caches it locally. Falls back to a hardcoded list only if the API is unreachable.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

NSE_HOLIDAY_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
NSE_BASE_URL = "https://www.nseindia.com"
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_CACHE_FILE = _CACHE_DIR / "nse_holidays.json"

_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

# In-memory cache: { year: set[date] }
_holiday_cache: dict[int, set[date]] = {}


def _fetch_from_nse() -> list[dict]:
    """Fetch holiday list from NSE API (needs session cookie)."""
    with httpx.Client(timeout=15, follow_redirects=True) as client:
        # NSE requires a session cookie — hit homepage first
        client.get(NSE_BASE_URL, headers=_NSE_HEADERS)
        resp = client.get(NSE_HOLIDAY_URL, headers=_NSE_HEADERS)
        resp.raise_for_status()
        data = resp.json()
        # CM = Capital Market segment (equities + F&O)
        return data.get("CM", [])


def _parse_holidays(raw: list[dict]) -> set[date]:
    """Parse NSE holiday JSON into a set of dates."""
    holidays: set[date] = set()
    for entry in raw:
        try:
            dt = datetime.strptime(entry["tradingDate"], "%d-%b-%Y").date()
            holidays.add(dt)
        except (KeyError, ValueError) as exc:
            logger.debug("Skipping invalid holiday entry: %s (%s)", entry, exc)
    return holidays


def _save_cache(year: int, raw: list[dict]) -> None:
    """Persist fetched holidays to a local JSON file."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Load existing cache file if present
        all_data = {}
        if _CACHE_FILE.exists():
            all_data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        all_data[str(year)] = raw
        _CACHE_FILE.write_text(
            json.dumps(all_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        logger.debug("Could not save holiday cache", exc_info=True)


def _load_cache(year: int) -> Optional[list[dict]]:
    """Load holidays from local cache file."""
    try:
        if _CACHE_FILE.exists():
            all_data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            return all_data.get(str(year))
    except Exception:
        logger.debug("Could not read holiday cache", exc_info=True)
    return None


def _get_holidays_for_year(year: int) -> set[date]:
    """Get NSE holidays: merge API/cache data WITH hardcoded fallback.

    Always includes hardcoded holidays so known dates are never missed
    even if the NSE API returns incomplete data.
    """
    if year in _holiday_cache:
        return _holiday_cache[year]

    # Start with hardcoded holidays (guaranteed baseline)
    holidays = set(_FALLBACK_HOLIDAYS.get(year, set()))

    # 1. Try fetching from NSE — merge into baseline
    try:
        raw = _fetch_from_nse()
        api_holidays = {d for d in _parse_holidays(raw) if d.year == year}
        if api_holidays:
            holidays |= api_holidays
            logger.info(
                "[Holidays] Merged %d API + %d hardcoded holidays for %d (total %d)",
                len(api_holidays), len(_FALLBACK_HOLIDAYS.get(year, set())),
                year, len(holidays),
            )
            _save_cache(year, [e for e in raw if str(year) in e.get("tradingDate", "")])
            _holiday_cache[year] = holidays
            return holidays
    except Exception:
        logger.warning("[Holidays] Could not fetch from NSE API — trying cache")

    # 2. Try local disk cache — merge into baseline
    cached = _load_cache(year)
    if cached:
        cache_holidays = _parse_holidays(cached)
        if cache_holidays:
            holidays |= cache_holidays
            logger.info(
                "[Holidays] Merged %d cached + %d hardcoded holidays for %d (total %d)",
                len(cache_holidays), len(_FALLBACK_HOLIDAYS.get(year, set())),
                year, len(holidays),
            )
            _holiday_cache[year] = holidays
            return holidays

    # 3. Only hardcoded holidays available
    if holidays:
        logger.info("[Holidays] Using %d hardcoded holidays for %d", len(holidays), year)
    else:
        logger.warning("[Holidays] No holiday data available for %d", year)
    _holiday_cache[year] = holidays
    return holidays


# Hardcoded fallback — only used if both NSE API and cache fail
_FALLBACK_HOLIDAYS: dict[int, set[date]] = {
    2026: {
        date(2026, 1, 15),   # Municipal Corporation Election
        date(2026, 1, 26),   # Republic Day
        date(2026, 2, 15),   # Mahashivratri
        date(2026, 3, 3),    # Holi
        date(2026, 3, 21),   # Id-Ul-Fitr
        date(2026, 3, 26),   # Ram Navami
        date(2026, 3, 31),   # Mahavir Jayanti
        date(2026, 4, 3),    # Good Friday
        date(2026, 4, 14),   # Dr. Ambedkar Jayanti
        date(2026, 5, 1),    # Maharashtra Day
        date(2026, 5, 28),   # Bakri Id
        date(2026, 6, 26),   # Muharram
        date(2026, 8, 15),   # Independence Day
        date(2026, 9, 14),   # Ganesh Chaturthi
        date(2026, 10, 2),   # Mahatma Gandhi Jayanti
        date(2026, 10, 20),  # Dussehra
        date(2026, 11, 8),   # Diwali — Laxmi Pujan
        date(2026, 11, 10),  # Diwali — Balipratipada
        date(2026, 11, 24),  # Guru Nanak Jayanti
        date(2026, 12, 25),  # Christmas
    },
}


def is_market_holiday(d: date) -> bool:
    """Check if a given date is an NSE holiday or weekend."""
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return True
    return d in _get_holidays_for_year(d.year)


def next_trading_date(d: date) -> date:
    """Get the next trading date after the given date (skips weekends + holidays)."""
    candidate = d + timedelta(days=1)
    while is_market_holiday(candidate):
        candidate += timedelta(days=1)
    return candidate
