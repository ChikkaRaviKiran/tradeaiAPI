"""DhanHQ expired options data fetcher — downloads 1-min option candles to PostgreSQL.

Fetches from DhanHQ rollingoption API and stores in the option_candles table,
compatible with the existing backtest/strategy_evaluator pipeline.

Usage:
    # Initial 90-day backfill (run once):
    python -m app.data.dhan_option_fetcher --batch --days 90

    # Fetch today's data only:
    python -m app.data.dhan_option_fetcher --today

    # Continue historical backfill (one 30-day chunk per instrument):
    python -m app.data.dhan_option_fetcher --backfill-chunk

    # Full 5-year backfill (runs in chunks, respects rate limits):
    python -m app.data.dhan_option_fetcher --batch --days 1825

Rate limits: 10/sec, 250/min, 1000/hr, 7000/day
Each API call returns up to 30 days of 1-min candles for one (instrument, strike, optionType).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time as time_mod
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pytz
import requests
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.holidays import is_market_holiday
from app.db.models import Base

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

# ── DhanHQ API config ─────────────────────────────────────────────────────────

DHAN_BASE_URL = "https://api.dhan.co/v2"
MAX_DAYS_PER_CALL = 30
RATE_LIMIT_DELAY = 0.15  # 150ms between calls (safe for 10/sec)

# Progress file — tracks what date ranges have been fetched
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_PROGRESS_FILE = _DATA_DIR / "dhan_fetch_progress.json"

# Instrument configs for DhanHQ rollingoption API
DHAN_INSTRUMENTS = {
    "NIFTY": {
        "securityId": 13,
        "exchangeSegment": "NSE_FNO",
        "instrument": "OPTIDX",
        "expiryFlag": "WEEK",
        "maxStrikes": 5,     # ATM ±5 (11 positions)
        "expiryWeekday": 3,  # Thursday
        "strikeInterval": 50,
    },
    "SENSEX": {
        "securityId": 51,
        "exchangeSegment": "BSE_FNO",
        "instrument": "OPTIDX",
        "expiryFlag": "WEEK",
        "maxStrikes": 3,     # ATM ±3 (7 positions)
        "expiryWeekday": 4,  # Friday
        "strikeInterval": 100,
    },
}

# Relative strike labels to fetch
def _strike_labels(max_offset: int) -> list[str]:
    labels = ["ATM"]
    for i in range(1, max_offset + 1):
        labels.append(f"ATM+{i}")
        labels.append(f"ATM-{i}")
    return labels


class DhanOptionFetcher:
    """Fetches expired option candles from DhanHQ and stores in option_candles DB."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token or getattr(settings, "dhan_access_token", "")
        if not self.token:
            raise ValueError("DhanHQ access token not configured. Set DHAN_ACCESS_TOKEN in .env")
        self.headers = {
            "Content-Type": "application/json",
            "access-token": self.token,
        }
        self._engine = create_async_engine(settings.database_url, echo=False)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        self._api_calls_today = 0
        self._api_calls_minute = 0
        self._minute_start = time_mod.time()
        self._progress = self._load_progress()

    # ── Progress tracking ─────────────────────────────────────────────────

    def _load_progress(self) -> dict:
        if _PROGRESS_FILE.exists():
            try:
                return json.loads(_PROGRESS_FILE.read_text())
            except Exception:
                pass
        return {}

    def _save_progress(self) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _PROGRESS_FILE.write_text(json.dumps(self._progress, indent=2))

    def _mark_fetched(self, instrument: str, from_date: str, to_date: str) -> None:
        if instrument not in self._progress:
            self._progress[instrument] = {"windows": [], "latest": None, "backfill_frontier": None}
        entry = self._progress[instrument]
        entry["windows"].append({"from": from_date, "to": to_date, "ts": datetime.now().isoformat()})
        # Update boundaries
        if not entry["latest"] or to_date > entry["latest"]:
            entry["latest"] = to_date
        if not entry["backfill_frontier"] or from_date < entry["backfill_frontier"]:
            entry["backfill_frontier"] = from_date
        self._save_progress()

    def _is_window_fetched(self, instrument: str, from_date: str, to_date: str) -> bool:
        entry = self._progress.get(instrument, {})
        frontier = entry.get("backfill_frontier")
        latest = entry.get("latest")
        if frontier and latest and from_date >= frontier and to_date <= latest:
            return True
        return False

    # ── Rate limiting ─────────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        # Per-second control
        time_mod.sleep(RATE_LIMIT_DELAY)
        # Per-minute control
        now = time_mod.time()
        if now - self._minute_start > 60:
            self._api_calls_minute = 0
            self._minute_start = now
        self._api_calls_minute += 1
        if self._api_calls_minute >= 240:  # Stay under 250/min
            wait = 60 - (now - self._minute_start)
            if wait > 0:
                logger.info("Rate limit: waiting %.0fs for minute window", wait)
                time_mod.sleep(wait)
            self._api_calls_minute = 0
            self._minute_start = time_mod.time()
        self._api_calls_today += 1

    # ── DhanHQ API call ───────────────────────────────────────────────────

    def _fetch_rolling_option(
        self,
        instrument_key: str,
        strike_label: str,
        option_type: str,  # "CALL" or "PUT"
        from_date: str,
        to_date: str,
    ) -> dict | None:
        cfg = DHAN_INSTRUMENTS[instrument_key]
        payload = {
            "exchangeSegment": cfg["exchangeSegment"],
            "interval": "1",
            "securityId": cfg["securityId"],
            "instrument": cfg["instrument"],
            "expiryFlag": cfg["expiryFlag"],
            "expiryCode": 1,  # NOTE: DhanHQ treats 0 as missing, use 1 (near expiry)
            "strike": strike_label,
            "drvOptionType": option_type,
            "requiredData": ["open", "high", "low", "close", "volume", "strike", "spot"],
            "fromDate": from_date,
            "toDate": to_date,
        }
        self._rate_limit()
        try:
            resp = requests.post(
                f"{DHAN_BASE_URL}/charts/rollingoption",
                headers=self.headers,
                json=payload,
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                error = resp.json() if resp.text else {}
                logger.warning(
                    "DhanHQ %s %s %s %s→%s: HTTP %d - %s",
                    instrument_key, strike_label, option_type, from_date, to_date,
                    resp.status_code, error.get("errorMessage", resp.text[:100]),
                )
                return None
        except Exception as e:
            logger.error("DhanHQ API error: %s", e)
            return None

    # ── Parse & store ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_weekly_expiry(dt: date, expiry_weekday: int) -> date:
        days_ahead = (expiry_weekday - dt.weekday()) % 7
        if days_ahead == 0:
            return dt
        return dt + timedelta(days=days_ahead)

    def _parse_candles(
        self,
        instrument_key: str,
        option_type_code: str,  # "CE" or "PE"
        data: dict,
    ) -> list[dict]:
        """Parse DhanHQ response into list of candle dicts for DB insertion."""
        cfg = DHAN_INSTRUMENTS[instrument_key]
        # Response key: "ce" for CALL, "pe" for PUT
        side_key = "ce" if option_type_code == "CE" else "pe"
        side_data = (data.get("data", {}).get(side_key)) or {}

        timestamps = side_data.get("timestamp", [])
        opens = side_data.get("open", [])
        highs = side_data.get("high", [])
        lows = side_data.get("low", [])
        closes = side_data.get("close", [])
        volumes = side_data.get("volume", [])
        strikes = side_data.get("strike", [])

        if not timestamps:
            return []

        candles = []
        for i in range(len(timestamps)):
            ts = datetime.fromtimestamp(timestamps[i])
            actual_strike = strikes[i] if i < len(strikes) and strikes else 0
            if actual_strike == 0:
                continue  # Skip if no strike data
            date_str = ts.strftime("%Y-%m-%d")
            expiry_date = self._compute_weekly_expiry(ts.date(), cfg["expiryWeekday"])
            expiry_str = expiry_date.strftime("%d%b%y").upper()
            trading_symbol = f"{instrument_key}{int(actual_strike)}{option_type_code}"

            candles.append({
                "instrument": instrument_key,
                "expiry": expiry_str,
                "strike": actual_strike,
                "option_type": option_type_code,
                "trading_symbol": trading_symbol,
                "date": date_str,
                "timestamp": ts,
                "open": opens[i] if i < len(opens) else 0,
                "high": highs[i] if i < len(highs) else 0,
                "low": lows[i] if i < len(lows) else 0,
                "close": closes[i] if i < len(closes) else 0,
                "volume": volumes[i] if i < len(volumes) else 0,
            })
        return candles

    async def _save_candles(self, candles: list[dict]) -> int:
        """Batch insert candles into DB. Returns count of new rows inserted."""
        if not candles:
            return 0

        async with self._session_factory() as session:
            # Get count before insert
            before = await session.execute(text("SELECT count(*) FROM option_candles"))
            count_before = before.scalar()

            # Use INSERT ... ON CONFLICT DO NOTHING for idempotency
            insert_sql = text("""
                INSERT INTO option_candles
                    (instrument, expiry, strike, option_type, trading_symbol,
                     date, timestamp, open, high, low, close, volume)
                VALUES
                    (:instrument, :expiry, :strike, :option_type, :trading_symbol,
                     :date, :timestamp, :open, :high, :low, :close, :volume)
                ON CONFLICT (trading_symbol, timestamp) DO NOTHING
            """)
            # Batch in chunks of 1000
            chunk_size = 1000
            for i in range(0, len(candles), chunk_size):
                chunk = candles[i : i + chunk_size]
                try:
                    await session.execute(insert_sql, chunk)
                except Exception as e:
                    logger.warning("Batch insert error (trying row-by-row): %s", e)
                    await session.rollback()
                    for c in chunk:
                        try:
                            await session.execute(insert_sql, c)
                        except Exception:
                            pass
                    await session.commit()
                    continue
            await session.commit()

            # Get count after insert to compute actual new rows
            after = await session.execute(text("SELECT count(*) FROM option_candles"))
            count_after = after.scalar()
        return count_after - count_before

    # ── Public methods ────────────────────────────────────────────────────

    async def init_db(self) -> None:
        """Ensure table and unique index exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            try:
                await conn.execute(text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_option_candles_unique "
                    "ON option_candles (trading_symbol, timestamp)"
                ))
            except Exception:
                pass

    async def fetch_window(
        self,
        instrument_key: str,
        from_date: str,
        to_date: str,
    ) -> dict:
        """Fetch all strikes/types for one instrument and one date window.

        Returns summary dict with counts.
        """
        cfg = DHAN_INSTRUMENTS[instrument_key]
        strike_labels = _strike_labels(cfg["maxStrikes"])
        total_candles = 0
        total_inserted = 0
        total_calls = 0

        for option_type, ot_code in [("CALL", "CE"), ("PUT", "PE")]:
            for strike_label in strike_labels:
                data = self._fetch_rolling_option(
                    instrument_key, strike_label, option_type, from_date, to_date
                )
                total_calls += 1
                if not data:
                    continue
                candles = self._parse_candles(instrument_key, ot_code, data)
                if candles:
                    inserted = await self._save_candles(candles)
                    total_candles += len(candles)
                    total_inserted += inserted
                    logger.debug(
                        "  %s %s %s: %d candles, %d new",
                        instrument_key, strike_label, ot_code, len(candles), inserted,
                    )

        return {
            "instrument": instrument_key,
            "from": from_date,
            "to": to_date,
            "api_calls": total_calls,
            "candles_parsed": total_candles,
            "candles_inserted": total_inserted,
        }

    async def fetch_batch(self, days: int = 90, instruments: list[str] | None = None) -> list[dict]:
        """Fetch historical data for the last N days. Main entry for initial backfill.

        Splits into 30-day windows and fetches all instruments, strikes, types.
        """
        await self.init_db()
        instruments = instruments or list(DHAN_INSTRUMENTS.keys())
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        results = []
        for inst in instruments:
            if inst not in DHAN_INSTRUMENTS:
                logger.warning("Unknown instrument %s, skipping", inst)
                continue

            # Split into 30-day windows
            window_start = start_date
            while window_start < end_date:
                window_end = min(window_start + timedelta(days=MAX_DAYS_PER_CALL), end_date)
                from_str = window_start.strftime("%Y-%m-%d")
                to_str = window_end.strftime("%Y-%m-%d")

                if self._is_window_fetched(inst, from_str, to_str):
                    logger.info("Skipping %s %s→%s (already fetched)", inst, from_str, to_str)
                    window_start = window_end
                    continue

                logger.info("Fetching %s %s → %s ...", inst, from_str, to_str)
                result = await self.fetch_window(inst, from_str, to_str)
                results.append(result)
                self._mark_fetched(inst, from_str, to_str)
                logger.info(
                    "  %s: %d candles parsed, %d inserted (%d API calls)",
                    inst, result["candles_parsed"], result["candles_inserted"], result["api_calls"],
                )
                window_start = window_end

        total_calls = sum(r["api_calls"] for r in results)
        total_inserted = sum(r["candles_inserted"] for r in results)
        logger.info(
            "Batch complete: %d total API calls, %d total candles inserted",
            total_calls, total_inserted,
        )
        return results

    async def fetch_today(self, instruments: list[str] | None = None) -> list[dict]:
        """Fetch today's data for all instruments."""
        await self.init_db()
        instruments = instruments or list(DHAN_INSTRUMENTS.keys())
        today = date.today().strftime("%Y-%m-%d")
        # Fetch today + 1 day (toDate is non-inclusive in DhanHQ)
        tomorrow = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

        results = []
        for inst in instruments:
            if inst not in DHAN_INSTRUMENTS:
                continue
            logger.info("Fetching today's data for %s ...", inst)
            result = await self.fetch_window(inst, today, tomorrow)
            results.append(result)
            logger.info(
                "  %s today: %d candles, %d new",
                inst, result["candles_parsed"], result["candles_inserted"],
            )
        return results

    async def fetch_backfill_chunk(self, instruments: list[str] | None = None) -> list[dict]:
        """Fetch one 30-day historical chunk going further back. Called daily."""
        await self.init_db()
        instruments = instruments or list(DHAN_INSTRUMENTS.keys())
        results = []

        for inst in instruments:
            if inst not in DHAN_INSTRUMENTS:
                continue
            entry = self._progress.get(inst, {})
            frontier = entry.get("backfill_frontier")
            if frontier:
                frontier_date = datetime.strptime(frontier, "%Y-%m-%d").date()
            else:
                # No previous fetch — start from 90 days ago
                frontier_date = date.today() - timedelta(days=90)

            # Go back 30 more days
            window_end = frontier_date
            window_start = window_end - timedelta(days=MAX_DAYS_PER_CALL)

            # Don't go beyond 5 years
            min_date = date.today() - timedelta(days=5 * 365)
            if window_start < min_date:
                window_start = min_date
            if window_start >= window_end:
                logger.info("%s: backfill complete (reached %s)", inst, min_date)
                continue

            from_str = window_start.strftime("%Y-%m-%d")
            to_str = window_end.strftime("%Y-%m-%d")

            logger.info("Backfill %s: %s → %s", inst, from_str, to_str)
            result = await self.fetch_window(inst, from_str, to_str)
            results.append(result)
            self._mark_fetched(inst, from_str, to_str)
            logger.info(
                "  %s backfill: %d candles, %d new (%d API calls)",
                inst, result["candles_parsed"], result["candles_inserted"], result["api_calls"],
            )
        return results

    async def cleanup(self) -> None:
        await self._engine.dispose()


# ── CLI ───────────────────────────────────────────────────────────────────────

async def _main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="DhanHQ expired options data fetcher")
    parser.add_argument("--batch", action="store_true", help="Batch fetch historical data")
    parser.add_argument("--days", type=int, default=90, help="Days to fetch (default: 90)")
    parser.add_argument("--today", action="store_true", help="Fetch today's data only")
    parser.add_argument("--backfill-chunk", action="store_true", help="Fetch one 30-day backfill chunk")
    parser.add_argument("--token", type=str, default=None, help="DhanHQ access token (overrides env)")
    parser.add_argument("--instruments", type=str, default=None, help="Comma-separated instruments (e.g. NIFTY,SENSEX)")
    args = parser.parse_args()

    instruments = args.instruments.split(",") if args.instruments else None

    fetcher = DhanOptionFetcher(token=args.token)

    try:
        if args.today:
            results = await fetcher.fetch_today(instruments)
        elif args.backfill_chunk:
            results = await fetcher.fetch_backfill_chunk(instruments)
        elif args.batch:
            results = await fetcher.fetch_batch(days=args.days, instruments=instruments)
        else:
            parser.print_help()
            return

        # Summary
        total_calls = sum(r["api_calls"] for r in results)
        total_inserted = sum(r["candles_inserted"] for r in results)
        total_parsed = sum(r["candles_parsed"] for r in results)
        print(f"\n{'='*60}")
        print(f"DONE: {total_calls} API calls | {total_parsed} candles parsed | {total_inserted} new rows inserted")
        for r in results:
            print(f"  {r['instrument']} {r['from']}→{r['to']}: {r['candles_inserted']}/{r['candles_parsed']} new/total")
        print(f"{'='*60}")
    finally:
        await fetcher.cleanup()


if __name__ == "__main__":
    asyncio.run(_main())
