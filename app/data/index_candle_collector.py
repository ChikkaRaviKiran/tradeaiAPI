"""Index candle data collector — downloads and caches 1-min index OHLCV to PostgreSQL.

Usage:
    # Daily collection (today's data):
    python -m app.data.index_candle_collector --today

    # Bulk historical download (last N days):
    python -m app.data.index_candle_collector --bulk --days 60

    # Specific date range:
    python -m app.data.index_candle_collector --from 2026-02-01 --to 2026-04-09
"""

from __future__ import annotations

import asyncio
import logging
import time as time_mod
from datetime import date, datetime, timedelta
from typing import Optional

import pytz
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.instruments import get_enabled_instruments, InstrumentConfig
from app.data.angelone_client import AngelOneClient
from app.db.models import Base, IndexCandle

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")


class IndexCandleCollector:
    """Downloads 1-min index candles from AngelOne and stores in PostgreSQL."""

    def __init__(self) -> None:
        self.angel = AngelOneClient()
        self._engine = create_async_engine(settings.database_url, echo=False)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            try:
                await conn.execute(text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_index_candles_unique "
                    "ON index_candles (instrument, timestamp)"
                ))
            except Exception:
                pass

    def _authenticate(self) -> None:
        self.angel.authenticate()
        logger.info("AngelOne authenticated for index candle collection")

    def _get_trading_dates(self, start: date, end: date) -> list[date]:
        dates = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                dates.append(current)
            current += timedelta(days=1)
        return dates

    async def _already_cached(self, session: AsyncSession, instrument: str, date_str: str) -> bool:
        result = await session.execute(
            text("SELECT 1 FROM index_candles WHERE instrument = :inst AND date = :dt LIMIT 1"),
            {"inst": instrument, "dt": date_str},
        )
        return result.scalar() is not None

    async def collect_date(
        self,
        instrument: InstrumentConfig,
        dt: date,
    ) -> dict:
        """Collect 1-min index candles for one instrument on one date."""
        date_str = dt.strftime("%Y-%m-%d")
        stats = {"date": date_str, "instrument": instrument.symbol, "candles": 0, "skipped": False}

        async with self._session_factory() as session:
            if await self._already_cached(session, instrument.symbol, date_str):
                stats["skipped"] = True
                return stats

        candles = self.angel.get_candle_data(
            instrument.token, instrument.exchange.value, "ONE_MINUTE",
            f"{date_str} 09:15", f"{date_str} 15:30",
        )
        time_mod.sleep(0.3)

        if not candles:
            logger.warning("[%s] No candles for %s", instrument.symbol, date_str)
            return stats

        async with self._session_factory() as session:
            for c in candles:
                ts_naive = c.timestamp.replace(tzinfo=None) if c.timestamp.tzinfo else c.timestamp
                session.add(IndexCandle(
                    instrument=instrument.symbol,
                    date=date_str,
                    timestamp=ts_naive,
                    open=c.open,
                    high=c.high,
                    low=c.low,
                    close=c.close,
                    volume=c.volume,
                ))
            try:
                await session.commit()
                stats["candles"] = len(candles)
            except Exception:
                await session.rollback()
                # Insert one-by-one skipping duplicates
                count = 0
                for c in candles:
                    try:
                        async with self._session_factory() as s2:
                            ts_naive = c.timestamp.replace(tzinfo=None) if c.timestamp.tzinfo else c.timestamp
                            s2.add(IndexCandle(
                                instrument=instrument.symbol,
                                date=date_str,
                                timestamp=ts_naive,
                                open=c.open, high=c.high,
                                low=c.low, close=c.close,
                                volume=c.volume,
                            ))
                            await s2.commit()
                            count += 1
                    except Exception:
                        pass
                stats["candles"] = count

        logger.info("[%s] %s — saved %d candles", instrument.symbol, date_str, stats["candles"])
        return stats

    async def collect_bulk(self, days: int = 60, start: Optional[date] = None, end: Optional[date] = None) -> None:
        """Bulk collect index candles for all enabled instruments."""
        await self.init()
        self._authenticate()

        if end is None:
            end = datetime.now(_IST).date()
        if start is None:
            start = end - timedelta(days=days)

        trading_dates = self._get_trading_dates(start, end)
        instruments = get_enabled_instruments()
        total_candles = 0
        skipped = 0

        logger.info("Bulk collecting index candles: %d dates × %d instruments",
                     len(trading_dates), len(instruments))

        for instrument in instruments:
            for dt in trading_dates:
                stats = await self.collect_date(instrument, dt)
                total_candles += stats["candles"]
                if stats["skipped"]:
                    skipped += 1

        logger.info("Bulk collection done: %d candles saved, %d dates skipped", total_candles, skipped)

    async def collect_today(self) -> None:
        """Collect today's index candles for all enabled instruments."""
        await self.init()
        self._authenticate()

        today = datetime.now(_IST).date()
        if today.weekday() >= 5:
            logger.info("Weekend — skipping index candle collection")
            return

        instruments = get_enabled_instruments()
        for instrument in instruments:
            await self.collect_date(instrument, today)

    async def get_candles(
        self,
        instrument: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch candles from DB for a date range. Returns list of dicts."""
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT timestamp, open, high, low, close, volume "
                    "FROM index_candles "
                    "WHERE instrument = :inst AND date >= :start AND date <= :end "
                    "ORDER BY timestamp"
                ),
                {"inst": instrument, "start": start_date, "end": end_date},
            )
            rows = result.fetchall()
            return [
                {"timestamp": r[0], "open": r[1], "high": r[2],
                 "low": r[3], "close": r[4], "volume": r[5]}
                for r in rows
            ]


async def _main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Index candle collector")
    parser.add_argument("--today", action="store_true", help="Collect today's candles")
    parser.add_argument("--bulk", action="store_true", help="Bulk historical download")
    parser.add_argument("--days", type=int, default=60, help="Days to look back for bulk")
    parser.add_argument("--from", dest="from_date", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", type=str, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    collector = IndexCandleCollector()

    if args.today:
        await collector.collect_today()
    elif args.bulk:
        start = date.fromisoformat(args.from_date) if args.from_date else None
        end = date.fromisoformat(args.to_date) if args.to_date else None
        await collector.collect_bulk(days=args.days, start=start, end=end)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(_main())
