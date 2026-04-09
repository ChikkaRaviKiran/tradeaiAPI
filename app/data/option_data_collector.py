"""Option candle data collector — downloads and caches 1-min option OHLCV to PostgreSQL.

Usage:
    # Daily collection (today's data for near-ATM strikes):
    python -m app.data.option_data_collector --today

    # Bulk historical download (last N days):
    python -m app.data.option_data_collector --bulk --days 60

    # Specific date range:
    python -m app.data.option_data_collector --from 2026-02-01 --to 2026-04-09
"""

from __future__ import annotations

import asyncio
import logging
import time as time_mod
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import pytz
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.instruments import get_enabled_instruments, InstrumentConfig
from app.data.angelone_client import AngelOneClient
from app.db.models import Base, OptionCandle

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

# Extra strikes beyond the day's high/low ATM to capture (covers OTM offset)
STRIKE_BUFFER = 3


class OptionDataCollector:
    """Downloads 1-min option candles from AngelOne and stores them in PostgreSQL."""

    def __init__(self) -> None:
        self.angel = AngelOneClient()
        self._engine = create_async_engine(settings.database_url, echo=False)
        self._session_factory = sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)

    async def init(self) -> None:
        """Ensure tables exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            try:
                await conn.execute(text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_option_candles_unique "
                    "ON option_candles (trading_symbol, timestamp)"
                ))
            except Exception:
                pass

    def _authenticate(self) -> None:
        self.angel.authenticate()
        logger.info("AngelOne authenticated for option data collection")

    def _get_trading_dates(self, start: date, end: date) -> list[date]:
        """Return list of trading dates (weekdays only, holidays skipped by data availability)."""
        dates = []
        current = start
        while current <= end:
            if current.weekday() < 5:  # Mon-Fri
                dates.append(current)
            current += timedelta(days=1)
        return dates

    def _get_day_range(self, instrument: InstrumentConfig, dt: date) -> Optional[tuple[float, float]]:
        """Get (day_low, day_high) for instrument from AngelOne spot candles."""
        date_str = dt.strftime("%Y-%m-%d")
        try:
            candles = self.angel.get_candle_data(
                instrument.token, instrument.exchange.value, "ONE_MINUTE",
                f"{date_str} 09:15", f"{date_str} 15:30",
            )
            time_mod.sleep(0.3)
            if not candles:
                return None
            df = self.angel.candles_to_dataframe(candles)
            if df.empty or len(df) < 10:
                return None
            return (float(df["low"].min()), float(df["high"].max()))
        except Exception as e:
            logger.warning("Failed to get day range for %s on %s: %s", instrument.symbol, dt, e)
            return None

    def _get_strikes_for_range(self, instrument: InstrumentConfig, day_low: float, day_high: float) -> list[float]:
        """Generate strikes covering the full day range + buffer.

        Computes ATM for day_low and day_high, then adds STRIKE_BUFFER
        extra strikes on each side to cover OTM entries at any intraday level.
        """
        interval = instrument.strike_interval
        atm_low = round(day_low / interval) * interval
        atm_high = round(day_high / interval) * interval
        lowest = atm_low - STRIKE_BUFFER * interval
        highest = atm_high + STRIKE_BUFFER * interval
        strikes = []
        s = lowest
        while s <= highest:
            strikes.append(s)
            s += interval
        return strikes

    @staticmethod
    def _compute_weekly_expiry(dt: date) -> str:
        """Compute the NIFTY weekly expiry (Thursday) for a given trade date.

        Returns expiry in DDMMMYY format (e.g. '13FEB26').
        Weekly options expire on Thursday. For any date, find the nearest
        Thursday on or after that date.
        """
        # Thursday = weekday 3
        days_ahead = (3 - dt.weekday()) % 7
        if days_ahead == 0:
            # dt is Thursday — expiry is today
            expiry_date = dt
        else:
            expiry_date = dt + timedelta(days=days_ahead)
        return expiry_date.strftime("%d%b%y").upper()

    async def _save_candles(
        self,
        session: AsyncSession,
        instrument: str,
        expiry: str,
        strike: float,
        option_type: str,
        trading_symbol: str,
        date_str: str,
        df: pd.DataFrame,
    ) -> int:
        """Insert candles into DB, skip duplicates."""
        if df.empty:
            return 0

        count = 0
        for ts, row in df.iterrows():
            ts_naive = ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts
            candle = OptionCandle(
                instrument=instrument,
                expiry=expiry,
                strike=strike,
                option_type=option_type,
                trading_symbol=trading_symbol,
                date=date_str,
                timestamp=ts_naive,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row.get("volume", 0)),
            )
            session.add(candle)
            count += 1

        try:
            await session.commit()
        except Exception:
            await session.rollback()
            # Likely duplicate key — insert one-by-one skipping duplicates
            count = 0
            for ts, row in df.iterrows():
                try:
                    ts_naive = ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts
                    async with self._session_factory() as s2:
                        s2.add(OptionCandle(
                            instrument=instrument,
                            expiry=expiry,
                            strike=strike,
                            option_type=option_type,
                            trading_symbol=trading_symbol,
                            date=date_str,
                            timestamp=ts_naive,
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                            volume=int(row.get("volume", 0)),
                        ))
                        await s2.commit()
                        count += 1
                except Exception:
                    pass  # Skip duplicate
        return count

    async def _already_cached(self, session: AsyncSession, trading_symbol: str, date_str: str) -> bool:
        """Check if we already have data for this symbol+date."""
        result = await session.execute(
            text("SELECT 1 FROM option_candles WHERE trading_symbol = :sym AND date = :dt LIMIT 1"),
            {"sym": trading_symbol, "dt": date_str},
        )
        return result.scalar() is not None

    async def collect_date(
        self,
        instrument: InstrumentConfig,
        dt: date,
        expiry_str: Optional[str] = None,
    ) -> dict:
        """Collect option candles for one instrument on one date.

        Uses AngelOne spot candles to get day high/low, then fetches option
        candles for all strikes covering that range + buffer.

        Returns: {"date": str, "contracts": int, "candles": int, "skipped": int}
        """
        date_str = dt.strftime("%Y-%m-%d")
        stats = {"date": date_str, "contracts": 0, "candles": 0, "skipped": 0}

        # Get day range from AngelOne spot candles
        day_range = self._get_day_range(instrument, dt)
        if day_range is None:
            logger.warning("[%s] No spot data for %s — skipping", instrument.symbol, date_str)
            return stats
        day_low, day_high = day_range

        # Get expiry — compute algorithmically (nearest Thursday)
        if expiry_str is None:
            expiry_str = self._compute_weekly_expiry(dt)

        strikes = self._get_strikes_for_range(instrument, day_low, day_high)
        logger.info(
            "[%s] %s | Low=%.0f High=%.0f | Expiry=%s | Strikes: %s to %s (%d)",
            instrument.symbol, date_str, day_low, day_high, expiry_str,
            int(min(strikes)), int(max(strikes)), len(strikes),
        )

        for strike in strikes:
            for opt_type in ("CE", "PE"):
                symbol = instrument.build_option_symbol(expiry_str, strike, opt_type)

                async with self._session_factory() as session:
                    if await self._already_cached(session, symbol, date_str):
                        stats["skipped"] += 1
                        continue

                # Fetch from AngelOne
                token_info = self.angel._search_symbol(symbol)
                if not token_info:
                    continue

                candles = self.angel.get_candle_data(
                    token_info["symboltoken"], "NFO", "ONE_MINUTE",
                    f"{date_str} 09:15", f"{date_str} 15:30",
                )
                time_mod.sleep(0.4)  # Rate limit

                if not candles:
                    continue

                df = self.angel.candles_to_dataframe(candles)

                async with self._session_factory() as session:
                    saved = await self._save_candles(
                        session, instrument.symbol, expiry_str,
                        strike, opt_type, symbol, date_str, df,
                    )
                    stats["contracts"] += 1
                    stats["candles"] += saved

        logger.info(
            "[%s] %s — Done: %d contracts, %d candles saved, %d skipped (cached)",
            instrument.symbol, date_str, stats["contracts"], stats["candles"], stats["skipped"],
        )
        return stats

    async def collect_today(self) -> list[dict]:
        """Collect today's option candles for all enabled instruments."""
        self._authenticate()
        await self.init()
        today = datetime.now(_IST).date()
        if today.weekday() >= 5:
            logger.info("Weekend — nothing to collect")
            return []

        results = []
        for instrument in get_enabled_instruments():
            stats = await self.collect_date(instrument, today)
            results.append(stats)
        return results

    async def collect_bulk(self, days: int = 60) -> list[dict]:
        """Bulk download historical option candles for the last N days."""
        self._authenticate()
        await self.init()
        end = datetime.now(_IST).date()
        start = end - timedelta(days=days)
        return await self.collect_range(start, end)

    async def collect_range(self, start: date, end: date) -> list[dict]:
        """Collect option candles for a date range."""
        self._authenticate()
        await self.init()
        trading_dates = self._get_trading_dates(start, end)
        logger.info("Collecting option data for %d trading days: %s to %s", len(trading_dates), start, end)

        results = []
        for instrument in get_enabled_instruments():
            for dt in trading_dates:
                try:
                    stats = await self.collect_date(instrument, dt)
                    results.append(stats)
                except Exception as e:
                    logger.error("[%s] %s — Error: %s", instrument.symbol, dt, e)
                    results.append({"date": str(dt), "contracts": 0, "candles": 0, "error": str(e)})
        return results

    async def get_cached_candles(
        self,
        instrument: str,
        date_str: str,
        strike: float,
        option_type: str,
    ) -> pd.DataFrame:
        """Retrieve cached option candles from DB — for use in backtests."""
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT timestamp, open, high, low, close, volume "
                    "FROM option_candles "
                    "WHERE instrument = :inst AND date = :dt "
                    "AND strike = :strike AND option_type = :ot "
                    "ORDER BY timestamp"
                ),
                {"inst": instrument, "dt": date_str, "strike": strike, "ot": option_type},
            )
            rows = result.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.index = df.index.tz_localize(_IST) if df.index.tz is None else df.index.tz_convert(_IST)
        return df

    async def get_stats(self) -> dict:
        """Get summary stats of cached data."""
        async with self._session_factory() as session:
            result = await session.execute(text(
                "SELECT instrument, COUNT(DISTINCT date) as days, "
                "COUNT(DISTINCT trading_symbol) as contracts, COUNT(*) as candles, "
                "MIN(date) as min_date, MAX(date) as max_date "
                "FROM option_candles GROUP BY instrument"
            ))
            rows = result.fetchall()
        return [
            {
                "instrument": r[0], "days": r[1], "contracts": r[2],
                "candles": r[3], "min_date": r[4], "max_date": r[5],
            }
            for r in rows
        ]

    async def cleanup(self) -> None:
        await self._engine.dispose()


# ═══════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════
async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Option Data Collector — cache 1-min option candles")
    parser.add_argument("--today", action="store_true", help="Collect today's data")
    parser.add_argument("--bulk", action="store_true", help="Bulk download last N days")
    parser.add_argument("--days", type=int, default=60, help="Number of days for bulk download (default: 60)")
    parser.add_argument("--from", dest="from_date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--stats", action="store_true", help="Show stats of cached data")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    collector = OptionDataCollector()

    try:
        if args.stats:
            await collector.init()
            stats = await collector.get_stats()
            if not stats:
                print("No cached option data yet.")
            else:
                print("\n=== Cached Option Data ===")
                for s in stats:
                    print(
                        f"  {s['instrument']}: {s['days']} days, {s['contracts']} contracts, "
                        f"{s['candles']:,} candles ({s['min_date']} to {s['max_date']})"
                    )

        elif args.today:
            results = await collector.collect_today()
            total_candles = sum(r.get("candles", 0) for r in results)
            print(f"\nToday: {total_candles:,} candles saved")

        elif args.from_date and args.to_date:
            start = datetime.strptime(args.from_date, "%Y-%m-%d").date()
            end = datetime.strptime(args.to_date, "%Y-%m-%d").date()
            results = await collector.collect_range(start, end)
            total_candles = sum(r.get("candles", 0) for r in results)
            print(f"\n{args.from_date} to {args.to_date}: {total_candles:,} candles saved")

        elif args.bulk:
            results = await collector.collect_bulk(days=args.days)
            total_candles = sum(r.get("candles", 0) for r in results)
            print(f"\nBulk ({args.days} days): {total_candles:,} candles saved")

        else:
            parser.print_help()

    finally:
        await collector.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
