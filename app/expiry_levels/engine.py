"""Computes weekly + monthly Classic-pivot (3 resistance + 3 support)
expiry-planning levels for NIFTY/SENSEX.

Week definition: Wednesday -> following Tuesday, matching NIFTY/SENSEX's
weekly options expiry cycle (NOT the calendar Mon-Fri week).
Month definition: prior COMPLETED calendar month.

Deliberately Classic pivots ONLY (not Camarilla): Classic is literally
defined as 3 resistance + 3 support levels - the standard swing-level
formula for weekly/monthly planning. Camarilla is an intraday
mean-reversion tool (4 levels/side around the day's own close) and isn't
the right lens here, so it's intentionally left out of this feature to
avoid the "which one do I follow" ambiguity of mixing both.

Reads from TradeAI's own `index_candles` table (same source `level_zones`
and `condor_setup` already use) via `load_daily_ohlc` - no live broker call
from this background job.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import select

from app.condor_setup.levels import classic_pivots
from app.expiry_levels.db_models import ExpiryLevelSnapshot
from app.level_zones.timeframe_levels import load_daily_ohlc

logger = logging.getLogger(__name__)


def prior_week_bounds(today: date, weeks_back: int = 1) -> tuple[date, date]:
    """Wed-Tue bounds of the week `weeks_back` weeks before the current one."""
    days_since_wed = (today.weekday() - 2) % 7  # Mon=0..Sun=6, Wed=2
    this_week_wed = today - timedelta(days=days_since_wed)
    start = this_week_wed - timedelta(days=7 * weeks_back)
    end = start + timedelta(days=6)  # Wed + 6 days = following Tue
    return start, end


def prior_month_bounds(today: date, months_back: int = 1) -> tuple[date, date]:
    """First/last calendar day of the month `months_back` months before this one."""
    y, m = today.year, today.month
    for _ in range(months_back):
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    start = date(y, m, 1)
    end = date(y, 12, 31) if m == 12 else date(y, m + 1, 1) - timedelta(days=1)
    return start, end


async def compute_and_save_snapshot(
    session, symbol: str, timeframe: str, as_of: Optional[date] = None
) -> ExpiryLevelSnapshot:
    as_of = as_of or date.today()
    daily = await load_daily_ohlc(session, symbol)

    if timeframe == "weekly":
        start, end = prior_week_bounds(as_of)
    elif timeframe == "monthly":
        start, end = prior_month_bounds(as_of)
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    row = ExpiryLevelSnapshot(
        symbol=symbol, timeframe=timeframe,
        period_start=start.isoformat(), period_end=end.isoformat(),
    )

    if daily.empty:
        row.status, row.notes = "no_data", "No daily candle history available yet."
        session.add(row)
        await session.commit()
        return row

    window = daily[(daily.index.date >= start) & (daily.index.date <= end)]
    if window.empty:
        row.status, row.notes = "no_data", f"No candles in {start}..{end} yet."
        session.add(row)
        await session.commit()
        return row

    h, l, c = float(window["high"].max()), float(window["low"].min()), float(window["close"].iloc[-1])
    piv = classic_pivots(h, l, c)

    row.high, row.low, row.close = h, l, c
    row.r1, row.r2, row.r3 = piv["R1"], piv["R2"], piv["R3"]
    row.s1, row.s2, row.s3 = piv["S1"], piv["S2"], piv["S3"]
    row.status = "ok"

    session.add(row)
    await session.commit()
    logger.info(
        "expiry_levels: computed %s %s (%s..%s) R1=%.1f S1=%.1f",
        symbol, timeframe, start, end, piv["R1"], piv["S1"],
    )
    return row


async def get_latest_snapshot(session, symbol: str, timeframe: str) -> Optional[ExpiryLevelSnapshot]:
    stmt = (
        select(ExpiryLevelSnapshot)
        .where(ExpiryLevelSnapshot.symbol == symbol, ExpiryLevelSnapshot.timeframe == timeframe)
        .order_by(ExpiryLevelSnapshot.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
