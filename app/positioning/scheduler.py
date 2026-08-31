"""Server-side collector for the Market Story option-chain series.

The AgenticTrading original (`app/tools/watch_positioning.py`) was a foreground
script, and its own docstring admits the problem: "a loop that only survives
while a terminal window stays open is not a data collection strategy". The
browser-driven poll in the UI has the same defect wearing a different hat - it
records only while somebody is watching, at whatever cadence the tab happens to
refresh at, and a headless server records nothing at all.

This is that loop as a background task owned by the API process, registered in
`lifespan` next to the pattern_engine / condor_setup / expiry_levels
schedulers. It writes to the same `chain_snapshots` table the read endpoints
already serve, so the UI needs no change: open the page at any hour and the
whole day is already there.

Disable with POSITIONING_SCHEDULER=0.

WHY IT SLEEPS TO THE CLOCK AND NOT FOR A FIXED INTERVAL
-------------------------------------------------------
`fetch_chain` floors `captured_at` onto the five-minute grid, so the grid is not
optional - it is what the snapshot is keyed by. Sleeping a fixed 300 seconds
starts the series wherever the process happened to launch and then drifts by
however long each fetch took, and the drift does not merely shift the labels:
two polls that straddle a boundary the wrong way both floor into the same
bucket and the bucket between them is never written at all. A gap that arrives
without an error message is the worst kind, so the wait is computed to the next
wall-clock boundary every time and cannot accumulate.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, datetime, time as dtime, timedelta
from typing import Any, Optional

from app.positioning import storage
from app.positioning.dhan import DhanError
from app.positioning.option_chain import IST, fetch_chain, nearest_expiry

logger = logging.getLogger(__name__)

OPEN = dtime(9, 15)
CLOSE = dtime(15, 30)

SYMBOL = os.environ.get("POSITIONING_SYMBOL", "NIFTY")
EVERY_SECONDS = int(os.environ.get("POSITIONING_EVERY_SECONDS", "300"))
WINDOW = int(os.environ.get("POSITIONING_WINDOW", "15"))

# A candle closes at the boundary; the closing print lands a moment after it.
# Fetching exactly on the second reads the previous candle's last tick often
# enough to matter, and the cost of waiting is five seconds of staleness.
SETTLE_SECONDS = 5

_status: dict[str, Any] = {
    "enabled": False,
    "symbol": SYMBOL,
    "expiry": None,
    "last_captured_at": None,
    "last_strikes": None,
    "last_spot": None,
    "last_poll_at": None,
    "next_poll_at": None,
    "last_error": None,
    "captures_today": 0,
    "session_date": None,
}


def get_scheduler_status() -> dict:
    return dict(_status)


def in_session(now: datetime) -> bool:
    """Weekday, not a market holiday, and inside market hours.

    Compared at minute resolution because the poll is deliberately a few
    seconds late: the 15:30 close would otherwise be judged out of session by
    its own settle delay and the last candle of the day would never be stored.
    """
    if now.weekday() >= 5:
        return False
    if not (OPEN <= now.replace(second=0, microsecond=0).time() <= CLOSE):
        return False
    try:
        from app.core.holidays import is_market_holiday
        if is_market_holiday(now.date()):
            return False
    except Exception:
        # Holiday data is an optimisation, not a correctness requirement: a
        # poll on a closed exchange stores a stale chain, it does not corrupt
        # anything. Never let a lookup failure stop the collector.
        pass
    return True


def next_tick(now: datetime, every: int = EVERY_SECONDS,
              settle: int = SETTLE_SECONDS) -> datetime:
    """The next wall-clock candle close, plus the settle delay.

    Anchored to midnight rather than to the process start, so restarting the
    API mid-session rejoins the same grid the morning was recorded on.
    """
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = (now - midnight).total_seconds()
    n = int((elapsed - settle) // every) + 1
    return midnight + timedelta(seconds=n * every + settle)


def following_tick(target: datetime, now: datetime, every: int = EVERY_SECONDS,
                   settle: int = SETTLE_SECONDS) -> datetime:
    """The tick after `target`, skipping any that have already gone by.

    Anchored to the target just used rather than to the clock, because a sleep
    may return a few milliseconds EARLY: recomputing from a `now` that is still
    a hair before the boundary hands back the boundary we have just polled, and
    the candle gets fetched twice. Taking the later of the two anchors also
    skips forward correctly when a fetch overran its own candle.
    """
    return next_tick(max(target, now), every, settle)


_expiry_cache: tuple[date, str] | None = None


def _expiry_for(day: date) -> str:
    """Front expiry, resolved once per day.

    The original script resolved it once at startup, which is correct for a
    process that lives for one session and wrong for one that stays up across
    an expiry rollover - it would keep polling a contract that no longer
    trades.
    """
    global _expiry_cache
    if _expiry_cache is not None and _expiry_cache[0] == day:
        return _expiry_cache[1]
    expiry = nearest_expiry(SYMBOL)
    _expiry_cache = (day, expiry)
    return expiry


def _poll_once_blocking(expiry: str) -> dict:
    """The synchronous fetch+store. Runs in a worker thread, never on the loop."""
    snapshot = fetch_chain(SYMBOL, expiry, window=WINDOW)
    written = storage.save_chain_snapshot(snapshot)
    return {
        "captured_at": snapshot["captured_at"],
        "spot": snapshot["spot"],
        "expiry": snapshot["expiry"],
        "strikes": written,
    }


async def poll_once() -> dict:
    """One capture on demand. Used by the loop and by the manual endpoint."""
    today = datetime.now(IST).date()
    expiry = await asyncio.to_thread(_expiry_for, today)
    out = await asyncio.to_thread(_poll_once_blocking, expiry)

    day = str(out["captured_at"])[:10]
    if _status["session_date"] != day:
        _status["session_date"] = day
        _status["captures_today"] = 0
    _status["captures_today"] += 1
    _status["expiry"] = out["expiry"]
    _status["last_captured_at"] = out["captured_at"]
    _status["last_spot"] = out["spot"]
    _status["last_strikes"] = out["strikes"]
    _status["last_poll_at"] = datetime.now(IST).isoformat()
    _status["last_error"] = None
    return out


async def _sleep_until(target: datetime) -> None:
    """Sleep until `target` has genuinely passed, not merely until sleep returns."""
    while True:
        remaining = (target - datetime.now(IST)).total_seconds()
        if remaining <= 0:
            return
        await asyncio.sleep(remaining)


async def _scheduler_loop() -> None:
    logger.info(
        "positioning: collector started (%s, every %ds +%ds settle, %02d:%02d-%02d:%02d IST)",
        SYMBOL, EVERY_SECONDS, SETTLE_SECONDS,
        OPEN.hour, OPEN.minute, CLOSE.hour, CLOSE.minute,
    )
    target = next_tick(datetime.now(IST))
    while True:
        try:
            _status["next_poll_at"] = target.isoformat()
            await _sleep_until(target)

            now = datetime.now(IST)
            if in_session(now):
                try:
                    out = await poll_once()
                    logger.info(
                        "positioning: %s spot %.2f, %d strikes",
                        out["captured_at"], out["spot"], out["strikes"],
                    )
                except DhanError as exc:
                    # Keep going. A single failed poll is a gap in the series;
                    # an exit is the end of the series.
                    _status["last_error"] = str(exc)
                    logger.warning("positioning: fetch failed at %s: %s",
                                   now.strftime("%H:%M:%S"), exc)

            target = following_tick(target, datetime.now(IST))
        except asyncio.CancelledError:
            logger.info("positioning: collector cancelled")
            raise
        except Exception as e:
            _status["last_error"] = str(e)
            logger.exception("positioning collector error: %s", e)
            await asyncio.sleep(60)
            target = following_tick(target, datetime.now(IST))


_task: Optional[asyncio.Task] = None


def start_scheduler() -> None:
    global _task
    if _task is not None and not _task.done():
        return
    storage.init_db()
    _status["enabled"] = True
    _task = asyncio.create_task(_scheduler_loop())


def stop_scheduler() -> None:
    global _task
    _status["enabled"] = False
    if _task is not None:
        _task.cancel()
        _task = None
