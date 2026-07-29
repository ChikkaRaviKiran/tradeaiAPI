"""Independent background scheduler for weekly + monthly expiry-level
snapshots (Classic pivot 3R/3S for NIFTY + SENSEX).

Self-contained: started as a FastAPI startup hook next to the condor_setup /
level_zones schedulers. Read-only/informational — no strategy or
order-placement is attached. Safe to disable via
EXPIRY_LEVELS_SCHEDULER=0 env var.

Schedule (IST, skips weekends/market holidays):
- 08:15 daily check:
    - WEEKLY snapshot recomputed every Wednesday (the week's own start day,
      matching NIFTY/SENSEX's Wed->Tue expiry cycle) - also self-heals by
      recomputing if the current week has no snapshot yet (e.g. missed due
      to downtime).
    - MONTHLY snapshot recomputed in the first 3 calendar days of a new
      month (handles month-start falling on a weekend/holiday) - also
      self-heals if the current month has no snapshot yet.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pytz

from app.core.holidays import is_market_holiday
from app.db import models as db_models
from app.expiry_levels.engine import (
    compute_and_save_snapshot,
    get_latest_snapshot,
    prior_month_bounds,
    prior_week_bounds,
)

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

RUN_HOUR_IST = 8
RUN_MINUTE_IST = 15
INDICES_TO_COMPUTE = ["NIFTY", "SENSEX"]

_last_run: dict = {
    "computed_at": None,
    "results": [],       # list of {symbol, timeframe, status}
    "next_run_at": None,
    "last_error": None,
    "running": False,
}


def get_scheduler_status() -> dict:
    return dict(_last_run)


async def _seconds_until(hour_ist: int, minute_ist: int) -> float:
    now = datetime.now(_IST)
    target = now.replace(hour=hour_ist, minute=minute_ist, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return (target - now).total_seconds()


async def _should_recompute(session, symbol: str, timeframe: str, as_of: date) -> bool:
    """True on the timeframe's trigger day, OR if the current period has no
    snapshot yet (self-heal for a missed run)."""
    if timeframe == "weekly":
        is_trigger_day = as_of.weekday() == 2  # Wednesday
        start, end = prior_week_bounds(as_of)
    else:
        is_trigger_day = as_of.day <= 3
        start, end = prior_month_bounds(as_of)

    if is_trigger_day:
        return True

    latest = await get_latest_snapshot(session, symbol, timeframe)
    return latest is None or latest.period_start != start.isoformat() or latest.period_end != end.isoformat()


async def run_snapshot_now(target_date: Optional[str] = None) -> list[dict]:
    """Compute (and persist) weekly + monthly snapshots for all configured
    indices, for whichever timeframes are due. Called by both the scheduled
    loop and the manual "Recompute Now" API endpoint."""
    as_of = datetime.strptime(target_date, "%Y-%m-%d").date() if target_date else datetime.now(_IST).date()

    results = []
    async with db_models.AsyncSessionLocal() as session:
        for symbol in INDICES_TO_COMPUTE:
            for timeframe in ("weekly", "monthly"):
                if await _should_recompute(session, symbol, timeframe, as_of):
                    row = await compute_and_save_snapshot(session, symbol, timeframe, as_of)
                    results.append({"symbol": symbol, "timeframe": timeframe, "status": row.status})

    _last_run["computed_at"] = datetime.now(_IST).isoformat()
    _last_run["results"] = results
    _last_run["last_error"] = None
    logger.info("expiry_levels: snapshot run for %s: %s", as_of, results)
    return results


async def _scheduler_loop() -> None:
    logger.info("expiry_levels: scheduler started (%02d:%02d IST daily)", RUN_HOUR_IST, RUN_MINUTE_IST)
    while True:
        try:
            sleep_s = await _seconds_until(RUN_HOUR_IST, RUN_MINUTE_IST)
            next_run = datetime.now(_IST) + timedelta(seconds=sleep_s)
            _last_run["next_run_at"] = next_run.isoformat()
            await asyncio.sleep(sleep_s)

            today = datetime.now(_IST).date()
            if today.weekday() >= 5 or is_market_holiday(today):
                logger.info("expiry_levels: %s is a weekend/holiday — skipping", today)
                continue

            _last_run["running"] = True
            await run_snapshot_now(today.strftime("%Y-%m-%d"))
        except asyncio.CancelledError:
            logger.info("expiry_levels: scheduler cancelled")
            raise
        except Exception as e:
            _last_run["last_error"] = str(e)
            logger.exception("expiry_levels scheduler error: %s", e)
            await asyncio.sleep(600)   # back off 10 min on unexpected failure
        finally:
            _last_run["running"] = False


_task: Optional[asyncio.Task] = None


def start_scheduler() -> None:
    global _task
    if _task is not None and not _task.done():
        return
    _task = asyncio.create_task(_scheduler_loop())


def stop_scheduler() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        _task = None
