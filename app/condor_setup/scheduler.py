"""Independent background scheduler for the pre-market condor setup job.

Self-contained: started as a FastAPI startup hook next to the pattern_engine
scheduler. Does NOT touch the orchestrator or any live execution path.
Runs in the same event loop as the API. Safe to disable via
CONDOR_SETUP_SCHEDULER=0 env var.

Schedule (IST, weekdays only, skips market holidays):
- 09:32 : compute today's NIFTY + SENSEX confluence condor setups
          (the opening range, 09:15-09:30, has closed by then)

Weekday routing (validated in the AgenticTrade backtest, SCHEDULE run):
  Mon, Tue, Fri -> NIFTY is "today's scheduled pick"
  Wed, Thu      -> SENSEX is "today's scheduled pick"
Both indices are always computed (so the operator can manually override),
only the `is_recommended_today` flag differs.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta

import pytz

from app.core.holidays import is_market_holiday
from app.db import models as db_models
from app.condor_setup.setup_engine import compute_condor_setup

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

RUN_HOUR_IST = 9
RUN_MINUTE_IST = 32

# 0=Mon .. 4=Fri. Matches AgenticTrade's validated WEEKDAY_UNDERLYING_SCHEDULE.
WEEKDAY_SCHEDULE = {
    0: "NIFTY",
    1: "NIFTY",
    2: "SENSEX",
    3: "SENSEX",
    4: "NIFTY",
}
INDICES_TO_COMPUTE = ["NIFTY", "SENSEX"]

_last_run: dict = {
    "computed_at": None,
    "results": [],       # list of {index, status, is_recommended_today}
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


async def run_setup_now(target_date: str | None = None) -> list[dict]:
    """Compute (and persist) today's setup for all configured indices.

    Can be called both by the scheduled loop and by the manual
    "Recompute Now" API endpoint.
    """
    if target_date is None:
        target_date = datetime.now(_IST).strftime("%Y-%m-%d")

    weekday = datetime.strptime(target_date, "%Y-%m-%d").date().weekday()
    recommended = WEEKDAY_SCHEDULE.get(weekday)

    results = []
    # Access AsyncSessionLocal via the module (not a name captured at import
    # time) — app.db.models.init_db() rebinds it to a fresh engine on startup
    # (to avoid asyncpg "attached to a different loop" errors), and this
    # module is imported before that rebind happens.
    async with db_models.AsyncSessionLocal() as session:
        for symbol in INDICES_TO_COMPUTE:
            row = await compute_condor_setup(
                session,
                symbol,
                target_date,
                is_recommended_today=(symbol == recommended),
            )
            results.append({
                "index": symbol,
                "status": row.status,
                "is_recommended_today": row.is_recommended_today,
            })

    _last_run["computed_at"] = datetime.now(_IST).isoformat()
    _last_run["results"] = results
    _last_run["last_error"] = None
    logger.info("condor_setup: computed setups for %s: %s", target_date, results)
    return results


async def _scheduler_loop() -> None:
    logger.info(
        "condor_setup: pre-market scheduler started (%02d:%02d IST daily)",
        RUN_HOUR_IST, RUN_MINUTE_IST,
    )
    while True:
        try:
            sleep_s = await _seconds_until(RUN_HOUR_IST, RUN_MINUTE_IST)
            next_run = datetime.now(_IST) + timedelta(seconds=sleep_s)
            _last_run["next_run_at"] = next_run.isoformat()
            await asyncio.sleep(sleep_s)

            today = datetime.now(_IST).date()
            if today.weekday() >= 5 or is_market_holiday(today):
                logger.info("condor_setup: %s is a weekend/holiday — skipping", today)
                continue

            _last_run["running"] = True
            await run_setup_now(today.strftime("%Y-%m-%d"))
        except asyncio.CancelledError:
            logger.info("condor_setup: scheduler cancelled")
            raise
        except Exception as e:
            _last_run["last_error"] = str(e)
            logger.exception("condor_setup scheduler error: %s", e)
            await asyncio.sleep(600)   # back off 10 min on unexpected failure
        finally:
            _last_run["running"] = False


_task: asyncio.Task | None = None


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
