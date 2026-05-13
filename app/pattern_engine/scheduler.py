"""Nightly scheduler — refreshes pattern stats every day at 22:30 IST.

Self-contained: starts as a FastAPI startup hook (registered next to the
existing routes). Does NOT touch the orchestrator. Runs in the same event
loop as the API. Safe to disable via PATTERN_ENGINE_SCHEDULER=0 env var.

Schedule (IST):
- 22:30 daily : refresh_pattern_stats() across all patterns / all windows
- 22:35 daily : auto-tier sync (apply suggested_tier → status if rules met)

Auto-tier rules (additive — never auto-promote into 'live' without operator):
- Patterns with status=shadow whose 30d window has REJECT tier for 7+ days
  → demote to 'paused' (operator must re-enable)
- Patterns with status=live whose 30d window goes REJECT for 3+ days
  → demote to 'shadow' (auto, with telegram alert if configured)

Operator promotion (shadow → live) stays manual via the UI.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

import pytz
from sqlalchemy import desc, select

from app.db.models import AsyncSessionLocal
from app.pattern_engine.db_models import PEPattern, PEPatternStats
from app.pattern_engine.stats import refresh_pattern_stats

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

# Last successful run snapshot (read by health endpoint)
_last_run: dict = {
    "stats_refresh_at": None,
    "stats_rows_written": 0,
    "auto_tier_at": None,
    "auto_tier_changes": 0,
    "next_run_at": None,
    "last_error": None,
}


def get_scheduler_status() -> dict:
    return dict(_last_run)


async def _seconds_until(hour_ist: int, minute_ist: int) -> float:
    now = datetime.now(_IST)
    target = now.replace(hour=hour_ist, minute=minute_ist, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return (target - now).total_seconds()


async def _do_stats_refresh() -> None:
    async with AsyncSessionLocal() as s:
        try:
            written = await refresh_pattern_stats(s)
            _last_run["stats_refresh_at"] = datetime.now(_IST).isoformat()
            _last_run["stats_rows_written"] = written
            _last_run["last_error"] = None
            logger.info("nightly: stats refreshed (%d rows)", written)
        except Exception as e:
            _last_run["last_error"] = f"stats_refresh: {e}"
            logger.exception("nightly stats refresh failed")


async def _do_auto_tier_sync() -> None:
    """Apply soft auto-demote rules. Promotion stays manual."""
    async with AsyncSessionLocal() as s:
        try:
            patterns = (await s.execute(select(PEPattern))).scalars().all()
            changes = 0
            for p in patterns:
                # Get latest 30d stat row
                stmt = (
                    select(PEPatternStats)
                    .where(
                        PEPatternStats.pattern_id == p.pattern_id,
                        PEPatternStats.window == "30d",
                    )
                    .order_by(desc(PEPatternStats.computed_at))
                    .limit(1)
                )
                stat = (await s.execute(stmt)).scalar_one_or_none()
                if not stat or stat.n_trades < 5:
                    continue  # not enough live data to judge

                if p.status == "live" and stat.suggested_tier == "REJECT":
                    p.status = "shadow"
                    p.notes = (
                        (p.notes or "")
                        + f"\n[auto] {datetime.now(_IST):%Y-%m-%d} "
                          f"demoted live→shadow (30d tier=REJECT, n={stat.n_trades}, "
                          f"wr={stat.win_rate:.0%}, pf={stat.profit_factor:.2f})"
                    )
                    changes += 1
                    logger.warning(
                        "nightly: auto-demoted %s live→shadow (REJECT, wr=%.0f%%)",
                        p.pattern_id, (stat.win_rate or 0) * 100,
                    )
            if changes:
                await s.commit()
            _last_run["auto_tier_at"] = datetime.now(_IST).isoformat()
            _last_run["auto_tier_changes"] = changes
            logger.info("nightly: auto-tier sync complete (%d changes)", changes)
        except Exception as e:
            _last_run["last_error"] = f"auto_tier: {e}"
            logger.exception("nightly auto-tier sync failed")


async def _scheduler_loop() -> None:
    """Forever loop — sleeps until next 22:30 IST, runs jobs, repeats."""
    logger.info("pattern_engine: nightly scheduler started (22:30 IST stats / 22:35 IST auto-tier)")
    while True:
        try:
            sleep_s = await _seconds_until(22, 30)
            next_run = datetime.now(_IST) + timedelta(seconds=sleep_s)
            _last_run["next_run_at"] = next_run.isoformat()
            await asyncio.sleep(sleep_s)
            await _do_stats_refresh()
            await asyncio.sleep(300)   # 5 min later
            await _do_auto_tier_sync()
        except asyncio.CancelledError:
            logger.info("pattern_engine: scheduler cancelled")
            raise
        except Exception as e:
            logger.exception("pattern_engine scheduler error: %s", e)
            await asyncio.sleep(600)   # back off 10 min on unexpected failure


_task: asyncio.Task | None = None


def start_scheduler(loop: asyncio.AbstractEventLoop | None = None) -> None:
    global _task
    if _task and not _task.done():
        return  # already running
    loop = loop or asyncio.get_event_loop()
    _task = loop.create_task(_scheduler_loop(), name="pattern_engine_scheduler")
    logger.info("pattern_engine: scheduler task created")


def stop_scheduler() -> None:
    global _task
    if _task and not _task.done():
        _task.cancel()
        _task = None


async def trigger_now() -> dict:
    """Manual trigger — useful for the UI 'Run nightly job now' button."""
    await _do_stats_refresh()
    await _do_auto_tier_sync()
    return get_scheduler_status()
