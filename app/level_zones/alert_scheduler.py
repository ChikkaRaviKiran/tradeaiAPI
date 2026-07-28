"""Background poller for the Level Zones breakout PAPER-TRADE alerts.

PAPER TRADING ONLY — never places a live order. Polls every
`POLL_INTERVAL_SECONDS` during market hours on trading days, delegating all
detection/tracking logic to `app.level_zones.alert_engine`.

Self-contained: mirrors the `app.condor_setup.scheduler` pattern. Safe to
disable via LEVEL_ZONE_ALERTS_SCHEDULER=0 env var.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import pytz

from app.core.holidays import is_market_holiday
from app.level_zones.alert_engine import (
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MIN,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MIN,
    POLL_INTERVAL_SECONDS,
    poll_once,
)

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")


def _within_market_hours(now: datetime) -> bool:
    open_t = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0, microsecond=0)
    close_t = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MIN, second=0, microsecond=0)
    return open_t <= now <= close_t


async def _scheduler_loop() -> None:
    logger.info(
        "level_zones: paper-trade alert poller started (every %ds, %02d:%02d-%02d:%02d IST)",
        POLL_INTERVAL_SECONDS, MARKET_OPEN_HOUR, MARKET_OPEN_MIN, MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN,
    )
    while True:
        try:
            now = datetime.now(_IST)
            today = now.date()
            if today.weekday() >= 5 or is_market_holiday(today) or not _within_market_hours(now):
                await asyncio.sleep(60)
                continue
            await poll_once()
        except asyncio.CancelledError:
            logger.info("level_zones: alert poller cancelled")
            raise
        except Exception as e:
            logger.exception("level_zones: alert poller error: %s", e)
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


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
