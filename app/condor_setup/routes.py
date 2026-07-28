"""FastAPI routes for the pre-market Condor Setup panel.

Mounted onto the main FastAPI app via:
    from app.condor_setup.routes import register_routes
    register_routes(app)

All endpoints prefixed with /api/condor-setup. Read-only + a manual
recompute trigger — nothing here places or touches a live order.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import pytz
from fastapi import FastAPI
from sqlalchemy import select

from app.condor_setup.db_models import CondorDailySetup
from app.condor_setup.scheduler import (
    WEEKDAY_SCHEDULE,
    get_scheduler_status,
    run_setup_now,
)
from app.db import models as db_models

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")


def _row_to_dict(row: CondorDailySetup) -> dict:
    return {
        "date": row.date,
        "index": row.index,
        "computed_at": row.computed_at.isoformat() if row.computed_at else None,
        "is_recommended_today": row.is_recommended_today,
        "spot": row.spot,
        "atm_strike": row.atm_strike,
        "strike_interval": row.strike_interval,
        "lot_size": row.lot_size,
        "expiry_weekday": row.expiry_weekday,
        "resistance_price": row.resistance_price,
        "resistance_source": row.resistance_source,
        "resistance_confidence": row.resistance_confidence,
        "support_price": row.support_price,
        "support_source": row.support_source,
        "support_confidence": row.support_confidence,
        "short_ce_strike": row.short_ce_strike,
        "short_pe_strike": row.short_pe_strike,
        "long_ce_strike": row.long_ce_strike,
        "long_pe_strike": row.long_pe_strike,
        "wing_width_points": row.wing_width_points,
        "status": row.status,
        "notes": row.notes,
        "levels_json": row.levels_json,
    }


def register_routes(app: FastAPI) -> None:

    @app.get("/api/condor-setup/today")
    async def get_todays_condor_setup():
        """Today's (or most recently computed) confluence condor setup for
        each configured index. Purely informational — the operator sets up
        the trade manually using these levels."""
        target_date = datetime.now(_IST).strftime("%Y-%m-%d")
        async with db_models.AsyncSessionLocal() as session:
            stmt = select(CondorDailySetup).where(CondorDailySetup.date == target_date)
            rows = (await session.execute(stmt)).scalars().all()
        weekday = datetime.now(_IST).date().weekday()
        return {
            "date": target_date,
            "recommended_index": WEEKDAY_SCHEDULE.get(weekday),
            "setups": [_row_to_dict(r) for r in rows],
        }

    @app.get("/api/condor-setup/status")
    async def get_condor_setup_status():
        return get_scheduler_status()

    @app.post("/api/condor-setup/recompute")
    async def trigger_condor_setup_recompute():
        """Manually recompute today's setup on-demand (e.g. if the pre-market
        job hasn't run yet, or the operator wants fresher levels intraday)."""
        async def _run():
            try:
                await run_setup_now()
            except Exception:
                logger.exception("Manual condor-setup recompute crashed")
        asyncio.create_task(_run())
        return {"message": "Recompute started"}
