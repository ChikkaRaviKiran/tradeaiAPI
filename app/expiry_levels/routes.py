"""FastAPI routes for weekly/monthly expiry-level snapshots + chart data.

Mounted onto the main FastAPI app via:
    from app.expiry_levels.routes import register_routes
    register_routes(app)

Read-only, informational-only subsystem — no strategy or order-placement
is attached.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException

from app.db import models as db_models
# Imported so ExpiryLevelSnapshot is registered on Base.metadata before
# init_db()'s create_all() runs (same pattern as condor_setup.db_models).
from app.expiry_levels.db_models import ExpiryLevelSnapshot
from app.expiry_levels.engine import get_latest_snapshot
from app.level_zones.timeframe_levels import load_daily_ohlc

logger = logging.getLogger(__name__)


def _snapshot_to_dict(row: Optional[ExpiryLevelSnapshot]) -> dict:
    if row is None:
        return {"status": "no_data", "notes": "No snapshot computed yet."}
    return {
        "symbol": row.symbol,
        "timeframe": row.timeframe,
        "period_start": row.period_start,
        "period_end": row.period_end,
        "computed_at": row.computed_at.isoformat() if row.computed_at else None,
        "high": row.high, "low": row.low, "close": row.close,
        "r1": row.r1, "r2": row.r2, "r3": row.r3,
        "s1": row.s1, "s2": row.s2, "s3": row.s3,
        "status": row.status, "notes": row.notes,
    }


def register_routes(app: FastAPI) -> None:

    @app.get("/api/expiry-levels/{symbol}/{timeframe}")
    async def get_expiry_level_snapshot(symbol: str, timeframe: str, days: int = 40):
        """Latest weekly/monthly Classic-pivot (3R/3S) snapshot + a daily
        OHLC series (for charting), covering `days` of recent context."""
        if timeframe not in ("weekly", "monthly"):
            raise HTTPException(400, "timeframe must be 'weekly' or 'monthly'")
        symbol = symbol.upper()

        async with db_models.AsyncSessionLocal() as session:
            snapshot = await get_latest_snapshot(session, symbol, timeframe)
            daily = await load_daily_ohlc(session, symbol)

        result = _snapshot_to_dict(snapshot)
        if daily.empty:
            result["candles"] = []
            return result

        recent = daily.tail(days)
        result["candles"] = [
            {
                "date": idx.strftime("%Y-%m-%d"),
                "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
            }
            for idx, r in recent.iterrows()
        ]
        return result

    @app.get("/api/expiry-levels/status")
    async def expiry_levels_scheduler_status():
        from app.expiry_levels.scheduler import get_scheduler_status
        return get_scheduler_status()

    @app.post("/api/expiry-levels/recompute")
    async def recompute_expiry_levels():
        from app.expiry_levels.scheduler import run_snapshot_now
        try:
            results = await run_snapshot_now()
            return {"status": "ok", "results": results}
        except Exception as e:
            logger.exception("Manual expiry-levels recompute crashed")
            raise HTTPException(500, str(e))
