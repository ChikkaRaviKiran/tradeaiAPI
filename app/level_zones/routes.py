"""FastAPI routes for the weekly/monthly S/R "Level Zones" panel.

Mounted onto the main FastAPI app via:
    from app.level_zones.routes import register_routes
    register_routes(app)

Read-only, informational-only subsystem — no strategy or order-placement
is attached. Intended purely to help the operator visualize higher-timeframe
support/resistance zones for planning manual directional (buy) trades.
"""
from __future__ import annotations

import json
import logging

from fastapi import FastAPI
from sqlalchemy import select

from app.condor_setup.setup_engine import _get_live_intraday_df
from app.db import models as db_models
# Imported so LevelZonePaperTrade is registered on Base.metadata before
# init_db()'s create_all() runs (same pattern as condor_setup.db_models).
from app.level_zones.db_models import LevelZonePaperTrade
from app.level_zones.timeframe_levels import compute_multi_timeframe_levels

logger = logging.getLogger(__name__)


def _trade_to_dict(row: LevelZonePaperTrade) -> dict:
    return {
        "id": row.id,
        "date": row.date,
        "symbol": row.symbol,
        "direction": row.direction,
        "zone_price": row.zone_price,
        "zone_confidence": row.zone_confidence,
        "zone_sources": json.loads(row.zone_sources) if row.zone_sources else None,
        "strike": row.strike,
        "expiry": row.expiry,
        "entry_price": row.entry_price,
        "sl_price": row.sl_price,
        "target_price": row.target_price,
        "entry_time": row.entry_time.isoformat() if row.entry_time else None,
        "spot_at_entry": row.spot_at_entry,
        "status": row.status,
        "exit_price": row.exit_price,
        "exit_time": row.exit_time.isoformat() if row.exit_time else None,
        "spot_at_exit": row.spot_at_exit,
        "pnl_points": row.pnl_points,
    }


def register_routes(app: FastAPI) -> None:

    # NOTE: registered before the "/{symbol}" route below so it isn't
    # shadowed by the path-parameter match.
    @app.get("/api/level-zone-alerts/trades")
    async def list_paper_trades(symbol: str | None = None, limit: int = 50):
        """Recent (or open) Level-Zone breakout PAPER TRADES. Informational
        only — mirrors what was sent via Telegram."""
        async with db_models.AsyncSessionLocal() as session:
            stmt = select(LevelZonePaperTrade).order_by(LevelZonePaperTrade.id.desc()).limit(limit)
            if symbol:
                stmt = stmt.where(LevelZonePaperTrade.symbol == symbol.upper())
            result = await session.execute(stmt)
            rows = result.scalars().all()
        return {"trades": [_trade_to_dict(r) for r in rows]}

    @app.get("/api/level-zone-alerts/status")
    async def alert_scheduler_status():
        from app.level_zones.alert_engine import get_status
        return get_status()

    @app.get("/api/level-zones/{symbol}")
    async def get_multi_timeframe_levels(symbol: str):
        """Weekly + monthly pivots and confluence zones for `symbol`
        (NIFTY / SENSEX). Spot is best-effort from the live feed if
        available, otherwise falls back to the last completed day's close."""
        symbol = symbol.upper()
        spot = None
        try:
            intraday, _reason = _get_live_intraday_df(symbol)
            if intraday is not None and not intraday.empty:
                spot = float(intraday["close"].iloc[-1])
        except Exception as e:  # pragma: no cover
            logger.warning("level_zones: live spot fetch failed for %s: %s", symbol, e)

        async with db_models.AsyncSessionLocal() as session:
            result = await compute_multi_timeframe_levels(session, symbol, spot=spot)
        return result
