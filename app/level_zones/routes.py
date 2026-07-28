"""FastAPI routes for the weekly/monthly S/R "Level Zones" panel.

Mounted onto the main FastAPI app via:
    from app.level_zones.routes import register_routes
    register_routes(app)

Read-only, informational-only subsystem — no strategy or order-placement
is attached. Intended purely to help the operator visualize higher-timeframe
support/resistance zones for planning manual directional (buy) trades.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI

from app.condor_setup.setup_engine import _get_live_intraday_df
from app.db import models as db_models
from app.level_zones.timeframe_levels import compute_multi_timeframe_levels

logger = logging.getLogger(__name__)


def register_routes(app: FastAPI) -> None:

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
