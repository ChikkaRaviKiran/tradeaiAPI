"""Computes one index's pre-market confluence-based iron condor setup.

Mirrors the AgenticTrade backtester's level-building logic (classic +
Camarilla pivots, prior-day H/L, opening range, round numbers -> confluence
clustering -> nearest resistance/support -> condor legs).

Data sources (deliberately split):
- Prior-day H/L/C: read from TradeAI's own `IndexCandle` table (populated by
  the nightly `IndexCandleCollector` job — by "today" it is always fully
  populated for all PRIOR days, so this is safe to read directly).
- Today's opening range / current spot: read from the orchestrator's
  ALREADY-CONNECTED live WebSocket candle buffer (`orchestrator.client.
  get_live_candles`). We deliberately do NOT open a second AngelOne
  session here — AngelOne allows only one active session per login, so an
  independent `authenticate()` call from this job could invalidate the live
  trading session's token. We also deliberately do NOT write today's partial
  candles into `IndexCandle` — that table's dedup check
  (`_already_cached`) assumes one row per day means "full day already
  collected", and writing a partial morning snapshot would make the
  end-of-day collector skip the real full-day collection.

This module is read-only with respect to trading — it never places an order.
It only computes a suggested setup and persists it to `condor_daily_setups`.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import pytz

from app.condor_setup.db_models import CondorDailySetup
from app.condor_setup.levels import (
    build_condor,
    build_confluence_levels,
    camarilla_pivots,
    classic_pivots,
    nearest_resistance,
    nearest_support,
    opening_range,
    round_levels,
)
from app.core.instruments import get_instrument
from app.pattern_engine.features import load_prior_day_levels

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

# Empirically tuned in the AgenticTrade backtest (SCHEDULE run, Apr-Jul):
# fixed 250-point wing width gave the best defined-risk/margin trade-off.
DEFAULT_WING_WIDTH_POINTS = 250
OPENING_RANGE_MINUTES = 15


def _get_live_intraday_df(symbol: str):
    """Read today's 1-min candles from the orchestrator's already-open
    WebSocket feed (read-only — does not create a broker session).

    Returns (dataframe_or_None, reason_if_none).
    """
    try:
        from app.api.routes import get_state  # lazy import: avoids circular import at module load
    except Exception as e:
        return None, f"Could not access shared app state: {e}"

    orch = get_state().get("orchestrator")
    if orch is None:
        return None, "Orchestrator is not running (no live client available)."

    client = getattr(orch, "client", None)
    if client is None:
        return None, "Live broker client not initialized yet."

    inst = get_instrument(symbol)
    if inst is None:
        return None, f"Unknown instrument: {symbol}"

    try:
        df = client.get_live_candles(inst.token, inst.exchange.value)
    except Exception as e:
        return None, f"Live candle fetch failed: {e}"

    if df is None or df.empty:
        return None, "No live candles yet (WebSocket not connected or market not open)."

    return df, None


async def compute_condor_setup(
    session,
    symbol: str,
    target_date: str,
    is_recommended_today: bool = False,
    wing_width_points: int = DEFAULT_WING_WIDTH_POINTS,
) -> CondorDailySetup:
    """Compute (and persist) today's confluence condor setup for `symbol`.

    Returns the saved `CondorDailySetup` row (also `status="no_data"` /
    `"error"` rows are persisted so the UI can show *why* nothing is ready
    yet, rather than silently showing nothing).
    """
    inst = get_instrument(symbol)
    if inst is None:
        raise ValueError(f"Unknown instrument: {symbol}")

    row = CondorDailySetup(
        date=target_date,
        index=symbol,
        computed_at=datetime.now(IST).replace(tzinfo=None),
        is_recommended_today=is_recommended_today,
        strike_interval=inst.strike_interval,
        lot_size=inst.lot_size,
        expiry_weekday=inst.expiry_weekday,
        wing_width_points=wing_width_points,
    )

    try:
        pdh, pdl, pdc = await load_prior_day_levels(session, symbol, target_date)
        if pdh is None:
            row.status = "no_data"
            row.notes = "No prior-day candles found yet — cannot compute pivots."
            await _save(session, row)
            return row

        intraday, no_data_reason = _get_live_intraday_df(symbol)
        if intraday is None:
            row.status = "no_data"
            row.notes = no_data_reason
            await _save(session, row)
            return row

        spot = float(intraday["close"].iloc[-1])
        row.spot = spot
        row.atm_strike = round(spot / inst.strike_interval) * inst.strike_interval

        level_sources: dict[str, float] = {}
        cp = classic_pivots(pdh, pdl, pdc)
        for name, price in cp.items():
            level_sources[f"pivot_{name}"] = price
        cam = camarilla_pivots(pdh, pdl, pdc)
        for name, price in cam.items():
            level_sources[f"camarilla_{name}"] = price
        level_sources["prior_day_high"] = pdh
        level_sources["prior_day_low"] = pdl

        orange = opening_range(intraday, minutes=OPENING_RANGE_MINUTES)
        if orange is not None:
            or_high, or_low = orange
            level_sources["opening_range_high"] = or_high
            level_sources["opening_range_low"] = or_low

        for rl in round_levels(spot, step=inst.strike_interval, span=2):
            level_sources[f"round_{int(rl)}"] = rl

        confluence = build_confluence_levels(level_sources)
        resistance = nearest_resistance(confluence, spot)
        support = nearest_support(confluence, spot)

        # Fallback if the confluence clustering finds nothing above/below spot
        # (e.g. very early in the session) — use prior day H/L directly.
        resistance_price = resistance.price if resistance else pdh
        resistance_source = "+".join(resistance.sources) if resistance else "prior_day_high (fallback)"
        resistance_confidence = resistance.confidence if resistance else 1

        support_price = support.price if support else pdl
        support_source = "+".join(support.sources) if support else "prior_day_low (fallback)"
        support_confidence = support.confidence if support else 1

        legs = build_condor(resistance_price, support_price, wing_width_points, int(inst.strike_interval))

        row.resistance_price = resistance_price
        row.resistance_source = resistance_source
        row.resistance_confidence = resistance_confidence
        row.support_price = support_price
        row.support_source = support_source
        row.support_confidence = support_confidence
        row.short_ce_strike = legs.short_ce_strike
        row.short_pe_strike = legs.short_pe_strike
        row.long_ce_strike = legs.long_ce_strike
        row.long_pe_strike = legs.long_pe_strike
        row.levels_json = json.dumps(
            {name: round(price, 2) for name, price in level_sources.items()}
        )
        row.status = "ok"
        await _save(session, row)
        return row

    except Exception as e:  # pragma: no cover
        logger.exception("compute_condor_setup failed for %s %s", symbol, target_date)
        row.status = "error"
        row.notes = str(e)[:500]
        await _save(session, row)
        return row


async def _save(session, row: CondorDailySetup) -> None:
    """Replace any existing row for this (date, index) then insert the new one."""
    from sqlalchemy import delete

    await session.execute(
        delete(CondorDailySetup).where(
            CondorDailySetup.date == row.date, CondorDailySetup.index == row.index
        )
    )
    session.add(row)
    await session.commit()

