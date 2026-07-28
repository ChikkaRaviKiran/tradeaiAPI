"""Breakout/bounce detection + paper-trade lifecycle for the Level Zones
weekly/monthly confluence panel.

PAPER TRADING ONLY. This module never places a live order. It:

  1. Watches live spot against the day's confluence zones
     (`app.level_zones.timeframe_levels`). When spot sustainedly breaks
     through a strong zone (2 consecutive polls beyond it — a cheap proxy
     for "candle close confirmation"), it treats that as a breakout signal:
     break above resistance -> hypothetical CE buy, break below support ->
     hypothetical PE buy.
  2. Opens a `LevelZonePaperTrade` row (real option premium fetched live)
     and sends a Telegram "BUY" alert.
  3. On every subsequent poll, checks each OPEN paper trade's live option
     LTP against its stop-loss / target, and force-closes any still-open
     trade at ~15:15 IST. Sends a Telegram alert on every exit.

Fully isolated subsystem: reuses the orchestrator's already-open broker
session (read-only) purely to read prices and send Telegram messages.
Never touches order placement, account state, or the live execution path.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pytz
from sqlalchemy import select

from app.condor_setup.setup_engine import _get_live_intraday_df
from app.core.holidays import compute_weekly_expiry
from app.core.instruments import get_instrument
from app.core.models import OptionType, StrategyName, StrategySignal
from app.db import models as db_models
from app.level_zones.db_models import LevelZonePaperTrade
from app.level_zones.timeframe_levels import compute_multi_timeframe_levels

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

SYMBOLS = ("NIFTY", "SENSEX")

POLL_INTERVAL_SECONDS = 180        # 3 minutes
MARKET_OPEN_HOUR, MARKET_OPEN_MIN = 9, 20
MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN = 15, 15

MIN_ZONE_CONFIDENCE = 3            # ignore weak/single-source zones
CONFIRM_POLLS = 2                  # consecutive polls beyond a zone before firing

# Simple, honest premium-based risk management (no options-greeks pricing
# model available for a precise underlying->premium SL/target translation,
# so this uses a fixed percentage-of-premium rule instead).
SL_PCT = 0.35                      # exit if premium falls 35% below entry
TARGET_PCT = 0.60                  # exit if premium rises 60% above entry

# In-memory per-symbol state (resets on process restart — acceptable for
# a paper-trade/alerting feature, not the live execution path).
_watch_state: dict = {}
_zones_cache: dict = {}

_status = {"last_poll_at": None, "last_error": None}


def get_status() -> dict:
    return dict(_status)


def _now() -> datetime:
    return datetime.now(_IST)


async def _get_cached_zones(session, symbol: str) -> dict:
    """Confluence zones only depend on prior-week/prior-month data, so
    they're computed once per day per symbol and reused across polls."""
    today = _now().strftime("%Y-%m-%d")
    cached = _zones_cache.get(symbol)
    if cached and cached["date"] == today:
        return cached["data"]
    data = await compute_multi_timeframe_levels(session, symbol)
    _zones_cache[symbol] = {"date": today, "data": data}
    return data


def _get_live_spot(symbol: str) -> Optional[float]:
    try:
        df, _reason = _get_live_intraday_df(symbol)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    return float(df["close"].iloc[-1])


async def _fetch_quote(orch, symbol: str, strike: float, option_type: OptionType, expiry: str) -> Optional[dict]:
    inst = get_instrument(symbol)
    if inst is None:
        return None
    signal = StrategySignal(
        strategy=StrategyName.RANGE_BREAKOUT,
        instrument=symbol,
        option_type=option_type,
        strike_price=strike,
    )
    try:
        return await orch._fetch_option_quote_for(inst, signal, expiry)
    except Exception as e:
        logger.warning("level_zones alert: quote fetch failed for %s %s %s: %s", symbol, strike, option_type, e)
        return None


async def _fire_breakout(session, orch, symbol: str, direction: str, zone_price: float, zone_confidence: int, zones: list) -> None:
    today_str = _now().strftime("%Y-%m-%d")

    # Dedup: don't open a second paper trade for the same zone/direction/day.
    existing = await session.execute(
        select(LevelZonePaperTrade).where(
            LevelZonePaperTrade.date == today_str,
            LevelZonePaperTrade.symbol == symbol,
            LevelZonePaperTrade.direction == direction,
            LevelZonePaperTrade.zone_price.between(zone_price - 1, zone_price + 1),
        )
    )
    if existing.scalars().first():
        return

    inst = get_instrument(symbol)
    spot = _get_live_spot(symbol)
    if inst is None or spot is None:
        return

    strike = inst.nearest_strike(spot, direction)
    expiry_date = compute_weekly_expiry(_now().date(), inst.expiry_weekday)
    expiry_str = expiry_date.strftime("%d%b%y").upper()
    option_type = OptionType.CALL if direction == "CE" else OptionType.PUT

    quote = await _fetch_quote(orch, symbol, strike, option_type, expiry_str)
    if not quote or not quote.get("ltp"):
        logger.warning("level_zones alert: no usable quote for %s %s %s — skipping alert", symbol, strike, direction)
        return

    entry_price = quote["best_ask"] if quote.get("best_ask") else quote["ltp"]
    sl_price = round(entry_price * (1 - SL_PCT), 2)
    target_price = round(entry_price * (1 + TARGET_PCT), 2)

    if direction == "CE":
        further = sorted(z["price"] for z in zones if z["price"] > zone_price)
    else:
        further = sorted((z["price"] for z in zones if z["price"] < zone_price), reverse=True)
    next_zone_price = further[0] if further else None

    row = LevelZonePaperTrade(
        date=today_str,
        symbol=symbol,
        direction=direction,
        zone_price=zone_price,
        zone_confidence=zone_confidence,
        strike=strike,
        expiry=expiry_str,
        entry_price=entry_price,
        sl_price=sl_price,
        target_price=target_price,
        entry_time=_now().replace(tzinfo=None),
        status="open",
    )
    session.add(row)
    await session.commit()

    side = "resistance" if direction == "CE" else "support"
    msg = (
        f"\U0001F4E3 PAPER TRADE (Level Zone breakout)\n"
        f"{symbol} {int(strike)} {direction} (exp {expiry_str})\n"
        f"Broke {side} zone {zone_price:.2f} (confidence x{zone_confidence})\n"
        f"BUY @ {entry_price:.2f}\n"
        f"SL: {sl_price:.2f}  |  Target: {target_price:.2f}\n"
        + (f"Next underlying zone: {next_zone_price:.2f}\n" if next_zone_price else "")
        + "This is a PAPER TRADE alert only \u2014 no real order placed."
    )
    await orch.alert_manager.telegram.send(msg)
    logger.info("level_zones alert: paper trade opened %s %s %s @ %.2f", symbol, strike, direction, entry_price)


async def _check_breakout(session, orch, symbol: str) -> None:
    zones_data = await _get_cached_zones(session, symbol)
    if zones_data.get("status") != "ok":
        return

    spot = _get_live_spot(symbol)
    if spot is None:
        return

    zones = zones_data.get("zones", [])
    strong = [z for z in zones if z.get("confidence", 0) >= MIN_ZONE_CONFIDENCE]
    resistance_candidates = sorted((z for z in strong if z["price"] > spot), key=lambda z: z["price"])
    support_candidates = sorted((z for z in strong if z["price"] < spot), key=lambda z: -z["price"])
    nearest_r = resistance_candidates[0] if resistance_candidates else None
    nearest_s = support_candidates[0] if support_candidates else None

    state = _watch_state.setdefault(
        symbol, {"resistance": None, "support": None, "pending_up": None, "pending_down": None}
    )

    prev_r = state["resistance"]
    if prev_r is not None and spot > prev_r["price"]:
        pending = state["pending_up"]
        if pending and abs(pending["zone_price"] - prev_r["price"]) < 0.01:
            pending["confirm_count"] += 1
        else:
            pending = {"zone_price": prev_r["price"], "confidence": prev_r["confidence"], "confirm_count": 1}
        state["pending_up"] = pending
        if pending["confirm_count"] >= CONFIRM_POLLS:
            await _fire_breakout(session, orch, symbol, "CE", pending["zone_price"], pending["confidence"], zones)
            state["pending_up"] = None
    else:
        state["pending_up"] = None

    prev_s = state["support"]
    if prev_s is not None and spot < prev_s["price"]:
        pending = state["pending_down"]
        if pending and abs(pending["zone_price"] - prev_s["price"]) < 0.01:
            pending["confirm_count"] += 1
        else:
            pending = {"zone_price": prev_s["price"], "confidence": prev_s["confidence"], "confirm_count": 1}
        state["pending_down"] = pending
        if pending["confirm_count"] >= CONFIRM_POLLS:
            await _fire_breakout(session, orch, symbol, "PE", pending["zone_price"], pending["confidence"], zones)
            state["pending_down"] = None
    else:
        state["pending_down"] = None

    state["resistance"] = nearest_r
    state["support"] = nearest_s


async def _monitor_open_trades(session, orch) -> None:
    result = await session.execute(select(LevelZonePaperTrade).where(LevelZonePaperTrade.status == "open"))
    open_trades = result.scalars().all()
    if not open_trades:
        return

    now = _now()
    is_eod = (now.hour, now.minute) >= (MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN)

    for trade in open_trades:
        option_type = OptionType.CALL if trade.direction == "CE" else OptionType.PUT
        quote = await _fetch_quote(orch, trade.symbol, trade.strike, option_type, trade.expiry)
        if not quote or not quote.get("ltp"):
            continue
        ltp = quote["ltp"]

        exit_reason = None
        if ltp <= trade.sl_price:
            exit_reason = "sl_hit"
        elif ltp >= trade.target_price:
            exit_reason = "target_hit"
        elif is_eod:
            exit_reason = "eod_close"

        if not exit_reason:
            continue

        exit_price = quote["best_bid"] if quote.get("best_bid") else ltp
        trade.status = exit_reason
        trade.exit_price = exit_price
        trade.exit_time = now.replace(tzinfo=None)
        trade.pnl_points = round(exit_price - trade.entry_price, 2)
        await session.commit()

        emoji = {"sl_hit": "\U0001F6D1", "target_hit": "\u2705", "eod_close": "\u23F1"}[exit_reason]
        label = {"sl_hit": "STOP-LOSS HIT", "target_hit": "TARGET HIT", "eod_close": "EOD SQUARE-OFF"}[exit_reason]
        msg = (
            f"{emoji} PAPER TRADE {label}\n"
            f"{trade.symbol} {int(trade.strike)} {trade.direction} (exp {trade.expiry})\n"
            f"Entry: {trade.entry_price:.2f}  \u2192  Exit: {exit_price:.2f}\n"
            f"PnL: {trade.pnl_points:+.2f} pts/lot"
        )
        await orch.alert_manager.telegram.send(msg)
        logger.info(
            "level_zones alert: paper trade closed (%s) %s %s %s pnl=%.2f",
            exit_reason, trade.symbol, trade.strike, trade.direction, trade.pnl_points,
        )


async def poll_once() -> None:
    """One full poll cycle across all symbols. Safe to call repeatedly;
    no-ops gracefully if the orchestrator/broker session isn't up yet."""
    try:
        from app.api.routes import get_state  # lazy import: avoids circular import at module load
    except Exception as e:
        _status["last_error"] = str(e)
        return

    orch = get_state().get("orchestrator")
    if orch is None:
        return  # orchestrator not running yet — skip this cycle silently

    try:
        async with db_models.AsyncSessionLocal() as session:
            for symbol in SYMBOLS:
                try:
                    await _check_breakout(session, orch, symbol)
                except Exception:
                    logger.exception("level_zones alert: breakout check failed for %s", symbol)
            try:
                await _monitor_open_trades(session, orch)
            except Exception:
                logger.exception("level_zones alert: monitor open trades failed")
        _status["last_poll_at"] = _now().isoformat()
        _status["last_error"] = None
    except Exception as e:
        _status["last_error"] = str(e)
        logger.exception("level_zones alert: poll_once failed: %s", e)
