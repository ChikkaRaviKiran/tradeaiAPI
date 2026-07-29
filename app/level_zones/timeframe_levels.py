"""Weekly & monthly support/resistance levels for NIFTY/SENSEX.

READ-ONLY, INFORMATIONAL ONLY — no strategy or order-placement is attached
to this module. It exists purely so the operator can see higher-timeframe
S/R zones to plan manual directional (buy) trades around.

Data source: TradeAI's own `index_candles` table (1-min OHLCV, populated by
the nightly IndexCandleCollector job) — the same source the Condor Setup
feature uses for prior-day levels — aggregated here into daily bars and then
into weekly / monthly bars.

Level sources included (as many as are useful to plot as lines/zones):
 - Classic pivots (P/R1-R3/S1-S3) from the prior COMPLETED week's H/L/C.
 - Camarilla pivots (R1-R4/S1-S4) from the prior COMPLETED week's H/L/C.
 - Classic + Camarilla pivots from the prior COMPLETED month's H/L/C.
 - Prior week high/low and prior month high/low (simple structural levels).
 - Recent weekly swing highs/lows (last SWING_LOOKBACK_WEEKS weeks) and
   recent monthly swing highs/lows (last SWING_LOOKBACK_MONTHS months) —
   catches structural turning points a single prior-period pivot can miss.
 - Round-number levels near current spot.
All of the above are merged through the same confluence-clustering engine
the Condor Setup uses (`build_confluence_levels`), with a wider tolerance
since weekly/monthly levels are naturally more spread out than daily ones —
a level backed by multiple independent sources (e.g. weekly S1 + monthly S1
+ a round number) is the "zone" worth paying attention to.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import select

from app.condor_setup.levels import (
    build_confluence_levels,
    camarilla_pivots,
    classic_pivots,
    round_levels,
)
from app.db.models import IndexCandle

# How many recent completed periods (besides the immediately-prior one) to
# also pull in as extra swing-high/low sources.
SWING_LOOKBACK_WEEKS = 8
SWING_LOOKBACK_MONTHS = 6

# Wider than the daily Condor Setup's default (0.15%) — weekly/monthly levels
# from different sources naturally sit further apart than intraday pivots.
CONFLUENCE_TOLERANCE_PCT = 0.004


async def load_daily_ohlc(session, symbol: str) -> pd.DataFrame:
    """Aggregate TradeAI's 1-min index_candles into one row per calendar day
    (open/high/low/close), across all history available for `symbol`.
    Returned frame is indexed by date (datetime64, no time component)."""
    stmt = (
        select(IndexCandle)
        .where(IndexCandle.instrument == symbol)
        .order_by(IndexCandle.timestamp.asc())
    )
    rows = (await session.execute(stmt)).scalars().all()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "date": r.date,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
        }
        for r in rows
    ])
    daily = df.groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


def _hlc_levels(high: float, low: float, close: float, prefix: str) -> dict:
    out = {f"{prefix}_high": high, f"{prefix}_low": low}
    for name, price in classic_pivots(high, low, close).items():
        out[f"{prefix}_pivot_{name}"] = price
    for name, price in camarilla_pivots(high, low, close).items():
        out[f"{prefix}_camarilla_{name}"] = price
    return out


def _period_ohlc(daily: pd.DataFrame, start: date, end: date) -> Optional[dict]:
    """H/L/C for daily bars within [start, end] inclusive, or None if empty."""
    window = daily[(daily.index.date >= start) & (daily.index.date <= end)]
    if window.empty:
        return None
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "high": float(window["high"].max()),
        "low": float(window["low"].min()),
        "close": float(window["close"].iloc[-1]),
    }


def _prior_week_bounds(today: date, weeks_back: int = 1) -> tuple:
    """Wed-Tue bounds of the week `weeks_back` weeks before the current one —
    NIFTY/SENSEX's trading week runs Wednesday -> following Tuesday (matching
    the weekly options expiry cycle), NOT the calendar Mon-Fri week."""
    days_since_wed = (today.weekday() - 2) % 7  # Mon=0..Sun=6, Wed=2
    this_week_wed = today - timedelta(days=days_since_wed)
    start = this_week_wed - timedelta(days=7 * weeks_back)
    end = start + timedelta(days=6)  # Wed + 6 days = following Tue
    return start, end


def _prior_month_bounds(today: date, months_back: int = 1) -> tuple:
    """First/last calendar day of the month `months_back` months before this one."""
    y, m = today.year, today.month
    for _ in range(months_back):
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    start = date(y, m, 1)
    if m == 12:
        end = date(y, 12, 31)
    else:
        end = date(y, m + 1, 1) - timedelta(days=1)
    return start, end


async def compute_multi_timeframe_levels(
    session, symbol: str, spot: Optional[float] = None
) -> dict:
    daily = await load_daily_ohlc(session, symbol)
    if daily.empty:
        return {"symbol": symbol, "status": "no_data", "notes": "No daily candle history available yet."}

    if spot is None:
        spot = float(daily["close"].iloc[-1])

    today = date.today()
    level_sources: dict[str, float] = {}

    # --- Prior completed week ---
    w_start, w_end = _prior_week_bounds(today, 1)
    prior_week = _period_ohlc(daily, w_start, w_end)
    weekly_block = None
    if prior_week:
        weekly_block = {**prior_week, **_hlc_levels(prior_week["high"], prior_week["low"], prior_week["close"], "weekly")}
        for k, v in weekly_block.items():
            if k not in ("start", "end"):
                level_sources[f"weekly_{k}"] = v

    # --- Prior completed month ---
    m_start, m_end = _prior_month_bounds(today, 1)
    prior_month = _period_ohlc(daily, m_start, m_end)
    monthly_block = None
    if prior_month:
        monthly_block = {**prior_month, **_hlc_levels(prior_month["high"], prior_month["low"], prior_month["close"], "monthly")}
        for k, v in monthly_block.items():
            if k not in ("start", "end"):
                level_sources[f"monthly_{k}"] = v

    # --- Recent weekly swing highs/lows (extra structural context) ---
    for i in range(2, SWING_LOOKBACK_WEEKS + 1):
        s, e = _prior_week_bounds(today, i)
        wk = _period_ohlc(daily, s, e)
        if wk:
            level_sources[f"week_swing_high_{i}"] = wk["high"]
            level_sources[f"week_swing_low_{i}"] = wk["low"]

    # --- Recent monthly swing highs/lows ---
    for i in range(2, SWING_LOOKBACK_MONTHS + 1):
        s, e = _prior_month_bounds(today, i)
        mo = _period_ohlc(daily, s, e)
        if mo:
            level_sources[f"month_swing_high_{i}"] = mo["high"]
            level_sources[f"month_swing_low_{i}"] = mo["low"]

    # --- Round numbers near spot ---
    # Bigger step than the intraday Condor Setup's round levels (which use the
    # strike interval, e.g. 100) — for weekly/monthly planning, round numbers
    # in steps of 500 are the ones that actually get psychological attention.
    for rl in round_levels(spot, step=500, span=4):
        level_sources[f"round_{int(rl)}"] = rl

    if not level_sources:
        return {"symbol": symbol, "status": "no_data", "notes": "Not enough history yet for a completed prior week/month."}

    zones = build_confluence_levels(level_sources, tolerance_pct=CONFLUENCE_TOLERANCE_PCT)
    zones_out = [
        {"price": round(z.price, 2), "confidence": z.confidence, "sources": z.sources}
        for z in zones
    ]
    resistance_zones = [z for z in zones_out if z["price"] > spot]
    support_zones = [z for z in zones_out if z["price"] < spot]

    return {
        "symbol": symbol,
        "status": "ok",
        "spot": spot,
        "weekly": weekly_block,
        "monthly": monthly_block,
        "zones": zones_out,
        "nearest_resistance_zone": min(resistance_zones, key=lambda z: z["price"]) if resistance_zones else None,
        "nearest_support_zone": max(support_zones, key=lambda z: z["price"]) if support_zones else None,
    }
