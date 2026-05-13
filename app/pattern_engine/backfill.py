"""Backfill engine — replays historical candles, generates snapshots,
evaluates seed patterns, simulates outcomes, persists everything.

Run via:
    python -m app.pattern_engine.backfill --days 540 --symbol NIFTY
    python -m app.pattern_engine.backfill --start 2024-11-01 --end 2026-05-13

Idempotent: skips dates that already have snapshots, can be re-run safely
after schema/pattern changes (use --reset-occurrences to clear and rebuild).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import pytz
from sqlalchemy import delete, distinct, select

from app.db.models import AsyncSessionLocal, IndexCandle, OptionCandle, init_db
from app.pattern_engine.db_models import (
    PEMarketSnapshot,
    PEPattern,
    PEPatternOccurrence,
)
from app.pattern_engine.dsl import evaluate_trigger, normalize_exit
from app.pattern_engine.features import compute_snapshot
from app.pattern_engine.seed import upsert_seed_patterns

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

# Snapshot cadence (5-min) — every 5 minutes from 09:20 to 15:25
SNAPSHOT_MINUTES = list(range(20, 60, 5)) + list(range(0, 60, 5)) * 6  # placeholder
SNAPSHOT_TIMES = []
for hour in range(9, 16):
    for m in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55):
        if hour == 9 and m < 20:
            continue
        if hour == 15 and m > 25:
            continue
        SNAPSHOT_TIMES.append((hour, m))


async def _get_trading_dates(session, symbol: str, start: date, end: date) -> list[str]:
    stmt = (
        select(distinct(IndexCandle.date))
        .where(
            IndexCandle.instrument == symbol,
            IndexCandle.date >= start.strftime("%Y-%m-%d"),
            IndexCandle.date <= end.strftime("%Y-%m-%d"),
        )
        .order_by(IndexCandle.date.asc())
    )
    rows = (await session.execute(stmt)).scalars().all()
    return list(rows)


async def _date_already_processed(session, symbol: str, target_date: str) -> bool:
    stmt = (
        select(PEMarketSnapshot.id)
        .where(
            PEMarketSnapshot.symbol == symbol,
            PEMarketSnapshot.ts >= datetime.strptime(target_date, "%Y-%m-%d"),
            PEMarketSnapshot.ts < datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=1),
        )
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none() is not None


async def _load_full_day_candles(session, symbol: str, target_date: str) -> pd.DataFrame:
    stmt = (
        select(IndexCandle)
        .where(IndexCandle.instrument == symbol, IndexCandle.date == target_date)
        .order_by(IndexCandle.timestamp.asc())
    )
    rows = (await session.execute(stmt)).scalars().all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        [
            {
                "timestamp": r.timestamp,
                "open": r.open, "high": r.high, "low": r.low, "close": r.close,
                "volume": r.volume or 0,
            }
            for r in rows
        ]
    )
    df.set_index("timestamp", inplace=True)
    return df


def _simulate_outcome(
    entry_ts: datetime,
    direction: str,
    entry_spot: float,
    full_day_df: pd.DataFrame,
    exit_rule: dict,
) -> dict:
    """Spot-based proxy outcome simulator (no option premium needed for backfill).

    Maps spot move % to a synthetic option premium move using a fixed leverage
    factor — good enough for relative ranking of patterns. Real-trade exits
    will use actual premium. Convention: CE wins on spot up, PE wins on spot down.
    """
    sl_pct = exit_rule.get("sl_pct", 25)
    target_pct = exit_rule.get("target_pct", 50)
    time_stop_min = exit_rule.get("time_stop_min", 60)
    leverage = 8.0  # 1% spot move ~ 8% premium move (rough ATM CE/PE)

    sub = full_day_df.loc[full_day_df.index > entry_ts]
    if sub.empty:
        return {
            "exit_premium": None, "outcome_pnl_pct": 0.0,
            "outcome_spot_pts": 0.0, "hold_minutes": 0,
            "exit_reason": "no_data", "mae_pct": 0.0, "mfe_pct": 0.0,
        }

    sign = 1.0 if direction == "CE" else -1.0
    end_ts = entry_ts + timedelta(minutes=time_stop_min)
    eod_ts = entry_ts.replace(hour=15, minute=20, second=0, microsecond=0)
    horizon = sub.loc[sub.index <= min(end_ts, eod_ts)]
    if horizon.empty:
        horizon = sub.iloc[: min(len(sub), 12)]  # at least try a few candles

    mae_pct = 0.0  # most adverse premium %
    mfe_pct = 0.0
    exit_reason = "time_stop"
    hit_ts = None

    for ts, row in horizon.iterrows():
        # Worst-case during candle = the side most against us
        adverse_spot = row["low"] if direction == "CE" else row["high"]
        favorable_spot = row["high"] if direction == "CE" else row["low"]

        adverse_prem = sign * (adverse_spot - entry_spot) / entry_spot * 100 * leverage
        favorable_prem = sign * (favorable_spot - entry_spot) / entry_spot * 100 * leverage

        mae_pct = min(mae_pct, adverse_prem)
        mfe_pct = max(mfe_pct, favorable_prem)

        # Stop loss hit?
        if adverse_prem <= -sl_pct:
            exit_reason = "stop_loss"
            hit_ts = ts
            outcome = -sl_pct
            break
        # Target hit?
        if favorable_prem >= target_pct:
            exit_reason = "target"
            hit_ts = ts
            outcome = target_pct
            break
    else:
        # Time stop — exit at last horizon close
        last_close = horizon["close"].iloc[-1]
        outcome = sign * (last_close - entry_spot) / entry_spot * 100 * leverage
        hit_ts = horizon.index[-1]

    hold_minutes = int((hit_ts - entry_ts).total_seconds() / 60) if hit_ts else 0
    spot_at_exit = horizon.loc[hit_ts]["close"] if hit_ts is not None else entry_spot
    outcome_spot_pts = float(sign * (spot_at_exit - entry_spot))

    return {
        "exit_premium": None,
        "outcome_pnl_pct": float(outcome),
        "outcome_spot_pts": outcome_spot_pts,
        "hold_minutes": hold_minutes,
        "exit_reason": exit_reason,
        "mae_pct": float(mae_pct),
        "mfe_pct": float(mfe_pct),
    }


async def backfill_date(
    session,
    symbol: str,
    target_date: str,
    force: bool = False,
    only_pattern_ids: Optional[set[str]] = None,
    skip_snapshots: bool = False,
) -> dict:
    """Generate snapshots + occurrences for a single date.

    - only_pattern_ids: if given, only those patterns are evaluated (used for
      single-pattern re-backfill from the UI)
    - skip_snapshots: if True, snapshots are not re-written (use during single-
      pattern re-backfill since snapshots already exist)

    Returns counters {snapshots, occurrences} for logging.
    """
    if not skip_snapshots and not force and await _date_already_processed(session, symbol, target_date):
        return {"snapshots": 0, "occurrences": 0, "skipped": True}

    full_day = await _load_full_day_candles(session, symbol, target_date)
    if full_day.empty:
        return {"snapshots": 0, "occurrences": 0, "no_data": True}

    # Load patterns from DB (so UI tweaks are honored)
    db_patterns = (await session.execute(select(PEPattern))).scalars().all()
    if only_pattern_ids:
        db_patterns = [p for p in db_patterns if p.pattern_id in only_pattern_ids]

    snap_count = 0
    occ_count = 0
    seen_pattern_today: set[str] = set()  # one entry per pattern per day

    for hour, minute in SNAPSHOT_TIMES:
        ts = datetime.strptime(target_date, "%Y-%m-%d").replace(hour=hour, minute=minute)
        if not (full_day.index <= ts).any():
            continue

        snap = await compute_snapshot(session, symbol, ts)
        if snap is None:
            continue

        if not skip_snapshots:
            session.add(
                PEMarketSnapshot(
                    **{k: v for k, v in snap.to_dict().items() if k != "ts" and k != "symbol"},
                    ts=ts,
                    symbol=symbol,
                )
            )
            snap_count += 1

        snap_dict = snap.to_dict()

        for p in db_patterns:
            if p.pattern_id in seen_pattern_today:
                continue
            try:
                if not evaluate_trigger(p.trigger_json, snap_dict):
                    continue
            except Exception as e:
                logger.debug("trigger %s eval error at %s: %s", p.pattern_id, ts, e)
                continue

            seen_pattern_today.add(p.pattern_id)
            outcome = _simulate_outcome(
                ts, p.direction, snap.spot, full_day, normalize_exit(p.exit_rule_json)
            )

            session.add(
                PEPatternOccurrence(
                    pattern_id=p.pattern_id,
                    ts=ts,
                    symbol=symbol,
                    direction=p.direction,
                    spot_at_entry=snap.spot,
                    strike=round(snap.spot / 50) * 50,
                    entry_premium=None,
                    exit_premium=outcome["exit_premium"],
                    outcome_pnl_pct=outcome["outcome_pnl_pct"],
                    outcome_spot_pts=outcome["outcome_spot_pts"],
                    hold_minutes=outcome["hold_minutes"],
                    exit_reason=outcome["exit_reason"],
                    mae_pct=outcome["mae_pct"],
                    mfe_pct=outcome["mfe_pct"],
                    regime_at_entry=snap.regime,
                    source="backfill",
                    features_json=snap_dict,
                )
            )
            occ_count += 1

    await session.commit()
    return {"snapshots": snap_count, "occurrences": occ_count}


async def run_backfill(
    symbol: str,
    start: Optional[date],
    end: Optional[date],
    days: Optional[int],
    reset_occurrences: bool = False,
    reset_snapshots: bool = False,
    only_pattern_id: Optional[str] = None,
) -> dict:
    """Execute backfill. Returns totals dict."""
    await init_db()

    if days and not start:
        end = end or date.today()
        start = end - timedelta(days=days)
    if not start or not end:
        raise SystemExit("Provide either --days or both --start and --end")

    async with AsyncSessionLocal() as session:
        await upsert_seed_patterns(session)

        if reset_occurrences:
            stmt = delete(PEPatternOccurrence).where(PEPatternOccurrence.source == "backfill")
            if only_pattern_id:
                stmt = stmt.where(PEPatternOccurrence.pattern_id == only_pattern_id)
            await session.execute(stmt)
            logger.warning("Deleted existing backfill occurrences (pattern=%s)", only_pattern_id or "ALL")
        if reset_snapshots:
            await session.execute(delete(PEMarketSnapshot))
            logger.warning("Deleted existing snapshots")
        if reset_occurrences or reset_snapshots:
            await session.commit()

        dates = await _get_trading_dates(session, symbol, start, end)
        logger.info(
            "Backfill %s: %s → %s (%d trading days available)%s",
            symbol, start, end, len(dates),
            f" only_pattern={only_pattern_id}" if only_pattern_id else "",
        )

    only_pattern_ids = {only_pattern_id} if only_pattern_id else None
    skip_snapshots = bool(only_pattern_id)  # snapshots already exist for re-backfill

    total = {"snapshots": 0, "occurrences": 0, "days": 0, "skipped": 0}
    # Fresh session per day so a single-day failure can't poison the rest
    for d in dates:
        try:
            async with AsyncSessionLocal() as day_session:
                result = await backfill_date(
                    day_session, symbol, d,
                    force=reset_snapshots,
                    only_pattern_ids=only_pattern_ids,
                    skip_snapshots=skip_snapshots,
                )
        except Exception as e:
            logger.exception("Failed to backfill %s: %s", d, e)
            continue
        if result.get("skipped"):
            total["skipped"] += 1
            continue
        total["snapshots"] += result.get("snapshots", 0)
        total["occurrences"] += result.get("occurrences", 0)
        total["days"] += 1
        if total["days"] % 10 == 0:
            logger.info(
                "Progress: %d days done, %d snapshots, %d occurrences",
                total["days"], total["snapshots"], total["occurrences"],
            )

    logger.info("=" * 60)
    logger.info("Backfill complete: %s", total)
    return total


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pattern Engine historical backfill")
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--days", type=int, default=None, help="Days back from today")
    p.add_argument("--reset-occurrences", action="store_true",
                   help="Delete existing backfill occurrences before running")
    p.add_argument("--reset-snapshots", action="store_true",
                   help="Delete ALL existing snapshots before running (destructive)")
    return p.parse_args()


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None
    await run_backfill(
        args.symbol, start, end, args.days,
        reset_occurrences=args.reset_occurrences,
        reset_snapshots=args.reset_snapshots,
    )

    # Refresh stats once backfill is done
    from app.pattern_engine.stats import refresh_pattern_stats
    async with AsyncSessionLocal() as session:
        await refresh_pattern_stats(session)


if __name__ == "__main__":
    asyncio.run(_main())
