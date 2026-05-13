"""Stats refresh + auto-tier classification.

Computes per-pattern rolling stats (all / 180d / 90d / 30d) from
pe_pattern_occurrences and writes a fresh row to pe_pattern_stats.

Tier rules (matches the design discussion):
  Tier S: n>=50, wr>=55, pf>=2.0, exp>=1.5, profitable in >=5/6 months
  Tier A: n>=40, wr>=50, pf>=1.6, exp>=1.0, profitable in >=4/6 months
  Tier B: n>=30, wr>=45, pf>=1.4, exp>=0.6, profitable in >=3/6 months
  REJECT: anything else
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select

from app.pattern_engine.db_models import (
    PEPattern,
    PEPatternOccurrence,
    PEPatternStats,
)

logger = logging.getLogger(__name__)

WINDOWS = {
    "all": None,
    "180d": 180,
    "90d": 90,
    "30d": 30,
}


def _classify_tier(s: dict) -> str:
    n = s["n_trades"]
    wr = s["win_rate"]
    pf = s["profit_factor"]
    exp_ = s["expectancy_pct"]
    months_ok = s["months_profitable"]
    months_total = max(1, s["months_total"])

    if n >= 50 and wr >= 0.55 and pf >= 2.0 and exp_ >= 1.5 and months_ok / months_total >= 5 / 6:
        return "S"
    if n >= 40 and wr >= 0.50 and pf >= 1.6 and exp_ >= 1.0 and months_ok / months_total >= 4 / 6:
        return "A"
    if n >= 30 and wr >= 0.45 and pf >= 1.4 and exp_ >= 0.6 and months_ok / months_total >= 3 / 6:
        return "B"
    return "REJECT"


def _compute_stats(rows: list[PEPatternOccurrence]) -> dict:
    if not rows:
        return {
            "n_trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "profit_factor": 0.0, "expectancy_pct": 0.0,
            "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "avg_hold_min": 0.0, "avg_mae_pct": 0.0, "avg_mfe_pct": 0.0,
            "total_pnl_pct": 0.0, "max_drawdown_pct": 0.0,
            "monthly_pnl": {}, "months_profitable": 0, "months_total": 0,
        }
    pnls = [r.outcome_pnl_pct or 0.0 for r in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n = len(pnls)
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    pf = (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)

    # Drawdown on cumulative PnL
    cum, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    # Monthly buckets
    monthly = defaultdict(float)
    for r in rows:
        if r.ts:
            monthly[r.ts.strftime("%Y-%m")] += (r.outcome_pnl_pct or 0.0)

    months_profitable = sum(1 for v in monthly.values() if v > 0)

    holds = [r.hold_minutes for r in rows if r.hold_minutes is not None]
    maes = [r.mae_pct for r in rows if r.mae_pct is not None]
    mfes = [r.mfe_pct for r in rows if r.mfe_pct is not None]

    return {
        "n_trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / n if n else 0.0,
        "profit_factor": float(pf),
        "expectancy_pct": sum(pnls) / n if n else 0.0,
        "avg_win_pct": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss_pct": (sum(losses) / len(losses)) if losses else 0.0,
        "avg_hold_min": (sum(holds) / len(holds)) if holds else 0.0,
        "avg_mae_pct": (sum(maes) / len(maes)) if maes else 0.0,
        "avg_mfe_pct": (sum(mfes) / len(mfes)) if mfes else 0.0,
        "total_pnl_pct": sum(pnls),
        "max_drawdown_pct": max_dd,
        "monthly_pnl": dict(monthly),
        "months_profitable": months_profitable,
        "months_total": len(monthly),
    }


async def refresh_pattern_stats(session, pattern_id: Optional[str] = None) -> int:
    """Recompute stats for one or all patterns. Returns number of stat rows written."""
    if pattern_id:
        patterns = [
            (await session.execute(select(PEPattern).where(PEPattern.pattern_id == pattern_id))).scalar_one_or_none()
        ]
        patterns = [p for p in patterns if p]
    else:
        patterns = (await session.execute(select(PEPattern))).scalars().all()

    now = datetime.utcnow()
    written = 0

    for p in patterns:
        # Load all occurrences (we'll filter by window in memory)
        stmt = select(PEPatternOccurrence).where(
            PEPatternOccurrence.pattern_id == p.pattern_id
        )
        all_rows = (await session.execute(stmt)).scalars().all()

        for window, days in WINDOWS.items():
            if days is None:
                rows = all_rows
            else:
                cutoff = now - timedelta(days=days)
                rows = [r for r in all_rows if r.ts and r.ts >= cutoff]

            stats = _compute_stats(rows)
            tier = _classify_tier(stats) if rows else "REJECT"

            session.add(
                PEPatternStats(
                    pattern_id=p.pattern_id,
                    computed_at=now,
                    window=window,
                    n_trades=stats["n_trades"],
                    wins=stats["wins"],
                    losses=stats["losses"],
                    win_rate=stats["win_rate"],
                    profit_factor=stats["profit_factor"],
                    expectancy_pct=stats["expectancy_pct"],
                    avg_win_pct=stats["avg_win_pct"],
                    avg_loss_pct=stats["avg_loss_pct"],
                    avg_hold_min=stats["avg_hold_min"],
                    avg_mae_pct=stats["avg_mae_pct"],
                    avg_mfe_pct=stats["avg_mfe_pct"],
                    total_pnl_pct=stats["total_pnl_pct"],
                    max_drawdown_pct=stats["max_drawdown_pct"],
                    monthly_pnl_json=stats["monthly_pnl"],
                    months_profitable=stats["months_profitable"],
                    months_total=stats["months_total"],
                    suggested_tier=tier,
                )
            )
            written += 1

    await session.commit()
    logger.info("pattern_engine: wrote %d stat rows for %d patterns", written, len(patterns))
    return written
