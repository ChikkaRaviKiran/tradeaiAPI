"""Live matcher + mini-sim gate.

Reads pattern definitions (trigger_json) FROM THE DB and evaluates them via
the JSON DSL. This means trigger tweaks made via the UI are picked up
immediately on the next live evaluation — no code change required.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select

from app.pattern_engine.db_models import (
    PELiveProbe,
    PEPattern,
    PEPatternOccurrence,
)
from app.pattern_engine.dsl import evaluate_trigger
from app.pattern_engine.features import FeatureSnapshot, compute_snapshot

logger = logging.getLogger(__name__)


async def mini_sim(
    session, pattern_id: str, days: int = 30
) -> dict:
    """Quick stats for the last `days` of occurrences for this pattern."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    rows = (
        await session.execute(
            select(PEPatternOccurrence).where(
                PEPatternOccurrence.pattern_id == pattern_id,
                PEPatternOccurrence.ts >= cutoff,
            )
        )
    ).scalars().all()

    if not rows:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "expectancy_pct": 0.0, "last_5_pnl": []}

    pnls = [r.outcome_pnl_pct or 0.0 for r in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = (sum(wins) / -sum(losses)) if losses and sum(losses) != 0 else (sum(wins) if wins else 0)

    last_5 = sorted(rows, key=lambda r: r.ts, reverse=True)[:5]
    last_5_pnl = [round(r.outcome_pnl_pct or 0.0, 2) for r in last_5]

    return {
        "n": len(rows),
        "wr": len(wins) / len(rows),
        "pf": float(pf),
        "expectancy_pct": sum(pnls) / len(rows),
        "last_5_pnl": last_5_pnl,
    }


def _passes_minisim(sim: dict, status: str) -> tuple[bool, str]:
    """Gate logic — different thresholds for live vs shadow."""
    if sim["n"] < 5:
        return True if status == "shadow" else False, "minisim_low_samples"
    if status == "live":
        if sim["wr"] < 0.45:
            return False, f"minisim_wr_low ({sim['wr']:.0%})"
        if sim["pf"] < 1.2:
            return False, f"minisim_pf_low ({sim['pf']:.2f})"
        last_5 = sim.get("last_5_pnl", [])
        if len(last_5) >= 5 and all(p < 0 for p in last_5):
            return False, "minisim_last_5_all_losses"
    return True, ""


async def evaluate_live(
    session, symbol: str = "NIFTY", at: Optional[datetime] = None
) -> list[dict]:
    """Evaluate all DB patterns at the given (or now) timestamp."""
    ts = at or datetime.now().replace(second=0, microsecond=0)
    snap = await compute_snapshot(session, symbol, ts)
    if snap is None:
        return []
    snap_dict = snap.to_dict()

    patterns = (await session.execute(select(PEPattern))).scalars().all()
    out: list[dict] = []

    for p in patterns:
        if p.status == "retired":
            continue
        try:
            if not evaluate_trigger(p.trigger_json, snap_dict):
                continue
        except Exception as e:
            logger.debug("trigger eval failed for %s: %s", p.pattern_id, e)
            continue

        sim = await mini_sim(session, p.pattern_id, days=30)
        passes, reason = _passes_minisim(sim, p.status)

        decision = "taken" if (passes and p.status == "live") else (
            "shadow_match" if p.status == "shadow" else (
                "skipped_minisim" if not passes else "skipped_status"
            )
        )

        session.add(PELiveProbe(
            ts=ts, pattern_id=p.pattern_id, symbol=symbol,
            edge_score=sim["expectancy_pct"],
            minisim_n=sim["n"], minisim_wr=sim["wr"], minisim_pf=sim["pf"],
            decision=decision, skip_reason=reason or None,
        ))

        out.append({
            "pattern_id": p.pattern_id,
            "name": p.name,
            "direction": p.direction,
            "status": p.status,
            "size_multiplier": p.size_multiplier,
            "minisim": sim,
            "decision": decision,
            "skip_reason": reason or None,
            "snapshot": snap_dict,
        })

    await session.commit()
    return out
