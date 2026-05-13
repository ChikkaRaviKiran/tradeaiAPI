"""FastAPI routes for the Pattern Engine UI.

Mounted onto the main FastAPI app via:
    from app.pattern_engine.routes import register_routes
    register_routes(app)

All endpoints prefixed with /api/pattern-engine.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import desc, func, select

from app.db.models import AsyncSessionLocal
from app.pattern_engine.db_models import (
    PELiveProbe,
    PEPattern,
    PEPatternOccurrence,
    PEPatternStats,
)
from app.pattern_engine.matcher import evaluate_live, mini_sim
from app.pattern_engine.seed import upsert_seed_patterns
from app.pattern_engine.stats import refresh_pattern_stats

logger = logging.getLogger(__name__)


class StatusUpdate(BaseModel):
    status: str  # research / shadow / live / paused / retired


class SizeUpdate(BaseModel):
    size_multiplier: float


def _latest_stats_subquery():
    """Subquery returning latest computed_at per (pattern_id, window)."""
    return (
        select(
            PEPatternStats.pattern_id,
            PEPatternStats.window,
            func.max(PEPatternStats.computed_at).label("max_ts"),
        )
        .group_by(PEPatternStats.pattern_id, PEPatternStats.window)
        .subquery()
    )


def register_routes(app: FastAPI) -> None:
    """Attach pattern-engine routes to the existing FastAPI app."""

    @app.get("/api/pattern-engine/health")
    async def pe_health():
        async with AsyncSessionLocal() as s:
            n_patterns = (await s.execute(select(func.count(PEPattern.pattern_id)))).scalar() or 0
            n_occ = (await s.execute(select(func.count(PEPatternOccurrence.id)))).scalar() or 0
            n_probes = (await s.execute(select(func.count(PELiveProbe.id)))).scalar() or 0
            last_occ = (
                await s.execute(select(func.max(PEPatternOccurrence.ts)))
            ).scalar()
            return {
                "patterns": n_patterns,
                "occurrences": n_occ,
                "live_probes": n_probes,
                "last_occurrence_ts": last_occ.isoformat() if last_occ else None,
            }

    @app.get("/api/pattern-engine/patterns")
    async def list_patterns():
        """List all patterns with their latest 'all'-window stats."""
        async with AsyncSessionLocal() as s:
            patterns = (await s.execute(select(PEPattern))).scalars().all()
            sub = _latest_stats_subquery()
            stmt = (
                select(PEPatternStats)
                .join(
                    sub,
                    (PEPatternStats.pattern_id == sub.c.pattern_id)
                    & (PEPatternStats.window == sub.c.window)
                    & (PEPatternStats.computed_at == sub.c.max_ts),
                )
                .where(PEPatternStats.window == "all")
            )
            stats_rows = (await s.execute(stmt)).scalars().all()
            stats_by_id = {r.pattern_id: r for r in stats_rows}

            sub30 = _latest_stats_subquery()
            stmt30 = (
                select(PEPatternStats)
                .join(
                    sub30,
                    (PEPatternStats.pattern_id == sub30.c.pattern_id)
                    & (PEPatternStats.window == sub30.c.window)
                    & (PEPatternStats.computed_at == sub30.c.max_ts),
                )
                .where(PEPatternStats.window == "30d")
            )
            stats30 = {r.pattern_id: r for r in (await s.execute(stmt30)).scalars().all()}

            out = []
            for p in patterns:
                st_all = stats_by_id.get(p.pattern_id)
                st_30 = stats30.get(p.pattern_id)
                out.append({
                    "pattern_id": p.pattern_id,
                    "name": p.name,
                    "tier": p.tier,
                    "direction": p.direction,
                    "description": p.description,
                    "status": p.status,
                    "size_multiplier": p.size_multiplier,
                    "stats_all": _stats_dict(st_all),
                    "stats_30d": _stats_dict(st_30),
                    "trigger": p.trigger_json,
                    "exit_rule": p.exit_rule_json,
                })
            return out

    @app.get("/api/pattern-engine/patterns/{pattern_id}")
    async def pattern_detail(pattern_id: str):
        async with AsyncSessionLocal() as s:
            p = (
                await s.execute(select(PEPattern).where(PEPattern.pattern_id == pattern_id))
            ).scalar_one_or_none()
            if not p:
                raise HTTPException(404, "Pattern not found")

            # All windows of stats
            sub = _latest_stats_subquery()
            stmt = (
                select(PEPatternStats)
                .join(
                    sub,
                    (PEPatternStats.pattern_id == sub.c.pattern_id)
                    & (PEPatternStats.window == sub.c.window)
                    & (PEPatternStats.computed_at == sub.c.max_ts),
                )
                .where(PEPatternStats.pattern_id == pattern_id)
            )
            stats = {r.window: _stats_dict(r) for r in (await s.execute(stmt)).scalars().all()}

            # Recent occurrences (last 50)
            occ_stmt = (
                select(PEPatternOccurrence)
                .where(PEPatternOccurrence.pattern_id == pattern_id)
                .order_by(desc(PEPatternOccurrence.ts))
                .limit(100)
            )
            occs = (await s.execute(occ_stmt)).scalars().all()

            # Equity curve (chronological)
            equity_curve = []
            cum = 0.0
            for r in sorted(occs, key=lambda x: x.ts):
                cum += r.outcome_pnl_pct or 0.0
                equity_curve.append({"ts": r.ts.isoformat(), "cum_pnl_pct": round(cum, 2)})

            # Win rate by time-bucket (from features_json)
            tb_buckets: dict[str, list[float]] = defaultdict(list)
            regime_buckets: dict[str, list[float]] = defaultdict(list)
            for r in occs:
                pnl = r.outcome_pnl_pct or 0.0
                if r.features_json:
                    tb = (r.features_json or {}).get("time_bucket", "unknown")
                    tb_buckets[tb].append(pnl)
                regime_buckets[r.regime_at_entry or "unknown"].append(pnl)

            def _agg(group: dict[str, list[float]]) -> list[dict]:
                rows = []
                for k, v in group.items():
                    wins = sum(1 for x in v if x > 0)
                    rows.append({
                        "key": k, "n": len(v),
                        "wr": (wins / len(v)) if v else 0,
                        "avg_pnl": (sum(v) / len(v)) if v else 0,
                    })
                rows.sort(key=lambda x: x["key"])
                return rows

            return {
                "pattern": {
                    "pattern_id": p.pattern_id,
                    "name": p.name,
                    "tier": p.tier,
                    "direction": p.direction,
                    "description": p.description,
                    "status": p.status,
                    "size_multiplier": p.size_multiplier,
                    "trigger": p.trigger_json,
                    "exit_rule": p.exit_rule_json,
                    "notes": p.notes,
                },
                "stats": stats,
                "recent_occurrences": [
                    {
                        "ts": r.ts.isoformat(),
                        "direction": r.direction,
                        "spot_at_entry": r.spot_at_entry,
                        "outcome_pnl_pct": r.outcome_pnl_pct,
                        "outcome_spot_pts": r.outcome_spot_pts,
                        "hold_minutes": r.hold_minutes,
                        "exit_reason": r.exit_reason,
                        "mae_pct": r.mae_pct,
                        "mfe_pct": r.mfe_pct,
                        "regime": r.regime_at_entry,
                    }
                    for r in occs[:50]
                ],
                "equity_curve": equity_curve,
                "by_time_bucket": _agg(tb_buckets),
                "by_regime": _agg(regime_buckets),
            }

    @app.post("/api/pattern-engine/patterns/{pattern_id}/status")
    async def set_pattern_status(pattern_id: str, body: StatusUpdate):
        valid = {"research", "shadow", "live", "paused", "retired"}
        if body.status not in valid:
            raise HTTPException(400, f"status must be one of {valid}")
        async with AsyncSessionLocal() as s:
            p = (
                await s.execute(select(PEPattern).where(PEPattern.pattern_id == pattern_id))
            ).scalar_one_or_none()
            if not p:
                raise HTTPException(404, "Pattern not found")
            p.status = body.status
            await s.commit()
            return {"ok": True, "pattern_id": pattern_id, "status": body.status}

    @app.post("/api/pattern-engine/patterns/{pattern_id}/size")
    async def set_pattern_size(pattern_id: str, body: SizeUpdate):
        if body.size_multiplier < 0 or body.size_multiplier > 5:
            raise HTTPException(400, "size_multiplier must be in [0, 5]")
        async with AsyncSessionLocal() as s:
            p = (
                await s.execute(select(PEPattern).where(PEPattern.pattern_id == pattern_id))
            ).scalar_one_or_none()
            if not p:
                raise HTTPException(404, "Pattern not found")
            p.size_multiplier = body.size_multiplier
            await s.commit()
            return {"ok": True, "pattern_id": pattern_id, "size_multiplier": body.size_multiplier}

    @app.get("/api/pattern-engine/live")
    async def live_state(symbol: str = "NIFTY"):
        """Run live evaluation right now and return current matches."""
        async with AsyncSessionLocal() as s:
            try:
                matches = await evaluate_live(s, symbol=symbol)
                return {"ts": datetime.utcnow().isoformat(), "matches": matches}
            except Exception as e:
                logger.exception("live eval failed: %s", e)
                return {"ts": datetime.utcnow().isoformat(), "matches": [], "error": str(e)}

    @app.get("/api/pattern-engine/performance")
    async def performance(days: int = 30):
        """Aggregate PnL stats across all patterns over the last N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        async with AsyncSessionLocal() as s:
            rows = (
                await s.execute(
                    select(PEPatternOccurrence).where(PEPatternOccurrence.ts >= cutoff)
                )
            ).scalars().all()

            if not rows:
                return _empty_perf(days)

            pnls = [r.outcome_pnl_pct or 0.0 for r in rows]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            n = len(pnls)
            pf = (sum(wins) / -sum(losses)) if losses and sum(losses) != 0 else (sum(wins) if wins else 0)

            # By week
            weekly: dict[str, list[float]] = defaultdict(list)
            daily: dict[str, list[float]] = defaultdict(list)
            monthly: dict[str, list[float]] = defaultdict(list)
            by_pattern: dict[str, list[float]] = defaultdict(list)
            for r in rows:
                ts = r.ts
                pnl = r.outcome_pnl_pct or 0.0
                week_start = (ts - timedelta(days=ts.weekday())).strftime("%Y-%m-%d")
                weekly[week_start].append(pnl)
                daily[ts.strftime("%Y-%m-%d")].append(pnl)
                monthly[ts.strftime("%Y-%m")].append(pnl)
                by_pattern[r.pattern_id].append(pnl)

            def _series(group: dict[str, list[float]]) -> list[dict]:
                out = []
                for k, v in sorted(group.items()):
                    w = sum(1 for x in v if x > 0)
                    out.append({
                        "key": k, "n": len(v),
                        "pnl": round(sum(v), 2),
                        "wr": round(w / len(v), 3) if v else 0,
                    })
                return out

            cum_eq, cum, peak, mdd = [], 0.0, 0.0, 0.0
            for r in sorted(rows, key=lambda x: x.ts):
                cum += r.outcome_pnl_pct or 0.0
                peak = max(peak, cum)
                mdd = min(mdd, cum - peak)
                cum_eq.append({"ts": r.ts.isoformat(), "cum_pnl_pct": round(cum, 2)})

            pattern_summary = []
            for pid, vals in by_pattern.items():
                w = sum(1 for x in vals if x > 0)
                pattern_summary.append({
                    "pattern_id": pid,
                    "n": len(vals),
                    "wins": w,
                    "losses": len(vals) - w,
                    "wr": round(w / len(vals), 3) if vals else 0,
                    "pnl_pct": round(sum(vals), 2),
                    "avg_pnl_pct": round(sum(vals) / len(vals), 2) if vals else 0,
                })
            pattern_summary.sort(key=lambda x: x["pnl_pct"], reverse=True)

            return {
                "window_days": days,
                "n_trades": n,
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(len(wins) / n, 3) if n else 0,
                "profit_factor": round(float(pf), 3),
                "expectancy_pct": round(sum(pnls) / n, 3) if n else 0,
                "total_pnl_pct": round(sum(pnls), 2),
                "avg_win_pct": round(sum(wins) / len(wins), 2) if wins else 0,
                "avg_loss_pct": round(sum(losses) / len(losses), 2) if losses else 0,
                "max_drawdown_pct": round(mdd, 2),
                "daily": _series(daily),
                "weekly": _series(weekly),
                "monthly": _series(monthly),
                "by_pattern": pattern_summary,
                "equity_curve": cum_eq,
            }

    @app.get("/api/pattern-engine/probes")
    async def list_probes(limit: int = 100):
        async with AsyncSessionLocal() as s:
            rows = (
                await s.execute(
                    select(PELiveProbe)
                    .order_by(desc(PELiveProbe.ts))
                    .limit(limit)
                )
            ).scalars().all()
            return [
                {
                    "ts": r.ts.isoformat(),
                    "pattern_id": r.pattern_id,
                    "symbol": r.symbol,
                    "edge_score": r.edge_score,
                    "minisim_n": r.minisim_n,
                    "minisim_wr": r.minisim_wr,
                    "minisim_pf": r.minisim_pf,
                    "decision": r.decision,
                    "skip_reason": r.skip_reason,
                }
                for r in rows
            ]

    @app.post("/api/pattern-engine/admin/refresh-stats")
    async def admin_refresh_stats():
        async with AsyncSessionLocal() as s:
            written = await refresh_pattern_stats(s)
            return {"ok": True, "stat_rows_written": written}

    @app.post("/api/pattern-engine/admin/seed")
    async def admin_seed_patterns():
        async with AsyncSessionLocal() as s:
            inserted = await upsert_seed_patterns(s)
            return {"ok": True, "inserted": inserted}

    @app.get("/api/pattern-engine/scheduler/status")
    async def scheduler_status():
        from app.pattern_engine.scheduler import get_scheduler_status
        return get_scheduler_status()

    @app.post("/api/pattern-engine/scheduler/run-now")
    async def scheduler_run_now():
        from app.pattern_engine.scheduler import trigger_now
        return await trigger_now()

    logger.info("pattern_engine: routes registered under /api/pattern-engine/*")


def _empty_perf(days: int) -> dict:
    return {
        "window_days": days, "n_trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0, "profit_factor": 0, "expectancy_pct": 0,
        "total_pnl_pct": 0, "avg_win_pct": 0, "avg_loss_pct": 0,
        "max_drawdown_pct": 0,
        "daily": [], "weekly": [], "monthly": [],
        "by_pattern": [], "equity_curve": [],
    }


def _stats_dict(r) -> dict | None:
    if not r:
        return None
    return {
        "window": r.window,
        "computed_at": r.computed_at.isoformat() if r.computed_at else None,
        "n_trades": r.n_trades,
        "wins": r.wins,
        "losses": r.losses,
        "win_rate": round(r.win_rate or 0, 3),
        "profit_factor": round(r.profit_factor or 0, 3),
        "expectancy_pct": round(r.expectancy_pct or 0, 3),
        "avg_win_pct": round(r.avg_win_pct or 0, 2),
        "avg_loss_pct": round(r.avg_loss_pct or 0, 2),
        "avg_hold_min": round(r.avg_hold_min or 0, 1),
        "avg_mae_pct": round(r.avg_mae_pct or 0, 2),
        "avg_mfe_pct": round(r.avg_mfe_pct or 0, 2),
        "total_pnl_pct": round(r.total_pnl_pct or 0, 2),
        "max_drawdown_pct": round(r.max_drawdown_pct or 0, 2),
        "monthly_pnl": r.monthly_pnl_json or {},
        "months_profitable": r.months_profitable or 0,
        "months_total": r.months_total or 0,
        "suggested_tier": r.suggested_tier,
    }
