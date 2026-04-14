"""Strategy optimizer — builds recommendations and provides live query API.

Two modes:
  1. Build mode (analysis): analyze all trades → rank strategies per DayType × TimeWindow
  2. Query mode (live):     load recommendations → return active strategies for current conditions

Usage in live orchestrator:
    optimizer = StrategyOptimizer.load("data/strategy_recommendations.json")
    strategies = optimizer.get_active_strategies("NIFTY", DayType.TREND, dtime(10, 30))
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import time as dtime
from typing import List, Dict, Optional

import numpy as np

from app.core.models import DayType

logger = logging.getLogger(__name__)


@dataclass
class StrategyRecommendation:
    instrument: str
    day_type: str
    time_window: str
    strategy: str
    rank: int
    win_rate: float
    profit_factor: float
    avg_pnl: float
    sharpe: float
    total_trades: int
    score: float


class StrategyOptimizer:
    """Build and query strategy recommendations per DayType × TimeWindow."""

    MIN_TRADES = 5

    def __init__(self):
        self.recommendations: List[StrategyRecommendation] = []
        self._lookup: Dict = {}

    # ── Build from analysis trades ────────────────────────────────────

    def build_from_trades(self, trades: list, top_n: int = 2):
        """Analyze TradeResult objects and build recommendation table."""
        self.recommendations = []

        # Group by (instrument, day_type_hindsight, time_window, strategy)
        groups: Dict[tuple, list] = {}
        for t in trades:
            key = (t.instrument, t.day_type_hindsight, t.time_window, t.strategy)
            groups.setdefault(key, []).append(t)

        # For each (instrument, day_type, time_window): rank strategies
        combos: Dict[tuple, list] = {}
        for (inst, dt, tw, strat), strat_trades in groups.items():
            if len(strat_trades) < self.MIN_TRADES:
                continue
            metrics = self._compute_metrics(strat_trades)
            combo_key = (inst, dt, tw)
            combos.setdefault(combo_key, []).append((strat, metrics))

        for combo_key, strat_list in combos.items():
            inst, dt, tw = combo_key
            ranked = sorted(strat_list, key=lambda x: x[1]["score"], reverse=True)
            for rank, (strat, metrics) in enumerate(ranked[:top_n], 1):
                self.recommendations.append(StrategyRecommendation(
                    instrument=inst, day_type=dt, time_window=tw,
                    strategy=strat, rank=rank,
                    win_rate=metrics["win_rate"],
                    profit_factor=metrics["profit_factor"],
                    avg_pnl=metrics["avg_pnl"],
                    sharpe=metrics["sharpe"],
                    total_trades=metrics["total_trades"],
                    score=metrics["score"],
                ))

        self._build_lookup()
        logger.info("Built %d recommendations from %d trades", len(self.recommendations), len(trades))

    # ── Also build aggregated recommendations (across all time windows) ──

    def build_aggregated(self, trades: list, top_n: int = 2):
        """Build per-DayType recommendations (no time window split).

        Useful for the core question: which strategies for which day type?
        """
        agg_recs = []
        groups: Dict[tuple, list] = {}
        for t in trades:
            key = (t.instrument, t.day_type_hindsight, t.strategy)
            groups.setdefault(key, []).append(t)

        combos: Dict[tuple, list] = {}
        for (inst, dt, strat), strat_trades in groups.items():
            if len(strat_trades) < self.MIN_TRADES:
                continue
            metrics = self._compute_metrics(strat_trades)
            combos.setdefault((inst, dt), []).append((strat, metrics))

        for (inst, dt), strat_list in combos.items():
            ranked = sorted(strat_list, key=lambda x: x[1]["score"], reverse=True)
            for rank, (strat, metrics) in enumerate(ranked[:top_n], 1):
                agg_recs.append(StrategyRecommendation(
                    instrument=inst, day_type=dt, time_window="ALL",
                    strategy=strat, rank=rank,
                    win_rate=metrics["win_rate"],
                    profit_factor=metrics["profit_factor"],
                    avg_pnl=metrics["avg_pnl"],
                    sharpe=metrics["sharpe"],
                    total_trades=metrics["total_trades"],
                    score=metrics["score"],
                ))

        return agg_recs

    # ── Live query API ────────────────────────────────────────────────

    def get_active_strategies(
        self,
        instrument: str,
        day_type: DayType,
        current_time: dtime,
    ) -> List[str]:
        """Return recommended strategy names for current conditions.

        Called by live orchestrator at trading time.
        Falls back to aggregated (any time window) if specific window has no data.
        """
        from .engine import TimeWindow

        dt_str = day_type.value if hasattr(day_type, "value") else str(day_type)
        tw = TimeWindow.from_time(current_time)

        # Try exact match first
        recs = self._lookup.get((instrument, dt_str, tw), [])
        if recs:
            return [r.strategy for r in recs]

        # Fallback: any time window for this day type
        all_recs = [
            r for r in self.recommendations
            if r.instrument == instrument and r.day_type == dt_str
        ]
        if not all_recs:
            return []

        seen = set()
        result = []
        for r in sorted(all_recs, key=lambda x: x.score, reverse=True):
            if r.strategy not in seen:
                result.append(r.strategy)
                seen.add(r.strategy)
            if len(result) >= 2:
                break
        return result

    def get_best_time_windows(self, instrument: str, day_type_str: str, strategy: str) -> List[str]:
        """Get the best time windows for a specific strategy + day type combo."""
        recs = [
            r for r in self.recommendations
            if r.instrument == instrument and r.day_type == day_type_str
            and r.strategy == strategy and r.time_window != "ALL"
        ]
        return [r.time_window for r in sorted(recs, key=lambda x: x.score, reverse=True)]

    # ── Metrics computation ───────────────────────────────────────────

    @staticmethod
    def _compute_metrics(trades) -> dict:
        pnls = [t.pnl for t in trades]
        total = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        gp = sum(p for p in pnls if p > 0) or 0
        gl = abs(sum(p for p in pnls if p <= 0)) or 1

        win_rate = wins / total * 100 if total > 0 else 0
        profit_factor = gp / gl
        avg_pnl = float(np.mean(pnls)) if pnls else 0
        sharpe = (
            float(np.mean(pnls)) / float(np.std(pnls)) * np.sqrt(252)
            if len(pnls) > 1 and np.std(pnls) > 0
            else 0
        )

        # Composite score
        score = (
            min(win_rate, 100) * 0.30
            + min(profit_factor * 25, 100) * 0.30
            + min(max(sharpe, 0) * 20, 100) * 0.20
            + min(total / 2, 100) * 0.20
        )

        return {
            "total_trades": total,
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "avg_pnl": round(avg_pnl, 0),
            "sharpe": round(sharpe, 2),
            "score": round(score, 1),
        }

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, filepath: str):
        """Save recommendations to JSON file."""
        data = [asdict(r) for r in self.recommendations]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d recommendations to %s", len(data), filepath)

    @classmethod
    def load(cls, filepath: str) -> StrategyOptimizer:
        """Load recommendations from JSON file."""
        opt = cls()
        with open(filepath) as f:
            data = json.load(f)
        opt.recommendations = [StrategyRecommendation(**d) for d in data]
        opt._build_lookup()
        logger.info("Loaded %d recommendations from %s", len(opt.recommendations), filepath)
        return opt

    async def save_to_db(self, conn):
        """Save recommendations to PostgreSQL (asyncpg connection)."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_recommendations (
                id SERIAL PRIMARY KEY,
                instrument VARCHAR(20),
                day_type VARCHAR(20),
                time_window VARCHAR(30),
                strategy VARCHAR(50),
                rank INTEGER,
                win_rate FLOAT,
                profit_factor FLOAT,
                avg_pnl FLOAT,
                sharpe FLOAT,
                total_trades INTEGER,
                score FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("DELETE FROM strategy_recommendations")
        for rec in self.recommendations:
            await conn.execute(
                "INSERT INTO strategy_recommendations "
                "(instrument, day_type, time_window, strategy, rank, "
                "win_rate, profit_factor, avg_pnl, sharpe, total_trades, score) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
                rec.instrument, rec.day_type, rec.time_window, rec.strategy,
                rec.rank, rec.win_rate, rec.profit_factor, rec.avg_pnl,
                rec.sharpe, rec.total_trades, rec.score,
            )
        logger.info("Saved %d recommendations to DB", len(self.recommendations))

    @classmethod
    async def load_from_db(cls, conn, instrument: str = None) -> StrategyOptimizer:
        """Load recommendations from DB."""
        opt = cls()
        if instrument:
            rows = await conn.fetch(
                "SELECT * FROM strategy_recommendations WHERE instrument = $1 ORDER BY score DESC",
                instrument,
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM strategy_recommendations ORDER BY score DESC"
            )
        for row in rows:
            opt.recommendations.append(StrategyRecommendation(
                instrument=row["instrument"], day_type=row["day_type"],
                time_window=row["time_window"], strategy=row["strategy"],
                rank=row["rank"], win_rate=row["win_rate"],
                profit_factor=row["profit_factor"], avg_pnl=row["avg_pnl"],
                sharpe=row["sharpe"], total_trades=row["total_trades"],
                score=row["score"],
            ))
        opt._build_lookup()
        return opt

    def _build_lookup(self):
        self._lookup = {}
        for rec in self.recommendations:
            key = (rec.instrument, rec.day_type, rec.time_window)
            self._lookup.setdefault(key, []).append(rec)
