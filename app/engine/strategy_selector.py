"""Strategy Selector — picks the best strategies for today's market conditions.

Uses the StrategyConditionPerformance table (populated by the evaluator) to
select strategies whose historical win-rate and profitability are highest
under conditions matching today's market.

Called pre-market (08:00-09:15) by the orchestrator.  The selected strategy
list drives which strategies are allowed to generate signals during live
trading — the others are parked for the day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import DayType, MarketRegime

logger = logging.getLogger(__name__)

# Minimum trades a strategy needs under a condition to be considered
MIN_TRADES_FOR_SELECTION = 3
# Minimum composite score to be considered (0-100)
MIN_COMPOSITE_SCORE = 25.0
# Maximum strategies to select per day
MAX_STRATEGIES_PER_DAY = 5


@dataclass
class MarketConditions:
    """Today's market conditions used for strategy selection."""
    instrument: str = "NIFTY"
    gap_pct: float = 0.0           # Opening gap % from prev close
    vix: float = 0.0               # India VIX value
    regime: MarketRegime = MarketRegime.INSUFFICIENT_DATA
    day_type: DayType = DayType.PENDING
    prev_day_range_pct: float = 0.0  # Previous day's range as % of close
    oi_bias: str = "neutral"        # Options OI bias: bullish/bearish/neutral
    global_bias: str = "neutral"    # Global market bias

    @property
    def gap_bucket(self) -> str:
        abs_gap = abs(self.gap_pct)
        if abs_gap < 0.2:
            return "flat"
        elif abs_gap < 0.5:
            return "small"
        elif abs_gap < 1.0:
            return "medium"
        return "large"

    @property
    def vix_bucket(self) -> str:
        if self.vix <= 0:
            return "unknown"
        elif self.vix < 13:
            return "low"
        elif self.vix < 18:
            return "medium"
        return "high"

    def condition_key(self) -> str:
        """Generate a condition key string for DB lookup.

        Format: gap_{bucket}_vix_{bucket}_{day_type}
        """
        dt = self.day_type.value if self.day_type != DayType.PENDING else "unknown"
        return f"gap_{self.gap_bucket}_vix_{self.vix_bucket}_{dt}"

    def partial_keys(self) -> list[str]:
        """Generate all partial condition keys for fuzzy matching.

        Returns keys from most specific to least specific so we can
        fall back when we don't have enough data for the exact condition.
        """
        dt = self.day_type.value if self.day_type != DayType.PENDING else "unknown"
        gb = self.gap_bucket
        vb = self.vix_bucket

        return [
            f"gap_{gb}_vix_{vb}_{dt}",         # Most specific
            f"gap_{gb}_vix_{vb}_any",           # Gap+VIX match, any day type
            f"gap_{gb}_any_{dt}",               # Gap+day type, any VIX
            f"any_vix_{vb}_{dt}",               # VIX+day type, any gap
            f"gap_{gb}_any_any",                # Gap only
            f"any_vix_{vb}_any",                # VIX only
            f"any_any_{dt}",                    # Day type only
            "any_any_any",                      # Global fallback
        ]


@dataclass
class StrategyPick:
    """One selected strategy with probability and reasoning."""
    strategy: str = ""
    probability: float = 0.0    # 0-100: win probability for today's conditions
    composite_score: float = 0.0
    condition_key: str = ""
    total_trades: int = 0
    avg_pnl: float = 0.0
    profit_factor: float = 0.0
    best_entry_window: str = ""
    reason: str = ""
    confidence: str = "medium"  # low/medium/high

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "probability": round(self.probability, 1),
            "composite_score": round(self.composite_score, 1),
            "condition_key": self.condition_key,
            "total_trades": self.total_trades,
            "avg_pnl": round(self.avg_pnl, 2),
            "profit_factor": round(self.profit_factor, 2),
            "best_entry_window": self.best_entry_window,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class SelectionResult:
    """Full output of strategy selection."""
    instrument: str = ""
    conditions: Optional[MarketConditions] = None
    condition_key: str = ""
    selected: list[StrategyPick] = field(default_factory=list)
    avoided: list[str] = field(default_factory=list)
    selection_time: str = ""
    fallback_used: bool = False

    def strategy_names(self) -> list[str]:
        return [p.strategy for p in self.selected]

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "condition_key": self.condition_key,
            "conditions": {
                "gap_pct": round(self.conditions.gap_pct, 2) if self.conditions else 0,
                "vix": round(self.conditions.vix, 1) if self.conditions else 0,
                "gap_bucket": self.conditions.gap_bucket if self.conditions else "",
                "vix_bucket": self.conditions.vix_bucket if self.conditions else "",
                "day_type": self.conditions.day_type.value if self.conditions else "",
                "regime": self.conditions.regime.value if self.conditions else "",
            },
            "selected": [p.to_dict() for p in self.selected],
            "avoided": self.avoided,
            "selection_time": self.selection_time,
            "fallback_used": self.fallback_used,
        }


class StrategySelector:
    """Selects best strategies for current market conditions.

    Queries the strategy_condition_performance table for strategies
    that have historically performed well under matching conditions.
    Uses cascading key matching: exact match → partial matches → global fallback.
    """

    def __init__(self) -> None:
        self._latest_selections: dict[str, SelectionResult] = {}

    @property
    def latest_selections(self) -> dict[str, SelectionResult]:
        return self._latest_selections

    async def select(
        self,
        session: AsyncSession,
        conditions: MarketConditions,
    ) -> SelectionResult:
        """Select the best strategies for the given market conditions.

        Uses cascading condition key matching:
        1. Exact condition key match
        2. Progressively less specific partial keys
        3. Global fallback (all conditions aggregated)
        """
        from datetime import datetime
        import pytz
        _IST = pytz.timezone("Asia/Kolkata")

        instrument = conditions.instrument
        result = SelectionResult(
            instrument=instrument,
            conditions=conditions,
            condition_key=conditions.condition_key(),
            selection_time=datetime.now(_IST).strftime("%Y-%m-%d %H:%M"),
        )

        # Try condition keys from most specific to least specific
        partial_keys = conditions.partial_keys()
        best_picks: dict[str, StrategyPick] = {}

        for key in partial_keys:
            rows = await self._query_performance(session, instrument, key)
            for row in rows:
                strat = row["strategy"]
                if strat in best_picks:
                    continue  # Already found from a more specific key
                if row["total_trades"] < MIN_TRADES_FOR_SELECTION:
                    continue
                if row["composite_score"] < MIN_COMPOSITE_SCORE:
                    continue

                confidence = "high" if key == partial_keys[0] else (
                    "medium" if key in partial_keys[:3] else "low"
                )
                pick = StrategyPick(
                    strategy=strat,
                    probability=row["probability"],
                    composite_score=row["composite_score"],
                    condition_key=key,
                    total_trades=row["total_trades"],
                    avg_pnl=row["avg_pnl"],
                    profit_factor=row["profit_factor"],
                    best_entry_window=row["best_entry_window"] or "",
                    confidence=confidence,
                    reason=self._build_reason(strat, row, key, confidence),
                )
                best_picks[strat] = pick

                if key != partial_keys[0]:
                    result.fallback_used = True

        # Sort by composite score and take top N
        picks = sorted(best_picks.values(), key=lambda p: p.composite_score, reverse=True)
        result.selected = picks[:MAX_STRATEGIES_PER_DAY]

        # Track which strategies were available but not selected
        all_strategies = await self._get_all_evaluated_strategies(session, instrument)
        selected_names = {p.strategy for p in result.selected}
        result.avoided = [s for s in all_strategies if s not in selected_names]

        # Cache
        self._latest_selections[instrument] = result

        logger.info(
            "[%s] Strategy selection: %s (conditions=%s, fallback=%s)",
            instrument,
            [f"{p.strategy}({p.probability:.0f}%%)" for p in result.selected],
            conditions.condition_key(),
            result.fallback_used,
        )

        return result

    async def _query_performance(
        self,
        session: AsyncSession,
        instrument: str,
        condition_key: str,
    ) -> list[dict]:
        """Query strategy_condition_performance for a specific condition key."""
        result = await session.execute(
            text(
                "SELECT strategy, condition_key, total_trades, win_rate, "
                "avg_pnl, profit_factor, sharpe_ratio, max_drawdown, "
                "composite_score, probability, best_entry_window "
                "FROM strategy_condition_performance "
                "WHERE instrument = :inst AND condition_key = :key "
                "ORDER BY composite_score DESC"
            ),
            {"inst": instrument, "key": condition_key},
        )
        rows = result.fetchall()
        return [
            {
                "strategy": r[0], "condition_key": r[1], "total_trades": r[2],
                "win_rate": r[3], "avg_pnl": r[4], "profit_factor": r[5],
                "sharpe_ratio": r[6], "max_drawdown": r[7],
                "composite_score": r[8], "probability": r[9],
                "best_entry_window": r[10],
            }
            for r in rows
        ]

    async def _get_all_evaluated_strategies(
        self,
        session: AsyncSession,
        instrument: str,
    ) -> list[str]:
        """Get all strategies that have been evaluated for this instrument."""
        result = await session.execute(
            text(
                "SELECT DISTINCT strategy FROM strategy_condition_performance "
                "WHERE instrument = :inst"
            ),
            {"inst": instrument},
        )
        return [r[0] for r in result.fetchall()]

    def _build_reason(self, strategy: str, row: dict, key: str, confidence: str) -> str:
        wr = row["win_rate"]
        pf = row["profit_factor"]
        trades = row["total_trades"]
        parts = [f"{wr:.0f}% win rate over {trades} trades"]
        if pf > 1.5:
            parts.append(f"profit factor {pf:.1f}")
        if confidence == "high":
            parts.append("exact condition match")
        elif confidence == "low":
            parts.append("broad condition match")
        return " | ".join(parts)

    def get_active_strategies(self, instrument: str) -> list[str]:
        """Get the currently selected strategy names for an instrument."""
        sel = self._latest_selections.get(instrument)
        if sel and sel.selected:
            return sel.strategy_names()
        # Fallback: return all strategies if no selection has been made
        return []
