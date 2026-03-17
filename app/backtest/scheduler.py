"""Evaluation scheduler — runs strategy evaluation post-market or on-demand.

Stores results to DB and makes them available via API for next-day planning.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import pytz

from app.core.config import settings
from app.core.instruments import InstrumentConfig, get_instrument
from app.backtest.strategy_evaluator import (
    EvaluationReport,
    StrategyEvaluator,
    StrategyRecommendation,
)

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")


class EvaluationScheduler:
    """Manages running strategy evaluations and persisting results."""

    def __init__(self, lookback_days: int = 20) -> None:
        self.evaluator = StrategyEvaluator(lookback_days=lookback_days)
        self._latest_report: EvaluationReport | None = None
        self._running = False
        self._status: str = "idle"          # idle | running | completed | failed
        self._status_message: str = ""
        self._last_run_time: str | None = None

    @property
    def latest_report(self) -> EvaluationReport | None:
        return self._latest_report

    @property
    def latest_recommendations(self) -> list[StrategyRecommendation]:
        if self._latest_report:
            return self._latest_report.recommendations
        return []

    @property
    def eval_status(self) -> dict:
        return {
            "status": self._status,
            "message": self._status_message,
            "running": self._running,
            "last_run_time": self._last_run_time,
            "has_results": self._latest_report is not None and len(self._latest_report.recommendations) > 0,
        }

    async def run_evaluation(
        self,
        instruments: list[InstrumentConfig],
    ) -> EvaluationReport:
        """Run a full evaluation (blocking). Safe to call from API or orchestrator."""
        if self._running:
            logger.warning("Evaluation already in progress — skipping")
            self._status_message = "Already running"
            if self._latest_report:
                return self._latest_report
            return EvaluationReport()

        self._running = True
        self._status = "running"
        self._status_message = f"Evaluating {len(instruments)} instruments..."
        try:
            logger.info(
                "Starting strategy evaluation for %d instruments...",
                len(instruments),
            )
            report = await self.evaluator.evaluate(instruments)

            self._latest_report = report

            # Persist to DB
            await self._save_to_db(report)

            self._status = "completed"
            self._last_run_time = datetime.now(_IST).strftime("%Y-%m-%d %H:%M")
            self._status_message = (
                f"{len(report.recommendations)} recommendations from "
                f"{len(report.instruments_evaluated)} instruments in "
                f"{report.run_time_seconds:.0f}s"
            )

            logger.info(
                "Evaluation complete: %d recommendations | Top: %s %s (score=%.1f)",
                len(report.recommendations),
                report.recommendations[0].strategy if report.recommendations else "N/A",
                report.recommendations[0].instrument if report.recommendations else "",
                report.recommendations[0].composite_score if report.recommendations else 0,
            )
            return report

        except Exception:
            logger.exception("Error during strategy evaluation")
            self._status = "failed"
            self._status_message = "Evaluation failed — check server logs"
            return EvaluationReport()
        finally:
            self._running = False

    async def _save_to_db(self, report: EvaluationReport) -> None:
        """Persist evaluation results to the database."""
        try:
            from app.db.models import AsyncSessionLocal, StrategyEvalRecord
            from sqlalchemy import delete

            async with AsyncSessionLocal() as session:
                # Clear old evaluations for today (re-run replaces)
                await session.execute(
                    delete(StrategyEvalRecord).where(
                        StrategyEvalRecord.eval_date == report.eval_date
                    )
                )

                for rec in report.recommendations:
                    record = StrategyEvalRecord(
                        eval_date=report.eval_date,
                        instrument=rec.instrument,
                        strategy=rec.strategy,
                        rank=rec.rank,
                        win_rate=rec.win_rate,
                        profit_factor=rec.profit_factor,
                        sharpe_ratio=rec.sharpe_ratio,
                        total_pnl=rec.total_pnl,
                        total_trades=rec.total_trades,
                        avg_pnl=rec.avg_pnl,
                        max_drawdown=rec.max_drawdown,
                        composite_score=rec.composite_score,
                        current_regime=rec.current_regime,
                        signal_frequency=rec.signal_frequency,
                        eval_days=rec.eval_days,
                    )
                    session.add(record)

                await session.commit()
                logger.info("Saved %d evaluation records to DB", len(report.recommendations))

        except Exception:
            logger.exception("Error saving evaluation to DB")

    async def load_latest_from_db(self) -> list[StrategyRecommendation]:
        """Load the most recent evaluation from DB (for cold starts)."""
        try:
            from app.db.models import AsyncSessionLocal, StrategyEvalRecord
            from sqlalchemy import select

            async with AsyncSessionLocal() as session:
                # Get the most recent eval_date
                result = await session.execute(
                    select(StrategyEvalRecord)
                    .order_by(
                        StrategyEvalRecord.eval_date.desc(),
                        StrategyEvalRecord.rank.asc(),
                    )
                    .limit(50)
                )
                rows = result.scalars().all()

                if not rows:
                    return []

                # Convert to StrategyRecommendation objects
                recs = []
                for r in rows:
                    recs.append(StrategyRecommendation(
                        rank=r.rank,
                        instrument=r.instrument,
                        strategy=r.strategy,
                        win_rate=r.win_rate,
                        profit_factor=r.profit_factor,
                        sharpe_ratio=r.sharpe_ratio,
                        total_pnl=r.total_pnl,
                        total_trades=r.total_trades,
                        avg_pnl=r.avg_pnl,
                        max_drawdown=r.max_drawdown,
                        composite_score=r.composite_score,
                        current_regime=r.current_regime,
                        signal_frequency=r.signal_frequency,
                        eval_days=r.eval_days,
                        eval_date=r.eval_date,
                    ))

                # Cache as latest report
                if recs and self._latest_report is None:
                    self._latest_report = EvaluationReport(
                        eval_date=recs[0].eval_date,
                        recommendations=recs,
                    )

                logger.info("Loaded %d evaluation records from DB (date: %s)", len(recs), recs[0].eval_date)
                return recs

        except Exception:
            logger.exception("Error loading evaluation from DB")
            return []
