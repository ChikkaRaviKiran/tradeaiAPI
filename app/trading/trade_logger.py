"""Trade logger — persists trades to database and provides query methods."""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime
from typing import Optional

import pytz
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import PerformanceMetrics, Trade
from app.db.models import DailyReport, TradeRecord, create_new_async_session_factory

logger = logging.getLogger(__name__)

_IST = pytz.timezone("Asia/Kolkata")

_thread_local = threading.local()


def _get_session_factory():
    if not hasattr(_thread_local, 'session_factory'):
        _thread_local.session_factory, _thread_local.engine = create_new_async_session_factory()
    return _thread_local.session_factory


class TradeLogger:
    """Persist and query trade records in PostgreSQL."""

    async def log_trade(self, trade: Trade) -> None:
        """Insert or update a trade record."""
        SessionLocal = _get_session_factory()
        async with SessionLocal() as session:
            async with session.begin():
                existing = await session.execute(
                    select(TradeRecord).where(TradeRecord.trade_id == trade.trade_id)
                )
                record = existing.scalar_one_or_none()

                if record:
                    await session.execute(
                        update(TradeRecord)
                        .where(TradeRecord.trade_id == trade.trade_id)
                        .values(
                            exit_price=trade.exit_price,
                            exit_time=trade.exit_time,
                            pnl=trade.pnl,
                            status=trade.status.value,
                        )
                    )
                else:
                    record = TradeRecord(
                        trade_id=trade.trade_id,
                        date=trade.date,
                        time=trade.time,
                        exit_time=trade.exit_time,
                        symbol=trade.symbol,
                        strike=trade.strike,
                        option_type=trade.option_type.value,
                        strategy=trade.strategy.value,
                        entry_price=trade.entry_price,
                        exit_price=trade.exit_price,
                        stoploss=trade.stoploss,
                        target1=trade.target1,
                        target2=trade.target2,
                        confidence=trade.confidence,
                        pnl=trade.pnl,
                        status=trade.status.value,
                        lot_size=trade.lot_size,
                        reason=trade.reason,
                        instrument=trade.instrument,
                    )
                    session.add(record)

        logger.info("Trade logged: %s (%s)", trade.trade_id, trade.status.value)

    async def get_today_trades(self) -> list[Trade]:
        """Fetch all trades for today."""
        today = datetime.now(_IST).date().isoformat()
        SessionLocal = _get_session_factory()
        async with SessionLocal() as session:
            result = await session.execute(
                select(TradeRecord).where(TradeRecord.date == today)
            )
            records = result.scalars().all()
            return [self._to_trade(r) for r in records]

    async def get_trades_by_date(self, target_date: str) -> list[Trade]:
        """Fetch trades for a specific date."""
        SessionLocal = _get_session_factory()
        async with SessionLocal() as session:
            result = await session.execute(
                select(TradeRecord).where(TradeRecord.date == target_date)
            )
            records = result.scalars().all()
            return [self._to_trade(r) for r in records]

    async def get_all_trades(self, limit: int = 500) -> list[Trade]:
        """Fetch recent trades."""
        SessionLocal = _get_session_factory()
        async with SessionLocal() as session:
            result = await session.execute(
                select(TradeRecord).order_by(TradeRecord.id.desc()).limit(limit)
            )
            records = result.scalars().all()
            return [self._to_trade(r) for r in records]

    async def compute_performance(self, trades: Optional[list[Trade]] = None) -> PerformanceMetrics:
        """Compute aggregate performance metrics."""
        if trades is None:
            trades = await self.get_all_trades()

        closed = [t for t in trades if t.status.value == "closed"]
        if not closed:
            return PerformanceMetrics()

        winners = [t for t in closed if (t.pnl or 0) > 0]
        losers = [t for t in closed if (t.pnl or 0) < 0]

        total_profit = sum(t.pnl for t in winners if t.pnl)
        total_loss = abs(sum(t.pnl for t in losers if t.pnl))

        # Max drawdown (running)
        running_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in closed:
            running_pnl += t.pnl or 0
            peak = max(peak, running_pnl)
            dd = peak - running_pnl
            max_dd = max(max_dd, dd)

        total_pnl = sum(t.pnl or 0 for t in closed)

        return PerformanceMetrics(
            total_trades=len(closed),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=round(len(winners) / len(closed) * 100, 1) if closed else 0,
            total_pnl=round(total_pnl, 2),
            profit_factor=round(total_profit / total_loss, 2) if total_loss > 0 else 0,
            max_drawdown=round(max_dd, 2),
            avg_pnl_per_trade=round(total_pnl / len(closed), 2) if closed else 0,
        )

    async def save_daily_report(self, metrics: PerformanceMetrics) -> None:
        """Save daily summary to database."""
        today = datetime.now(_IST).date().isoformat()
        SessionLocal = _get_session_factory()
        async with SessionLocal() as session:
            async with session.begin():
                existing = await session.execute(
                    select(DailyReport).where(DailyReport.date == today)
                )
                report = existing.scalar_one_or_none()
                if report:
                    report.total_trades = metrics.total_trades
                    report.winning_trades = metrics.winning_trades
                    report.losing_trades = metrics.losing_trades
                    report.total_pnl = metrics.total_pnl
                    report.win_rate = metrics.win_rate
                    report.max_drawdown = metrics.max_drawdown
                else:
                    session.add(
                        DailyReport(
                            date=today,
                            total_trades=metrics.total_trades,
                            winning_trades=metrics.winning_trades,
                            losing_trades=metrics.losing_trades,
                            total_pnl=metrics.total_pnl,
                            win_rate=metrics.win_rate,
                            max_drawdown=metrics.max_drawdown,
                        )
                    )

    @staticmethod
    def _to_trade(record: TradeRecord) -> Trade:
        from app.core.models import OptionType, StrategyName, TradeStatus

        return Trade(
            trade_id=record.trade_id,
            date=record.date,
            time=record.time,
            exit_time=record.exit_time,
            symbol=record.symbol,
            strike=record.strike,
            option_type=OptionType(record.option_type),
            strategy=StrategyName(record.strategy),
            entry_price=record.entry_price,
            exit_price=record.exit_price,
            stoploss=record.stoploss,
            target1=record.target1,
            target2=record.target2,
            confidence=record.confidence,
            pnl=record.pnl,
            status=TradeStatus(record.status),
            lot_size=record.lot_size,
            reason=record.reason,
            instrument=record.instrument,
        )
