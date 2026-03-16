"""History logger — persists market snapshots and alerts to database."""

from __future__ import annotations

import logging
import threading
from datetime import datetime

from sqlalchemy import select

from app.core.models import AlertItem, MarketSnapshot
from app.db.models import AlertRecord, MarketSnapshotRecord, create_new_async_session_factory

logger = logging.getLogger(__name__)

# Thread-local storage so each event loop gets its own engine
_thread_local = threading.local()

# Track DB errors to avoid spamming logs/alerts
_db_error_count = 0
_DB_ERROR_LOG_INTERVAL = 10  # Only log every N failures


def _get_session_factory():
    """Return a session factory bound to the current thread's event loop."""
    if not hasattr(_thread_local, 'session_factory'):
        _thread_local.session_factory, _thread_local.engine = create_new_async_session_factory()
    return _thread_local.session_factory


async def _ensure_tables():
    """Ensure our tables exist (called once per thread)."""
    if not hasattr(_thread_local, '_tables_created'):
        try:
            _get_session_factory()  # ensure engine is created
            from app.db.models import Base
            engine = _thread_local.engine
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            _thread_local._tables_created = True
        except Exception:
            # Reset thread-local state so next call retries fresh
            for attr in ('session_factory', 'engine', '_tables_created'):
                if hasattr(_thread_local, attr):
                    delattr(_thread_local, attr)
            raise


class HistoryLogger:
    """Persist market snapshots and alerts for historical review."""

    async def save_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Save a single market snapshot to the database."""
        global _db_error_count
        try:
            await _ensure_tables()
            now = snapshot.timestamp
            record = MarketSnapshotRecord(
                date=now.strftime("%Y-%m-%d"),
                time=now.strftime("%H:%M:%S"),
                nifty_price=snapshot.nifty_price,
                vwap=snapshot.vwap,
                regime=snapshot.regime.value,
                global_bias=snapshot.global_bias.value,
                ema9=snapshot.indicators.ema9,
                ema20=snapshot.indicators.ema20,
                ema50=snapshot.indicators.ema50,
                ema200=snapshot.indicators.ema200,
                rsi=snapshot.indicators.rsi,
                macd=snapshot.indicators.macd,
                macd_signal=snapshot.indicators.macd_signal,
                macd_hist=snapshot.indicators.macd_hist,
                atr=snapshot.indicators.atr,
                adx=snapshot.indicators.adx,
                bollinger_upper=snapshot.indicators.bollinger_upper,
                bollinger_middle=snapshot.indicators.bollinger_middle,
                bollinger_lower=snapshot.indicators.bollinger_lower,
                pcr=snapshot.options_metrics.pcr,
                max_pain=snapshot.options_metrics.max_pain,
                call_oi_cluster=snapshot.options_metrics.call_oi_cluster,
                put_oi_cluster=snapshot.options_metrics.put_oi_cluster,
                oi_change=snapshot.options_metrics.oi_change,
            )
            SessionLocal = _get_session_factory()
            async with SessionLocal() as session:
                async with session.begin():
                    session.add(record)
            if _db_error_count > 0:
                logger.info("DB connection recovered after %d failures", _db_error_count)
                _db_error_count = 0
            logger.debug("Snapshot saved: NIFTY=%.2f at %s", snapshot.nifty_price, record.time)
        except Exception as e:
            _db_error_count += 1
            if _db_error_count == 1 or _db_error_count % _DB_ERROR_LOG_INTERVAL == 0:
                logger.error(
                    "DB save_snapshot failed (%d consecutive): %s",
                    _db_error_count, str(e),
                )

    async def save_alert(self, alert: AlertItem) -> None:
        """Save an alert to the database."""
        try:
            await _ensure_tables()
            record = AlertRecord(
                date=alert.timestamp.strftime("%Y-%m-%d"),
                alert_type=alert.alert_type,
                title=alert.title,
                message=alert.message,
                trade_id=alert.trade_id,
                strategy=alert.strategy,
                pnl=alert.pnl,
            )
            SessionLocal = _get_session_factory()
            async with SessionLocal() as session:
                async with session.begin():
                    session.add(record)
        except Exception as e:
            # Reset thread-local state so next attempt retries fresh
            for attr in ('session_factory', 'engine', '_tables_created'):
                if hasattr(_thread_local, attr):
                    delattr(_thread_local, attr)
            logger.error("Error saving alert")
            logger.debug("save_alert detail: %s", str(e), exc_info=True)

    async def get_snapshots_by_date(self, target_date: str) -> list[dict]:
        """Get all snapshots for a specific date (YYYY-MM-DD)."""
        SessionLocal = _get_session_factory()
        async with SessionLocal() as session:
            result = await session.execute(
                select(MarketSnapshotRecord)
                .where(MarketSnapshotRecord.date == target_date)
                .order_by(MarketSnapshotRecord.time)
            )
            records = result.scalars().all()
            return [self._snapshot_to_dict(r) for r in records]

    async def get_snapshots_by_range(self, start_date: str, end_date: str) -> list[dict]:
        """Get snapshots between two dates."""
        SessionLocal = _get_session_factory()
        async with SessionLocal() as session:
            result = await session.execute(
                select(MarketSnapshotRecord)
                .where(
                    MarketSnapshotRecord.date >= start_date,
                    MarketSnapshotRecord.date <= end_date,
                )
                .order_by(MarketSnapshotRecord.date, MarketSnapshotRecord.time)
            )
            records = result.scalars().all()
            return [self._snapshot_to_dict(r) for r in records]

    async def get_daily_summary(self, target_date: str) -> dict:
        """Get a summary of a specific day: first/last snapshot, high/low, etc."""
        snapshots = await self.get_snapshots_by_date(target_date)
        if not snapshots:
            return {"date": target_date, "has_data": False}

        prices = [s["nifty_price"] for s in snapshots]
        rsi_values = [s["rsi"] for s in snapshots if s["rsi"] is not None]
        adx_values = [s["adx"] for s in snapshots if s["adx"] is not None]
        return {
            "date": target_date,
            "has_data": True,
            "total_snapshots": len(snapshots),
            "open_price": snapshots[0]["nifty_price"],
            "close_price": snapshots[-1]["nifty_price"],
            "high": max(prices),
            "low": min(prices),
            "first_time": snapshots[0]["time"],
            "last_time": snapshots[-1]["time"],
            "avg_rsi": round(sum(rsi_values) / len(rsi_values), 1) if rsi_values else 0,
            "avg_adx": round(sum(adx_values) / len(adx_values), 1) if adx_values else 0,
            "regimes": list(set(s["regime"] for s in snapshots)),
            "last_pcr": snapshots[-1]["pcr"],
            "last_max_pain": snapshots[-1]["max_pain"],
        }

    async def get_calendar_data(self, year: int, month: int) -> list[dict]:
        """Get daily summaries for an entire month — for the calendar view."""
        start = f"{year}-{month:02d}-01"
        if month == 12:
            end = f"{year + 1}-01-01"
        else:
            end = f"{year}-{month + 1:02d}-01"

        # Get unique dates that have snapshots
        SessionLocal = _get_session_factory()
        async with SessionLocal() as session:
            from sqlalchemy import func, distinct
            result = await session.execute(
                select(
                    MarketSnapshotRecord.date,
                    func.count(MarketSnapshotRecord.id).label("cnt"),
                    func.min(MarketSnapshotRecord.nifty_price).label("low"),
                    func.max(MarketSnapshotRecord.nifty_price).label("high"),
                    func.min(MarketSnapshotRecord.time).label("first_time"),
                    func.max(MarketSnapshotRecord.time).label("last_time"),
                )
                .where(
                    MarketSnapshotRecord.date >= start,
                    MarketSnapshotRecord.date < end,
                )
                .group_by(MarketSnapshotRecord.date)
                .order_by(MarketSnapshotRecord.date)
            )
            rows = result.all()

        days = []
        for row in rows:
            # Get first and last snapshot for open/close
            snaps = await self.get_snapshots_by_date(row.date)
            open_p = snaps[0]["nifty_price"] if snaps else 0
            close_p = snaps[-1]["nifty_price"] if snaps else 0
            days.append({
                "date": row.date,
                "snapshots": row.cnt,
                "open": open_p,
                "close": close_p,
                "high": row.high,
                "low": row.low,
                "change": round(close_p - open_p, 2) if open_p else 0,
                "change_pct": round(((close_p - open_p) / open_p) * 100, 2) if open_p else 0,
            })
        return days

    async def get_alerts_by_date(self, target_date: str) -> list[dict]:
        """Get all alerts for a specific date."""
        SessionLocal = _get_session_factory()
        async with SessionLocal() as session:
            result = await session.execute(
                select(AlertRecord)
                .where(AlertRecord.date == target_date)
                .order_by(AlertRecord.created_at.desc())
            )
            records = result.scalars().all()
            return [self._alert_to_dict(r) for r in records]

    async def get_alerts_by_range(self, start_date: str, end_date: str) -> list[dict]:
        """Get alerts between two dates."""
        SessionLocal = _get_session_factory()
        async with SessionLocal() as session:
            result = await session.execute(
                select(AlertRecord)
                .where(
                    AlertRecord.date >= start_date,
                    AlertRecord.date <= end_date,
                )
                .order_by(AlertRecord.created_at.desc())
            )
            records = result.scalars().all()
            return [self._alert_to_dict(r) for r in records]

    def _snapshot_to_dict(self, r: MarketSnapshotRecord) -> dict:
        return {
            "date": r.date,
            "time": r.time,
            "nifty_price": r.nifty_price,
            "vwap": r.vwap,
            "regime": r.regime,
            "global_bias": r.global_bias,
            "ema9": r.ema9,
            "ema20": r.ema20,
            "ema50": r.ema50,
            "ema200": r.ema200,
            "rsi": r.rsi,
            "macd": r.macd,
            "macd_signal": r.macd_signal,
            "macd_hist": r.macd_hist,
            "atr": r.atr,
            "adx": r.adx,
            "bollinger_upper": r.bollinger_upper,
            "bollinger_middle": r.bollinger_middle,
            "bollinger_lower": r.bollinger_lower,
            "pcr": r.pcr,
            "max_pain": r.max_pain,
            "call_oi_cluster": r.call_oi_cluster,
            "put_oi_cluster": r.put_oi_cluster,
            "oi_change": r.oi_change,
        }

    def _alert_to_dict(self, r: AlertRecord) -> dict:
        return {
            "id": r.id,
            "date": r.date,
            "alert_type": r.alert_type,
            "title": r.title,
            "message": r.message,
            "trade_id": r.trade_id,
            "strategy": r.strategy,
            "pnl": r.pnl,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
