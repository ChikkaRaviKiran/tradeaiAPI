"""FastAPI application and REST API endpoints."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date, datetime

import pytz
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.models import AlertItem, MarketSnapshot, PerformanceMetrics, Trade
from app.db.models import init_db
from app.trading.history_logger import HistoryLogger
from app.trading.trade_logger import TradeLogger

logger = logging.getLogger(__name__)

_IST = pytz.timezone("Asia/Kolkata")

# Shared state — populated by the orchestrator
_state: dict = {
    "snapshot": None,
    "open_trades": [],
    "orchestrator": None,
    "global_indices": [],
}


def get_state() -> dict:
    return _state


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    await init_db()
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="TradeAI — NIFTY Options Decision System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

trade_logger = TradeLogger()
history_logger = HistoryLogger()


# ── Market Overview ───────────────────────────────────────────────────────

@app.get("/api/market/snapshot", response_model=MarketSnapshot)
async def get_market_snapshot():
    """Current market snapshot: NIFTY price, VWAP, regime, global bias."""
    snapshot = _state.get("snapshot")
    if snapshot is None:
        raise HTTPException(status_code=503, detail="Market data not yet available")
    return snapshot


@app.get("/api/market/global-indices")
async def get_global_indices():
    """Current global index data with individual change percentages."""
    return _state.get("global_indices", [])


# ── Trades ────────────────────────────────────────────────────────────────

@app.get("/api/trades/active", response_model=list[Trade])
async def get_active_trades():
    """Get currently open trades."""
    return _state.get("open_trades", [])


@app.get("/api/trades/today", response_model=list[Trade])
async def get_today_trades():
    """Get all trades for today."""
    return await trade_logger.get_today_trades()


@app.get("/api/trades/history", response_model=list[Trade])
async def get_trade_history(limit: int = 100):
    """Get recent trade history."""
    if limit < 1 or limit > 1000:
        limit = 100
    return await trade_logger.get_all_trades(limit=limit)


@app.get("/api/trades/date/{target_date}", response_model=list[Trade])
async def get_trades_by_date(target_date: str):
    """Get trades for a specific date (YYYY-MM-DD)."""
    return await trade_logger.get_trades_by_date(target_date)


# ── Performance ──────────────────────────────────────────────────────────

@app.get("/api/performance", response_model=PerformanceMetrics)
async def get_performance():
    """Aggregate performance metrics."""
    return await trade_logger.compute_performance()


@app.get("/api/performance/today", response_model=PerformanceMetrics)
async def get_today_performance():
    """Today's performance metrics."""
    today_trades = await trade_logger.get_today_trades()
    return await trade_logger.compute_performance(today_trades)


# ── Alerts (UI) ──────────────────────────────────────────────────────────

@app.get("/api/alerts", response_model=list[AlertItem])
async def get_alerts(limit: int = 50):
    """Get recent alerts for the dashboard."""
    from app.alerts.alert_manager import alert_store

    if limit < 1 or limit > 200:
        limit = 50
    return alert_store.get_recent(limit)


# ── System Control ───────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Lightweight health check for load balancers."""
    return {"status": "ok"}


@app.get("/api/system/status")
async def get_system_status():
    """System health and status."""
    orch = _state.get("orchestrator")
    snapshot = _state.get("snapshot")

    # Quick DB connectivity check
    db_ok = False
    try:
        from app.db.models import AsyncSessionLocal
        from sqlalchemy import text
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        logger.warning("DB health check failed: %s", e)

    return {
        "status": "running" if orch and getattr(orch, "running", False) else "stopped",
        "paper_trading": settings.paper_trading,
        "capital": settings.initial_capital,
        "max_trades_per_day": settings.max_trades_per_day,
        "cycle_count": getattr(orch, "_cycle_count", 0) if orch else 0,
        "expiry": getattr(orch, "_expiry", None) if orch else None,
        "last_snapshot_time": snapshot.timestamp.isoformat() if snapshot else None,
        "open_trades_count": len(_state.get("open_trades", [])),
        "db_connected": db_ok,
    }


@app.get("/api/system/logs")
async def get_recent_logs(lines: int = 100):
    """Return recent lines from the log file."""
    import os
    log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "logs", "tradeai.log")
    log_file = os.path.abspath(log_file)
    if not os.path.exists(log_file):
        return {"logs": "Log file not found", "path": log_file}
    if lines < 1 or lines > 1000:
        lines = 100
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()
    return {"logs": "".join(all_lines[-lines:]), "total_lines": len(all_lines)}


@app.post("/api/system/start")
async def start_system():
    """Start the trading system for the current day."""
    orch = _state.get("orchestrator")
    if not orch:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    if orch.running:
        return {"message": "System already running"}
    # Launch a new trading day in the orchestrator's thread
    import threading, asyncio
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(orch._run_trading_day())
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"message": "System started"}


@app.post("/api/system/stop")
async def stop_system():
    """Stop the trading system."""
    orch = _state.get("orchestrator")
    if orch:
        orch.running = False
        return {"message": "System stopped"}
    raise HTTPException(status_code=503, detail="Orchestrator not initialized")


# ── History (Snapshots / Alerts / Calendar) ──────────────────────────────

@app.get("/api/history/snapshots/{target_date}")
async def get_snapshots_by_date(target_date: str):
    """Get all market snapshots for a specific date (YYYY-MM-DD)."""
    return await history_logger.get_snapshots_by_date(target_date)


@app.get("/api/history/snapshots")
async def get_snapshots_by_range(start: str, end: str):
    """Get snapshots between two dates. ?start=YYYY-MM-DD&end=YYYY-MM-DD"""
    return await history_logger.get_snapshots_by_range(start, end)


@app.get("/api/history/summary/{target_date}")
async def get_daily_summary(target_date: str):
    """Get a summary of a specific trading day."""
    return await history_logger.get_daily_summary(target_date)


@app.get("/api/history/calendar/{year}/{month}")
async def get_calendar_data(year: int, month: int):
    """Get daily summaries for a month — for calendar view."""
    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="Invalid month")
    return await history_logger.get_calendar_data(year, month)


@app.get("/api/history/alerts/{target_date}")
async def get_alerts_by_date(target_date: str):
    """Get all alerts for a specific date."""
    return await history_logger.get_alerts_by_date(target_date)


@app.get("/api/history/alerts")
async def get_alerts_by_range(start: str, end: str):
    """Get alerts between two dates."""
    return await history_logger.get_alerts_by_range(start, end)


@app.get("/api/history/day/{target_date}")
async def get_full_day_data(target_date: str):
    """Get complete data for a day: summary, snapshots, trades, alerts, performance.

    For today's date, falls back to live in-memory data if DB has nothing.
    """
    summary = await history_logger.get_daily_summary(target_date)
    snapshots = await history_logger.get_snapshots_by_date(target_date)
    trades = await trade_logger.get_trades_by_date(target_date)
    alerts = await history_logger.get_alerts_by_date(target_date)
    perf = await trade_logger.compute_performance(trades)

    # For today: if DB has no data, fall back to live in-memory data
    today_str = datetime.now(_IST).date().isoformat()
    if target_date == today_str and not snapshots:
        snapshot = _state.get("snapshot")
        if snapshot:
            # Build a snapshot dict from live data
            live_snap = {
                "date": today_str,
                "time": snapshot.timestamp.strftime("%H:%M:%S") if snapshot.timestamp else "",
                "nifty_price": snapshot.nifty_price,
                "vwap": snapshot.vwap,
                "regime": snapshot.regime.value if snapshot.regime else "unknown",
                "global_bias": snapshot.global_bias.value if snapshot.global_bias else "unavailable",
                "ema9": snapshot.indicators.ema9 if snapshot.indicators else None,
                "ema20": snapshot.indicators.ema20 if snapshot.indicators else None,
                "ema50": snapshot.indicators.ema50 if snapshot.indicators else None,
                "rsi": snapshot.indicators.rsi if snapshot.indicators else None,
                "macd": snapshot.indicators.macd if snapshot.indicators else None,
                "macd_signal": snapshot.indicators.macd_signal if snapshot.indicators else None,
                "macd_hist": snapshot.indicators.macd_hist if snapshot.indicators else None,
                "atr": snapshot.indicators.atr if snapshot.indicators else None,
                "adx": snapshot.indicators.adx if snapshot.indicators else None,
                "bollinger_upper": snapshot.indicators.bollinger_upper if snapshot.indicators else None,
                "bollinger_middle": snapshot.indicators.bollinger_middle if snapshot.indicators else None,
                "bollinger_lower": snapshot.indicators.bollinger_lower if snapshot.indicators else None,
                "pcr": snapshot.options_metrics.pcr if snapshot.options_metrics else None,
                "max_pain": snapshot.options_metrics.max_pain if snapshot.options_metrics else None,
                "call_oi_cluster": snapshot.options_metrics.call_oi_cluster if snapshot.options_metrics else None,
                "put_oi_cluster": snapshot.options_metrics.put_oi_cluster if snapshot.options_metrics else None,
                "oi_change": snapshot.options_metrics.oi_change if snapshot.options_metrics else 0,
            }
            snapshots = [live_snap]

            # Build summary from live snapshot
            summary = {
                "date": today_str,
                "has_data": True,
                "total_snapshots": 1,
                "open_price": snapshot.nifty_price,
                "close_price": snapshot.nifty_price,
                "high": snapshot.nifty_price,
                "low": snapshot.nifty_price,
                "first_time": live_snap["time"],
                "last_time": live_snap["time"],
                "avg_rsi": round(snapshot.indicators.rsi, 1) if snapshot.indicators and snapshot.indicators.rsi else 0,
                "avg_adx": round(snapshot.indicators.adx, 1) if snapshot.indicators and snapshot.indicators.adx else 0,
                "regimes": [live_snap["regime"]],
                "last_pcr": live_snap["pcr"],
                "last_max_pain": live_snap["max_pain"],
                "source": "live",
            }

    # For today: also include in-memory alerts if DB has none
    if target_date == today_str and not alerts:
        from app.alerts.alert_manager import alert_store
        mem_alerts = alert_store.get_all()
        if mem_alerts:
            alerts = [
                {
                    "id": a.id,
                    "date": a.timestamp.strftime("%Y-%m-%d") if a.timestamp else today_str,
                    "alert_type": a.alert_type,
                    "title": a.title,
                    "message": a.message,
                    "trade_id": a.trade_id,
                    "strategy": a.strategy,
                    "pnl": a.pnl,
                    "created_at": a.timestamp.isoformat() if a.timestamp else None,
                }
                for a in mem_alerts
            ]

    return {
        "date": target_date,
        "summary": summary,
        "snapshots": snapshots,
        "trades": [t.model_dump() for t in trades],
        "alerts": alerts,
        "performance": perf.model_dump(),
    }
