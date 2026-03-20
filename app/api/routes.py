"""FastAPI application and REST API endpoints.

Multi-instrument aware: serves per-instrument snapshots, stock rankings,
and ML predictions alongside existing trade/performance endpoints.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date, datetime

import pytz
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.instruments import get_all_instruments, get_instrument
from app.core.models import AlertItem, MarketSnapshot, PerformanceMetrics, Trade
from app.db.models import init_db
from app.trading.history_logger import HistoryLogger
from app.trading.trade_logger import TradeLogger

logger = logging.getLogger(__name__)

_IST = pytz.timezone("Asia/Kolkata")

# Shared state — populated by the orchestrator
_state: dict = {
    "snapshot": None,
    "snapshots": {},       # {symbol: MarketSnapshot}
    "open_trades": [],
    "orchestrator": None,
    "global_indices": [],
    "stock_rankings": [],  # Latest stock rankings
    "predictions": {},     # {symbol: MarketPrediction}
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
    title="TradeAI — Multi-Instrument AI Trading System",
    version="2.0.0",
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
async def get_alerts(limit: int = 50, target_date: str | None = None):
    """Get alerts for the dashboard, filtered by date (defaults to today)."""
    from app.alerts.alert_manager import alert_store

    if limit < 1 or limit > 200:
        limit = 50

    # Determine filter date (default = today IST)
    filter_date = target_date or datetime.now(_IST).strftime("%Y-%m-%d")

    # Try in-memory alerts first
    all_alerts = alert_store.get_all()
    filtered = [
        a for a in all_alerts
        if a.timestamp and a.timestamp.strftime("%Y-%m-%d") == filter_date
    ][:limit]

    # If no in-memory alerts for requested date, try DB
    if not filtered and target_date:
        db_alerts = await history_logger.get_alerts_by_date(filter_date)
        if db_alerts:
            return db_alerts[:limit]

    return filtered


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

    # Quick DB connectivity check (uses its own engine to avoid event-loop
    # mismatch with the orchestrator thread's loop)
    db_ok = False
    try:
        from sqlalchemy import text
        from app.db.models import create_new_async_session_factory
        if not hasattr(get_system_status, "_health_sf"):
            sf, eng = create_new_async_session_factory()
            get_system_status._health_sf = sf
            get_system_status._health_eng = eng
        async with get_system_status._health_sf() as session:
            await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        logger.warning("DB health check failed: %s", e)
        # Reset so it rebinds to current loop on next call
        for attr in ("_health_sf", "_health_eng"):
            if hasattr(get_system_status, attr):
                delattr(get_system_status, attr)

    return {
        "status": "running" if orch and getattr(orch, "running", False) else "stopped",
        "paper_trading": settings.paper_trading,
        "capital": settings.initial_capital,
        "max_trades_per_day": settings.max_trades_per_day,
        "auto_select": settings.auto_select_instruments,
        "active_instruments": (
            [i.symbol for i in orch._active_instruments]
            if orch and hasattr(orch, "_active_instruments") and orch._active_instruments
            else settings.get_active_instrument_list() or ["NIFTY"]
        ),
        "cycle_count": getattr(orch, "_cycle_count", 0) if orch else 0,
        "expiries": getattr(orch, "_expiries", {}) if orch else {},
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


@app.get("/api/system/activity")
async def get_system_activity(limit: int = 200):
    """Get pipeline activity log and data source health for dashboard visibility."""
    orch = _state.get("orchestrator")
    if not orch:
        return {"events": [], "data_sources": {}, "cycle": 0}

    if limit < 1 or limit > 500:
        limit = 200

    events = list(orch.activity_log)[-limit:] if hasattr(orch, "activity_log") else []
    sources = getattr(orch, "data_sources", {})
    cycle = getattr(orch, "_cycle_count", 0)

    # Add per-instrument regime and HTF bias
    regimes = {}
    for sym, snap in (getattr(orch, "snapshots", {}) or {}).items():
        regimes[sym] = {
            "regime": snap.regime.value if snap and snap.regime else "unknown",
            "htf_trend": (getattr(orch, "_htf_biases", {}) or {}).get(sym, "unknown"),
            "price": round(snap.price, 2) if snap and snap.price else None,
        }

    return {
        "events": events,
        "data_sources": sources,
        "cycle": cycle,
        "regimes": regimes,
        "paper_trading": settings.paper_trading,
        "open_trades": len(_state.get("open_trades", [])),
    }


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
        try:
            loop.run_until_complete(orch._run_trading_day())
        except Exception as e:
            logger.exception("Trading day thread crashed: %s", e)
            orch.running = False
            if hasattr(orch, '_log_event'):
                orch._log_event("error", f"Trading day crashed: {e}")
    t = threading.Thread(target=_run, daemon=True, name="trading-day")
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


@app.post("/api/system/trading-mode")
async def set_trading_mode(body: dict):
    """Toggle between paper and live trading at runtime."""
    mode = body.get("mode", "").lower()
    if mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")

    orch = _state.get("orchestrator")
    if orch and getattr(orch, "running", False):
        raise HTTPException(
            status_code=409,
            detail="Cannot switch trading mode while system is running. Stop the system first.",
        )

    settings.paper_trading = mode == "paper"
    # Reset auto-pause flag when user explicitly switches to live
    orch = _state.get("orchestrator")
    if orch and mode == "live":
        orch._live_paused_insufficient_margin = False
    logger.info("Trading mode switched to: %s", mode.upper())
    return {"paper_trading": settings.paper_trading, "message": f"Switched to {mode.upper()} trading"}


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
            live_price = snapshot.price or snapshot.nifty_price
            live_snap = {
                "instrument": snapshot.instrument,
                "date": today_str,
                "time": snapshot.timestamp.strftime("%H:%M:%S") if snapshot.timestamp else "",
                "nifty_price": snapshot.nifty_price,
                "price": live_price,
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

            # Build summary from live snapshot — use actual price (not nifty_price which can be 0)
            summary = {
                "date": today_str,
                "has_data": True,
                "total_snapshots": 1,
                "open_price": live_price,
                "close_price": live_price,
                "high": live_price,
                "low": live_price,
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


# ── Multi-Instrument Endpoints (SRS rebuild) ────────────────────────────

@app.get("/api/instruments")
async def list_instruments():
    """List all registered instruments and their config."""
    instruments = get_all_instruments()
    return [
        {
            "symbol": i.symbol,
            "display_name": i.display_name,
            "exchange": i.exchange.value,
            "type": i.instrument_type.value,
            "lot_size": i.lot_size,
            "strike_interval": i.strike_interval,
            "is_index": i.is_index,
            "enabled": i.enabled,
        }
        for i in instruments
    ]


@app.get("/api/instruments/active")
async def list_active_instruments():
    """List instruments currently being monitored."""
    return settings.get_active_instrument_list()


@app.get("/api/market/snapshot/{symbol}")
async def get_instrument_snapshot(symbol: str):
    """Get market snapshot for a specific instrument."""
    snapshots = _state.get("snapshots", {})
    snap = snapshots.get(symbol.upper())
    if snap is None:
        raise HTTPException(status_code=404, detail=f"No snapshot for {symbol}")
    return snap


@app.get("/api/market/snapshots")
async def get_all_snapshots():
    """Get current snapshots for all active instruments."""
    snapshots = _state.get("snapshots", {})
    return {k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in snapshots.items()}


@app.get("/api/rankings")
async def get_stock_rankings():
    """Get latest AI stock rankings."""
    return _state.get("stock_rankings", [])


@app.get("/api/predictions")
async def get_predictions():
    """Get latest ML predictions for all instruments."""
    preds = _state.get("predictions", {})
    return {k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in preds.items()}


@app.get("/api/predictions/{symbol}")
async def get_prediction_for(symbol: str):
    """Get ML prediction for a specific instrument."""
    preds = _state.get("predictions", {})
    pred = preds.get(symbol.upper())
    if pred is None:
        raise HTTPException(status_code=404, detail=f"No prediction for {symbol}")
    return pred.model_dump() if hasattr(pred, "model_dump") else pred


# ── Strategy Evaluation / Recommendations ────────────────────────────────

@app.get("/api/recommendations")
async def get_recommendations():
    """Get latest strategy recommendations ranked by composite score."""
    scheduler = _state.get("eval_scheduler")
    if scheduler is None:
        return {"recommendations": [], "eval_date": None}
    recs = scheduler.latest_recommendations
    report = scheduler.latest_report
    return {
        "eval_date": report.eval_date if report else None,
        "run_time_seconds": round(report.run_time_seconds, 1) if report else None,
        "total_simulated_trades": len(report.all_trades) if report else 0,
        "recommendations": [r.to_dict() for r in recs],
    }


@app.post("/api/evaluate/run")
async def trigger_evaluation():
    """Trigger a strategy evaluation on-demand — evaluates ALL registered instruments."""
    import asyncio
    scheduler = _state.get("eval_scheduler")
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Evaluation scheduler not initialized")

    from app.core.instruments import get_enabled_instruments
    instruments = get_enabled_instruments()
    if not instruments:
        raise HTTPException(status_code=400, detail="No instruments registered")

    # Run evaluation in a background task so the API doesn't block
    async def _run():
        try:
            await scheduler.run_evaluation(instruments)
        except Exception:
            logger.exception("Background evaluation task crashed")
    asyncio.create_task(_run())

    return {"message": "Evaluation started", "instruments": [i.symbol for i in instruments]}


# ── Market Intelligence ──────────────────────────────────────────────────

@app.get("/api/intelligence")
async def get_intelligence():
    """Get current pre-market intelligence and AI insights."""
    insight = _state.get("intelligence")
    analyst = _state.get("pre_market_analyst")

    result = {
        "insight": insight,
        "has_insight": insight is not None,
    }

    # Add live FII/DII data
    if analyst and analyst.institutional_flow:
        flow = analyst.institutional_flow
        result["fii_dii"] = {
            "fii_buy": flow.fii_buy,
            "fii_sell": flow.fii_sell,
            "fii_net": flow.fii_net,
            "dii_buy": flow.dii_buy,
            "dii_sell": flow.dii_sell,
            "dii_net": flow.dii_net,
            "net_institutional": flow.net_institutional,
            "signal": flow.signal,
        }

    # Add live market breadth
    if analyst and analyst.market_breadth:
        breadth = analyst.market_breadth
        result["breadth"] = {
            "total_advancing": breadth.total_advancing,
            "total_declining": breadth.total_declining,
            "total_unchanged": breadth.total_unchanged,
            "advance_decline_ratio": breadth.advance_decline_ratio,
            "breadth_signal": breadth.breadth_signal,
            "strong_sectors": breadth.strong_sectors,
            "weak_sectors": breadth.weak_sectors,
            "sectors": [
                {"name": s.name, "change_pct": round(s.change_pct, 2)}
                for s in breadth.sectors
            ],
        }

    return result


@app.get("/api/intelligence/news")
async def get_intelligence_news(days: int = 1):
    """Get recent Telegram news items."""
    if days < 1 or days > 30:
        days = 1
    from app.data.telegram_news import get_recent_news
    news = await get_recent_news(days=days)
    return {"news": news, "count": len(news)}


@app.get("/api/intelligence/history")
async def get_intelligence_history(limit: int = 7):
    """Get historical AI insights."""
    if limit < 1 or limit > 90:
        limit = 7
    from app.db.models import DailyAIInsight, AsyncSessionLocal
    from sqlalchemy import select
    import json as _json

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(DailyAIInsight)
                .order_by(DailyAIInsight.created_at.desc())
                .limit(limit)
            )
            records = result.scalars().all()
            return [
                {
                    "date": r.date,
                    "insight_type": r.insight_type,
                    "market_bias": r.market_bias,
                    "confidence": r.confidence,
                    "fii_dii_signal": r.fii_dii_signal,
                    "fii_net": r.fii_net,
                    "dii_net": r.dii_net,
                    "breadth_signal": r.breadth_signal,
                    "advance_decline_ratio": r.advance_decline_ratio,
                    "news_sentiment": r.news_sentiment,
                    "strong_sectors": r.strong_sectors,
                    "weak_sectors": r.weak_sectors,
                    "ai_summary": r.ai_summary,
                    "trading_plan": r.trading_plan,
                    "key_levels": _json.loads(r.key_levels) if r.key_levels else {},
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in records
            ]
    except Exception:
        logger.exception("Error fetching intelligence history")
        return []


@app.post("/api/intelligence/refresh")
async def refresh_intelligence():
    """Trigger a fresh pre-market intelligence analysis."""
    import asyncio
    analyst = _state.get("pre_market_analyst")
    if analyst is None:
        raise HTTPException(status_code=503, detail="Pre-market analyst not initialized")

    async def _run():
        try:
            insight = await analyst.run_analysis()
            if insight:
                _state["intelligence"] = insight
        except Exception:
            logger.exception("Intelligence refresh failed")

    asyncio.create_task(_run())
    return {"message": "Intelligence refresh started"}


@app.get("/api/evaluate/status")
async def get_evaluation_status():
    """Get current evaluation run status — used by frontend to poll progress."""
    scheduler = _state.get("eval_scheduler")
    if scheduler is None:
        return {"status": "idle", "message": "Scheduler not initialized", "running": False, "has_results": False}
    return scheduler.eval_status


@app.get("/api/evaluate/history/{target_date}")
async def get_evaluation_history(target_date: str):
    """Get evaluation results for a specific date from DB."""
    from app.db.models import AsyncSessionLocal, StrategyEvalRecord
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(StrategyEvalRecord)
            .where(StrategyEvalRecord.eval_date == target_date)
            .order_by(StrategyEvalRecord.rank.asc())
        )
        rows = result.scalars().all()

    return {
        "eval_date": target_date,
        "recommendations": [
            {
                "rank": r.rank,
                "instrument": r.instrument,
                "strategy": r.strategy,
                "win_rate": round(r.win_rate, 1),
                "profit_factor": round(r.profit_factor, 2),
                "sharpe_ratio": round(r.sharpe_ratio, 2),
                "total_pnl": round(r.total_pnl, 2),
                "total_trades": r.total_trades,
                "avg_pnl": round(r.avg_pnl, 2),
                "max_drawdown": round(r.max_drawdown, 2),
                "composite_score": round(r.composite_score, 1),
                "current_regime": r.current_regime,
                "signal_frequency": round(r.signal_frequency, 2),
                "eval_days": r.eval_days,
            }
            for r in rows
        ],
    }
