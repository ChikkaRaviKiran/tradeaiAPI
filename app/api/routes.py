"""FastAPI application and REST API endpoints.

Multi-instrument aware: serves per-instrument snapshots, stock rankings,
and ML predictions alongside existing trade/performance endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Any, Optional

import pyotp
import pytz
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core import cache
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


def _safe_num(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return int(default)
        return int(float(v))
    except Exception:
        return int(default)


def _map_product_type(product: str):
    from app.execution.broker_base import ProductType

    p = str(product or "").strip().upper()
    if p in {"NRML", "CARRYFORWARD", "CARRY", "MARGIN"}:
        return ProductType.CARRYFORWARD
    if p in {"CNC", "DELIVERY"}:
        return ProductType.DELIVERY
    return ProductType.INTRADAY


def _get_active_broker():
    """Return broker adapter matching TRADING_ACCOUNT with safe fallback."""
    account = (settings.trading_account or "angel").strip().lower()
    try:
        if account == "kite":
            from app.execution.kite_broker import KiteBroker
            return KiteBroker(), "kite"
        if account == "dhan":
            from app.execution.dhan_broker import DhanBroker
            return DhanBroker(), "dhan"
    except Exception:
        logger.exception("Failed to build broker adapter for account=%s", account)

    orch = _state.get("orchestrator")
    broker = getattr(orch, "broker", None) if orch else None
    if broker is not None:
        return broker, "angel"

    from app.execution.angelone_broker import AngelOneBroker
    return AngelOneBroker(), "angel"


async def _list_all_active_brokers() -> list[dict]:
    """Return every wired-up broker adapter (multi-account aware).

    Each entry: ``{"broker", "account_id", "account_name", "broker_type"}``.

    Falls back to the legacy env-based single broker (``account_id=0``)
    when no BrokerAccount rows exist so existing single-account setups
    keep working unchanged.
    """
    out: list[dict] = []
    try:
        from sqlalchemy import select
        from app.db.account_models import BrokerAccount
        from app.db.models import AsyncSessionLocal
        from app.execution.broker_factory import build_broker_from_row

        async with AsyncSessionLocal() as s:
            rows = (
                await s.execute(
                    select(BrokerAccount).where(
                        BrokerAccount.is_active == True  # noqa: E712
                    ).order_by(BrokerAccount.id.asc())
                )
            ).scalars().all()
            row_dicts = [
                {
                    "id": r.id,
                    "name": r.name,
                    "broker": r.broker,
                    "client_id": r.client_id,
                    "api_key": r.api_key,
                    "api_secret": r.api_secret,
                    "password": r.password,
                    "mpin": r.mpin,
                    "totp_secret": r.totp_secret,
                    "access_token": r.access_token,
                    "refresh_token": r.refresh_token,
                    "proxy_url": r.proxy_url,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else "",
                }
                for r in rows
            ]
    except Exception:
        logger.exception("Failed to enumerate BrokerAccount rows")
        row_dicts = []

    for row in row_dicts:
        try:
            broker = build_broker_from_row(row)
        except Exception:
            logger.exception("Failed to build broker for account_id=%s", row.get("id"))
            broker = None
        if broker is None:
            continue
        out.append({
            "broker": broker,
            "account_id": int(row.get("id") or 0),
            "account_name": str(row.get("name") or f"account-{row.get('id')}"),
            "broker_type": (row.get("broker") or "").lower(),
        })

    if not out:
        # Env-based fallback (legacy single-account setups).
        broker, account_type = _get_active_broker()
        out.append({
            "broker": broker,
            "account_id": 0,
            "account_name": account_type.upper(),
            "broker_type": account_type,
        })
    return out


async def _find_broker_for_account_id(account_id: int) -> Optional[dict]:
    """Return the broker entry matching ``account_id`` or ``None``."""
    all_brokers = await _list_all_active_brokers()
    for entry in all_brokers:
        if int(entry["account_id"]) == int(account_id or 0):
            return entry
    return None


def _extract_positions_for_ui(broker: Any, account_name: str, account_id: int = 0) -> tuple[list[dict], dict]:
    """Read raw broker positions and normalize shape used by PositionsPage."""
    positions: list[dict] = []
    realised_sum = 0.0
    unrealised_sum = 0.0

    # Kite
    if hasattr(broker, "_client") and broker.__class__.__name__.lower().startswith("kite"):
        raw = broker._client.get_positions() or {}
        for p in (raw.get("net", []) or []):
            buy_qty = _safe_int(p.get("buy_quantity") or p.get("buy_qty"), 0)
            sell_qty = _safe_int(p.get("sell_quantity") or p.get("sell_qty"), 0)
            net_qty = _safe_int(p.get("quantity") or p.get("net_quantity") or p.get("net_qty"), 0)
            realised = _safe_num(p.get("realised") or p.get("realized"), 0.0)
            unrealised = _safe_num(p.get("unrealised") or p.get("unrealized") or p.get("m2m"), 0.0)
            pnl = _safe_num(p.get("pnl"), realised + unrealised)
            item = {
                "id": f"{account_id}:{account_name}:{p.get('exchange', '')}:{p.get('tradingsymbol', '')}",
                "tradingsymbol": p.get("tradingsymbol", ""),
                "symboltoken": str(p.get("instrument_token", "") or ""),
                "exchange": p.get("exchange", "NFO"),
                "product": p.get("product", ""),
                "buy_qty": buy_qty,
                "buy_avg": _safe_num(p.get("buy_price") or p.get("buy_average_price"), 0.0),
                "sell_qty": sell_qty,
                "sell_avg": _safe_num(p.get("sell_price") or p.get("sell_average_price"), 0.0),
                "net_qty": net_qty,
                "ltp": _safe_num(p.get("last_price"), 0.0),
                "pnl": pnl,
                "realised": realised,
                "unrealised": unrealised,
                "account_id": account_id,
                "account_name": account_name,
            }
            positions.append(item)
            realised_sum += realised
            unrealised_sum += unrealised

    # Dhan
    elif hasattr(broker, "_client") and broker.__class__.__name__.lower().startswith("dhan"):
        if broker._client is None:
            logger.warning(
                "Positions: Dhan client not initialised \u2014 check credentials in DB "
                "(broker_credentials table) or DHAN_CLIENT_ID/DHAN_ACCESS_TOKEN env"
            )
            raw = []
        else:
            raw = broker._client.get_positions() or []
            logger.info(
                "Positions: Dhan account=%s (id=%s) returned %d raw rows",
                account_name, account_id, len(raw),
            )
        for p in raw:
            buy_qty = _safe_int(p.get("buyQty") or p.get("buyQuantity"), 0)
            sell_qty = _safe_int(p.get("sellQty") or p.get("sellQuantity"), 0)
            net_qty = _safe_int(p.get("netQty") or p.get("quantity"), 0)
            realised = _safe_num(p.get("realizedProfit") or p.get("realised"), 0.0)
            unrealised = _safe_num(p.get("unrealizedProfit") or p.get("unrealised"), 0.0)
            pnl = _safe_num(p.get("pnl"), realised + unrealised)
            item = {
                "id": f"{account_id}:{account_name}:{p.get('exchangeSegment', '')}:{p.get('tradingSymbol', '')}",
                "tradingsymbol": p.get("tradingSymbol", ""),
                "symboltoken": str(p.get("securityId", "") or ""),
                "exchange": p.get("exchangeSegment", "NFO"),
                "product": p.get("productType", ""),
                "buy_qty": buy_qty,
                "buy_avg": _safe_num(p.get("buyAvg") or p.get("buyAverage"), 0.0),
                "sell_qty": sell_qty,
                "sell_avg": _safe_num(p.get("sellAvg") or p.get("sellAverage"), 0.0),
                "net_qty": net_qty,
                "ltp": _safe_num(p.get("lastTradedPrice") or p.get("ltp"), 0.0),
                "pnl": pnl,
                "realised": realised,
                "unrealised": unrealised,
                "account_id": account_id,
                "account_name": account_name,
            }
            positions.append(item)
            realised_sum += realised
            unrealised_sum += unrealised

    # Angel (or generic fallback)
    else:
        raw = []
        try:
            resp = broker.client._smart_api.position()  # noqa: SLF001
            raw = (resp or {}).get("data") or []
        except Exception:
            logger.exception("Failed to fetch Angel positions")
        for p in raw:
            buy_qty = _safe_int(p.get("buyqty") or p.get("buy_qty"), 0)
            sell_qty = _safe_int(p.get("sellqty") or p.get("sell_qty"), 0)
            net_qty = _safe_int(p.get("netqty") or p.get("quantity"), 0)
            realised = _safe_num(p.get("realised") or p.get("realized"), 0.0)
            pnl = _safe_num(p.get("pnl"), 0.0)
            unrealised = _safe_num(p.get("unrealised") or p.get("unrealized"), pnl - realised)
            item = {
                "id": f"{account_id}:{account_name}:{p.get('exchange', '')}:{p.get('tradingsymbol', '')}",
                "tradingsymbol": p.get("tradingsymbol", ""),
                "symboltoken": str(p.get("symboltoken", "") or ""),
                "exchange": p.get("exchange", "NFO"),
                "product": p.get("producttype", ""),
                "buy_qty": buy_qty,
                "buy_avg": _safe_num(p.get("buyavgprice") or p.get("buyaverageprice") or p.get("averageprice"), 0.0),
                "sell_qty": sell_qty,
                "sell_avg": _safe_num(p.get("sellavgprice") or p.get("sellaverageprice"), 0.0),
                "net_qty": net_qty,
                "ltp": _safe_num(p.get("ltp"), 0.0),
                "pnl": pnl,
                "realised": realised,
                "unrealised": unrealised,
                "account_id": account_id,
                "account_name": account_name,
            }
            positions.append(item)
            realised_sum += realised
            unrealised_sum += unrealised

    broker_day = {
        "realised": round(realised_sum, 2),
        "unrealised": round(unrealised_sum, 2),
        "total": round(realised_sum + unrealised_sum, 2),
    }
    return positions, broker_day


def _mark_scanners_completed_for_today(reason: str = "manual_user_exit") -> None:
    """Prevent immediate re-entry after manual position exit from UI."""
    orch = _state.get("orchestrator")
    if orch is None:
        return

    now_str = datetime.now(_IST).strftime("%H:%M")

    md = getattr(orch, "move_detection_scanner", None)
    if md is not None:
        try:
            md._signal_found_today = True  # noqa: SLF001
            tr = getattr(md, "_active_trade", None)
            if tr is not None and not getattr(tr, "exited", False):
                tr.exited = True
                tr.exit_reason = reason
                tr.exit_time = now_str
        except Exception:
            logger.exception("Failed to mark MoveDet complete after manual exit")

    mdb = getattr(orch, "move_detection_scanner_bull", None)
    if mdb is not None:
        try:
            mdb._signal_found_today = True  # noqa: SLF001
            tr = getattr(mdb, "_active_trade", None)
            if tr is not None and not getattr(tr, "exited", False):
                tr.exited = True
                tr.exit_reason = reason
                tr.exit_time = now_str
        except Exception:
            logger.exception("Failed to mark MoveDetBull complete after manual exit")

    pdh = getattr(orch, "pdh_pdl_scanner", None)
    if pdh is not None:
        try:
            pdh._signal_found_today = True  # noqa: SLF001
            tr = getattr(pdh, "_active_trade", None)
            if tr is not None and not getattr(tr, "exited", False):
                tr.exited = True
                tr.exit_reason = reason
                tr.exit_time = now_str
        except Exception:
            logger.exception("Failed to mark PDH/PDL complete after manual exit")

    atl = getattr(orch, "atl_straddle_scanner", None)
    if atl is not None:
        try:
            st = getattr(atl, "_state", None)  # noqa: SLF001
            if st is not None:
                st.done_for_day = True
                st.halted = False
                st.halt_reason = reason
                st.last_event = f"manual_exit:{reason}"
        except Exception:
            logger.exception("Failed to mark ATM complete after manual exit")


def _rearm_scanners_for_today() -> None:
    """Allow scanner entries again after user explicitly re-arms."""
    orch = _state.get("orchestrator")
    if orch is None:
        return

    md = getattr(orch, "move_detection_scanner", None)
    if md is not None:
        try:
            md._signal_found_today = False  # noqa: SLF001
        except Exception:
            logger.exception("Failed to re-arm MoveDet")

    mdb = getattr(orch, "move_detection_scanner_bull", None)
    if mdb is not None:
        try:
            mdb._signal_found_today = False  # noqa: SLF001
        except Exception:
            logger.exception("Failed to re-arm MoveDetBull")

    pdh = getattr(orch, "pdh_pdl_scanner", None)
    if pdh is not None:
        try:
            pdh._signal_found_today = False  # noqa: SLF001
        except Exception:
            logger.exception("Failed to re-arm PDH/PDL")

    atl = getattr(orch, "atl_straddle_scanner", None)
    if atl is not None:
        try:
            atl.reset_halt()
        except Exception:
            logger.exception("Failed to re-arm ATM")

    # Per-instance registry scanners — the legacy alias above only
    # points at the "primary" instance, so without this loop only ONE
    # of the user's per-account strategies would actually be re-armed.
    # We call ``reset_halt`` on each so that (a) the circuit breaker is
    # cleared, (b) the in-memory ``entered``/``phase``/leg state is
    # wiped, and (c) the settings-change ``future_rearm_blocked`` guard
    # in run_cycle() no longer fires — letting the scanner attempt a
    # fresh entry when its ``entry_time`` is reached.
    registry = getattr(orch, "strategy_registry", None)
    if registry is not None:
        try:
            scanners = list(registry.scanners.items())
        except Exception:
            scanners = []
        for iid, sc in scanners:
            try:
                sc.reset_halt()
            except Exception:
                logger.exception(
                    "Failed to re-arm registry scanner instance %s", iid,
                )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    await init_db()

    # Pattern Engine: seed defaults + start nightly scheduler (isolated subsystem)
    import os
    if os.environ.get("PATTERN_ENGINE_SCHEDULER", "1") != "0":
        try:
            from app.db.models import AsyncSessionLocal as _PE_Session
            from app.pattern_engine.seed import upsert_seed_patterns
            from app.pattern_engine.scheduler import start_scheduler

            async with _PE_Session() as _s:
                inserted = await upsert_seed_patterns(_s)
                logger.info("pattern_engine: seed upsert ok (inserted=%d)", inserted)
            start_scheduler()
        except Exception as _pe_e:
            logger.warning("pattern_engine startup hook failed: %s", _pe_e)

    # Condor Setup: pre-market confluence-based condor levels job (isolated,
    # read-only, informational subsystem — never places live orders).
    if os.environ.get("CONDOR_SETUP_SCHEDULER", "1") != "0":
        try:
            from app.condor_setup.scheduler import start_scheduler as _cs_start
            _cs_start()
        except Exception as _cs_e:
            logger.warning("condor_setup startup hook failed: %s", _cs_e)

    if os.environ.get("SKIP_ORCHESTRATOR") == "1":
        logger.info("SKIP_ORCHESTRATOR=1 — skipping orchestrator (backtest-only mode)")
        yield
        return

    # Start orchestrator as a background task in the SAME event loop
    # (avoids asyncpg "Future attached to a different loop" errors)
    from app.engine.orchestrator import Orchestrator

    async def _run_orchestrator():
        orchestrator = Orchestrator()
        state = get_state()
        state["orchestrator"] = orchestrator
        state["eval_scheduler"] = orchestrator.eval_scheduler
        await orchestrator.start()

    orchestrator_task = asyncio.create_task(_run_orchestrator())

    # Account-level kill-switch watchdog (Dhan only — polls positions
    # every few seconds and force-closes everything if total PnL
    # crosses the configured daily loss limit).
    try:
        from app.execution.account_kill_switch import start_watchdog as _aks_start
        _aks_start()
    except Exception:
        logger.exception("Failed to start account kill switch watchdog")

    yield

    # Shutdown
    logger.info("Shutting down...")
    # Stop pattern-engine scheduler if running
    try:
        from app.pattern_engine.scheduler import stop_scheduler
        stop_scheduler()
    except Exception:
        pass
    # Stop the condor-setup scheduler if running
    try:
        from app.condor_setup.scheduler import stop_scheduler as _cs_stop
        _cs_stop()
    except Exception:
        pass
    # Stop the kill-switch watchdog.
    try:
        from app.execution.account_kill_switch import stop_watchdog as _aks_stop
        await _aks_stop()
    except Exception:
        logger.debug("Kill switch stop hook failed", exc_info=True)
    # Safety kill-switch: prevent any new real order execution during shutdown.
    settings.paper_trading = True
    orch = _state.get("orchestrator")
    if orch is not None:
        try:
            await orch.stop(reason="app_shutdown")
        except Exception:
            logger.exception("Orchestrator stop hook failed during shutdown")
    orchestrator_task.cancel()
    try:
        await orchestrator_task
    except asyncio.CancelledError:
        pass


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


# ── Condor Setup routes (isolated, read-only subsystem) ─────────────────
try:
    from app.condor_setup.routes import register_routes as _register_cs_routes
    _register_cs_routes(app)
except Exception as _cs_err:  # pragma: no cover
    logger.warning("condor_setup routes not registered: %s", _cs_err)


# ── Level Zones routes (isolated, read-only, informational-only subsystem) ──
try:
    from app.level_zones.routes import register_routes as _register_lz_routes
    _register_lz_routes(app)
except Exception as _lz_err:  # pragma: no cover
    logger.warning("level_zones routes not registered: %s", _lz_err)


# ── Pattern Engine routes (isolated subsystem) ───────────────────────────
try:
    from app.pattern_engine.routes import register_routes as _register_pe_routes
    _register_pe_routes(app)
except Exception as _pe_err:  # pragma: no cover
    logger.warning("pattern_engine routes not registered: %s", _pe_err)


# ── Multi-account / multi-strategy CRUD routes ───────────────────────────
try:
    from app.api.multi_account_routes import register_multi_account_routes
    register_multi_account_routes(app)
except Exception as _ma_err:  # pragma: no cover
    logger.warning("multi_account routes not registered: %s", _ma_err)


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


# ── V2 Engine ────────────────────────────────────────────────────────────

@app.get("/api/strategy-selection")
async def get_strategy_selection():
    """Get today's strategy selection with conditions and probabilities."""
    orch = _state.get("orchestrator")
    selector = _state.get("strategy_selector")

    if not selector and orch:
        selector = getattr(orch, "strategy_selector", None)

    if not selector or not selector.latest_selections:
        # Return default info
        return {
            "selections": [],
            "day_type": _state.get("day_type", "pending"),
            "active_strategies": ["TREND_PULLBACK", "MOMENTUM_BREAKOUT"],
            "message": "No condition-based selection yet — using defaults",
        }

    selections = []
    for symbol, result in selector.latest_selections.items():
        selections.append(result.to_dict())

    day_type = "pending"
    if orch and hasattr(orch, "day_type"):
        day_type = orch.day_type.value if orch.day_type else "pending"

    active = []
    if orch and hasattr(orch, "strategies"):
        active = [type(s).__name__.replace("Strategy", "").upper() for s in orch.strategies]

    return {
        "selections": selections,
        "day_type": day_type,
        "active_strategies": active,
    }


@app.get("/api/performance/comparison")
async def get_performance_comparison():
    """Performance breakdown by strategy."""
    all_trades = await trade_logger.get_today_trades()
    # Group by strategy instead of V1/V2
    from collections import defaultdict
    by_strategy: dict[str, list] = defaultdict(list)
    for t in all_trades:
        strat = t.strategy.value if hasattr(t.strategy, 'value') else str(t.strategy)
        by_strategy[strat].append(t)

    result = {}
    for strat, trades in by_strategy.items():
        metrics = await trade_logger.compute_performance(trades)
        result[strat] = metrics

    return result


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

    # If no in-memory alerts for requested date, fall back to DB
    if not filtered:
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

    # Quick DB connectivity check (cached briefly to avoid DB hit per dashboard tick)
    db_ok = False
    db_error: str | None = None
    cached_db = await cache.get_json("system:db_ping")
    if cached_db is not None:
        db_ok = bool(cached_db.get("ok"))
        db_error = cached_db.get("error")
    else:
        try:
            from sqlalchemy import text
            from app.db.models import AsyncSessionLocal
            async with AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
            db_ok = True
        except Exception as e:
            db_error = str(e)[:300]
            logger.warning("DB health check failed: %s", e)
        await cache.set_json(
            "system:db_ping",
            {"ok": db_ok, "error": db_error},
            ttl_seconds=5,
        )

    # WebSocket status
    ws_status = "not_started"
    ws_last_tick = None
    ws_subscriptions = 0
    if orch and hasattr(orch, "client") and orch.client.ws:
        ws = orch.client.ws
        ws_subscriptions = len(ws._subscriptions)
        if ws.is_connected:
            if ws.is_stale:
                ws_status = "stale"
            else:
                ws_status = "connected"
        elif ws._running:
            ws_status = "reconnecting"
        else:
            ws_status = "disconnected"
        tick_time = ws.last_tick_time
        if tick_time:
            ws_last_tick = tick_time.strftime("%H:%M:%S")

    # Scanner status
    scanner_info = {}
    if orch:
        move_det = getattr(orch, "move_detection_scanner", None)
        move_det_bull = getattr(orch, "move_detection_scanner_bull", None)
        pdh_pdl = getattr(orch, "pdh_pdl_scanner", None)
        atm = getattr(orch, "atl_straddle_scanner", None)
        if move_det:
            md_trade = move_det._active_trade
            scanner_info["move_det"] = {
                "active": True,
                "day_tradeable": move_det._day_tradeable,
                "signal_found": move_det._signal_found_today,
                "in_trade": md_trade is not None and not md_trade.exited if md_trade else False,
                "last_trade_week": move_det._last_trade_week,
            }
        if move_det_bull:
            mdb_trade = move_det_bull._active_trade
            scanner_info["move_det_bull"] = {
                "active": True,
                "day_tradeable": move_det_bull._day_tradeable,
                "signal_found": move_det_bull._signal_found_today,
                "in_trade": mdb_trade is not None and not mdb_trade.exited if mdb_trade else False,
                "last_trade_week": move_det_bull._last_trade_week,
            }
        if pdh_pdl:
            p_trade = pdh_pdl._active_trade
            scanner_info["pdh_pdl"] = {
                "active": settings.pdh_pdl_scanner_enabled,
                "setup_checked": pdh_pdl._setup_checked,
                "is_tradeable_day": pdh_pdl._is_tradeable_day,
                "signal_found": pdh_pdl._signal_found_today,
                "in_trade": p_trade is not None and not p_trade.exited if p_trade else False,
                "prev_high": pdh_pdl._prev_high,
                "prev_low": pdh_pdl._prev_low,
                "trade": (
                    {
                        "side": p_trade.side,
                        "entry_time": p_trade.entry_time,
                        "entry_spot": p_trade.entry_spot,
                        "stop_level": p_trade.stop_level,
                        "target_level": p_trade.target_level,
                        "option_symbol": p_trade.option_symbol,
                        "option_entry_price": p_trade.option_entry_price,
                        "exited": p_trade.exited,
                        "exit_time": p_trade.exit_time,
                        "exit_spot": p_trade.exit_spot,
                        "exit_reason": p_trade.exit_reason,
                        "option_exit_price": p_trade.option_exit_price,
                    }
                    if p_trade else None
                ),
            }
        if atm:
            try:
                rt = atm.get_runtime_state()
            except Exception:
                rt = {}
            latest_event = ""
            try:
                ev = (rt.get("events") or [])
                if ev:
                    latest_event = str(ev[-1].get("message") or "")
            except Exception:
                latest_event = ""
            scanner_info["atm"] = {
                "active": bool(rt.get("enabled", False)),
                "in_trade": bool(rt.get("in_trade", False)),
                "phase": rt.get("phase", "IDLE"),
                "halted": bool(rt.get("halted", False)),
                "halt_reason": rt.get("halt_reason", ""),
                "index": rt.get("index", "NIFTY"),
                "expiry": rt.get("expiry", ""),
                "live_mode": bool(rt.get("live_mode", False)),
                "last_event": latest_event,
            }

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
        "db_error": db_error,
        "scanners": scanner_info,
        "websocket": {
            "status": ws_status,
            "last_tick": ws_last_tick,
            "subscriptions": ws_subscriptions,
        },
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
        "missed_signals": getattr(orch, "_missed_signals", [])[-20:],
    }


@app.post("/api/system/start")
async def start_system():
    """Start the trading system for the current day."""
    orch = _state.get("orchestrator")
    if not orch:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    if orch.running:
        return {"message": "System already running"}
    # Guard against duplicate starts: only one manual trading-day task at a time.
    task = _state.get("manual_start_task")
    if task is not None and not task.done():
        return {"message": "System start already in progress"}

    async def _run_manual_day_once():
        try:
            await orch._run_trading_day()
        except asyncio.CancelledError:
            logger.info("Manual trading day task cancelled")
            raise
        except Exception as e:
            logger.exception("Manual trading day task crashed: %s", e)
            orch.running = False
            if hasattr(orch, "_log_event"):
                orch._log_event("error", f"Manual trading day crashed: {e}")
        finally:
            _state["manual_start_task"] = None

    _state["manual_start_task"] = asyncio.create_task(_run_manual_day_once())
    return {"message": "System start requested"}


@app.post("/api/system/stop")
async def stop_system():
    """Stop the trading system."""
    orch = _state.get("orchestrator")
    if orch:
        orch.running = False
        task = _state.get("manual_start_task")
        if task is not None and not task.done():
            task.cancel()
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
    # Persist so the mode survives container restarts
    try:
        _persist_env_vars({"PAPER_TRADING": "false" if mode == "live" else "true"})
    except Exception:
        logger.exception("Failed to persist PAPER_TRADING to .env")
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


# ── Broker Replica Positions + Trade History ─────────────────────────────

@app.get("/api/positions")
async def get_broker_positions():
    """Return active broker positions in UI-friendly shape (multi-account)."""
    cache_key = "positions:all"

    cached_payload = await cache.get_json(cache_key)
    if cached_payload is not None:
        return cached_payload

    try:
        brokers = await _list_all_active_brokers()

        async def _fetch_one(entry: dict) -> tuple[list[dict], dict]:
            try:
                return await asyncio.to_thread(
                    _extract_positions_for_ui,
                    entry["broker"],
                    entry["account_name"].upper(),
                    int(entry["account_id"]),
                )
            except Exception:
                logger.exception(
                    "Failed to fetch broker positions for account_id=%s",
                    entry.get("account_id"),
                )
                return [], {"realised": 0.0, "unrealised": 0.0, "total": 0.0}

        results = await asyncio.gather(*(_fetch_one(e) for e in brokers))
    except Exception as exc:
        logger.exception("Failed to fetch broker positions (multi-account)")
        raise HTTPException(status_code=500, detail=str(exc))

    all_positions: list[dict] = []
    total_realised = 0.0
    total_unrealised = 0.0
    per_account: list[dict] = []
    for entry, (pos, day) in zip(brokers, results):
        all_positions.extend(pos)
        total_realised += float(day.get("realised") or 0.0)
        total_unrealised += float(day.get("unrealised") or 0.0)
        per_account.append({
            "account_id": int(entry["account_id"]),
            "account_name": entry["account_name"].upper(),
            "broker_type": entry["broker_type"],
            "realised": day.get("realised", 0.0),
            "unrealised": day.get("unrealised", 0.0),
            "total": day.get("total", 0.0),
        })

    payload = {
        "status": "ok",
        "positions": all_positions,
        "broker_day_pnl": {
            "realised": round(total_realised, 2),
            "unrealised": round(total_unrealised, 2),
            "total": round(total_realised + total_unrealised, 2),
        },
        "accounts": per_account,
    }
    # Short TTL: dashboard polls every 10s, broker call typically 200-500ms.
    # 3s cap keeps positions essentially live while collapsing duplicate hits
    # from multiple tabs/users.
    await cache.set_json(cache_key, payload, ttl_seconds=3)
    return payload


@app.post("/api/positions/exit")
async def exit_broker_positions(body: dict):
    """Exit selected broker positions and mark strategy done for the day.

    Each position item MUST carry its ``account_id`` and either
    ``security_id`` (Dhan) or ``symboltoken`` so the exit is routed to
    the correct account's broker without re-resolving the display
    trading symbol.
    """
    from app.execution.broker_base import OrderRequest, OrderSide, OrderType

    items = (body or {}).get("positions") or []
    if not isinstance(items, list) or not items:
        raise HTTPException(status_code=400, detail="No positions provided")

    all_brokers = await _list_all_active_brokers()
    brokers_by_id: dict[int, dict] = {
        int(e["account_id"]): e for e in all_brokers
    }

    results: list[dict] = []
    used_accounts: set[int] = set()

    for p in items:
        qty = abs(_safe_int(p.get("net_qty"), 0))
        if qty <= 0:
            continue
        account_id = int(_safe_int(p.get("account_id"), 0))
        entry = brokers_by_id.get(account_id)
        if entry is None:
            # Fall back to the sole active broker when the caller didn't
            # send an account_id (legacy single-account clients).
            if len(all_brokers) == 1:
                entry = all_brokers[0]
                account_id = int(entry["account_id"])
            else:
                results.append({
                    "tradingsymbol": str(p.get("tradingsymbol", "") or ""),
                    "quantity": qty,
                    "ok": False,
                    "order_id": "",
                    "message": f"account_id={account_id} not found among active brokers",
                })
                continue
        broker = entry["broker"]
        side = OrderSide.SELL if _safe_int(p.get("net_qty"), 0) > 0 else OrderSide.BUY
        symbol_token = str(p.get("symboltoken", "") or "")
        req = OrderRequest(
            instrument=None,
            trading_symbol=str(p.get("tradingsymbol", "") or ""),
            symbol_token=symbol_token,
            exchange=str(p.get("exchange", "NFO") or "NFO"),
            side=side,
            order_type=OrderType.MARKET,
            product_type=_map_product_type(str(p.get("product", "") or "")),
            quantity=qty,
            price=0.0,
            trigger_price=0.0,
            # Dhan-native identifiers so the broker skips symbol
            # re-resolution (display symbol is broker-formatted and
            # cannot be reparsed back to strike/expiry).
            broker_security_id=str(p.get("security_id") or symbol_token) or None,
            broker_exchange_segment=str(p.get("exchange", "") or "") or None,
        )
        try:
            resp = await asyncio.to_thread(broker.place_order, req)
            ok = bool(resp and str(getattr(resp, "status", "")).upper() != "REJECTED")
            results.append({
                "account_id": account_id,
                "account_name": entry["account_name"],
                "tradingsymbol": req.trading_symbol,
                "quantity": qty,
                "side": side.value,
                "ok": ok,
                "order_id": getattr(resp, "order_id", "") if resp else "",
                "message": getattr(resp, "message", "") if resp else "",
            })
            if ok:
                used_accounts.add(account_id)
        except Exception as exc:
            logger.exception(
                "Manual exit failed account_id=%s symbol=%s",
                account_id, req.trading_symbol,
            )
            results.append({
                "account_id": account_id,
                "account_name": entry["account_name"],
                "tradingsymbol": req.trading_symbol,
                "quantity": qty,
                "side": side.value,
                "ok": False,
                "order_id": "",
                "message": str(exc),
            })

    if used_accounts:
        _mark_scanners_completed_for_today()

    # Bust positions cache so the next poll reflects the exit immediately.
    await cache.delete_prefix("positions:")

    return {
        "status": "ok",
        "results": results,
        "accounts_hit": sorted(used_accounts),
        "completed_for_today": bool(used_accounts),
    }


@app.post("/api/positions/rearm")
async def rearm_after_manual_exit():
    """Clear manual completion flags so strategies can place entries again today."""
    _rearm_scanners_for_today()
    await cache.delete_prefix("positions:")
    return {"status": "ok", "message": "Strategies re-armed for today"}


@app.post("/api/eod/snapshot")
async def capture_broker_eod_snapshot():
    """Persist current broker PnL/positions snapshot for history calendar."""
    from app.db.models import AsyncSessionLocal, BrokerEODSnapshot

    brokers = await _list_all_active_brokers()
    today = datetime.now(_IST).strftime("%Y-%m-%d")

    per_account: list[dict] = []
    for entry in brokers:
        try:
            positions, broker_day = await asyncio.to_thread(
                _extract_positions_for_ui,
                entry["broker"],
                entry["account_name"].upper(),
                int(entry["account_id"]),
            )
        except Exception:
            logger.exception(
                "EOD snapshot: extract failed account_id=%s",
                entry.get("account_id"),
            )
            continue

        async with AsyncSessionLocal() as session:
            async with session.begin():
                session.add(
                    BrokerEODSnapshot(
                        date=today,
                        account_name=entry["account_name"].upper(),
                        broker_pnl=float(broker_day.get("total", 0.0) or 0.0),
                        realised=float(broker_day.get("realised", 0.0) or 0.0),
                        unrealised=float(broker_day.get("unrealised", 0.0) or 0.0),
                        positions_count=len(positions),
                        payload_json=json.dumps(positions),
                    )
                )
        per_account.append({
            "account_id": int(entry["account_id"]),
            "account_name": entry["account_name"].upper(),
            "positions_count": len(positions),
            "broker_day_pnl": broker_day,
        })

    return {
        "status": "ok",
        "date": today,
        "accounts": per_account,
    }


@app.get("/api/trade-history")
async def get_trade_history(
    date: str | None = None,
    month: str | None = None,
    account_id: int = 0,
):
    """Unified history API consumed by HistoryPage.

    Query params:
      - date=YYYY-MM-DD   -> day detail (with per-account broker snapshots)
      - month=YYYY-MM     -> month summary calendar + per-account rollup
      - account_id=<int>  -> filter to a specific BrokerAccount (0/omit = all)
    """
    from collections import defaultdict
    from sqlalchemy import select
    from app.db.account_models import BrokerAccount
    from app.db.models import AsyncSessionLocal, BrokerEODSnapshot, TradeRecord

    target_date = (date or "").strip()
    target_month = (month or "").strip()

    # ── Helpers ────────────────────────────────────────────────────
    def _snap_matches(snap: "BrokerEODSnapshot", acct: "BrokerAccount") -> bool:
        """Best-effort snapshot ↔ account matcher.

        Current EOD snapshot code stores account_name as UPPER(broker) e.g.
        "DHAN" (one row per broker type). Match by either broker or name.
        """
        got = (snap.account_name or "").strip().upper()
        candidates = {
            (acct.broker or "").strip().upper(),
            (acct.name or "").strip().upper(),
        }
        candidates.discard("")
        return got in candidates

    def _friendly_name(snap: "BrokerEODSnapshot", accts: list["BrokerAccount"]) -> str:
        for a in accts:
            if _snap_matches(snap, a):
                return a.name or (a.broker or "").upper() or "Primary"
        return snap.account_name or "Primary"

    def _account_id_for(name: str, accts: list["BrokerAccount"]) -> int | None:
        up = (name or "").strip().upper()
        for a in accts:
            if up and (up == (a.name or "").strip().upper() or up == (a.broker or "").strip().upper()):
                return a.id
        return None

    # ── Day detail ────────────────────────────────────────────────
    if target_date:
        async with AsyncSessionLocal() as session:
            accts = (await session.execute(select(BrokerAccount))).scalars().all()
            filter_acct = next((a for a in accts if a.id == account_id), None) if account_id else None

            trade_rows = (
                await session.execute(
                    select(TradeRecord).where(TradeRecord.date == target_date)
                )
            ).scalars().all()

            snap_rows = (
                await session.execute(
                    select(BrokerEODSnapshot).where(BrokerEODSnapshot.date == target_date)
                )
            ).scalars().all()

        if filter_acct is not None:
            snap_rows = [s for s in snap_rows if _snap_matches(s, filter_acct)]

        trade_pnl = round(sum(_safe_num(t.pnl, 0.0) for t in trade_rows), 2)
        broker_pnl = round(sum(_safe_num(s.broker_pnl, 0.0) for s in snap_rows), 2)

        broker_accounts = [
            {
                "account_id": _account_id_for(_friendly_name(s, accts), accts),
                "account_name": _friendly_name(s, accts),
                "broker_pnl": round(_safe_num(s.broker_pnl, 0.0), 2),
                "positions": _safe_int(s.positions_count, 0),
            }
            for s in snap_rows
        ]

        return {
            "status": "ok",
            "date": target_date,
            "total_trades": len(trade_rows),
            "month_pnl": trade_pnl,
            "broker_pnl": broker_pnl if snap_rows else trade_pnl,
            "broker_accounts": broker_accounts,
        }

    # ── Month calendar ────────────────────────────────────────────
    if not target_month:
        target_month = datetime.now(_IST).strftime("%Y-%m")

    try:
        year, mon = target_month.split("-")
        start_date = f"{int(year):04d}-{int(mon):02d}-01"
        dt = datetime(int(year), int(mon), 1)
        next_month_dt = datetime(
            dt.year + (1 if dt.month == 12 else 0),
            1 if dt.month == 12 else dt.month + 1,
            1,
        )
        end_date = next_month_dt.strftime("%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid month format. Use YYYY-MM")

    async with AsyncSessionLocal() as session:
        accts = (await session.execute(select(BrokerAccount))).scalars().all()
        filter_acct = next((a for a in accts if a.id == account_id), None) if account_id else None

        trade_rows = (
            await session.execute(
                select(TradeRecord).where(
                    TradeRecord.date >= start_date,
                    TradeRecord.date < end_date,
                )
            )
        ).scalars().all()

        snap_rows_all = (
            await session.execute(
                select(BrokerEODSnapshot).where(
                    BrokerEODSnapshot.date >= start_date,
                    BrokerEODSnapshot.date < end_date,
                )
            )
        ).scalars().all()

    if filter_acct is not None:
        snap_rows_all = [s for s in snap_rows_all if _snap_matches(s, filter_acct)]

    # Aggregate trade rows per date
    trade_map: dict[str, dict] = defaultdict(lambda: {"trades": 0, "trade_pnl": 0.0})
    for t in trade_rows:
        d = t.date
        trade_map[d]["trades"] += 1
        trade_map[d]["trade_pnl"] += _safe_num(t.pnl, 0.0)

    # Group snapshots by date
    snap_by_date: dict[str, list] = defaultdict(list)
    for s in snap_rows_all:
        snap_by_date[s.date].append(s)

    # Per-account monthly rollup
    acct_monthly_pnl: dict[str, float] = defaultdict(float)
    acct_monthly_days: dict[str, set] = defaultdict(set)

    all_dates = sorted(set(trade_map.keys()) | set(snap_by_date.keys()))
    days: list[dict] = []
    for d in all_dates:
        trades = int(trade_map[d]["trades"])
        t_pnl = float(trade_map[d]["trade_pnl"])
        d_snaps = snap_by_date.get(d, [])

        day_accounts: list[dict] = []
        b_pnl_total = 0.0
        for s in d_snaps:
            name = _friendly_name(s, accts)
            pnl_val = _safe_num(s.broker_pnl, 0.0)
            b_pnl_total += pnl_val
            day_accounts.append({
                "account_id": _account_id_for(name, accts),
                "account_name": name,
                "pnl": round(pnl_val, 2),
                "positions": _safe_int(s.positions_count, 0),
            })
            acct_monthly_pnl[name] += pnl_val
            acct_monthly_days[name].add(d)

        has_snap = bool(d_snaps)
        days.append({
            "date": d,
            "trades": trades,
            "pnl": round(t_pnl, 2),
            "broker_pnl": round(b_pnl_total, 2) if has_snap else round(t_pnl, 2),
            "source": "eod_snapshot" if has_snap else "trades",
            "accounts": day_accounts,
        })

    month_pnl = round(
        sum(
            (d.get("broker_pnl") if d.get("source") == "eod_snapshot" else d.get("pnl", 0.0))
            for d in days
        ),
        2,
    )
    winning_days = len(
        [d for d in days if (d.get("broker_pnl") if d.get("source") == "eod_snapshot" else d.get("pnl", 0.0)) > 0]
    )
    losing_days = len(
        [d for d in days if (d.get("broker_pnl") if d.get("source") == "eod_snapshot" else d.get("pnl", 0.0)) < 0]
    )

    accounts_summary = [
        {
            "account_id": _account_id_for(name, accts),
            "account_name": name,
            "pnl": round(pnl, 2),
            "trading_days": len(acct_monthly_days[name]),
        }
        for name, pnl in sorted(acct_monthly_pnl.items())
    ]

    return {
        "status": "ok",
        "month": target_month,
        "month_pnl": month_pnl,
        "trading_days": len(days),
        "winning_days": winning_days,
        "losing_days": losing_days,
        "month_trades": int(sum(d.get("trades", 0) for d in days)),
        "days": days,
        "accounts": accounts_summary,
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
        finally:
            await cache.delete_prefix("strategy_analytics:db:")
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

    async def _load() -> list:
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

    return await cache.cached(
        f"intel:history:{limit}", ttl_seconds=60, loader=_load,
    ) or []


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
        finally:
            await cache.delete_prefix("intel:history:")

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


# ── Broker Settings ─────────────────────────────────────────────────────

@app.get("/api/broker/status")
async def get_broker_status():
    """Get AngelOne broker connection status and auth health."""
    orch = _state.get("orchestrator")
    client = orch.client if orch and hasattr(orch, "client") else None

    result = {
        "configured": bool(settings.angelone_api_key and settings.angelone_client_id),
        "api_key_set": bool(settings.angelone_api_key),
        "client_id_set": bool(settings.angelone_client_id),
        "mpin_set": bool(settings.angelone_mpin),
        "totp_secret_set": bool(settings.angelone_totp_secret),
        "authenticated": False,
        "last_auth": None,
        "auth_error": None,
        "client_id": settings.angelone_client_id[:2] + "***" if settings.angelone_client_id else None,
    }

    if client:
        result["authenticated"] = client._auth_token is not None
        if client._last_auth:
            result["last_auth"] = client._last_auth.isoformat()
            # Check if token is stale
            age = (datetime.now(_IST) - client._last_auth).total_seconds()
            result["token_age_minutes"] = round(age / 60, 1)
            result["token_stale"] = age > 7200  # >2 hours

    return result


@app.post("/api/broker/test")
async def test_broker_connection():
    """Test AngelOne authentication with current credentials.

    Attempts a fresh login and reports success/failure with details.
    """
    from SmartApi import SmartConnect

    api_key = settings.angelone_api_key
    client_id = settings.angelone_client_id
    credential = settings.angelone_mpin or settings.angelone_password
    totp_secret = settings.angelone_totp_secret

    if not api_key:
        return {"success": False, "error": "ANGELONE_API_KEY is not configured"}
    if not client_id:
        return {"success": False, "error": "ANGELONE_CLIENT_ID is not configured"}
    if not credential:
        return {"success": False, "error": "ANGELONE_MPIN / ANGELONE_PASSWORD is not configured"}
    if not totp_secret:
        return {"success": False, "error": "ANGELONE_TOTP_SECRET is not configured"}

    try:
        smart_api = SmartConnect(api_key=api_key)
        totp = pyotp.TOTP(totp_secret).now()
        data = smart_api.generateSession(client_id, credential, totp)

        if not data or data.get("status") is False:
            error_msg = data.get("message", "Unknown error") if data else "No response"
            error_code = data.get("errorcode", "") if data else ""
            return {
                "success": False,
                "error": f"{error_msg} (code: {error_code})" if error_code else error_msg,
                "raw_response": str(data),
            }

        return {
            "success": True,
            "message": "Authentication successful",
            "client_id": client_id[:2] + "***",
        }
    except Exception as e:
        logger.exception("Broker test connection failed")
        return {"success": False, "error": str(e)}


@app.post("/api/broker/update-credentials")
async def update_broker_credentials(body: dict):
    """Update AngelOne credentials in .env file and reload settings.

    Only updates fields that are provided (non-empty).
    Requires system to be stopped.
    """
    import os
    import re

    orch = _state.get("orchestrator")
    if orch and getattr(orch, "running", False):
        raise HTTPException(
            status_code=409,
            detail="Cannot update credentials while system is running. Stop the system first.",
        )

    field_map = {
        "api_key": "ANGELONE_API_KEY",
        "client_id": "ANGELONE_CLIENT_ID",
        "password": "ANGELONE_PASSWORD",
        "mpin": "ANGELONE_MPIN",
        "totp_secret": "ANGELONE_TOTP_SECRET",
    }

    updates = {}
    for field, env_var in field_map.items():
        value = body.get(field, "").strip()
        if value:
            updates[env_var] = value

    if not updates:
        raise HTTPException(status_code=400, detail="No credentials provided")

    # Update .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    env_path = os.path.abspath(env_path)

    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = ""

    for env_var, value in updates.items():
        # Replace existing line or append
        pattern = rf"^{re.escape(env_var)}=.*$"
        if re.search(pattern, content, re.MULTILINE):
            content = re.sub(pattern, f"{env_var}={value}", content, flags=re.MULTILINE)
        else:
            content = content.rstrip() + f"\n{env_var}={value}\n"

    with open(env_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Reload settings in memory
    for env_var, value in updates.items():
        attr_name = env_var.lower()
        if hasattr(settings, attr_name):
            object.__setattr__(settings, attr_name, value)

    # Force re-auth on next use
    if orch and hasattr(orch, "client"):
        orch.client._last_auth = None
        orch.client._smart_api = None
        orch.client._auth_token = None

    updated_fields = list(updates.keys())
    logger.info("Broker credentials updated: %s", updated_fields)
    return {"success": True, "updated_fields": updated_fields}


@app.post("/api/broker/re-authenticate")
async def re_authenticate_broker():
    """Force re-authentication with AngelOne using current credentials."""
    orch = _state.get("orchestrator")
    if not orch or not hasattr(orch, "client"):
        raise HTTPException(status_code=503, detail="AngelOne client not initialized")

    client = orch.client
    client._last_auth = None
    client._smart_api = None
    client._auth_token = None

    success = client.authenticate()
    if success:
        return {"success": True, "message": "Re-authenticated successfully"}
    return {"success": False, "error": "Authentication failed — check logs for details"}


# ── Trading account selector (Angel / Kite / Dhan) ────────────────────────

_VALID_TRADING_ACCOUNTS = {"angel", "kite", "dhan"}


def _persist_env_vars(updates: dict[str, str]) -> None:
    """Write/replace env vars in backend/.env and reload `settings` in-memory.

    Also pushes the values into ``os.environ`` so a subsequent
    ``Settings()`` reload picks them up, and re-runs ``settings.__init__``
    so any validators / aliasing logic re-applies the new values.
    """
    import os
    import re

    env_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    )
    content = ""
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read()
    for env_var, value in updates.items():
        pattern = rf"^{re.escape(env_var)}=.*$"
        if re.search(pattern, content, re.MULTILINE):
            content = re.sub(pattern, f"{env_var}={value}", content, flags=re.MULTILINE)
        else:
            content = content.rstrip() + f"\n{env_var}={value}\n"
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Push into the process environment so pydantic-settings sees the new
    # values on re-init (env vars take precedence over the .env file).
    for env_var, value in updates.items():
        os.environ[env_var] = value
        attr = env_var.lower()
        if hasattr(settings, attr):
            try:
                # Coerce string env value to the existing attribute's type so
                # we don't clobber booleans/ints with raw strings (e.g.
                # PAPER_TRADING="false" → bool False, not the truthy string).
                current = getattr(settings, attr)
                coerced: object = value
                if isinstance(current, bool):
                    coerced = str(value).strip().lower() in ("1", "true", "yes", "on")
                elif isinstance(current, int) and not isinstance(current, bool):
                    try:
                        coerced = int(value)
                    except (TypeError, ValueError):
                        coerced = current
                elif isinstance(current, float):
                    try:
                        coerced = float(value)
                    except (TypeError, ValueError):
                        coerced = current
                object.__setattr__(settings, attr, coerced)
            except Exception:
                logger.exception("Failed to set settings.%s in-memory", attr)

    # Re-instantiate broker credentials so the running adapter picks up
    # rotated tokens without a process restart. Best-effort — failures are
    # logged but don't block the credential save.
    #
    # NOTE: orch.broker is always AngelOneBroker. The Dhan/Kite brokers used
    # for live order routing live on the scanners (built by `_build_atl_broker`
    # at orchestrator init). We must reload all of them so a freshly-rotated
    # DHAN_ACCESS_TOKEN / KITE_ACCESS_TOKEN actually takes effect mid-session.
    try:
        orch = _state.get("orchestrator")
        if orch is not None:
            seen_brokers: set[int] = set()
            broker_holders = [
                orch,
                getattr(orch, "move_detection_scanner", None),
                getattr(orch, "move_detection_scanner_bull", None),
                getattr(orch, "pdh_pdl_scanner", None),
                getattr(orch, "atl_straddle_scanner", None),
            ]
            for holder in broker_holders:
                if holder is None:
                    continue
                broker = getattr(holder, "broker", None)
                if broker is None or id(broker) in seen_brokers:
                    continue
                seen_brokers.add(id(broker))
                if hasattr(broker, "reload_credentials"):
                    try:
                        broker.reload_credentials()
                        logger.info(
                            "Reloaded credentials on %s (held by %s)",
                            type(broker).__name__, type(holder).__name__,
                        )
                    except Exception:
                        logger.exception(
                            "reload_credentials failed on %s", type(broker).__name__,
                        )
    except Exception:
        logger.exception("Failed to reload broker credentials after env update")


@app.get("/api/broker/trading-account")
async def get_trading_account():
    """Return the currently active trading account and orchestrator state."""
    orch = _state.get("orchestrator")
    selected = (settings.trading_account or "angel").lower()
    return {
        "selected": selected,
        "active": selected,
        "running": bool(orch and getattr(orch, "running", False)),
        "options": sorted(_VALID_TRADING_ACCOUNTS),
    }


@app.post("/api/broker/trading-account")
async def set_trading_account(body: dict):
    """Switch the active trading account. System must be stopped."""
    account = (body.get("account") or "").strip().lower()
    if account not in _VALID_TRADING_ACCOUNTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid account '{account}'. Must be one of {sorted(_VALID_TRADING_ACCOUNTS)}.",
        )
    orch = _state.get("orchestrator")
    if orch and getattr(orch, "running", False):
        raise HTTPException(
            status_code=409,
            detail="Cannot switch trading account while system is running. Stop the system first.",
        )
    _persist_env_vars({"TRADING_ACCOUNT": account})
    logger.info("Trading account switched to %s", account)
    return {"success": True, "selected": account, "message": f"Trading account set to {account}"}


# ── Kite (Zerodha) — daily OAuth + credential management ──────────────────

def _kite_configured() -> bool:
    return bool(settings.kite_api_key and settings.kite_api_secret)


@app.get("/api/broker/kite/status")
async def get_kite_status():
    """Kite credential + access-token health for the BrokerSettings UI."""
    api_key = settings.kite_api_key
    api_secret = settings.kite_api_secret
    access_token = settings.kite_access_token

    result = {
        "configured": bool(api_key and api_secret),
        "api_key_set": bool(api_key),
        "api_secret_set": bool(api_secret),
        "access_token_set": bool(access_token),
        "authenticated": False,
        "user_id": None,
        "user_name": None,
        "auth_error": None,
        "redirect_url": settings.kite_redirect_url,
        "proxy_url": settings.kite_proxy_url,
        "proxy_enabled": bool(settings.kite_proxy_url),
    }
    if not (api_key and access_token):
        return result

    try:
        from app.data.kite_client import KiteClient

        client = KiteClient(
            api_key=api_key,
            access_token=access_token,
            proxy_url=settings.kite_proxy_url,
        )
        profile = client.get_profile() or {}
        if profile.get("user_id"):
            result["authenticated"] = True
            result["user_id"] = profile.get("user_id")
            result["user_name"] = profile.get("user_name")
    except Exception as exc:
        result["auth_error"] = str(exc)
    return result


@app.post("/api/broker/kite/update-credentials")
async def update_kite_credentials(body: dict):
    """Save Kite api_key / api_secret to .env. System must be stopped."""
    orch = _state.get("orchestrator")
    if orch and getattr(orch, "running", False):
        raise HTTPException(
            status_code=409,
            detail="Cannot update credentials while system is running. Stop the system first.",
        )
    field_map = {
        "api_key": "KITE_API_KEY",
        "api_secret": "KITE_API_SECRET",
        "access_token": "KITE_ACCESS_TOKEN",
        "redirect_url": "KITE_REDIRECT_URL",
        "proxy_url": "KITE_PROXY_URL",
    }
    updates: dict[str, str] = {}
    for field, env_var in field_map.items():
        # proxy_url may legitimately be cleared (empty string disables proxy)
        raw = body.get(field)
        if raw is None:
            continue
        value = str(raw).strip()
        if field == "proxy_url":
            updates[env_var] = value
        elif value:
            updates[env_var] = value
    if not updates:
        raise HTTPException(status_code=400, detail="No Kite credentials provided")
    _persist_env_vars(updates)
    logger.info("Kite credentials updated: %s", list(updates.keys()))
    return {"success": True, "updated_fields": list(updates.keys())}


@app.get("/api/auth/kite/login-url")
async def kite_login_url():
    """Return the Kite Connect OAuth login URL for the configured api_key."""
    if not settings.kite_api_key:
        return {"login_url": None, "message": "KITE_API_KEY not configured"}
    try:
        from app.data.kite_client import KiteClient

        client = KiteClient(
            api_key=settings.kite_api_key,
            proxy_url=settings.kite_proxy_url,
        )
        url = client.kite.login_url()
        return {"login_url": url, "redirect_url": settings.kite_redirect_url}
    except Exception as exc:
        logger.exception("Failed to build Kite login URL")
        return {"login_url": None, "message": str(exc)}


@app.get("/api/auth/kite/callback")
async def kite_oauth_callback(
    request: Request,
    request_token: str = "",
    status: str = "",
    action: str = "",
):
    """OAuth callback target registered in the Kite developer console.

    Exchanges the request_token for an access_token, persists it to .env,
    then redirects the browser back to the BrokerSettings page.

    Redirect target resolution (first match wins):
      1. KITE_POST_LOGIN_REDIRECT env var (if set to non-default)
      2. Origin/Referer header from the request (so prod auto-uses
         tradeai.tavabharat.com and dev auto-uses localhost:3000)
      3. Hard fallback to "/settings" on the same host
    """
    from fastapi.responses import RedirectResponse
    from urllib.parse import urlencode, urlparse

    _DEFAULT = "http://localhost:3000/settings"
    configured = (settings.kite_post_login_redirect or "").strip()
    base = configured if configured and configured != _DEFAULT else ""

    if not base:
        # Derive from the public host the callback was reached at. When the
        # backend sits behind nginx (prod), X-Forwarded-* headers carry the
        # original scheme/host that the browser used.
        fwd_proto = request.headers.get("x-forwarded-proto")
        fwd_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
        if fwd_host:
            scheme = fwd_proto or request.url.scheme or "https"
            base = f"{scheme}://{fwd_host}/settings"
        else:
            base = _DEFAULT

    def _redirect(params: dict) -> RedirectResponse:
        sep = "&" if urlparse(base).query else "?"
        return RedirectResponse(url=f"{base}{sep}{urlencode(params)}", status_code=302)

    if not request_token:
        return _redirect({"kite_auth_error": status or "missing_request_token"})
    if not (settings.kite_api_key and settings.kite_api_secret):
        return _redirect({"kite_auth_error": "kite_credentials_not_configured"})

    try:
        from app.data.kite_client import KiteClient

        client = KiteClient(
            api_key=settings.kite_api_key,
            proxy_url=settings.kite_proxy_url,
        )
        data = client.kite.generate_session(
            request_token, api_secret=settings.kite_api_secret
        )
        access_token = (data or {}).get("access_token", "")
        user_id = (data or {}).get("user_id", "")
        if not access_token:
            return _redirect({"kite_auth_error": "no_access_token_in_response"})
        _persist_env_vars({"KITE_ACCESS_TOKEN": access_token})
        logger.info("Kite OAuth complete for user_id=%s", user_id)
        return _redirect({"kite_auth": "success", "user_id": user_id})
    except Exception as exc:
        logger.exception("Kite OAuth callback failed")
        return _redirect({"kite_auth_error": str(exc)[:200]})


# ── Dhan — daily access-token rotation + credential management ────────────

@app.get("/api/broker/dhan/status")
async def get_dhan_status():
    from app.db.broker_credentials import get_dhan_credentials

    creds = get_dhan_credentials()
    client_id = creds.get("client_id") or ""
    access_token = creds.get("access_token") or ""
    return {
        "configured": bool(client_id and access_token),
        "client_id_set": bool(client_id),
        "access_token_set": bool(access_token),
        "client_id": (client_id[:2] + "***") if client_id else None,
    }


@app.post("/api/broker/dhan/test")
async def test_dhan_connection():
    from app.db.broker_credentials import get_dhan_credentials

    creds = get_dhan_credentials()
    client_id = creds.get("client_id") or ""
    access_token = creds.get("access_token") or ""
    if not (client_id and access_token):
        return {"success": False, "error": "Dhan client_id or access_token not configured"}
    try:
        from app.data.dhan_client import DhanClient

        client = DhanClient(client_id, access_token)
        # Hit the SDK directly so we can see the raw status / remarks /
        # errorCode envelope. The DhanClient.get_fund_limits() wrapper
        # collapses failures into an empty dict, which used to hide
        # invalid / expired token errors and made every test "succeed".
        try:
            resp = await asyncio.to_thread(client._dhan.get_fund_limits)
        except Exception as exc:
            return {"success": False, "error": f"Dhan API call failed: {exc}"}

        if not isinstance(resp, dict):
            return {
                "success": False,
                "error": f"Unexpected Dhan response type: {type(resp).__name__}",
            }

        status = str(resp.get("status") or "").lower()
        data = resp.get("data") or {}
        remarks = resp.get("remarks") or {}
        # Dhan returns remarks as either a string or a dict {error_code, error_message, ...}
        if isinstance(remarks, dict):
            err_code = remarks.get("error_code") or remarks.get("errorCode") or ""
            err_msg = (
                remarks.get("error_message")
                or remarks.get("errorMessage")
                or remarks.get("message")
                or ""
            )
        else:
            err_code = ""
            err_msg = str(remarks)

        if status != "success":
            detail = " | ".join(filter(None, [err_code, err_msg])) or str(resp)[:200]
            return {
                "success": False,
                "error": f"Dhan auth rejected: {detail}",
                "raw": resp,
            }

        # status=success but no actual fund fields → token may be valid
        # but account is restricted; surface that instead of pretending
        # everything is fine.
        balance_keys = (
            "availabelBalance",
            "availableBalance",
            "sodLimit",
            "collateralAmount",
            "withdrawableBalance",
        )
        if not any(k in data for k in balance_keys):
            return {
                "success": False,
                "error": "Dhan returned success but no fund fields — account may be restricted",
                "raw": resp,
            }

        return {
            "success": True,
            "message": "Dhan authentication successful",
            "client_id": (client_id[:2] + "***"),
            "funds": data,
        }
    except Exception as exc:
        logger.exception("Dhan test connection failed")
        return {"success": False, "error": str(exc)}


@app.post("/api/broker/dhan/update-credentials")
async def update_dhan_credentials(body: dict):
    """Persist Dhan credentials to the DB and hot-reload the broker.

    Stored in the ``broker_credentials`` table — survives container
    restarts and does not require editing ``.env``. The active broker
    adapter (if any) is reloaded immediately so the new token takes
    effect for the next order without restarting the system.
    """
    from app.db.broker_credentials import set_dhan_credentials

    client_id = (body.get("client_id") or "").strip()
    access_token = (body.get("access_token") or "").strip()
    if not client_id and not access_token:
        raise HTTPException(status_code=400, detail="No Dhan credentials provided")
    try:
        set_dhan_credentials(client_id=client_id or None,
                             access_token=access_token or None)
    except Exception as exc:
        logger.exception("Failed to persist Dhan credentials to DB")
        raise HTTPException(status_code=500, detail=f"DB write failed: {exc}")

    updated = [k for k, v in (("client_id", client_id), ("access_token", access_token)) if v]

    # Hot-reload all broker adapters so the new token is picked up
    # immediately. NOTE: orch.broker is always AngelOneBroker. The Dhan
    # broker instances used for live order routing are held by the scanners
    # (built by `_build_atl_broker` at orchestrator init), so we must walk
    # every known holder — otherwise rotating the token here leaves the
    # scanners using the expired one and orders keep getting rejected with
    # "Client ID or user generated access token is invalid or expired."
    reloaded: list[str] = []
    try:
        orch = _state.get("orchestrator")
        if orch is not None:
            seen_brokers: set[int] = set()
            broker_holders = [
                orch,
                getattr(orch, "move_detection_scanner", None),
                getattr(orch, "move_detection_scanner_bull", None),
                getattr(orch, "pdh_pdl_scanner", None),
                getattr(orch, "atl_straddle_scanner", None),
            ]
            for holder in broker_holders:
                if holder is None:
                    continue
                broker = getattr(holder, "broker", None)
                if broker is None or id(broker) in seen_brokers:
                    continue
                seen_brokers.add(id(broker))
                if hasattr(broker, "reload_credentials"):
                    try:
                        broker.reload_credentials()
                        reloaded.append(
                            f"{type(broker).__name__}@{type(holder).__name__}"
                        )
                    except Exception:
                        logger.exception(
                            "reload_credentials failed on %s", type(broker).__name__,
                        )
    except Exception:
        logger.exception("Failed to reload broker credentials after Dhan update")

    logger.info("Dhan credentials updated in DB: %s (reloaded: %s)", updated, reloaded)
    return {"success": True, "updated_fields": updated, "reloaded": reloaded}


@app.post("/api/broker/dhan/refresh-instruments")
async def refresh_dhan_instruments():
    from app.db.broker_credentials import get_dhan_credentials

    creds = get_dhan_credentials()
    client_id = creds.get("client_id") or ""
    access_token = creds.get("access_token") or ""
    if not (client_id and access_token):
        return {"ok": False, "error": "Dhan client_id or access_token not configured"}
    try:
        from app.data.dhan_client import DhanClient

        client = DhanClient(client_id, access_token)
        n = await asyncio.to_thread(client.refresh_scrip_master, True)
        return {"ok": True, "instruments_loaded": n}
    except Exception as exc:
        logger.exception("Dhan scrip master refresh failed")
        return {"ok": False, "error": str(exc)}


# ── Account-level kill switch (per BrokerAccount) ──────────────────────

@app.get("/api/risk/kill-switch")
async def get_kill_switch_state():
    """Return the current kill-switch state for every wired account.

    Response shape::

        {
          "ok": true,
          "states": [
             {"account_id": 3, "account_name": "primary", "broker": "dhan",
              "enabled": true, "limit": 6000.0, "locked": false, ...},
             ...
          ],
          # Backwards-compat: first (or env) account's state
          "state": { ... }
        }
    """
    from app.execution.account_kill_switch import get_all_states, ENV_ACCOUNT_ID

    # Warm the state cache by touching the DB for active Dhan accounts —
    # ensures the UI can render freshly-created accounts before the first
    # watchdog tick lands.
    try:
        await _refresh_kill_switch_states_from_db()
    except Exception:
        logger.debug("kill switch DB warmup failed (non-fatal)", exc_info=True)

    states = get_all_states()
    # Sort so ENV_ACCOUNT_ID (0) is last; real accounts first.
    states.sort(key=lambda s: (s.account_id == ENV_ACCOUNT_ID, s.account_id))
    state_dicts = [s.to_dict() for s in states]
    return {
        "ok": True,
        "states": state_dicts,
        # Legacy compat: many old callers still read `data.state.limit`.
        "state": state_dicts[0] if state_dicts else None,
    }


@app.put("/api/risk/kill-switch")
async def update_kill_switch(body: dict):
    """Update kill-switch enable flag and/or daily loss limit for one account.

    Body::

        { "account_id": 3, "enabled": true, "limit": 8000 }

    ``account_id`` is required whenever more than one BrokerAccount row
    exists. For legacy env-based single-account setups, ``account_id=0``
    (or omitted) is accepted and the value is also persisted to ``.env``
    so it survives a container restart.
    """
    from app.execution.account_kill_switch import ENV_ACCOUNT_ID, update_settings

    enabled = body.get("enabled")
    limit = body.get("limit")
    account_id = int(body.get("account_id") or 0)
    try:
        limit_f = float(limit) if limit is not None else None
    except (TypeError, ValueError):
        return {"ok": False, "error": "limit must be a number"}

    if account_id and account_id != ENV_ACCOUNT_ID:
        # Persist to DB so it survives restarts and the watchdog picks up
        # the new value on its next tick.
        try:
            from sqlalchemy import select
            from app.db.account_models import BrokerAccount
            from app.db.models import AsyncSessionLocal

            async with AsyncSessionLocal() as s:
                async with s.begin():
                    row = await s.get(BrokerAccount, account_id)
                    if row is None:
                        return {"ok": False, "error": f"account_id={account_id} not found"}
                    if enabled is not None:
                        row.kill_switch_enabled = bool(enabled)
                    if limit_f is not None and limit_f > 0:
                        row.daily_loss_limit = float(limit_f)
                    # Capture display metadata BEFORE the transaction
                    # commits. Do NOT call s.refresh(..., [cols]) here —
                    # it expires+re-reads those columns from the DB and
                    # SILENTLY DISCARDS the pending assignments above,
                    # so the UPDATE never fires and the row stays at
                    # its old values.
                    account_name = row.name
                    broker_type = row.broker
                    # Force the UPDATE onto the wire inside this
                    # transaction so the outer `async with s.begin()`
                    # exit performs a real COMMIT of the new values.
                    await s.flush()
        except Exception:
            logger.exception("Failed to persist kill-switch settings for account %s", account_id)
            return {"ok": False, "error": "database write failed"}

        # Seed / update in-memory state so the UI reflects the change
        # immediately (before the next watchdog tick).
        from app.execution.account_kill_switch import _get_or_create_state, _state_lock, _snapshot
        with _state_lock:
            st = _get_or_create_state(account_id, account_name=account_name, broker=broker_type)
            if enabled is not None:
                st.enabled = bool(enabled)
            if limit_f is not None and limit_f > 0:
                st.limit = float(limit_f)
            snap = _snapshot(st)
        return {"ok": True, "state": snap.to_dict()}

    # Env-based path (account_id=0)
    state = update_settings(ENV_ACCOUNT_ID, enabled=enabled, limit=limit_f)
    env_updates: dict[str, str] = {}
    if enabled is not None:
        env_updates["ACCOUNT_KILL_SWITCH_ENABLED"] = "true" if bool(enabled) else "false"
    if limit_f is not None and limit_f > 0:
        env_updates["ACCOUNT_MAX_DAILY_LOSS"] = (
            str(int(limit_f)) if float(limit_f).is_integer() else str(limit_f)
        )
    if env_updates:
        try:
            _persist_env_vars(env_updates)
        except Exception:
            logger.exception("Failed to persist kill-switch settings to .env")
    return {"ok": True, "state": state.to_dict()}


@app.post("/api/risk/kill-switch/reset")
async def reset_kill_switch_route(body: dict | None = None):
    """Manually clear a tripped lock for one account so the bot can place
    new entries again. Typically used after the user resolves the
    underlying issue (added funds, reviewed the loss, etc.).

    Body: ``{ "account_id": 3, "reason": "manual_reset_via_ui" }``. When
    ``account_id`` is missing, the env-based (legacy) singleton is
    reset.
    """
    from app.execution.account_kill_switch import ENV_ACCOUNT_ID, reset_kill_switch

    account_id = int((body or {}).get("account_id") or 0) or ENV_ACCOUNT_ID
    reason = (body or {}).get("reason") or "manual_reset_via_ui"
    state = reset_kill_switch(account_id, reason=str(reason))
    return {"ok": True, "state": state.to_dict()}


async def _refresh_kill_switch_states_from_db() -> None:
    """Warm the in-memory kill-switch state cache from active Dhan rows
    so newly-added accounts appear in ``GET /api/risk/kill-switch``
    before the watchdog has run its first poll.
    """
    from sqlalchemy import select
    from app.db.account_models import BrokerAccount
    from app.db.models import AsyncSessionLocal
    from app.execution.account_kill_switch import _get_or_create_state, _state_lock

    async with AsyncSessionLocal() as s:
        rows = (
            await s.execute(
                select(BrokerAccount).where(
                    BrokerAccount.is_active == True,  # noqa: E712
                    BrokerAccount.broker == "dhan",
                ).order_by(BrokerAccount.id.asc())
            )
        ).scalars().all()
        with _state_lock:
            for r in rows:
                st = _get_or_create_state(
                    int(r.id),
                    account_name=str(r.name or f"account-{r.id}"),
                    broker=str(r.broker or "dhan"),
                )
                if r.kill_switch_enabled is not None:
                    st.enabled = bool(r.kill_switch_enabled)
                if r.daily_loss_limit is not None:
                    try:
                        v = float(r.daily_loss_limit)
                        if v > 0:
                            st.limit = v
                    except (TypeError, ValueError):
                        pass


# ── Strategy Settings (per-scanner exec params + ATL straddle) ────────────

@app.get("/api/strategy-settings/atl-straddle")
async def get_atl_straddle_settings():
    from app.engine.atl_settings import load_atl_settings

    return load_atl_settings()


@app.put("/api/strategy-settings/atl-straddle")
async def update_atl_straddle_settings(body: dict):
    from app.engine.atl_settings import save_atl_settings

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    try:
        saved = save_atl_settings(body)
        logger.info("ATL straddle settings updated")
        return saved
    except Exception as exc:
        logger.exception("Failed to save ATL straddle settings")
        raise HTTPException(status_code=500, detail=str(exc))


def _scanner_exec_get(scanner: str) -> dict:
    from app.engine import scanner_exec_settings as exec_settings

    return exec_settings.load(scanner)


def _scanner_exec_put(scanner: str, body: dict) -> dict:
    from app.engine import scanner_exec_settings as exec_settings

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    try:
        return exec_settings.save(scanner, body)
    except Exception as exc:
        logger.exception("Failed to save %s exec settings", scanner)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/strategy-settings/move-det")
async def get_move_det_settings():
    return _scanner_exec_get("move_det")


@app.put("/api/strategy-settings/move-det")
async def update_move_det_settings(body: dict):
    return _scanner_exec_put("move_det", body)


@app.get("/api/strategy-settings/move-det-bull")
async def get_move_det_bull_settings():
    return _scanner_exec_get("move_det_bull")


@app.put("/api/strategy-settings/move-det-bull")
async def update_move_det_bull_settings(body: dict):
    return _scanner_exec_put("move_det_bull", body)


@app.get("/api/strategy-settings/pdh-pdl")
async def get_pdh_pdl_settings():
    return _scanner_exec_get("pdh_pdl")


@app.put("/api/strategy-settings/pdh-pdl")
async def update_pdh_pdl_settings(body: dict):
    return _scanner_exec_put("pdh_pdl", body)


@app.get("/api/strategy-settings/priority-handoff")
async def get_priority_handoff_settings():
    from app.engine import priority_handoff_settings as phs

    return phs.load()


@app.put("/api/strategy-settings/priority-handoff")
async def update_priority_handoff_settings(body: dict):
    from app.engine import priority_handoff_settings as phs

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    try:
        saved = phs.save(body)
        logger.info("Priority handoff settings updated: %s", saved)
        return saved
    except Exception as exc:
        logger.exception("Failed to save priority handoff settings")
        raise HTTPException(status_code=500, detail=str(exc))


# ── ATM Straddle Strategy runtime (ATLStraddleScanner) ────────────────────

def _get_atl_scanner():
    """Return the live ATL straddle scanner instance, or None."""
    orch = _state.get("orchestrator")
    if orch is None:
        return None
    return getattr(orch, "atl_straddle_scanner", None)


@app.get("/api/atm/runtime")
async def get_atm_runtime():
    """Snapshot of the ATM straddle scanner: settings, phase, in_trade, events.

    Always returns 200 with a stable shape so the UI can render even when
    the orchestrator is still warming up.
    """
    scanner = _get_atl_scanner()
    if scanner is None:
        from app.engine.atl_settings import load_atl_settings
        return {
            "runtime": {
                "live_mode": not bool(getattr(settings, "paper_trading", True)),
                "phase": "INIT",
                "in_trade": False,
                "settings": load_atl_settings(),
                "events": [],
                "scanner_ready": False,
            }
        }
    try:
        rt = scanner.get_runtime_state()
    except Exception as exc:
        logger.exception("ATL get_runtime_state failed")
        raise HTTPException(status_code=500, detail=str(exc))
    rt["scanner_ready"] = True
    rt["live_mode"] = not bool(getattr(settings, "paper_trading", True))
    rt["trading_account"] = getattr(settings, "trading_account", "angel")
    return {"runtime": rt}


@app.post("/api/atm/force-close")
async def force_close_atm():
    scanner = _get_atl_scanner()
    if scanner is None:
        raise HTTPException(status_code=503, detail="ATL scanner not initialised")
    if not scanner.is_in_trade():
        return {"ok": True, "message": "No open ATL trade"}
    orch = _state.get("orchestrator")
    df_today = None
    instrument = None
    try:
        # Pull the live frame for the configured index from the orchestrator
        from app.engine.atl_settings import load_atl_settings
        idx = (load_atl_settings().get("index") or "NIFTY").upper()
        for inst in getattr(orch, "_active_instruments", []) or []:
            if inst.symbol == idx:
                instrument = inst
                df_today = getattr(orch, "_df_today_cache", {}).get(idx)
                break
    except Exception:
        logger.exception("ATL force-close: could not assemble live frame")
    if df_today is None or instrument is None:
        raise HTTPException(status_code=409, detail="Live frame for ATL index not available")
    try:
        await scanner.force_close(df_today, instrument)
        return {"ok": True}
    except Exception as exc:
        logger.exception("ATL force_close failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/atm/place-now")
async def place_now_atm():
    """Force the ATL scanner to attempt entry on its next cycle.

    Clears phantom in-memory position state (in case the user manually
    exited at the broker) and bypasses the entry-time gate.
    """
    scanner = _get_atl_scanner()
    if scanner is None:
        raise HTTPException(status_code=503, detail="ATL scanner not initialised")
    try:
        scanner.request_force_entry()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "message": "ATL scanner will attempt entry on next cycle"}


@app.post("/api/atm/reset")
async def reset_atm():
    """Clear the per-day circuit breaker so the scanner may retry entries.

    Use this after fixing whatever caused the broker rejection (e.g. IP
    whitelist, margin, credentials). Without this, the scanner stays halted
    for the day after the first failed entry.
    """
    scanner = _get_atl_scanner()
    if scanner is None:
        raise HTTPException(status_code=503, detail="ATL scanner not initialised")
    try:
        scanner.reset_halt()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "message": "ATL scanner halt cleared"}


@app.get("/api/atm/instances")
async def get_atm_instances():
    """Per-instance runtime state for every StrategyInstance the registry
    is running. The legacy ``/api/atm/runtime`` only surfaces the primary
    (first) scanner — so when a user has 2+ strategies bound to different
    accounts / indices, the others are invisible in the UI and their
    ``order_error`` events are hidden. This endpoint returns them all.

    Always returns 200 with a stable shape so the UI can render even while
    the orchestrator is still warming up.
    """
    orch = _state.get("orchestrator")
    registry = getattr(orch, "strategy_registry", None) if orch else None
    if registry is None or not registry.scanners:
        return {"instances": []}
    out = []
    for iid, sc in sorted(registry.scanners.items()):
        try:
            rt = sc.get_runtime_state()
        except Exception:
            logger.exception("[ATL] runtime snapshot failed for instance %s", iid)
            rt = {"error": "runtime snapshot failed"}
        rt["instance_id"] = iid
        out.append(rt)
    return {"instances": out}


@app.post("/api/atm/instances/{instance_id}/force-close")
async def force_close_instance(instance_id: int):
    """Force-close a specific instance's legs (not just the primary)."""
    orch = _state.get("orchestrator")
    registry = getattr(orch, "strategy_registry", None) if orch else None
    scanner = registry.scanners.get(instance_id) if registry else None
    if scanner is None:
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
    if not scanner.is_in_trade():
        return {"ok": True, "message": "No open trade for this instance"}
    df_today = None
    instrument = None
    try:
        idx = (scanner._settings.get("index") or "NIFTY").upper()
        for inst in getattr(orch, "_active_instruments", []) or []:
            if inst.symbol == idx:
                instrument = inst
                df_today = getattr(orch, "_df_today_cache", {}).get(idx)
                break
    except Exception:
        logger.exception("[ATL] force-close instance: could not assemble live frame")
    if df_today is None or instrument is None:
        raise HTTPException(status_code=409, detail=f"Live frame for {instance_id} index not available")
    try:
        await scanner.force_close(df_today, instrument)
        return {"ok": True}
    except Exception as exc:
        logger.exception("[ATL] force_close for instance %s failed", instance_id)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/atm/instances/{instance_id}/place-now")
async def place_now_instance(instance_id: int):
    """Force a specific instance to attempt entry on its next cycle."""
    orch = _state.get("orchestrator")
    registry = getattr(orch, "strategy_registry", None) if orch else None
    scanner = registry.scanners.get(instance_id) if registry else None
    if scanner is None:
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
    try:
        scanner.request_force_entry()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "message": f"Instance {instance_id} will attempt entry on next cycle"}


@app.post("/api/atm/instances/{instance_id}/reset")
async def reset_instance(instance_id: int):
    """Clear the per-day circuit breaker for a specific instance."""
    orch = _state.get("orchestrator")
    registry = getattr(orch, "strategy_registry", None) if orch else None
    scanner = registry.scanners.get(instance_id) if registry else None
    if scanner is None:
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
    try:
        scanner.reset_halt()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "message": f"Instance {instance_id} halt cleared"}


# ── Research Multi-Index Straddle (new live modes) ────────────────────────
# Single-index time-based mode keeps using the legacy /api/atm/* endpoints
# above. Anything multi-index OR indicator-gated runs through this scanner.

def _get_research_scanner():
    orch = _state.get("orchestrator")
    if orch is None:
        return None
    return getattr(orch, "research_straddle_scanner", None)


@app.get("/api/atm-research/settings")
async def get_atm_research_settings():
    from app.engine.atl_research_settings import load_research_settings
    return load_research_settings()


@app.put("/api/atm-research/settings")
async def update_atm_research_settings(body: dict):
    from app.engine.atl_research_settings import save_research_settings
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    try:
        saved = save_research_settings(body)
        logger.info("Research straddle settings updated")
        return saved
    except Exception as exc:
        logger.exception("Failed to save research straddle settings")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/atm-research/defaults")
async def get_atm_research_defaults():
    """Return the two preset schedules (time-based + indicator-based)
    so the UI can pre-fill its grid on demand."""
    from app.engine.atl_research_settings import (
        DEFAULT_SCHEDULE_MULTI_INDICATOR,
        DEFAULT_SCHEDULE_MULTI_TIME,
    )
    return {
        "multi_indicator": DEFAULT_SCHEDULE_MULTI_INDICATOR,
        "multi_time": DEFAULT_SCHEDULE_MULTI_TIME,
    }


@app.get("/api/atm-research/runtime")
async def get_atm_research_runtime():
    scanner = _get_research_scanner()
    if scanner is None:
        from app.engine.atl_research_settings import load_research_settings
        return {
            "runtime": {
                "scanner_ready": False,
                "live_mode": not bool(getattr(settings, "paper_trading", True)),
                "settings": load_research_settings(),
                "indices": {},
            }
        }
    try:
        rt = scanner.get_runtime_state()
    except Exception as exc:
        logger.exception("Research get_runtime_state failed")
        raise HTTPException(status_code=500, detail=str(exc))
    rt["scanner_ready"] = True
    rt["live_mode"] = not bool(getattr(settings, "paper_trading", True))
    # Surface a warning if SENSEX/NIFTY isn't in active instruments
    orch = _state.get("orchestrator")
    active = {i.symbol for i in getattr(orch, "_active_instruments", []) or []}
    rt["active_instruments"] = sorted(active)
    rt["warnings"] = []
    mode = (rt.get("settings", {}) or {}).get("mode", "")
    if mode.startswith("multi_"):
        if "NIFTY" not in active:
            rt["warnings"].append("NIFTY is not in ACTIVE_INSTRUMENTS — NIFTY trades will be skipped.")
        if "SENSEX" not in active:
            rt["warnings"].append("SENSEX is not in ACTIVE_INSTRUMENTS — SENSEX trades will be skipped.")
    return {"runtime": rt}


def _find_instrument_by_symbol(orch, symbol: str):
    if orch is None:
        return None
    for inst in getattr(orch, "_active_instruments", []) or []:
        if inst.symbol == symbol:
            return inst
    return None


@app.post("/api/atm-research/force-close")
async def force_close_atm_research(body: dict | None = None):
    """Close any open research-mode positions. Optional body: {"index": "NIFTY"|"SENSEX"}.
    If omitted, closes both."""
    scanner = _get_research_scanner()
    if scanner is None:
        raise HTTPException(status_code=503, detail="Research scanner not initialised")
    orch = _state.get("orchestrator")
    targets = []
    requested = (body or {}).get("index") if isinstance(body, dict) else None
    if requested:
        targets = [str(requested).upper()]
    else:
        targets = ["NIFTY", "SENSEX"]
    results: dict[str, bool] = {}
    for sym in targets:
        inst = _find_instrument_by_symbol(orch, sym)
        if inst is None:
            results[sym] = False
            continue
        try:
            results[sym] = await scanner.force_close_one(inst, reason="manual_force_close")
        except Exception:
            logger.exception("Research force-close failed for %s", sym)
            results[sym] = False
    return {"ok": True, "results": results}


@app.post("/api/atm-research/reset")
async def reset_atm_research(body: dict | None = None):
    scanner = _get_research_scanner()
    if scanner is None:
        raise HTTPException(status_code=503, detail="Research scanner not initialised")
    sym = (body or {}).get("index") if isinstance(body, dict) else None
    try:
        scanner.reset_halt(sym)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True}


# ── Strategy Analytics & Today's Plan ─────────────────────────────────────

async def _load_strategy_analytics_db(today: str) -> dict:
    """Heavy DB-only portion of /api/strategy-analytics — cached separately."""
    from app.db.models import AsyncSessionLocal
    from sqlalchemy import text as _sql_text

    db_result: dict = {
        "strategy_rankings": [],
        "condition_performance": [],
        "eval_history": [],
        "trade_stats": [],
        "data_coverage": {},
    }

    try:
        active_syms = settings.get_active_instrument_list()
        if not active_syms:
            from app.core.instruments import get_enabled_instruments
            active_syms = [i.symbol for i in get_enabled_instruments()]

        async with AsyncSessionLocal() as session:
            # 1. Latest strategy evaluation rankings (filtered to active instruments)
            rows = await session.execute(
                _sql_text(
                    "SELECT eval_date, instrument, strategy, rank, win_rate, "
                    "profit_factor, sharpe_ratio, total_pnl, total_trades, "
                    "avg_pnl, max_drawdown, composite_score, signal_frequency, eval_days "
                    "FROM strategy_evaluations "
                    "WHERE eval_date = (SELECT MAX(eval_date) FROM strategy_evaluations) "
                    "  AND instrument = ANY(:instruments) "
                    "ORDER BY rank"
                ),
                {"instruments": active_syms},
            )
            for r in rows:
                db_result["strategy_rankings"].append({
                    "eval_date": r[0],
                    "instrument": r[1],
                    "strategy": r[2],
                    "rank": r[3],
                    "win_rate": round(r[4], 1),
                    "profit_factor": round(r[5], 2),
                    "sharpe_ratio": round(r[6], 2),
                    "total_pnl": round(r[7], 2),
                    "total_trades": r[8],
                    "avg_pnl": round(r[9], 2),
                    "max_drawdown": round(r[10], 2),
                    "composite_score": round(r[11], 1),
                    "signal_frequency": round(r[12], 3),
                    "eval_days": r[13],
                })

            # 2. Best condition performance per strategy (latest eval, top conditions)
            rows = await session.execute(
                _sql_text(
                    "SELECT instrument, strategy, condition_key, day_type, "
                    "gap_bucket, vix_bucket, total_trades, win_rate, avg_pnl, "
                    "profit_factor, composite_score, best_entry_window, probability, lookback_days "
                    "FROM strategy_condition_performance "
                    "WHERE eval_date = (SELECT MAX(eval_date) FROM strategy_condition_performance) "
                    "  AND instrument = ANY(:instruments) "
                    "  AND condition_key NOT LIKE 'any%%' "
                    "  AND total_trades >= 3 "
                    "ORDER BY composite_score DESC "
                    "LIMIT 50"
                ),
                {"instruments": active_syms},
            )
            for r in rows:
                db_result["condition_performance"].append({
                    "instrument": r[0],
                    "strategy": r[1],
                    "condition_key": r[2],
                    "day_type": r[3],
                    "gap_bucket": r[4],
                    "vix_bucket": r[5],
                    "total_trades": r[6],
                    "win_rate": round(r[7], 1),
                    "avg_pnl": round(r[8], 2),
                    "profit_factor": round(r[9], 2),
                    "composite_score": round(r[10], 1),
                    "best_entry_window": r[11],
                    "probability": round(r[12], 1),
                    "lookback_days": r[13],
                })

            # 3. Evaluation history — composite scores over last 30 days (active instruments only)
            rows = await session.execute(
                _sql_text(
                    "SELECT eval_date, instrument, strategy, composite_score, "
                    "win_rate, profit_factor, total_trades "
                    "FROM strategy_evaluations "
                    "WHERE eval_date >= :cutoff "
                    "  AND instrument = ANY(:instruments) "
                    "ORDER BY eval_date DESC, rank"
                ),
                {"cutoff": (datetime.now(_IST) - timedelta(days=30)).strftime("%Y-%m-%d"), "instruments": active_syms},
            )
            for r in rows:
                db_result["eval_history"].append({
                    "eval_date": r[0],
                    "instrument": r[1],
                    "strategy": r[2],
                    "composite_score": round(r[3], 1),
                    "win_rate": round(r[4], 1),
                    "profit_factor": round(r[5], 2),
                    "total_trades": r[6],
                })

            # 4. Trade stats from actual trades (all time)
            rows = await session.execute(
                _sql_text(
                    "SELECT instrument, strategy, "
                    "COUNT(*) as total, "
                    "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins, "
                    "COALESCE(SUM(pnl), 0) as total_pnl, "
                    "COALESCE(AVG(pnl), 0) as avg_pnl, "
                    "MIN(date) as first_trade, MAX(date) as last_trade "
                    "FROM trades WHERE status = 'closed' "
                    "GROUP BY instrument, strategy "
                    "ORDER BY total_pnl DESC"
                )
            )
            trade_stats = []
            for r in rows:
                total = r[2]
                wins = r[3]
                trade_stats.append({
                    "instrument": r[0],
                    "strategy": r[1],
                    "total_trades": total,
                    "wins": wins,
                    "win_rate": round(wins / total * 100, 1) if total else 0,
                    "total_pnl": round(r[4], 2),
                    "avg_pnl": round(r[5], 2),
                    "first_trade": r[6],
                    "last_trade": r[7],
                })
            db_result["trade_stats"] = trade_stats

            # 5. Data coverage — how many days of index/option candles
            rows = await session.execute(
                _sql_text(
                    "SELECT instrument, COUNT(DISTINCT date) as days, "
                    "MIN(date) as from_date, MAX(date) as to_date "
                    "FROM index_candles GROUP BY instrument"
                )
            )
            for r in rows:
                db_result["data_coverage"][r[0]] = {
                    "index_candle_days": r[1],
                    "from": r[2],
                    "to": r[3],
                }

            rows = await session.execute(
                _sql_text(
                    "SELECT instrument, COUNT(DISTINCT date) as days, "
                    "MIN(date) as from_date, MAX(date) as to_date "
                    "FROM option_candles GROUP BY instrument"
                )
            )
            for r in rows:
                sym = r[0]
                if sym not in db_result["data_coverage"]:
                    db_result["data_coverage"][sym] = {}
                db_result["data_coverage"][sym]["option_candle_days"] = r[1]
                db_result["data_coverage"][sym]["option_from"] = r[2]
                db_result["data_coverage"][sym]["option_to"] = r[3]

    except Exception:
        logger.exception("Error fetching strategy analytics")

    return db_result


@app.get("/api/strategy-analytics")
async def get_strategy_analytics():
    """Comprehensive strategy performance data for the dashboard.

    Returns:
        - Per-strategy per-instrument historical backtest metrics
        - Evaluation history (last 30 days)
        - Today's trading plan (strategies, capital allocation)
        - Overall system stats
    """
    orch = _state.get("orchestrator")
    today = datetime.now(_IST).strftime("%Y-%m-%d")

    # DB-heavy portion: cache 60s. Live `today_plan` always merged below.
    cache_key = f"strategy_analytics:db:{today}"
    db_part = await cache.cached(
        cache_key,
        ttl_seconds=60,
        loader=lambda: _load_strategy_analytics_db(today),
    ) or {}

    result = {
        "today": today,
        "today_plan": {},
        "strategy_rankings": db_part.get("strategy_rankings", []),
        "condition_performance": db_part.get("condition_performance", []),
        "eval_history": db_part.get("eval_history", []),
        "trade_stats": db_part.get("trade_stats", []),
        "data_coverage": db_part.get("data_coverage", {}),
    }

    # 6. Today's plan from orchestrator (always live, never cached)
    if orch:
        plan = {}
        for inst in getattr(orch, "_active_instruments", []):
            sym = inst.symbol
            strats = getattr(orch, "_instrument_strategies", {}).get(sym, [])
            strat_names = [type(s).__name__.replace("Strategy", "").upper().replace("_", "_") for s in strats]
            # Clean up names
            clean_names = []
            for s in strats:
                name = type(s).__name__
                # Map class names to DB strategy names
                name_map = {
                    "TrendPullbackStrategy": "TREND_PULLBACK",
                    "MomentumBreakoutStrategy": "MOMENTUM_BREAKOUT",
                    "ORBStrategy": "ORB",
                    "VWAPReclaimStrategy": "VWAP_RECLAIM",
                    "RangeBreakoutStrategy": "RANGE_BREAKOUT",
                    "LiquiditySweepStrategy": "LIQUIDITY_SWEEP",
                }
                clean_names.append(name_map.get(name, name))

            cap = getattr(orch, "_instrument_capital", {}).get(sym, 0)
            plan[sym] = {
                "strategies": clean_names,
                "allocated_capital": round(cap, 0),
                "lot_size": inst.lot_size,
            }

        day_type = "pending"
        if hasattr(orch, "day_type") and orch.day_type:
            day_type = orch.day_type.value

        result["today_plan"] = {
            "instruments": plan,
            "day_type": day_type,
            "total_capital": settings.initial_capital,
            "max_concurrent": settings.max_concurrent_positions,
            "max_per_instrument": settings.max_concurrent_per_instrument,
            "max_trades_per_day": settings.max_trades_per_day,
        }

    return result


# ── Backtest Endpoints ────────────────────────────────────────────────────

@app.post("/api/backtest/run")
async def api_run_backtest(body: dict):
    """Launch a new backtest simulation.

    Body:
        start_date: str (YYYY-MM-DD) — required
        end_date: str (YYYY-MM-DD) — required
        instruments: list[str] | null — optional, defaults to enabled instruments
        strategies: list[str] | null — optional, defaults to all
    """
    from app.backtest.runner import run_backtest

    start = body.get("start_date")
    end = body.get("end_date")
    if not start or not end:
        raise HTTPException(status_code=400, detail="start_date and end_date are required")

    # Validate date format
    try:
        datetime.strptime(start, "%Y-%m-%d")
        datetime.strptime(end, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD format")

    if start > end:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    instruments = body.get("instruments")
    strategies = body.get("strategies")

    job_id = await run_backtest(start, end, instruments, strategies)
    return {"job_id": job_id, "status": "started"}


@app.get("/api/backtest/status/{job_id}")
async def api_backtest_status(job_id: str):
    """Get the progress of a running backtest."""
    from app.backtest.runner import get_job_progress, get_job_result

    progress = get_job_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")

    resp = {
        "job_id": progress.job_id,
        "status": progress.status,
        "total_days": progress.total_days,
        "processed_days": progress.processed_days,
        "current_date": progress.current_date,
        "message": progress.message,
        "error": progress.error,
    }

    if progress.status == "completed":
        result = get_job_result(job_id)
        if result:
            resp["result"] = {
                "start_date": result.start_date,
                "end_date": result.end_date,
                "instruments": result.instruments,
                "initial_capital": result.initial_capital,
                "ending_capital": result.ending_capital,
                "total_pnl": result.total_pnl,
                "return_pct": result.return_pct,
                "total_trades": result.total_trades,
                "winners": result.winners,
                "losers": result.losers,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "max_drawdown_pct": result.max_drawdown_pct,
                "trades": result.trades,
                "equity_curve": result.equity_curve,
                "config_used": result.config_used,
            }

    return resp


@app.get("/api/backtest/export/{job_id}")
async def api_backtest_export(job_id: str):
    """Download backtest results as Excel file."""
    from fastapi.responses import Response
    from app.backtest.runner import generate_excel, get_job_progress

    progress = get_job_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")
    if progress.status != "completed":
        raise HTTPException(status_code=400, detail="Backtest not yet completed")

    excel_bytes = generate_excel(job_id)
    if not excel_bytes:
        raise HTTPException(status_code=404, detail="No trades to export")

    filename = f"backtest_{job_id}_{progress.current_date}.xlsx"
    return Response(
        content=excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/backtest/jobs")
async def api_backtest_jobs():
    """List all backtest jobs."""
    from app.backtest.runner import list_jobs
    return list_jobs()


@app.get("/api/backtest/config")
async def api_backtest_config():
    """Return backtest config for display in UI."""
    from app.core.instruments import get_enabled_instruments
    from app.backtest.runner import ALL_STRATEGIES

    enabled = get_enabled_instruments()
    return {
        "strategy": "RANGE_BREAKOUT",
        "initial_capital": 100_000,
        "sl_pct": 20.0,
        "instruments": [{"symbol": i.symbol, "name": i.display_name, "lot_size": i.lot_size} for i in enabled],
        "strategies": list(ALL_STRATEGIES.keys()),
    }
