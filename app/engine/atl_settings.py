"""ATL straddle settings persistence shared by API and runtime engine."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

ATL_SETTINGS_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "strategy_type": "ATM_STRADDLE",
    "index": "NIFTY",
    "trading_day": "Daily",
    "entry_time": "09:20",
    "exit_time": "15:15",
    "strike_mode": "ATM",
    "otm_strikes": 0,
    # When True: enter the two legs once at entry_time and HOLD them
    # unchanged until exit_time. No strangle-to-straddle conversion on
    # touch, no rolling, no adjustment/reform. This matches the backtest
    # simulator exactly (Phase-3a / ULTIMATE sweep results were produced
    # under this rule).
    "static_legs": False,
    "lots": 1,
    "strike_interval": 50,
    "offset_points": 500,
    "rolling_points": 300,
    "sl_type": "premium_pct",
    "sl_lower": 0,
    "sl_upper": 0,
    "first_straddle_sl_pct": 100,
    "reform_straddle_sl_pct": 60,
    "hedge_mode": "none",
    "hedge_enabled": False,
    "hedge_premium": 3,
    "hedge_otm_points": 500,
    "hedge_lots": 0,
    "execution_account": "Primary",
    # ── Smart-condition reform mode ─────────────────────────────────
    # Activated ONLY when `adjustment_points == 0`. When >0, the legacy
    # fixed-distance reform rule runs and these keys are ignored. This
    # preserves the exact current behavior for every existing user.
    #
    # `adjustment_points` is intentionally NOT in defaults — its absence
    # means "fall back to rolling_points" (legacy path). Setting it to 0
    # via the API/UI is the explicit opt-in for smart mode.
    "smart_ratio_trigger": 2.0,             # CE/PE premium imbalance to allow a reform
    "smart_ratio_trigger_expiry": 1.6,      # tighter on expiry day (gamma is brutal)
    "smart_use_5min_close": True,           # require trend confirm (5-min close avg vs LTP)
    "smart_reform_cooldown_min": 20,        # min minutes between consecutive reforms
    "smart_reform_cooldown_min_expiry": 10, # shorter cooldown on expiry day
    "smart_max_reforms_per_day": 4,         # hard cap on intraday reforms
    "smart_no_reform_after": "14:45",       # last clock time a reform may fire
    "smart_no_reform_after_expiry": "14:30",
    # Per-instrument minimum drift floor (in spot points) before any
    # smart reform is considered. 0 = use built-in default for that index.
    "smart_min_drift_nifty": 0,             # built-in default: 50
    "smart_min_drift_banknifty": 0,         # built-in default: 150
    "smart_min_drift_sensex": 0,            # built-in default: 200
    # Emergency override: if drift exceeds this multiplier × min_drift,
    # reform regardless of premium ratio (handles vol-crush edge cases
    # where ratio stays low despite a large directional move).
    "smart_force_reform_drift_mult": 3.0,
}


def atl_settings_path() -> str:
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(backend_root, "data", "atl_straddle_settings.json")


def load_atl_settings() -> dict[str, Any]:
    """Return the effective ATL settings.

    Preference order:

    1. First enabled ``StrategyInstance`` row in the DB (ATM_STRADDLE or
       OTM_STRANGLE).  The multi-account Settings UI writes here.
    2. Legacy ``atl_straddle_settings.json`` on disk.
    3. Built-in defaults.

    Results are cached in-process for ``_DB_SNAPSHOT_TTL_S`` seconds to
    avoid a DB round-trip on every scanner cycle (called 3× per lifecycle
    including inside the async scan loop).  Cache is thread-safe.
    ``invalidate_settings_cache()`` clears it immediately after a save.
    """
    db_snap = _load_from_db_cached()
    if db_snap is not None:
        merged = dict(ATL_SETTINGS_DEFAULTS)
        merged.update(db_snap)
        return normalize_atl_settings(merged)

    path = atl_settings_path()
    if not os.path.exists(path):
        return dict(ATL_SETTINGS_DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(ATL_SETTINGS_DEFAULTS)
        out = dict(ATL_SETTINGS_DEFAULTS)
        out.update(data)
        return normalize_atl_settings(out)
    except Exception:
        logger.exception("Failed to load ATL settings; using defaults")
        return dict(ATL_SETTINGS_DEFAULTS)


# ── DB snapshot cache ────────────────────────────────────────────────────
# The scanner reloads settings on nearly every cycle. Uncached, each call
# spins up a thread + fresh asyncpg pool → connection storm. The TTL is
# short enough that UI edits appear within seconds without dropping the
# scanner's live view of the config.
_DB_SNAPSHOT_TTL_S = 5.0
_db_cache_lock = threading.Lock()
_db_cache_snapshot: dict[str, Any] | None = None
_db_cache_expires_at: float = 0.0
_db_cache_negative: bool = False   # True → previous fetch returned None


def invalidate_settings_cache() -> None:
    """Drop the in-process DB snapshot cache.

    Called from ``save_atl_settings`` and the CRUD endpoints after they
    mutate a ``StrategyInstance`` row so the scanner sees the new value
    on its very next cycle instead of waiting up to the TTL.
    """
    global _db_cache_snapshot, _db_cache_expires_at, _db_cache_negative
    with _db_cache_lock:
        _db_cache_snapshot = None
        _db_cache_expires_at = 0.0
        _db_cache_negative = False
    # Also flush the per-instance cache used by multi-instance scanners.
    with _instance_cache_lock:
        _instance_cache_data.clear()
        _instance_cache_expires.clear()


# ── Per-instance settings loader (multi-instance scanner support) ────────
_instance_cache_lock = threading.Lock()
_instance_cache_data: dict[int, dict[str, Any]] = {}
_instance_cache_expires: dict[int, float] = {}


def make_instance_settings_loader(instance_id: int):
    """Return a zero-arg callable that reads the settings for ONE
    ``StrategyInstance`` row (identified by ``instance_id``) and returns
    them in the normalized ATL-settings shape.

    The returned loader is cached per-instance with the same TTL as the
    singleton path so parallel scanners don't each open a DB connection
    every cycle.

    If the row is missing or the DB is unreachable, the loader returns
    built-in defaults with ``enabled=False`` so the scanner idles
    quietly instead of crashing.
    """
    inst_id = int(instance_id)

    def _loader() -> dict[str, Any]:
        now = time.monotonic()
        with _instance_cache_lock:
            exp = _instance_cache_expires.get(inst_id, 0.0)
            if now < exp:
                cached = _instance_cache_data.get(inst_id)
                if cached is not None:
                    return dict(cached)

        snap = _load_instance_from_db_sync(inst_id)
        if snap is None:
            out = dict(ATL_SETTINGS_DEFAULTS)
            out["enabled"] = False
            with _instance_cache_lock:
                _instance_cache_data[inst_id] = out
                _instance_cache_expires[inst_id] = time.monotonic() + _DB_SNAPSHOT_TTL_S
            return dict(out)

        merged = dict(ATL_SETTINGS_DEFAULTS)
        merged.update(snap)
        normalized = normalize_atl_settings(merged)
        with _instance_cache_lock:
            _instance_cache_data[inst_id] = normalized
            _instance_cache_expires[inst_id] = time.monotonic() + _DB_SNAPSHOT_TTL_S
        return dict(normalized)

    return _loader


def _load_instance_from_db_sync(instance_id: int) -> dict[str, Any] | None:
    """Fetch a specific StrategyInstance row → ATL settings dict.

    Uses a plain sync SQLAlchemy engine (built from the app's async
    URL with ``+asyncpg`` stripped) so it works from inside the scanner's
    running event loop without the ``asyncio.run`` + ThreadPoolExecutor
    dance.  That dance would spin up a fresh asyncpg pool per call and
    was observed to intermittently fail for one instance while
    succeeding for another spawned in the same cycle, causing the
    loader to silently fall back to defaults (``enabled=False``) — so
    the NIFTY OTM STRANGLE instance idled every cycle while SENSEX ATM
    STRADDLE placed orders correctly.
    """
    try:
        from sqlalchemy import create_engine, text
        from app.core.config import settings

        url = (
            settings.database_url
            .replace("postgresql+asyncpg://", "postgresql://")
            .replace("sqlite+aiosqlite://", "sqlite://")
        )
        engine = create_engine(url, pool_pre_ping=True)
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        'SELECT id, account_id, "index", trading_day, '
                        "entry_time, exit_time, lots, strike_interval, "
                        "strike_mode, otm_strikes, static_legs, "
                        "adjustment_points, rolling_points, sl_type, "
                        "sl_lower, sl_upper, first_straddle_sl_pct, "
                        "reform_straddle_sl_pct, hedge_mode, hedge_premium, "
                        "hedge_otm_points, hedge_lots, is_active "
                        "FROM strategy_instances WHERE id = :iid"
                    ),
                    {"iid": int(instance_id)},
                ).mappings().first()
                if row is None:
                    return None

                account_label = "Primary"
                account_id = row["account_id"]
                if account_id is not None:
                    acct = conn.execute(
                        text(
                            "SELECT broker, paper_trading FROM broker_accounts "
                            "WHERE id = :aid"
                        ),
                        {"aid": int(account_id)},
                    ).mappings().first()
                    if acct is not None:
                        account_label = (
                            "Paper" if acct["paper_trading"]
                            else f"Live ({(acct['broker'] or '').title()})"
                        )
        finally:
            engine.dispose()

        return {
            "enabled": bool(row["is_active"]),
            "strategy_type": "ATM_STRADDLE",
            "index": row["index"],
            "trading_day": row["trading_day"],
            "entry_time": row["entry_time"],
            "exit_time": row["exit_time"],
            "strike_mode": row["strike_mode"],
            "otm_strikes": row["otm_strikes"],
            "static_legs": bool(row["static_legs"]),
            "lots": row["lots"],
            "strike_interval": row["strike_interval"],
            "rolling_points": row["rolling_points"],
            "adjustment_points": row["adjustment_points"],
            "sl_type": row["sl_type"],
            "sl_lower": row["sl_lower"] or 0,
            "sl_upper": row["sl_upper"] or 0,
            "first_straddle_sl_pct": row["first_straddle_sl_pct"],
            "reform_straddle_sl_pct": row["reform_straddle_sl_pct"],
            "hedge_mode": row["hedge_mode"],
            "hedge_premium": row["hedge_premium"] or 3,
            "hedge_otm_points": row["hedge_otm_points"] or 500,
            "hedge_lots": row["hedge_lots"],
            "execution_account": account_label,
            "_source": "db",
            "_instance_id": row["id"],
            "_account_id": account_id,
        }
    except Exception:
        logger.warning(
            "Per-instance settings load failed for id=%s", instance_id, exc_info=True,
        )
        return None


def _load_from_db_cached() -> dict[str, Any] | None:
    global _db_cache_snapshot, _db_cache_expires_at, _db_cache_negative
    now = time.monotonic()
    with _db_cache_lock:
        if now < _db_cache_expires_at:
            if _db_cache_negative:
                return None
            if _db_cache_snapshot is not None:
                # Return a copy so callers can mutate freely.
                return dict(_db_cache_snapshot)

    fresh = _load_from_db_sync()

    with _db_cache_lock:
        _db_cache_snapshot = dict(fresh) if fresh is not None else None
        _db_cache_negative = fresh is None
        _db_cache_expires_at = time.monotonic() + _DB_SNAPSHOT_TTL_S
    return fresh


def _load_from_db_sync() -> dict[str, Any] | None:
    """Load the first active strategy instance from the DB.

    Runs the async query on a fresh event loop to stay callable from
    both the sync scanner ``__init__`` and the running orchestrator.
    Returns ``None`` if no matching row exists or on any error.
    """
    try:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        async def _q() -> dict[str, Any] | None:
            from sqlalchemy import select
            from app.db.account_models import BrokerAccount, StrategyInstance
            from app.db.models import AsyncSessionLocal

            async with AsyncSessionLocal() as s:
                row = (
                    await s.execute(
                        select(StrategyInstance)
                        .where(StrategyInstance.is_active.is_(True))
                        .where(StrategyInstance.strategy_type.in_(
                            ("ATM_STRADDLE", "OTM_STRANGLE")
                        ))
                        .order_by(StrategyInstance.id.asc())
                        .limit(1)
                    )
                ).scalars().first()
                if row is None:
                    return None
                account_label = "Primary"
                if row.account_id is not None:
                    acct = await s.get(BrokerAccount, row.account_id)
                    if acct is not None:
                        account_label = f"Live ({acct.broker.title()})" if not acct.paper_trading else "Paper"
                return {
                    "enabled": bool(row.is_active),
                    "strategy_type": "ATM_STRADDLE",  # scanner always thinks ATM_STRADDLE
                    "index": row.index,
                    "trading_day": row.trading_day,
                    "entry_time": row.entry_time,
                    "exit_time": row.exit_time,
                    "strike_mode": row.strike_mode,
                    "otm_strikes": row.otm_strikes,
                    "static_legs": bool(row.static_legs),
                    "lots": row.lots,
                    "strike_interval": row.strike_interval,
                    "rolling_points": row.rolling_points,
                    "adjustment_points": row.adjustment_points,
                    "sl_type": row.sl_type,
                    "sl_lower": row.sl_lower or 0,
                    "sl_upper": row.sl_upper or 0,
                    "first_straddle_sl_pct": row.first_straddle_sl_pct,
                    "reform_straddle_sl_pct": row.reform_straddle_sl_pct,
                    "hedge_mode": row.hedge_mode,
                    "hedge_premium": row.hedge_premium or 3,
                    "hedge_otm_points": row.hedge_otm_points or 500,
                    "hedge_lots": row.hedge_lots,
                    "execution_account": account_label,
                    "_source": "db",
                    "_instance_id": row.id,
                    "_account_id": row.account_id,
                }

        if loop is None:
            # Called from sync context — spin a short-lived loop.
            return asyncio.run(_q())

        # We're inside a running event loop. Schedule on a worker thread
        # to avoid nested-loop issues.
        import concurrent.futures

        def _runner() -> dict[str, Any] | None:
            return asyncio.run(_q())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_runner)
            return fut.result(timeout=5)
    except Exception:
        logger.debug("DB-backed ATL settings unavailable; falling back", exc_info=True)
        return None


def normalize_atl_settings(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(ATL_SETTINGS_DEFAULTS)
    out.update(payload or {})

    out["enabled"] = bool(out.get("enabled", False))
    out["strategy_type"] = "ATM_STRADDLE"
    out["index"] = str(out.get("index", "NIFTY")).upper()
    if out["index"] not in {"NIFTY", "BANKNIFTY", "SENSEX"}:
        out["index"] = "NIFTY"

    day = str(out.get("trading_day", "Daily")).strip().title()
    if day not in {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Daily"}:
        day = "Daily"
    out["trading_day"] = day

    out["entry_time"] = str(out.get("entry_time", "09:20"))
    out["exit_time"] = str(out.get("exit_time", "15:15"))
    mode = str(out.get("strike_mode", "ATM")).upper()
    if mode not in {"ATM", "ITM", "STRANGLE", "MAXPAIN"}:
        mode = "ATM"
    out["strike_mode"] = mode
    out["lots"] = max(1, int(out.get("lots", 1)))
    out["strike_interval"] = max(1, int(out.get("strike_interval", 50)))
    # otm_strikes is the OTM offset expressed in strike-steps (0/1/2/3...).
    # When strike_mode == STRANGLE, this drives offset_points so users can
    # pick "+2 OTM" without hand-computing points.
    out["otm_strikes"] = max(0, int(out.get("otm_strikes", 0)))
    out["static_legs"] = bool(out.get("static_legs", False))
    if mode == "STRANGLE" and out["otm_strikes"] > 0:
        out["offset_points"] = out["otm_strikes"] * out["strike_interval"]
    elif mode == "MAXPAIN":
        out["offset_points"] = out["otm_strikes"] * out["strike_interval"] if out["otm_strikes"] > 0 else 0
    elif mode == "ATM":
        out["offset_points"] = 0
    else:
        out["offset_points"] = max(0, int(out.get("offset_points", 500)))
    out["rolling_points"] = max(1, int(out.get("rolling_points", 300)))

    sl_type = str(out.get("sl_type", "premium_pct")).strip().lower()
    if sl_type not in {"none", "premium_pct", "spot"}:
        sl_type = "premium_pct"
    out["sl_type"] = sl_type
    out["sl_lower"] = max(0, int(out.get("sl_lower", 0)))
    out["sl_upper"] = max(0, int(out.get("sl_upper", 0)))

    out["first_straddle_sl_pct"] = max(1, int(out.get("first_straddle_sl_pct", 100)))
    out["reform_straddle_sl_pct"] = max(1, int(out.get("reform_straddle_sl_pct", 60)))

    hedge_mode = str(out.get("hedge_mode", "none")).strip().lower()
    if hedge_mode not in {"none", "premium", "otm_points"}:
        # Backward compatibility with previous boolean-only flag.
        hedge_mode = "premium" if bool(out.get("hedge_enabled", False)) else "none"
    out["hedge_mode"] = hedge_mode
    out["hedge_enabled"] = hedge_mode != "none"
    out["hedge_premium"] = max(1, int(out.get("hedge_premium", 3)))
    out["hedge_otm_points"] = max(1, int(out.get("hedge_otm_points", out.get("offset_points", 500))))
    out["hedge_lots"] = max(0, int(out.get("hedge_lots", 0)))
    out["execution_account"] = str(out.get("execution_account", "Primary"))

    if sl_type == "none":
        out["sl_lower"] = 0
        out["sl_upper"] = 0

    if sl_type == "spot":
        out["first_straddle_sl_pct"] = 100

    if hedge_mode == "none":
        out["hedge_lots"] = 0

    # ── Smart-mode normalization ─────────────────────────────────────
    # `adjustment_points` is optional. If the user provided it, accept it
    # as a non-negative int (0 == smart mode). If absent, leave it absent
    # so the scanner's legacy fallback (`rolling_points`) kicks in.
    if "adjustment_points" in (payload or {}):
        try:
            adj = int(payload.get("adjustment_points"))
        except Exception:
            adj = 0
        out["adjustment_points"] = max(0, adj)

    def _clamp_float(key: str, lo: float, hi: float, default: float) -> None:
        try:
            v = float(out.get(key, default))
        except Exception:
            v = default
        out[key] = max(lo, min(hi, v))

    def _clamp_int(key: str, lo: int, hi: int, default: int) -> None:
        try:
            v = int(out.get(key, default))
        except Exception:
            v = default
        out[key] = max(lo, min(hi, v))

    def _clamp_time(key: str, default: str) -> None:
        raw = str(out.get(key, default))
        try:
            h, m = [int(x) for x in raw.split(":")]
            if 0 <= h <= 23 and 0 <= m <= 59:
                out[key] = f"{h:02d}:{m:02d}"
                return
        except Exception:
            pass
        out[key] = default

    _clamp_float("smart_ratio_trigger", 1.05, 10.0, 2.0)
    _clamp_float("smart_ratio_trigger_expiry", 1.05, 10.0, 1.6)
    out["smart_use_5min_close"] = bool(out.get("smart_use_5min_close", True))
    _clamp_int("smart_reform_cooldown_min", 0, 240, 20)
    _clamp_int("smart_reform_cooldown_min_expiry", 0, 240, 10)
    _clamp_int("smart_max_reforms_per_day", 0, 50, 4)
    _clamp_time("smart_no_reform_after", "14:45")
    _clamp_time("smart_no_reform_after_expiry", "14:30")
    _clamp_int("smart_min_drift_nifty", 0, 5000, 0)
    _clamp_int("smart_min_drift_banknifty", 0, 5000, 0)
    _clamp_int("smart_min_drift_sensex", 0, 10000, 0)
    _clamp_float("smart_force_reform_drift_mult", 1.0, 20.0, 3.0)

    return out


def save_atl_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Persist ATL settings.

    Writes to ``atl_straddle_settings.json`` (legacy path) *and* mirrors
    the values into the first active ``StrategyInstance`` row so the two
    stay coherent.  Never raises on the DB mirror — file write is the
    source of truth for the legacy path.
    """
    out = normalize_atl_settings(payload)
    path = atl_settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    try:
        _mirror_to_db_sync(out)
    except Exception:
        logger.debug("Mirroring ATL settings to DB failed (non-fatal)", exc_info=True)

    # Drop cache so the next scanner cycle sees the new values.
    invalidate_settings_cache()

    return out


def _mirror_to_db_sync(payload: dict[str, Any]) -> None:
    import asyncio

    async def _u() -> None:
        from sqlalchemy import select
        from app.db.account_models import StrategyInstance
        from app.db.models import AsyncSessionLocal

        async with AsyncSessionLocal() as s:
            async with s.begin():
                row = (
                    await s.execute(
                        select(StrategyInstance)
                        .where(StrategyInstance.strategy_type.in_(
                            ("ATM_STRADDLE", "OTM_STRANGLE")
                        ))
                        .order_by(StrategyInstance.id.asc())
                        .limit(1)
                    )
                ).scalars().first()
                if row is None:
                    return
                row.is_active = bool(payload.get("enabled", row.is_active))
                row.index = payload.get("index", row.index)
                row.trading_day = payload.get("trading_day", row.trading_day)
                row.entry_time = payload.get("entry_time", row.entry_time)
                row.exit_time = payload.get("exit_time", row.exit_time)
                row.lots = int(payload.get("lots", row.lots))
                row.strike_interval = int(payload.get("strike_interval", row.strike_interval))
                row.strike_mode = payload.get("strike_mode", row.strike_mode)
                row.otm_strikes = int(payload.get("otm_strikes", row.otm_strikes))
                row.static_legs = bool(payload.get("static_legs", row.static_legs))
                row.adjustment_points = int(payload.get("adjustment_points", row.adjustment_points))
                row.rolling_points = int(payload.get("rolling_points", row.rolling_points))
                row.sl_type = payload.get("sl_type", row.sl_type)
                row.sl_lower = float(payload.get("sl_lower", row.sl_lower or 0))
                row.sl_upper = float(payload.get("sl_upper", row.sl_upper or 0))
                row.first_straddle_sl_pct = int(payload.get("first_straddle_sl_pct", row.first_straddle_sl_pct))
                row.reform_straddle_sl_pct = int(payload.get("reform_straddle_sl_pct", row.reform_straddle_sl_pct))
                row.hedge_mode = payload.get("hedge_mode", row.hedge_mode)
                row.hedge_premium = float(payload.get("hedge_premium", row.hedge_premium or 3))
                row.hedge_otm_points = int(payload.get("hedge_otm_points", row.hedge_otm_points or 500))
                row.hedge_lots = int(payload.get("hedge_lots", row.hedge_lots))
                if row.strike_mode == "STRANGLE":
                    row.strategy_type = "OTM_STRANGLE"

    try:
        asyncio.get_running_loop()
        import concurrent.futures

        def _runner() -> None:
            asyncio.run(_u())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(_runner).result(timeout=5)
    except RuntimeError:
        asyncio.run(_u())
