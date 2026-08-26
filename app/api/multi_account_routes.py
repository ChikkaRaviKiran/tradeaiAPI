"""REST endpoints for the multi-account / multi-strategy Settings UI.

Registered from ``app.api.routes`` via ``register_multi_account_routes(app)``.

Endpoints
---------
Broker Accounts
  * GET    /api/accounts
  * POST   /api/accounts
  * PUT    /api/accounts/{account_id}
  * DELETE /api/accounts/{account_id}
  * POST   /api/accounts/{account_id}/test
  * POST   /api/accounts/{account_id}/set-primary
  * POST   /api/accounts/{account_id}/set-data-feed

Strategy Instances
  * GET    /api/strategy-instances
  * POST   /api/strategy-instances
  * PUT    /api/strategy-instances/{instance_id}
  * DELETE /api/strategy-instances/{instance_id}
  * POST   /api/strategy-instances/{instance_id}/toggle
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from sqlalchemy import select

from app.db.account_models import (
    SUPPORTED_BROKERS,
    SUPPORTED_STRATEGY_TYPES,
    BrokerAccount,
    StrategyInstance,
)
from app.db.models import AsyncSessionLocal

logger = logging.getLogger(__name__)


def _invalidate_settings_cache_safe() -> None:
    """Invalidate the shared ATL settings cache after any strategy
    mutation so the scanner picks up the change on the next cycle.
    Never raises — cache is best-effort.
    """
    try:
        from app.engine.atl_settings import invalidate_settings_cache
        invalidate_settings_cache()
    except Exception:
        logger.debug("invalidate_settings_cache failed (non-fatal)", exc_info=True)


def _invalidate_broker_cache_safe(account_id: Optional[int] = None) -> None:
    """Drop the cached per-account broker so the next scanner cycle
    rebuilds it with the latest credentials.  Best-effort — never raises.
    """
    try:
        from app.execution.broker_factory import invalidate_account_cache
        invalidate_account_cache(account_id)
    except Exception:
        logger.debug("invalidate_account_cache failed (non-fatal)", exc_info=True)


# ── Serialisation helpers ────────────────────────────────────────────────

_VALID_STRIKE_MODES = {"ATM", "STRANGLE", "ITM", "MAXPAIN"}
_VALID_HEDGE_MODES = {"none", "premium", "otm_points"}
_VALID_SL_TYPES = {"none", "premium_pct", "spot", "amount"}
_VALID_INDICES = {"NIFTY", "BANKNIFTY", "SENSEX"}
_VALID_TRADING_DAYS = {
    "Daily", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
}

_INDEX_STRIKE_INTERVAL_DEFAULTS: dict[str, int] = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "SENSEX": 100,
}


def _default_strike_interval_for_index(index_symbol: str) -> int:
    return int(_INDEX_STRIKE_INTERVAL_DEFAULTS.get(str(index_symbol).upper(), 50))


def _compute_proxy_status(a: BrokerAccount) -> str:
    """Match OptionSelling's proxy_status contract (active/provisioning/not_needed/none).
    Dhan doesn't strictly require a proxy — but we still allow one, so we
    never return 'not_needed' here; we simply say 'none' when no proxy is set.
    """
    if a.proxy_ip:
        return "active"
    if a.proxy_instance_name:
        return "provisioning"
    return "none"


def _account_to_dict(a: BrokerAccount) -> dict[str, Any]:
    return {
        "id": a.id,
        "name": a.name,
        "broker": a.broker,
        "client_id": a.client_id,
        "api_key": a.api_key or "",
        "api_key_set": bool(a.api_key),
        "api_secret_set": bool(a.api_secret),
        "password_set": bool(a.password),
        "mpin_set": bool(a.mpin),
        "totp_secret_set": bool(a.totp_secret),
        "access_token_set": bool(a.access_token),
        "login_method": a.login_method or "manual",
        "is_active": bool(a.is_active),
        "paper_trading": bool(a.paper_trading),
        "is_data_feed": bool(a.is_data_feed),
        "is_primary": bool(a.is_primary),
        "proxy_url": a.proxy_url or "",
        "proxy_ip": a.proxy_ip or "",
        "proxy_instance_name": a.proxy_instance_name or "",
        "proxy_status": _compute_proxy_status(a),
        "available_funds": a.available_funds or 0.0,
        "used_funds": a.used_funds or 0.0,
        "kill_switch_enabled": bool(a.kill_switch_enabled) if a.kill_switch_enabled is not None else True,
        "daily_loss_limit": float(a.daily_loss_limit) if a.daily_loss_limit is not None else 6000.0,
        "last_connection_status": a.last_connection_status or "unknown",
        "last_connection_error": a.last_connection_error or "",
        "last_connected_at": a.last_connected_at.isoformat()
        if a.last_connected_at else None,
        "created_at": a.created_at.isoformat() if a.created_at else None,
        "updated_at": a.updated_at.isoformat() if a.updated_at else None,
    }


def _instance_to_dict(i: StrategyInstance) -> dict[str, Any]:
    return {
        "id": i.id,
        "strategy_type": i.strategy_type,
        "account_id": i.account_id,
        "index": i.index,
        "trading_day": i.trading_day,
        "entry_time": i.entry_time,
        "exit_time": i.exit_time,
        "lots": i.lots,
        "strike_interval": i.strike_interval,
        "strike_mode": i.strike_mode,
        "otm_strikes": i.otm_strikes,
        "static_legs": bool(i.static_legs),
        "adjustment_points": i.adjustment_points,
        "rolling_points": i.rolling_points,
        "sl_type": i.sl_type,
        "sl_lower": i.sl_lower or 0,
        "sl_upper": i.sl_upper or 0,
        "sl_amount": i.sl_amount or 0,
        "first_straddle_sl_pct": i.first_straddle_sl_pct,
        "reform_straddle_sl_pct": i.reform_straddle_sl_pct,
        "hedge_mode": i.hedge_mode,
        "hedge_premium": i.hedge_premium or 0,
        "hedge_otm_points": i.hedge_otm_points or 0,
        "hedge_lots": i.hedge_lots,
        "is_active": bool(i.is_active),
        "live_execution": bool(i.live_execution),
        "display_name": i.display_name or "",
        "params_json": i.params_json or "",
        "created_at": i.created_at.isoformat() if i.created_at else None,
        "updated_at": i.updated_at.isoformat() if i.updated_at else None,
    }


# ── Validation ───────────────────────────────────────────────────────────

def _clean_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v).strip()


def _clean_int(v: Any, default: int, lo: int | None = None, hi: int | None = None) -> int:
    try:
        n = int(v)
    except Exception:
        return default
    if lo is not None:
        n = max(lo, n)
    if hi is not None:
        n = min(hi, n)
    return n


def _clean_float(v: Any, default: float, lo: float | None = None, hi: float | None = None) -> float:
    try:
        n = float(v)
    except Exception:
        return default
    if lo is not None:
        n = max(lo, n)
    if hi is not None:
        n = min(hi, n)
    return n


def _clean_time(v: Any, default: str) -> str:
    raw = _clean_str(v, default)
    try:
        h, m = [int(x) for x in raw.split(":")]
        if 0 <= h <= 23 and 0 <= m <= 59:
            return f"{h:02d}:{m:02d}"
    except Exception:
        pass
    return default


def _validate_broker(v: Any) -> str:
    b = _clean_str(v, "angel").lower()
    if b not in SUPPORTED_BROKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported broker '{b}'. Must be one of {list(SUPPORTED_BROKERS)}.",
        )
    return b


def _validate_strategy_type(v: Any) -> str:
    t = _clean_str(v, "ATM_STRADDLE").upper()
    if t not in SUPPORTED_STRATEGY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported strategy_type '{t}'. Must be one of {list(SUPPORTED_STRATEGY_TYPES)}.",
        )
    return t


def _validate_access_token(broker: str, token: str) -> str:
    """Reject a token that cannot possibly authenticate.

    Dhan/Kite tokens are JWTs. A mis-paste is otherwise stored happily and
    only surfaces at 09:30 as a broker rejection with no order placed.
    """
    if not token or (broker or "").lower() not in ("dhan", "kite"):
        return token
    if any(c.isspace() for c in token) or not token.startswith("eyJ") or token.count(".") != 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "That does not look like a valid access token. Expected a JWT "
                "(starts with 'eyJ', three dot-separated parts, no spaces). "
                "Copy the token only — not surrounding text."
            ),
        )
    return token


# ── FastAPI wiring ───────────────────────────────────────────────────────

def register_multi_account_routes(app: FastAPI) -> None:
    """Attach account + strategy-instance CRUD endpoints to ``app``."""

    # ---- BrokerAccount CRUD ---------------------------------------------

    @app.get("/api/accounts")
    async def list_accounts() -> dict[str, Any]:
        async with AsyncSessionLocal() as s:
            rows = (
                await s.execute(select(BrokerAccount).order_by(BrokerAccount.id.asc()))
            ).scalars().all()
        return {"accounts": [_account_to_dict(a) for a in rows]}

    @app.post("/api/accounts")
    async def create_account(body: dict) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Body must be JSON object")

        broker = _validate_broker(body.get("broker"))
        name = _clean_str(body.get("name"))
        client_id = _clean_str(body.get("client_id"))
        if not name:
            raise HTTPException(status_code=400, detail="'name' is required")
        if not client_id:
            raise HTTPException(status_code=400, detail="'client_id' is required")

        row = BrokerAccount(
            name=name,
            broker=broker,
            client_id=client_id,
            api_key=_clean_str(body.get("api_key")),
            api_secret=_clean_str(body.get("api_secret")),
            password=_clean_str(body.get("password")),
            mpin=_clean_str(body.get("mpin")),
            totp_secret=_clean_str(body.get("totp_secret")),
            access_token=_validate_access_token(broker, _clean_str(body.get("access_token"))) or None,
            refresh_token=_clean_str(body.get("refresh_token")) or None,
            login_method=_clean_str(body.get("login_method"), "manual") or "manual",
            paper_trading=bool(body.get("paper_trading", False)),
            is_active=bool(body.get("is_active", True)),
            is_data_feed=bool(body.get("is_data_feed", False)),
            is_primary=bool(body.get("is_primary", False)),
            proxy_url=_clean_str(body.get("proxy_url")),
            proxy_ip=_clean_str(body.get("proxy_ip")),
            proxy_instance_name=_clean_str(body.get("proxy_instance_name")),
        )
        async with AsyncSessionLocal() as s:
            async with s.begin():
                s.add(row)
                await s.flush()
                if row.is_primary:
                    await _clear_other_primaries(s, keep_id=row.id)
                if row.is_data_feed:
                    await _clear_other_data_feeds(s, keep_id=row.id)
            await s.refresh(row)
            _invalidate_broker_cache_safe(row.id)
            new_id = row.id
            result = _account_to_dict(row)

        # Fire-and-forget proxy provisioning if AWS is configured and
        # this broker benefits from a dedicated IP (Kite/Angel/Dhan all
        # supported for IP isolation). Non-blocking — UI polls
        # /api/proxy/status/{id} or the account list to watch progress.
        try:
            from app.core.config import settings as _s
            if _s.aws_access_key_id and _s.aws_secret_access_key and not row.proxy_url:
                # Mark as provisioning immediately so proxy_status='provisioning'
                # shows up in the UI before AWS finishes.
                async with AsyncSessionLocal() as s2:
                    async with s2.begin():
                        r2 = await s2.get(BrokerAccount, new_id)
                        if r2 is not None and not r2.proxy_instance_name:
                            r2.proxy_instance_name = f"tradeai-proxy-{new_id}"
                asyncio.create_task(_provision_proxy_background(new_id))
                result["proxy_provisioning"] = True
                result["proxy_status"] = "provisioning"
        except Exception:
            logger.debug("proxy auto-provision skipped", exc_info=True)
        return result

    @app.put("/api/accounts/{account_id}")
    async def update_account(account_id: int, body: dict) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Body must be JSON object")

        async with AsyncSessionLocal() as s:
            async with s.begin():
                row = await s.get(BrokerAccount, account_id)
                if row is None:
                    raise HTTPException(status_code=404, detail="Account not found")

                if "broker" in body:
                    row.broker = _validate_broker(body.get("broker"))
                if "name" in body:
                    n = _clean_str(body.get("name"))
                    if not n:
                        raise HTTPException(status_code=400, detail="'name' cannot be empty")
                    row.name = n
                if "client_id" in body:
                    c = _clean_str(body.get("client_id"))
                    if not c:
                        raise HTTPException(status_code=400, detail="'client_id' cannot be empty")
                    row.client_id = c
                for k in ("api_key", "api_secret", "password", "mpin", "totp_secret",
                          "access_token", "refresh_token",
                          "proxy_url", "proxy_ip", "proxy_instance_name",
                          "login_method"):
                    if k in body:
                        val = _clean_str(body.get(k))
                        if k == "access_token":
                            val = _validate_access_token(row.broker, val)
                        setattr(row, k, val)
                for k in ("paper_trading", "is_active", "is_data_feed", "is_primary"):
                    if k in body:
                        setattr(row, k, bool(body.get(k)))

                await s.flush()
                if row.is_primary:
                    await _clear_other_primaries(s, keep_id=row.id)
                if row.is_data_feed:
                    await _clear_other_data_feeds(s, keep_id=row.id)
            await s.refresh(row)
            _invalidate_broker_cache_safe(row.id)
            return _account_to_dict(row)

    @app.delete("/api/accounts/{account_id}")
    async def delete_account(account_id: int) -> dict[str, Any]:
        async with AsyncSessionLocal() as s:
            async with s.begin():
                row = await s.get(BrokerAccount, account_id)
                if row is None:
                    raise HTTPException(status_code=404, detail="Account not found")

                # Un-link any strategy instance pointing at this account
                # so we don't leave dangling FKs (schema doesn't enforce
                # them but the runtime resolver relies on this).
                instances = (
                    await s.execute(
                        select(StrategyInstance).where(
                            StrategyInstance.account_id == account_id
                        )
                    )
                ).scalars().all()
                for inst in instances:
                    inst.account_id = None
                    inst.is_active = False

                await s.delete(row)
            _invalidate_broker_cache_safe(account_id)
            _invalidate_settings_cache_safe()
            # Best-effort delete of the Lightsail proxy instance.
            try:
                from app.core.config import settings as _s
                if _s.aws_access_key_id and _s.aws_secret_access_key:
                    from app.infra.proxy_manager import proxy_manager
                    await asyncio.to_thread(proxy_manager.delete_proxy, account_id)
            except Exception:
                logger.exception("Failed to delete Lightsail proxy for account %s", account_id)
            return {"ok": True, "id": account_id, "detached_instances": len(instances)}

    @app.post("/api/accounts/{account_id}/set-primary")
    async def set_primary_account(account_id: int) -> dict[str, Any]:
        async with AsyncSessionLocal() as s:
            async with s.begin():
                row = await s.get(BrokerAccount, account_id)
                if row is None:
                    raise HTTPException(status_code=404, detail="Account not found")
                await _clear_other_primaries(s, keep_id=account_id)
                row.is_primary = True
            await s.refresh(row)
            return _account_to_dict(row)

    @app.post("/api/accounts/{account_id}/set-data-feed")
    async def set_data_feed_account(account_id: int) -> dict[str, Any]:
        async with AsyncSessionLocal() as s:
            async with s.begin():
                row = await s.get(BrokerAccount, account_id)
                if row is None:
                    raise HTTPException(status_code=404, detail="Account not found")
                await _clear_other_data_feeds(s, keep_id=account_id)
                row.is_data_feed = True
            await s.refresh(row)
            return _account_to_dict(row)

    @app.post("/api/accounts/{account_id}/test")
    async def test_account(account_id: int) -> dict[str, Any]:
        """Best-effort connectivity probe using THIS account's own
        credentials (not the process-wide env vars). Records result on
        the row so the Settings UI reflects the true state.
        """
        async with AsyncSessionLocal() as s:
            row = await s.get(BrokerAccount, account_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Account not found")
            creds = {
                "broker": (row.broker or "").lower(),
                "client_id": row.client_id or "",
                "api_key": row.api_key or "",
                "api_secret": row.api_secret or "",
                "password": row.password or "",
                "mpin": row.mpin or "",
                "totp_secret": row.totp_secret or "",
                "access_token": row.access_token or "",
                "proxy_url": row.proxy_url or "",
            }

        status, detail = await _probe_broker_account(creds)

        async with AsyncSessionLocal() as s:
            async with s.begin():
                row = await s.get(BrokerAccount, account_id)
                row.last_connection_status = status
                row.last_connection_error = detail if status != "connected" else None
                row.last_connected_at = datetime.utcnow() if status == "connected" else row.last_connected_at
            await s.refresh(row)
            return {"ok": status == "connected", "status": status, "detail": detail, "account": _account_to_dict(row)}

    # ---- Proxy management (Lightsail SOCKS5) ----------------------------
    # Mirrors OptionSelling's endpoints. Returns {"status":"ok"|"error", ...}
    # instead of raising HTTPException so the UI can reuse the same handlers.

    @app.get("/api/proxy/list")
    async def list_proxies() -> dict[str, Any]:
        from app.core.config import settings as _s
        if not _s.aws_access_key_id:
            return {"status": "ok", "proxies": [], "message": "AWS not configured"}
        try:
            from app.infra.proxy_manager import proxy_manager
            proxies = await asyncio.to_thread(proxy_manager.list_proxies)
            return {"status": "ok", "proxies": proxies}
        except Exception:
            logger.exception("Failed to list proxies")
            return {"status": "error", "message": "Failed to list proxies"}

    @app.get("/api/proxy/status/{account_id}")
    async def get_proxy_status(account_id: int) -> dict[str, Any]:
        from app.core.config import settings as _s
        if not _s.aws_access_key_id:
            return {"status": "error", "message": "AWS not configured"}
        try:
            from app.infra.proxy_manager import proxy_manager
            result = await asyncio.to_thread(proxy_manager.get_status, account_id)
            return {"status": "ok", **result}
        except Exception:
            logger.exception("Failed to get proxy status")
            return {"status": "error", "message": "Failed to get proxy status"}

    @app.post("/api/proxy/provision/{account_id}")
    async def provision_proxy(account_id: int) -> dict[str, Any]:
        """(Re)provision the Lightsail proxy for an account and persist it.
        Blocks until AWS reports the IP so the caller sees it immediately.
        """
        from app.core.config import settings as _s
        if not _s.aws_access_key_id:
            return {"status": "error", "message": "AWS not configured"}
        try:
            from app.infra.proxy_manager import proxy_manager
            result = await asyncio.to_thread(proxy_manager.create_proxy, account_id)
            proxy_url = result.get("proxy_url", "")
            proxy_ip = result.get("public_ip", "")
            proxy_instance_name = result.get("instance_name", "")
            broker = ""
            async with AsyncSessionLocal() as s:
                async with s.begin():
                    row = await s.get(BrokerAccount, account_id)
                    if row is not None:
                        row.proxy_url = proxy_url
                        row.proxy_ip = proxy_ip
                        row.proxy_instance_name = proxy_instance_name
                        broker = row.broker or ""
            _invalidate_broker_cache_safe(account_id)
            whitelist_target = (
                "Kite Connect developer console" if broker == "kite"
                else "SmartAPI portal" if broker == "angel"
                else "broker console"
            )
            return {
                "status": "ok",
                "proxy_ip": proxy_ip,
                "message": (
                    f"Proxy provisioned — whitelist IP {proxy_ip} in {whitelist_target}"
                    + (" and re-do OAuth login" if broker == "kite" else "")
                ),
            }
        except Exception:
            logger.exception("Failed to provision proxy")
            return {"status": "error", "message": "Failed to provision proxy"}

    @app.delete("/api/proxy/{account_id}")
    async def delete_proxy_endpoint(account_id: int) -> dict[str, Any]:
        from app.core.config import settings as _s
        if not _s.aws_access_key_id:
            return {"status": "error", "message": "AWS not configured"}
        try:
            from app.infra.proxy_manager import proxy_manager
            await asyncio.to_thread(proxy_manager.delete_proxy, account_id)
            async with AsyncSessionLocal() as s:
                async with s.begin():
                    row = await s.get(BrokerAccount, account_id)
                    if row is not None:
                        row.proxy_url = ""
                        row.proxy_ip = ""
                        row.proxy_instance_name = ""
            _invalidate_broker_cache_safe(account_id)
            return {"status": "ok", "message": "Proxy deleted"}
        except Exception:
            logger.exception("Failed to delete proxy")
            return {"status": "error", "message": "Failed to delete proxy"}

    # ---- StrategyInstance CRUD ------------------------------------------

    @app.get("/api/strategy-instances")
    async def list_strategy_instances() -> dict[str, Any]:
        async with AsyncSessionLocal() as s:
            rows = (
                await s.execute(
                    select(StrategyInstance).order_by(StrategyInstance.id.asc())
                )
            ).scalars().all()
            accounts = (
                await s.execute(select(BrokerAccount))
            ).scalars().all()
        acct_lookup = {a.id: a.name for a in accounts}
        return {
            "instances": [
                {
                    **_instance_to_dict(i),
                    "account_name": acct_lookup.get(i.account_id, "—"),
                }
                for i in rows
            ],
        }

    @app.post("/api/strategy-instances")
    async def create_strategy_instance(body: dict) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Body must be JSON object")
        clean = _normalize_instance_payload(body)

        async with AsyncSessionLocal() as s:
            async with s.begin():
                if clean["account_id"] is not None:
                    if await s.get(BrokerAccount, clean["account_id"]) is None:
                        raise HTTPException(status_code=400, detail="account_id does not exist")
                row = StrategyInstance(**clean)
                s.add(row)
                await s.flush()
            await s.refresh(row)
            _invalidate_settings_cache_safe()
            return _instance_to_dict(row)

    @app.put("/api/strategy-instances/{instance_id}")
    async def update_strategy_instance(instance_id: int, body: dict) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Body must be JSON object")

        async with AsyncSessionLocal() as s:
            async with s.begin():
                row = await s.get(StrategyInstance, instance_id)
                if row is None:
                    raise HTTPException(status_code=404, detail="Strategy instance not found")

                # Merge existing → payload → normalized
                merged = _instance_to_dict(row)
                merged.update({k: v for k, v in body.items() if k not in {"id", "created_at", "updated_at", "account_name"}})
                # If user switched index but did not explicitly provide
                # strike_interval, reset to the exchange step for that index.
                # Without this, a NIFTY interval (50) can leak into SENSEX
                # rows and produce wrong max-pain anchor offsets/orders.
                if "index" in body and "strike_interval" not in body:
                    target_idx = _clean_str(body.get("index"), merged.get("index", "NIFTY")).upper()
                    merged["strike_interval"] = _default_strike_interval_for_index(target_idx)
                clean = _normalize_instance_payload(merged)
                if clean["account_id"] is not None:
                    if await s.get(BrokerAccount, clean["account_id"]) is None:
                        raise HTTPException(status_code=400, detail="account_id does not exist")

                for k, v in clean.items():
                    setattr(row, k, v)
            await s.refresh(row)
            _invalidate_settings_cache_safe()
            return _instance_to_dict(row)

    @app.delete("/api/strategy-instances/{instance_id}")
    async def delete_strategy_instance(instance_id: int) -> dict[str, Any]:
        async with AsyncSessionLocal() as s:
            async with s.begin():
                row = await s.get(StrategyInstance, instance_id)
                if row is None:
                    raise HTTPException(status_code=404, detail="Strategy instance not found")
                await s.delete(row)
        _invalidate_settings_cache_safe()
        return {"ok": True, "id": instance_id}

    @app.post("/api/strategy-instances/{instance_id}/toggle")
    async def toggle_strategy_instance(instance_id: int, body: dict | None = None) -> dict[str, Any]:
        want = None if not isinstance(body, dict) else body.get("is_active")
        async with AsyncSessionLocal() as s:
            async with s.begin():
                row = await s.get(StrategyInstance, instance_id)
                if row is None:
                    raise HTTPException(status_code=404, detail="Strategy instance not found")
                row.is_active = bool(want) if want is not None else (not row.is_active)
            await s.refresh(row)
            _invalidate_settings_cache_safe()
            return _instance_to_dict(row)


# ── Helpers ──────────────────────────────────────────────────────────────

async def _clear_other_primaries(session, *, keep_id: int) -> None:
    others = (
        await session.execute(
            select(BrokerAccount).where(
                BrokerAccount.is_primary.is_(True),
                BrokerAccount.id != keep_id,
            )
        )
    ).scalars().all()
    for o in others:
        o.is_primary = False


async def _clear_other_data_feeds(session, *, keep_id: int) -> None:
    others = (
        await session.execute(
            select(BrokerAccount).where(
                BrokerAccount.is_data_feed.is_(True),
                BrokerAccount.id != keep_id,
            )
        )
    ).scalars().all()
    for o in others:
        o.is_data_feed = False


def _normalize_instance_payload(body: dict) -> dict[str, Any]:
    strategy_type = _validate_strategy_type(body.get("strategy_type", "ATM_STRADDLE"))

    idx = _clean_str(body.get("index"), "NIFTY").upper()
    if idx not in _VALID_INDICES:
        idx = "NIFTY"

    day = _clean_str(body.get("trading_day"), "Daily").title()
    if day not in _VALID_TRADING_DAYS:
        day = "Daily"

    strike_interval_default = _default_strike_interval_for_index(idx)

    strike_mode = _clean_str(body.get("strike_mode"), "ATM").upper()
    # Strategy-type canonical strike-mode mapping:
    # - ATM_STRADDLE  -> ATM
    # - OTM_STRANGLE  -> STRANGLE
    # - MAXPAIN_ROLL  -> MAXPAIN
    if strategy_type == "OTM_STRANGLE":
        strike_mode = "STRANGLE"
    elif strategy_type == "MAXPAIN_ROLL":
        strike_mode = "MAXPAIN"
    elif strategy_type == "ATM_STRADDLE":
        strike_mode = "ATM"
    if strike_mode not in _VALID_STRIKE_MODES:
        strike_mode = "ATM"

    hedge_mode = _clean_str(body.get("hedge_mode"), "none").lower()
    if hedge_mode not in _VALID_HEDGE_MODES:
        hedge_mode = "none"

    sl_type = _clean_str(body.get("sl_type"), "none").lower()
    if sl_type not in _VALID_SL_TYPES:
        sl_type = "none"

    account_id_raw = body.get("account_id")
    if account_id_raw in (None, "", "null"):
        account_id: int | None = None
    else:
        try:
            account_id = int(account_id_raw)
        except Exception:
            raise HTTPException(status_code=400, detail="account_id must be integer or null")

    params_json = body.get("params_json") or ""
    if isinstance(params_json, dict):
        params_json = json.dumps(params_json)
    else:
        params_json = _clean_str(params_json)

    return {
        "strategy_type": strategy_type,
        "account_id": account_id,
        "index": idx,
        "trading_day": day,
        "entry_time": _clean_time(body.get("entry_time"), "09:20"),
        "exit_time": _clean_time(body.get("exit_time"), "15:15"),
        "lots": _clean_int(body.get("lots"), 1, lo=1, hi=1000),
        "strike_interval": _clean_int(body.get("strike_interval"), strike_interval_default, lo=1, hi=10000),
        "strike_mode": strike_mode,
        "otm_strikes": _clean_int(body.get("otm_strikes"), 0, lo=0, hi=50),
        "static_legs": bool(body.get("static_legs", False)),
        "adjustment_points": _clean_int(body.get("adjustment_points"), 1, lo=0, hi=100000),
        "rolling_points": _clean_int(body.get("rolling_points"), 300, lo=1, hi=100000),
        "sl_type": sl_type,
        "sl_lower": _clean_float(body.get("sl_lower"), 0.0, lo=0.0),
        "sl_upper": _clean_float(body.get("sl_upper"), 0.0, lo=0.0),
        "sl_amount": _clean_float(body.get("sl_amount"), 0.0, lo=0.0),
        "first_straddle_sl_pct": _clean_int(body.get("first_straddle_sl_pct"), 100, lo=1, hi=1000),
        "reform_straddle_sl_pct": _clean_int(body.get("reform_straddle_sl_pct"), 60, lo=1, hi=1000),
        "hedge_mode": hedge_mode,
        "hedge_premium": _clean_float(body.get("hedge_premium"), 3.0, lo=0.0),
        "hedge_otm_points": _clean_int(body.get("hedge_otm_points"), 500, lo=0, hi=100000),
        "hedge_lots": _clean_int(body.get("hedge_lots"), 0, lo=0, hi=1000),
        "is_active": bool(body.get("is_active", True)),
        "live_execution": bool(body.get("live_execution", False)),
        "display_name": _clean_str(body.get("display_name")),
        "params_json": params_json or None,
    }


def _require_aws_configured() -> None:
    """Kept for internal helpers only — not raised from user-facing endpoints.
    Public /api/proxy/* endpoints follow OptionSelling's pattern of returning
    {'status': 'error', 'message': ...} instead of raising HTTPException.
    """
    from app.core.config import settings as _s
    if not (_s.aws_access_key_id and _s.aws_secret_access_key):
        raise RuntimeError("AWS credentials not configured")


async def _provision_proxy_background(account_id: int) -> None:
    """Provision a Lightsail SOCKS5 proxy for the given account and persist
    the resulting proxy_url / proxy_ip / proxy_instance_name back onto the
    BrokerAccount row. Safe to call from asyncio.create_task — catches
    everything so a failed provision never crashes the request handler.
    """
    try:
        from app.infra.proxy_manager import proxy_manager
        logger.info("[Proxy] Provisioning proxy for account %s ...", account_id)
        info = await asyncio.to_thread(proxy_manager.create_proxy, account_id)
        proxy_url = info.get("proxy_url") or ""
        public_ip = info.get("public_ip") or ""
        name = info.get("instance_name") or ""
        async with AsyncSessionLocal() as s:
            async with s.begin():
                row = await s.get(BrokerAccount, account_id)
                if row is None:
                    logger.warning("[Proxy] Account %s vanished before proxy could be saved", account_id)
                    return
                if proxy_url:
                    row.proxy_url = proxy_url
                if public_ip:
                    row.proxy_ip = public_ip
                if name:
                    row.proxy_instance_name = name
        _invalidate_broker_cache_safe(account_id)
        logger.info("[Proxy] Account %s provisioned -> ip=%s", account_id, public_ip)
    except Exception:
        logger.exception("[Proxy] Provisioning failed for account %s", account_id)


async def _probe_broker_account(creds: dict) -> tuple[str, str]:
    """Best-effort broker health probe using THIS account's own credentials.

    Unlike the legacy ``_probe_broker(broker)`` (which read process env
    vars and lied to the UI when the account row was missing creds),
    this uses the DB row's own client_id / access_token / proxy_url so
    the "Connected" pill in Settings actually reflects whether THIS
    account can trade. Never raises.
    """
    broker = (creds.get("broker") or "").lower()
    try:
        if broker == "angel":
            return await _probe_angel_account(creds)
        if broker == "kite":
            return await _probe_kite_account(creds)
        if broker == "dhan":
            return await _probe_dhan_account(creds)
    except Exception as exc:
        return "error", f"{type(exc).__name__}: {exc}"
    return "unknown", f"No probe implemented for broker '{broker}'"


async def _probe_broker(broker: str) -> tuple[str, str]:
    """Legacy env-based probe. Retained only so any external caller that
    imports this name keeps working. Prefer ``_probe_broker_account``.
    """
    try:
        if broker == "angel":
            return await _probe_angel()
        if broker == "kite":
            return await _probe_kite()
        if broker == "dhan":
            return await _probe_dhan()
    except Exception as exc:
        return "error", f"{type(exc).__name__}: {exc}"
    return "unknown", "No probe implemented"


async def _probe_angel_account(creds: dict) -> tuple[str, str]:
    api_key = creds.get("api_key") or ""
    client_id = creds.get("client_id") or ""
    credential = creds.get("mpin") or creds.get("password") or ""
    totp_secret = creds.get("totp_secret") or ""

    if not api_key:
        return "disconnected", "api_key not set on this account"
    if not client_id:
        return "disconnected", "client_id not set on this account"
    if not credential:
        return "disconnected", "mpin / password not set on this account"
    if not totp_secret:
        return "disconnected", "totp_secret not set on this account"

    try:
        import pyotp
        from SmartApi import SmartConnect

        def _sync_login() -> tuple[str, str]:
            smart_api = SmartConnect(api_key=api_key)
            totp = pyotp.TOTP(totp_secret).now()
            data = smart_api.generateSession(client_id, credential, totp)
            if not data or data.get("status") is False:
                msg = (data or {}).get("message", "Unknown error")
                code = (data or {}).get("errorcode", "")
                return "error", f"{msg} (code: {code})" if code else msg
            return "connected", "AngelOne authenticated OK"

        import asyncio
        return await asyncio.to_thread(_sync_login)
    except Exception as exc:
        return "error", f"{type(exc).__name__}: {exc}"


async def _probe_kite_account(creds: dict) -> tuple[str, str]:
    api_key = creds.get("api_key") or ""
    access_token = creds.get("access_token") or ""
    proxy_url = creds.get("proxy_url") or ""
    if not api_key:
        return "disconnected", "api_key not set on this account"
    if not access_token:
        return "disconnected", "access_token not set on this account (Kite OAuth required daily)"

    try:
        from app.data.kite_client import KiteClient
        client = KiteClient(
            api_key=api_key,
            access_token=access_token,
            proxy_url=proxy_url or None,
        )
        profile = None
        if hasattr(client, "get_profile"):
            profile = await client.get_profile() if _is_coro(client.get_profile) else _to_coro(client.get_profile)
        return "connected", f"Kite OK ({(profile or {}).get('user_id', 'unknown user')})"
    except Exception as exc:
        return "error", f"{type(exc).__name__}: {exc}"


async def _probe_dhan_account(creds: dict) -> tuple[str, str]:
    client_id = creds.get("client_id") or ""
    access_token = creds.get("access_token") or ""
    proxy_url = creds.get("proxy_url") or ""
    if not client_id:
        return "disconnected", "client_id not set on this account"
    if not access_token:
        return (
            "disconnected",
            "access_token not set on this account — paste it from "
            "web.dhan.co → Profile → DhanHQ Trading APIs (rotates periodically)",
        )

    try:
        from app.data.dhan_client import DhanClient
        client = DhanClient(
            client_id=client_id,
            access_token=access_token,
            proxy_url=proxy_url,
        )
        # get_fund_limits is the lightest available probe and matches what
        # DhanBroker.authenticate() uses, so this "Connected" verdict is
        # exactly what the trading path will see.
        import asyncio
        data = await asyncio.to_thread(client.get_fund_limits)
        if data:
            avail = data.get("availabelBalance") or data.get("availableBalance") or "?"
            return "connected", f"Dhan OK (available={avail})"
        return "error", "get_fund_limits returned empty — token likely expired or IP not whitelisted"
    except Exception as exc:
        return "error", f"{type(exc).__name__}: {exc}"


async def _probe_angel() -> tuple[str, str]:
    """Fresh AngelOne SmartAPI login using currently-configured creds.
    Mirrors POST /api/broker/test.
    """
    from app.core.config import settings

    api_key = settings.angelone_api_key
    client_id = settings.angelone_client_id
    credential = settings.angelone_mpin or settings.angelone_password
    totp_secret = settings.angelone_totp_secret

    if not api_key:
        return "disconnected", "ANGELONE_API_KEY not configured"
    if not client_id:
        return "disconnected", "ANGELONE_CLIENT_ID not configured"
    if not credential:
        return "disconnected", "ANGELONE_MPIN / ANGELONE_PASSWORD not configured"
    if not totp_secret:
        return "disconnected", "ANGELONE_TOTP_SECRET not configured"

    try:
        import pyotp
        from SmartApi import SmartConnect

        def _sync_login() -> tuple[str, str]:
            smart_api = SmartConnect(api_key=api_key)
            totp = pyotp.TOTP(totp_secret).now()
            data = smart_api.generateSession(client_id, credential, totp)
            if not data or data.get("status") is False:
                msg = (data or {}).get("message", "Unknown error")
                code = (data or {}).get("errorcode", "")
                return "error", f"{msg} (code: {code})" if code else msg
            return "connected", "AngelOne authenticated OK"

        import asyncio
        return await asyncio.to_thread(_sync_login)
    except Exception as exc:
        return "error", f"{type(exc).__name__}: {exc}"


async def _probe_kite() -> tuple[str, str]:
    """Verify Kite access token by fetching the profile."""
    from app.core.config import settings

    api_key = settings.kite_api_key
    access_token = settings.kite_access_token
    if not api_key:
        return "disconnected", "KITE_API_KEY not configured"
    if not access_token:
        return "disconnected", "KITE_ACCESS_TOKEN not set (login required)"

    try:
        from app.data.kite_client import KiteClient
        client = KiteClient(
            api_key=api_key,
            access_token=access_token,
            proxy_url=getattr(settings, "kite_proxy_url", None),
        )
        profile = None
        if hasattr(client, "get_profile"):
            profile = await client.get_profile() if _is_coro(client.get_profile) else _to_coro(client.get_profile)
        return "connected", f"Kite OK ({(profile or {}).get('user_id', 'unknown user')})"
    except Exception as exc:
        return "error", f"{type(exc).__name__}: {exc}"


async def _probe_dhan() -> tuple[str, str]:
    """Verify Dhan access token via a light fund-limit call."""
    from app.core.config import settings

    client_id = getattr(settings, "dhan_client_id", "") or ""
    access_token = getattr(settings, "dhan_access_token", "") or ""
    if not client_id:
        return "disconnected", "DHAN_CLIENT_ID not configured"
    if not access_token:
        return "disconnected", "DHAN_ACCESS_TOKEN not set (rotation required)"

    try:
        from app.data.dhan_client import DhanClient
        client = DhanClient(client_id=client_id, access_token=access_token)
        # Try the lightest probe available on this client class.
        for method_name in ("get_fund_limit", "get_profile", "get_positions"):
            fn = getattr(client, method_name, None)
            if fn is None:
                continue
            import inspect, asyncio
            if inspect.iscoroutinefunction(fn):
                result = await fn()
            else:
                result = await asyncio.to_thread(fn)
            if result:
                return "connected", f"Dhan OK ({method_name})"
        return "connected", "Dhan credentials accepted (no probe method available)"
    except Exception as exc:
        return "error", f"{type(exc).__name__}: {exc}"


def _is_coro(fn) -> bool:
    import inspect
    return inspect.iscoroutinefunction(fn)


async def _to_coro(fn):
    import asyncio
    return await asyncio.to_thread(fn)
